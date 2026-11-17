"""Exercise metric semantics independently of the heavyweight training imports."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.metric_reduction import reduce_training_metrics


class LocalDist:
    @staticmethod
    def is_available():
        return False


class RemoteDist:
    Reducing = None
    ReduceOp = dist.ReduceOp

    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 2

    def all_reduce(self, value, op):
        assert op == dist.ReduceOp.SUM
        assert value.shape == (4,)
        # Sorted average keys loss/num_custom, then sorted sample-count keys.
        value.add_(value.new_tensor([6.0, 10.0, 5.0, 7.0]))
        self.calls += 1


@pytest.mark.parametrize("reverse", [False, True])
def test_packed_average_count_and_dictionary_order(reverse):
    loss = torch.tensor(2.0, requires_grad=True)
    metrics = {"loss": loss, "num_custom": 4.0, "num_pol_samples": 3,
               "num_vm_samples": torch.tensor(1), "tag": "preserved"}
    if reverse:
        metrics = dict(reversed(list(metrics.items())))
    remote = RemoteDist()
    result = reduce_training_metrics(metrics, torch.device("cpu"), remote)
    assert result == {"loss": 4.0, "num_custom": 7.0, "num_pol_samples": 8.0,
                      "num_vm_samples": 8.0, "tag": "preserved"}
    assert remote.calls == 1
    assert loss.requires_grad and loss.grad is None
    assert loss.item() == 2.0


def test_local_dtype_precision_and_metadata():
    metrics = {"count": torch.tensor(2**54 + 1),
               "double": torch.tensor(1.0 + 2**-40, dtype=torch.float64),
               "bf16": torch.tensor(0.125, dtype=torch.bfloat16),
               "python": 1.0 + 2**-45, "tag": None}
    result = reduce_training_metrics(metrics, torch.device("cpu"), LocalDist)
    assert result == {"count": 2**54 + 1, "double": 1.0 + 2**-40,
                      "bf16": 0.125, "python": 1.0 + 2**-45, "tag": None}
    assert isinstance(result["count"], int)


def test_empty_and_nonscalar():
    assert reduce_training_metrics({}, "cpu", LocalDist) == {}
    with pytest.raises(ValueError, match="must be scalar"):
        reduce_training_metrics({"loss": torch.ones(2)}, "cpu", LocalDist)


def _gloo_worker(rank, rendezvous, result_dir):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        entries = [("loss", torch.tensor(rank * 2.0 + 1.0)),
                   ("num_pol_samples", 3 + rank), ("num_vm_samples", rank),
                   ("num_custom", 10.0 + rank * 2)]
        metrics = dict(entries if rank == 0 else reversed(entries))
        result = reduce_training_metrics(metrics, torch.device("cpu"))
        assert result == {"loss": 2.0, "num_pol_samples": 7.0,
                          "num_vm_samples": 1.0, "num_custom": 11.0}
        Path(result_dir, f"rank{rank}.ok").write_text("ok")
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo(tmp_path):
    mp.spawn(_gloo_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / "rank0.ok").exists() and (tmp_path / "rank1.ok").exists()


def _load_train_step():
    source = Path(__file__).resolve().parents[1] / "train/train.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "train_step")
    namespace = {"torch": torch, "Dict": dict, "Any": object, "SAMPLE_CLASS_KEY": "sample_class"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["train_step"]


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.75))

    def training_step(self, latents, actions, **kwargs):
        loss = ((latents.float() * self.weight - actions.float()) ** 2).mean()
        return {"total_loss": loss, "num_pol_samples": latents.numel()}


class Accumulation:
    def __init__(self, steps):
        self.steps, self.microstep, self.sync_gradients = steps, 0, False

    @contextmanager
    def accumulate(self, model):
        self.microstep += 1
        self.sync_gradients = self.microstep % self.steps == 0
        yield

    def backward(self, loss):
        (loss / self.steps).backward()


class BoundaryOptimizer:
    def __init__(self, optimizer, accelerator):
        self.optimizer, self.accelerator = optimizer, accelerator

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        if self.accelerator.sync_gradients:
            self.optimizer.step()

    def zero_grad(self):
        if self.accelerator.sync_gradients:
            self.optimizer.zero_grad()


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_actual_train_step_preserves_accumulation_updates(accumulation):
    method = _load_train_step()
    model, reference = ToyModel(), ToyModel()
    accelerator = Accumulation(accumulation)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    trainer = SimpleNamespace(
        model=model, accelerator=accelerator, optimizer=BoundaryOptimizer(optimizer, accelerator),
        config=SimpleNamespace(training=SimpleNamespace(grad_clip_norm=0.5)),
        device=torch.device("cpu"), dtype=torch.bfloat16, global_step=0, scheduler=None,
        _timing_active=False, _debug_step_enabled=lambda _: False,
    )
    last_metrics = None
    for microstep in range(accumulation * 2):
        batch = {"latents": torch.tensor([microstep + 1.0]), "actions": torch.tensor([0.25])}
        metrics, did_step, _ = method(trainer, batch)
        loss = reference.training_step(**batch)["total_loss"]
        (loss / accumulation).backward()
        if (microstep + 1) % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            trainer.global_step += 1
        assert did_step == ((microstep + 1) % accumulation == 0)
        # The kernel/helper port is independent of Trainer's reporting switch.
        # Both the scalar API and detached-Tensor API must retain
        # exactly the same accumulation and optimizer updates.
        reported = metrics["total_loss"]
        if torch.is_tensor(reported):
            assert reported.numel() == 1 and reported.grad_fn is None
        else:
            assert isinstance(reported, float)
        torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)
        last_metrics = metrics
    # Logging materialization remains possible after non-logging microsteps.
    logged = reduce_training_metrics(last_metrics, "cpu", LocalDist)
    assert isinstance(logged["total_loss"], float) and logged["num_pol_samples"] == 1
