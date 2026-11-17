from collections import defaultdict
import torch

from utils.evaluation_metrics import append_metrics, finalize_metrics


def test_uneven_batches_weight_surviving_samples_and_ignore_empty_metrics():
    storage = defaultdict(list)
    append_metrics(storage, 'policy', {'mse': 1., '_count/mse': 4})
    append_metrics(storage, 'policy', {'mse': 9., '_count/mse': 1})
    append_metrics(storage, 'policy', {'mse': float('nan'), '_count/mse': 0})
    result = finalize_metrics(storage, 'cpu')
    assert result == {'policy_mse': 2.6, '_count/policy_mse': 5.}


def test_distributed_reduction_keeps_empty_rank_and_per_metric_counts():
    class Distributed:
        ReduceOp = torch.distributed.ReduceOp
        is_available = staticmethod(lambda: True)
        is_initialized = staticmethod(lambda: True)
        get_world_size = staticmethod(lambda: 2)
        @staticmethod
        def all_gather_object(result, keys):
            result[:] = [keys, ['policy_mse', 'wm_mse']]
        @staticmethod
        def all_reduce(totals, op):
            assert op == torch.distributed.ReduceOp.SUM
            totals += torch.tensor([[8., 2.], [15., 3.]], dtype=totals.dtype)
    assert finalize_metrics({}, 'cpu', Distributed) == {
        'policy_mse': 4., '_count/policy_mse': 2., 'wm_mse': 5., '_count/wm_mse': 3.}


def test_actual_evaluator_weights_uneven_batches_and_restores_mode(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    path = Path(__file__).resolve().parents[1] / 'train/sample.py'
    spec = importlib.util.spec_from_file_location('memory_sample_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(actions_per_latent=1, fixed_chunk_size=1)
            self.dtype = torch.float32
        def _obs_latents(self):
            return 1
        def _prepare_noisy_action_blocks(self, actions, states, chunk_size, **kwargs):
            return {'flat_actions': actions[:, chunk_size:]}

    def batch(size, error):
        return {'latents': torch.zeros(size, 1, 2, 1, 1),
                'actions': torch.zeros(size, 2, 2), 'valid_len': torch.full((size,), 2),
                'pred': torch.full((size, 1, 2), error)}
    monkeypatch.setattr(module, 'inference_sample', lambda model, b, config, **kw: (b['latents'][:, :, 1:], b['pred']))
    monkeypatch.setattr(module, 'create_video_grid', lambda *args, **kwargs: None)
    model = Model().eval()
    result = module.evaluate_model(model, [batch(4, 1.), batch(4, 1.), batch(1, 3.), batch(1, 3.)], None,
        OmegaConf.create({'model': {'wm_ratio': 0, 'value': {}}, 'system': {'val_batches_per_mode': 2}}))
    for mode in ['policy_decouple', 'policy_full']:
        assert result[f'{mode}_action_mse_loss'] == 2.6
        assert result[f'_count/{mode}_action_mse_loss'] == 5
    assert not model.training
