"""Tests for clipping before the backend's optimizer update."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest

from utils.gradient_clipping import configure_gradient_clipping, clip_gradients


class GradientClippingTests(unittest.TestCase):
    def test_training_yaml_overrides_deepspeed_json_before_engine_creation(self):
        plugin = SimpleNamespace(deepspeed_config={"gradient_clipping": 0.0})
        kwargs = {"deepspeed_plugin": plugin}
        self.assertEqual(configure_gradient_clipping(kwargs, {"grad_clip_norm": 0.5}), 0.5)
        self.assertEqual(plugin.deepspeed_config["gradient_clipping"], 0.5)
        self.assertEqual(configure_gradient_clipping({}, {}), 1.0)
        configure_gradient_clipping(kwargs, {"grad_clip_norm": 0})
        self.assertEqual(plugin.deepspeed_config["gradient_clipping"], 0)

    def test_invalid_threshold_rejected_before_training(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure_gradient_clipping({}, {"grad_clip_norm": value})

    def test_backend_owns_clipping(self):
        backend = Enum("Backend", ["DEEPSPEED", "MULTI_GPU"])
        calls = []
        accelerator = SimpleNamespace(
            distributed_type=backend.DEEPSPEED,
            clip_grad_norm_=lambda params, **kw: calls.append((params, kw)),
        )
        clip_gradients(accelerator, ["parameter"], 0.5)
        self.assertEqual(calls, [])
        accelerator.distributed_type = backend.MULTI_GPU
        clip_gradients(accelerator, ["parameter"], 0.5)
        self.assertEqual(calls, [(["parameter"], {"max_norm": 0.5})])
        clip_gradients(accelerator, ["parameter"], 0)
        self.assertEqual(len(calls), 1)

    def test_trainer_configures_backend_before_accelerator(self):
        source = (Path(__file__).resolve().parents[1] / "train/train.py").read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        setup = [n.lineno for n in calls if isinstance(n.func, ast.Name)
                 and n.func.id == "configure_gradient_clipping"]
        construction = [n.lineno for n in calls if isinstance(n.func, ast.Name)
                        and n.func.id == "Accelerator"]
        self.assertEqual(len(setup), 1)
        self.assertEqual(len(construction), 1)
        self.assertLess(setup[0], construction[0])
        self.assertNotIn("torch.nn.utils.clip_grad_norm_", source)
        self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == "clip_gradients"
                            for n in calls))


if __name__ == "__main__":
    unittest.main()
