import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from wan.modules.model import rope_params


def test_wan_native_axis_spectrum():
    source = Path(__file__).resolve().parents[1] / 'third_party/wan/modules/model.py'
    tree = ast.parse(source.read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == 'freqs' for t in node.targets))
    model, dim = SimpleNamespace(), 128
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'wan-frequencies', 'exec'),
         {'self': model, 'd': dim, 'torch': torch, 'rope_params': rope_params})
    widths = [dim - 4 * (dim // 6), 2 * (dim // 6), 2 * (dim // 6)]
    for actual, width in zip(model.freqs.split([w // 2 for w in widths], dim=1), widths):
        torch.testing.assert_close(actual, rope_params(1024, width), rtol=0, atol=0)
