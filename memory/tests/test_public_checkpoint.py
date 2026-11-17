"""Standalone checkpoint saving and recovery."""
import ast
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace


def test_accelerate_checkpoint_roundtrip_ignores_external_environment(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "train/train.py"
    tree = ast.parse(source.read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    methods = [n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in {"save_checkpoint", "load_checkpoint", "_restore_scheduler_step"}]
    namespace = dict(os=os, re=re, Path=Path, logger=logging.getLogger(__name__))
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    calls = []
    def save(path):
        calls.append(("save", path))
        Path(path).mkdir()
    scheduler = SimpleNamespace(set_step=lambda step: calls.append(("scheduler", step)), get_last_lr=lambda: [1e-4])
    model = SimpleNamespace(checkpoint_dir=tmp_path, global_step=17, scheduler=scheduler,
        accelerator=SimpleNamespace(wait_for_everyone=lambda: None, save_state=save,
                                    load_state=lambda path: calls.append(("load", path))))
    model._restore_scheduler_step = lambda: namespace["_restore_scheduler_step"](model)
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "external-run"))
    monkeypatch.setenv("MEMORY_ROOT", "/nonexistent-external-workspace")
    namespace["save_checkpoint"](model)
    checkpoint = str(tmp_path / "checkpoint_step_17")
    model.global_step = 0
    namespace["load_checkpoint"](model, checkpoint)
    assert model.global_step == 17
    assert calls == [("save", checkpoint), ("load", checkpoint), ("scheduler", 17)]


def test_public_loader_imports_and_model_config():
    import importlib
    from models.motus import MotusConfig
    for name in ("data.robotwin2.robotwin_agilex_dataset", "data.tianji_wuji.tianji_wuji_dataset"):
        importlib.import_module(name)
    assert "use_mot" not in MotusConfig.__dataclass_fields__
    assert "video_rope_mode" not in MotusConfig.__dataclass_fields__
