import ast
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import GradientAccumulationPlugin

ROOT=Path(__file__).resolve().parents[1]


def scope_with_classes(path):
    tree=ast.parse(path.read_text());scope={'torch':torch}
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=classes,type_ignores=[]),'actual-schedulers','exec'),scope)
    return scope


def trainer_type():
    tree=ast.parse((ROOT/'train/train.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
    names={'load_checkpoint','_restore_scheduler_step'}
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'os':os,'re':re,'Path':Path,'logger':logging.getLogger('test'),'_formal_run_context':lambda:None}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual-resume','exec'),scope)
    return scope['Trainer']


def test_resume_missing_path_fails(tmp_path):
    trainer=object.__new__(trainer_type());trainer.global_step=0
    trainer.accelerator=SimpleNamespace(load_state=lambda p:pytest.fail('load must not run'))
    with pytest.raises(FileNotFoundError):trainer.load_checkpoint(str(tmp_path/'missing'))


@pytest.mark.parametrize('scheduler_name',['LambdaLinearScheduler','LambdaConstantScheduler'])
def test_resume_next_lr_matches_uninterrupted_and_ignores_parent_step(tmp_path,scheduler_name):
    scope=scope_with_classes(ROOT/'utils/scheduler.py')
    if scheduler_name not in scope:pytest.skip('scheduler not offered by this package')
    cls=scope[scheduler_name]
    def build():
        opt=torch.optim.SGD([nn.Parameter(torch.tensor(1.))],lr=.1)
        sched=cls(opt,10,100) if scheduler_name=='LambdaLinearScheduler' else cls(opt,10)
        return opt,sched
    ref_opt,ref=build()
    for _ in range(40):ref.step()
    opt,sched=build();trainer=object.__new__(trainer_type())
    trainer.global_step=0;trainer.scheduler=sched
    trainer.accelerator=SimpleNamespace(load_state=lambda p:None)
    trainer.train_dataloader=SimpleNamespace()
    path=tmp_path/'old_step_999'/'checkpoint_step_40';path.mkdir(parents=True)
    trainer.load_checkpoint(str(path))
    assert trainer.global_step==40
    assert opt.param_groups[0]['lr']==ref_opt.param_groups[0]['lr']
    sched.step();ref.step()
    assert opt.param_groups[0]['lr']==ref_opt.param_groups[0]['lr']


def test_accumulation_crosses_epoch_without_partial_update():
    tree=ast.parse((ROOT/'train/train.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignment=next((n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='grad_accum_plugin' for t in n.targets) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='GradientAccumulationPlugin'),None)
    assert assignment is not None
    env={'GradientAccumulationPlugin':GradientAccumulationPlugin,'grad_accum_steps':2,'zero_stage':2}
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-main-plugin','exec'),env)
    assert env['grad_accum_plugin'].sync_with_dataloader is False
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        acc=Accelerator(cpu=True,gradient_accumulation_plugin=env['grad_accum_plugin'])
        model=nn.Linear(1,1,bias=False);model.weight.data.fill_(1.)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        model,opt,loader=acc.prepare(model,opt,DataLoader(torch.ones(3,1),batch_size=1))
        flags=[]
        for _ in range(2):
            for batch in loader:
                with acc.accumulate(model):
                    acc.backward(model(batch).sum());flags.append(acc.sync_gradients)
                    opt.step();opt.zero_grad()
        assert flags==[False,True,False,True,False,True]
        assert model.weight.item()==pytest.approx(.7)
    finally:AcceleratorState._reset_state(reset_partial_state=True)
