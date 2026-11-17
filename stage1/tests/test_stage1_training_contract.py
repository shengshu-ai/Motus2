"""Small real-update/checkpoint/attention regressions, with no large model weights."""
import ast
import copy
import contextlib
import importlib.util
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Dict, Optional

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils.deepspeed import DeepSpeedEngineWrapper

ROOT = Path(__file__).resolve().parents[1]


def load_file(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(cls, name, scope):
    t = ast.parse((ROOT / 'train/train_video_pretrain.py').read_text())
    c = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-training-method>', 'exec'), scope)
    return scope[name]


def scheduler_class(name):
    tree = ast.parse((ROOT / 'utils/scheduler.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    scope = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-schedulers>', 'exec'), scope)
    return scope[name]


@pytest.mark.parametrize('name', ['WarmupConstantScheduler', 'LambdaLinearScheduler'])
def test_save_resume_next_update_matches_uninterrupted(tmp_path, name):
    helpers = load_file('utils/video_pretrain_checkpoint.py', 'checkpoint_helpers')
    cls = scheduler_class(name)
    def build():
        p = nn.Parameter(torch.tensor(5.))
        opt = torch.optim.SGD([p], lr=.1, momentum=.9)
        sched = cls(opt, 10) if name.startswith('Warmup') else cls(opt, 10, 100)
        return p, opt, sched
    def update(p, opt, sched):
        (p.square()/2).backward(); opt.step(); opt.zero_grad(); sched.step()
    p, opt, sched = build()
    for _ in range(4): update(p, opt, sched)
    helpers.save_trainer_state(tmp_path, 4, sched)
    old_p = p.detach().clone(); optimizer_state = copy.deepcopy(opt.state_dict())
    update(p, opt, sched)
    p2, opt2, sched2 = build(); p2.data.copy_(old_p); opt2.load_state_dict(optimizer_state)
    state = helpers.read_trainer_state(tmp_path, sched2)
    helpers.restore_scheduler(sched2, state)
    update(p2, opt2, sched2)
    torch.testing.assert_close(p2, p, rtol=0, atol=0)
    assert sched2.step_count == sched.step_count == 5
    assert opt2.param_groups[0]['lr'] == opt.param_groups[0]['lr']


def test_legacy_restore_leaf_step_and_missing_resume(tmp_path):
    h = load_file('utils/video_pretrain_checkpoint.py', 'checkpoint_helpers_legacy')
    cls = scheduler_class('WarmupConstantScheduler')
    opt = torch.optim.SGD([nn.Parameter(torch.tensor(1.))], lr=.1)
    sched = cls(opt, 100)
    ckpt = tmp_path/'parent_step_999'/'checkpoint_step_40'; ckpt.mkdir(parents=True)
    state = h.read_trainer_state(ckpt, sched); h.restore_scheduler(sched, state)
    assert state['global_step'] == sched.step_count == 40
    assert opt.param_groups[0]['lr'] == pytest.approx(.04)
    with pytest.raises(FileNotFoundError):h.read_trainer_state(tmp_path/'absent',sched)
    renamed = tmp_path/'renamed'; renamed.mkdir()
    with pytest.raises(ValueError):h.read_trainer_state(renamed,sched)


def test_actual_trainer_resume_restores_metadata_after_directory_rename(tmp_path):
    h = load_file('utils/video_pretrain_checkpoint.py', 'checkpoint_helpers_trainer')
    cls = scheduler_class('WarmupConstantScheduler')
    opt = torch.optim.SGD([nn.Parameter(torch.tensor(1.))], lr=.1);sched=cls(opt,100)
    sched.step_count=25;h.save_trainer_state(tmp_path,25,sched)
    fake=SimpleNamespace(global_step=0,scheduler=sched,accelerator=SimpleNamespace(load_state=lambda p:None))
    fn=method('VideoPretrainTrainer','load_checkpoint',{'read_trainer_state':h.read_trainer_state,'restore_scheduler':h.restore_scheduler,'Path':Path,'logger':logging.getLogger('test')})
    fn(fake,str(tmp_path));assert fake.global_step==25
    with pytest.raises(FileNotFoundError):fn(fake,str(tmp_path/'missing'))


def test_clipping_routes_before_actual_deepspeed_update():
    h = load_file('utils/gradient_clipping.py','clip_helpers')
    fn = method('VideoPretrainTrainer','train_step',{'torch':torch,'clip_gradients':h.clip_gradients})
    class Tiny(nn.Module):
        def __init__(self): super().__init__();self.weight=nn.Parameter(torch.tensor(10.))
        def forward(self,**kwargs):return {'total_loss':self.weight.square()/2}
    model=Tiny();opt=torch.optim.SGD(model.parameters(),lr=1.)
    plugin=SimpleNamespace(deepspeed_config={})
    h.configure_gradient_clipping({'deepspeed_plugin':plugin},{'grad_clip_norm':.5})
    class Engine:
        def set_gradient_accumulation_boundary(self,is_boundary):pass
        def backward(self,loss,**kwargs):loss.backward()
        def step(self):
            torch.nn.utils.clip_grad_norm_(model.parameters(),plugin.deepspeed_config['gradient_clipping'])
            opt.step();opt.zero_grad()
    wrapper=DeepSpeedEngineWrapper(Engine())
    class Accel:
        sync_gradients=True;distributed_type='DEEPSPEED'
        def accumulate(self,*args):return contextlib.nullcontext()
        def backward(self,loss):wrapper.backward(loss)
        def clip_grad_norm_(self,*args,**kwargs):raise AssertionError('late clip')
    noop=SimpleNamespace(step=lambda:None,zero_grad=lambda:None)
    trainer=SimpleNamespace(model=model,accelerator=Accel(),optimizer=noop,scheduler=None,config=SimpleNamespace(training={'grad_clip_norm':.5}),device='cpu',dtype=torch.float32)
    fn(trainer,{'latents':torch.zeros(1)})
    assert model.weight.item()==pytest.approx(9.5)


def test_padding_is_removed_before_attention_and_cannot_affect_valid_loss():
    timestep=load_file('utils/ti2v_timestep.py','ti2v_helper')
    scope={'torch':torch,'F':F,'Dict':Dict,'Optional':Optional,'build_ti2v_per_token_timestep':timestep.build_ti2v_per_token_timestep}
    fn=method('VideoPretrainModel','training_step',scope)
    seen=[]
    class Wan:
        text_len=2;text_dim=2;patch_size=(1,2,2)
        def __call__(self,x,t,context,seq_len,y):
            seen.append(([v.shape[1] for v in x],seq_len))
            return [v + v.mean() for v in x]
    fake=SimpleNamespace(dtype=torch.float32,cond_frame_prob_multi=0.,cond_noise_prob=0.,cond_noise_std=0.,timestep_sample_method='uniform',video_model=SimpleNamespace(wan_model=Wan(),precision=torch.bfloat16),scheduler=SimpleNamespace(shift=1.,num_train_timesteps=1000,training_weight=lambda t:torch.ones_like(t)))
    latents=torch.randn(2,1,6,2,2);a=latents.clone();b=latents.clone()
    a[0,:,3:]=1000;b[0,:,3:]=-1000
    a[1,:,5:]=1000;b[1,:,5:]=-1000
    a.requires_grad_();b.requires_grad_()
    torch.manual_seed(42);la=fn(fake,a,valid_len=torch.tensor([3,5]))['total_loss'];la.backward()
    torch.manual_seed(42);lb=fn(fake,b,valid_len=torch.tensor([3,5]))['total_loss'];lb.backward()
    torch.testing.assert_close(la,lb,rtol=0,atol=0)
    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert torch.count_nonzero(a.grad[0,:,3:])==0
    assert seen==[([3,5],5),([3,5],5)]


def test_sdpa_valid_tokens_match_unpadded_attention_and_gradients():
    wan='third_party/wan' if (ROOT/'third_party/wan').exists() else 'bak/wan'
    h=load_file(wan+'/modules/attention.py','wan_attention_test')
    h.FLASH_ATTN_2_AVAILABLE=h.FLASH_ATTN_3_AVAILABLE=False
    torch.manual_seed(10)
    q=torch.randn(2,5,2,4,requires_grad=True);k=torch.randn(2,5,2,4,requires_grad=True);v=torch.randn(2,5,2,4,requires_grad=True)
    out=h.flash_attention(q,k,v,k_lens=torch.tensor([3,5]),dtype=torch.bfloat16)
    loss=out[0,:3].square().sum()+out[1].square().sum();loss.backward()
    grads=[a.grad.clone() for a in [q,k,v]]
    for a in [q,k,v]:a.grad=None
    refs=[h.flash_attention(q[i:i+1,:n],k[i:i+1,:n],v[i:i+1,:n],dtype=torch.bfloat16) for i,n in enumerate([3,5])]
    (refs[0].square().sum()+refs[1].square().sum()).backward()
    for i,n in enumerate([3,5]):torch.testing.assert_close(out[i,:n],refs[i][0],rtol=0,atol=0)
    for a,g in zip([q,k,v],grads):torch.testing.assert_close(a.grad,g,rtol=0,atol=0)
