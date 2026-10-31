import json
import numpy as np
import pytest
import torch
from unsgrpo.reward_to_rl import *
from unsgrpo.core import reward_parts


def test_uniform_windows_and_three_regular_fixed_edge_graph():
    assert uniform_indices(60,6)==[5,15,25,35,45,55]
    assert uniform_indices(3,6)==[0,1,2]
    ee=selected_edges('video','window',list(range(8)))
    assert len(ee)==len({tuple(sorted(x)) for x in ee})==12
    assert all(sum(i in edge for edge in ee)==3 for i in range(8))
    assert ee==selected_edges('video','window',list(range(8)))


def test_swapped_identity_alignment_and_abstention():
    good=align_preference([3.,0.,-1.],[0.,3.,-1.])
    assert good['kept'] and good['preferred_first_clip'] and good['p1']==pytest.approx(good['p2'])
    assert good['weight']>.2
    assert not align_preference([3.,0.,-1.],[3.,0.,-1.])['kept']
    assert not align_preference([0.,0.,3.],[0.,0.,3.])['kept']
    assert not align_preference([.01,0.,-3.],[0.,.01,-3.])['kept']


def test_frozen_pair_reward_ties_empty_and_illegal():
    e=dict(align_preference([3,0,-1],[0,3,-1]),clip_i=0,clip_j=1)
    assert pair_reward('{"scores":[90,10]}',[0,1],[e])==1
    assert pair_reward('{"scores":[10,90]}',[0,1],[e])==0
    assert pair_reward('{"scores":[50,50]}',[0,1],[e])==.5
    assert pair_reward('{"scores":[90,10]}',[0,1],[])==.5
    assert pair_reward('bad',[0,1],[])==-1
    assert reliability('R_old',[0,1],[])==1
    assert reliability('R_pair',[0,1],[e])==pytest.approx(e['weight'])
    q=[.1,.9];text='{"scores":[10,90]}'
    assert rewards(text,q,[0,1],[e])['R_old']==reward_parts(text,q,.8).total


def test_actual_sequence_loss_and_u_survive_group_normalization():
    rr=[.45,.49,.5,.51];a,raw,std=group_advantages(rr,.4,True)
    assert std<.05 and torch.allclose(a,.4*(torch.tensor(rr)-np.mean(rr))/.0501,atol=1e-6)
    plain,_,_=group_advantages(rr,.4,False)
    assert torch.allclose(plain,raw)
    x=torch.tensor([.1,.2],requires_grad=True);z=torch.zeros_like(x)
    loss,d=sequence_grpo_loss(x,z,z,a[0],.4);loss.backward();g=x.grad.clone()
    x.grad.zero_();zero,_=sequence_grpo_loss(x,z,z,0.,0.);zero.backward()
    assert zero.item()==0 and torch.count_nonzero(x.grad)==0 and torch.count_nonzero(g)>0


def test_four_sequence_microbatch_loss_matches_registered_GRPO():
    from unsgrpo.round3_reward import grpo_loss
    rr=torch.tensor([.1,.2,.3,.4]);u=.6;adv,_,_=group_advantages(rr,u,True)
    x=torch.randn(4,5,requires_grad=True)*.1;old=torch.zeros_like(x);ref=torch.ones_like(x)*.02
    manual=sum(sequence_grpo_loss(x[i],old[i],ref[i],adv[i],u)[0] for i in range(4))/4
    expected,_=grpo_loss(x,old,ref,rr,torch.ones_like(x),[u],module='R1')
    assert torch.allclose(manual,expected,atol=1e-7)


def test_teacher_audit_survives_swift_destructive_collation(monkeypatch):
    from types import SimpleNamespace
    from unsgrpo import reward_to_rl_model as mod
    enc={'input_ids':[1,2,3],'image_grid_thw':torch.tensor([[1,2,2]]),
         'video_grid_thw':torch.tensor([[2,2,2],[2,2,2]]),'pixel_values':torch.zeros(1,4),
         'pixel_values_videos':torch.zeros(2,4)}
    monkeypatch.setattr(mod,'prompt_encoded',lambda engine,row:dict(enc))
    def consumes(engine,encoded):
        encoded.pop('image_grid_thw');encoded.pop('video_grid_thw');return {}
    monkeypatch.setattr(mod,'collate',consumes)
    class Model:
        def eval(self):pass
        def __call__(self,**kwargs):return SimpleNamespace(logits=torch.zeros(1,1,64))
    template=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda letter,**kwargs:[ord(letter)-33]),set_mode=lambda mode:None)
    scores,detail=mod.conditional_letters(SimpleNamespace(template=template,model=Model()),{},audit_context=False)
    assert detail['encoding']['processed_video_frames']==8
    assert detail['encoding']['image_visual_tokens']==1
    assert len(scores)==3 and detail['single_token_same_position']
