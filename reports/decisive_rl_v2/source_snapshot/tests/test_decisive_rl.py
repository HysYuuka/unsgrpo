import math
import pytest
import torch
import numpy as np
from unsgrpo.decisive_core import *


def test_disjoint_pairs_are_deterministic_and_cover_eight():
    ee=pairs('video','window',list(range(8)))
    assert len(ee)==4 and sorted(x for edge in ee for x in edge)==list(range(8))
    assert ee==pairs('video','window',list(range(8)))


def test_abcd_identity_alignment_and_tie_unknown_distinction():
    good=align(np.log([.8,.1,.05,.05]),np.log([.1,.8,.05,.05]))
    assert good['kept'] and good['preferred_clip_i'] and good['weight']==pytest.approx(.7)
    assert good['total_variation']==0
    assert align(np.log([.8,.1,.05,.05]),np.log([.8,.1,.05,.05]))['reason']=='order_conflict'
    assert align(np.log([.05,.05,.85,.05]),np.log([.05,.05,.85,.05]))['kind']=='tie'
    assert align(np.log([.05,.05,.05,.85]),np.log([.05,.05,.05,.85]))['kind']=='unknown'
    assert not align(np.log([.8,.1,.05,.05])-3,np.log([.1,.8,.05,.05])-3)['kept']


def test_reward_and_reliability_no_missing_evidence_labels():
    e=dict(align(np.log([.8,.1,.05,.05]),np.log([.1,.8,.05,.05])),clip_i=0,clip_j=1)
    assert reward('{"scores":[90,10]}',[0,1],[e])==1
    assert reward('{"scores":[10,90]}',[0,1],[e])==0
    assert reward('{"scores":[50,50]}',[0,1],[e])==.5
    assert reward('{"scores":[50,50]}',[0,1],[])==.5
    assert reward('bad',[0,1],[])==-1
    assert evidence(list(range(8)),[e])['u']==pytest.approx(.7*2/8)
    assert not evidence(list(range(8)),[e])['sufficient']


def test_dpo_sums_actual_completion_tokens_and_detaches_reference():
    pos=torch.tensor([-.1,-.2,-.3],requires_grad=True);neg=torch.tensor([-.4,-.5],requires_grad=True)
    rp=torch.tensor([-.3,-.3,-.3],requires_grad=True);rn=torch.tensor([-.4,-.4],requires_grad=True)
    loss,logit=dpo_loss(pos,neg,rp,rn,.4)
    expected=.1*((-.6+.9)-(-.9+.8))
    assert float(logit)==pytest.approx(expected,abs=1e-7)
    assert float(loss.detach())==pytest.approx(.4*math.log1p(math.exp(-expected)),abs=1e-7)
    loss.backward();assert rp.grad is None and rn.grad is None
    assert torch.all(pos.grad<0) and torch.all(neg.grad>0)


def test_zero_u_has_zero_actual_dpo_gradient():
    pos=torch.randn(5,requires_grad=True);neg=torch.randn(4,requires_grad=True)
    loss,_=dpo_loss(pos,neg,torch.zeros(5),torch.zeros(4),0.)
    loss.backward();assert loss.item()==0 and torch.count_nonzero(pos.grad)==0 and torch.count_nonzero(neg.grad)==0


def test_guard_has_one_reliability_factor_and_sample_std_floor():
    rr=[.45,.49,.5,.51];a,raw,std=group_advantages(rr,.4,True)
    assert std<.05 and torch.allclose(a,.4*(torch.tensor(rr)-torch.tensor(rr).mean())/.0501,atol=1e-6)
    cur=torch.tensor([.1,.2],requires_grad=True);zero=torch.zeros(2)
    loss,parts=sequence_grpo_loss(cur,zero,zero,a[0],.4)
    assert parts['actual_KL_weight']==pytest.approx(.04*.4)
    loss.backward();assert torch.count_nonzero(cur.grad)>0


def test_new_admission_does_not_reintroduce_accuracy_gate():
    new={'G':.002,'selected_minus_greedy':.001,'accuracy':.1};old={'G':.001}
    assert reward_gate(new,old,{'sufficient_windows':36,'sources_with_two':8},6)[0]
    assert not reward_gate(new,old,{'sufficient_windows':35,'sources_with_two':8},6)[0]
