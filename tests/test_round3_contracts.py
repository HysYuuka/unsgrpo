import json
import numpy as np
import pytest
import torch
from unsgrpo.core import reward_parts
from unsgrpo.round3_reward import reward_r1, advantages, grpo_loss, evidence
from unsgrpo.round3_teacher import combine, parse_events, parse_semantic
from unsgrpo.round3_trainer_adapter import sft_target, completion_token_weights, weighted_sft_loss


def answer(x):return json.dumps({'scores':x})


def test_pair_anchor_correct_reverse_constant_and_positive_scale():
    q=[0,.3,.7,1];c=[1]*4
    assert reward_r1(answer([0,30,70,100]),q,c).total==1
    assert reward_r1(answer([100,70,30,0]),q,c).total==0
    r=reward_r1(answer([50]*4),q,c)
    assert r.pair==r.anchor==r.total==.5
    shrunk=reward_r1(answer([25,40,60,75]),q,c)
    assert shrunk.pair==1 and shrunk.anchor==.75
    assert reward_parts(answer([0,30,70,100]),q,.8).total==1


def test_equal_confidence_gap_has_larger_weight_at_high_q():
    low=evidence([.1,.3],[1,1])[4][0];high=evidence([.7,.9],[1,1])[4][0]
    assert high>low and high/low==pytest.approx(1.9/1.3)


def test_low_confidence_no_evidence_invalid_and_single_component():
    assert reward_r1(answer([0,100]),[0,1],[.25,.25]).u_g==0
    r=reward_r1(answer([10,90]),[.4,.6],[1,1]);assert r.pair==1 and r.anchor is None and r.total==1
    r=reward_r1(answer([90,90]),[.9,.9],[1,1]);assert r.pair is None and r.anchor==.9
    r=reward_r1(answer([0,100]),[.45,.55],[1,1]);assert r.u_g==0 and r.total==0
    assert reward_r1('bad',[0,1],[1,1]).total==-1
    assert reward_r1(answer([0,101]),[0,1],[1,1]).total==-1


def test_actual_actor_and_kl_weight_not_normalized_away():
    x=torch.tensor([[.1,.2]]*4,requires_grad=True);old=torch.zeros_like(x);ref=torch.zeros_like(x)
    rewards=torch.tensor([.2,.4,.6,.8]);mask=torch.ones_like(x)
    a,da=grpo_loss(x,old,ref,rewards,mask,[1.]);a.backward();g=x.grad.clone();x.grad.zero_()
    b,db=grpo_loss(x,old,ref,rewards,mask,[.25]);b.backward()
    assert b.item()==pytest.approx(a.item()*.25,abs=1e-7)
    assert torch.allclose(x.grad,g*.25)
    assert torch.allclose(db['advantages'],da['advantages']*.25)
    x.grad.zero_();z,dz=grpo_loss(x,old,ref,rewards,mask,[0]);z.backward()
    assert z.item()==0 and not torch.count_nonzero(x.grad) and not torch.count_nonzero(dz['KL_reliability'])


def test_sample_std_floor_detachment_and_R0_unchanged():
    r=torch.tensor([.5,.5001,.4999,.5],requires_grad=True)
    a,raw,std=advantages(r,[.75]);assert not a.requires_grad
    assert torch.allclose(a,.75*(r.detach()-r.detach().mean())/.0501)
    old,_,_=advantages(r,[0],module='R0')
    assert torch.allclose(old,(r.detach()-r.detach().mean())/(r.detach().std(unbiased=True)+1e-4))
    from training_telemetry import recompute_grpo_loss
    x=torch.tensor([[.1,.2],[.3,.2],[-.1,.4],[.2,-.2]],requires_grad=True)
    loss,_=grpo_loss(x,torch.zeros_like(x),torch.zeros_like(x),r.detach(),torch.ones_like(x),[0],module='R0')
    audit=recompute_grpo_loss(x.detach(),torch.zeros_like(x),torch.zeros_like(x),old[:,None],torch.ones_like(x),beta=.04,epsilon_low=.2,epsilon_high=.2)
    assert loss.item()==pytest.approx(audit['recomputed_loss'],abs=1e-7)


def test_teacher_strict_parsing_and_mixture_fallback():
    assert parse_events('{"events":[{"id":"E1","description":"visible object"}]}')['events'][0]['id']=='E1'
    with pytest.raises(ValueError):parse_events('{"events":[]}')
    good={'scores':[4,1,2],'event_ids':[['E1'],[],[]],'roles':['result','context','uncertain']}
    parse_semantic(json.dumps(good),3,['E1'])
    with pytest.raises(ValueError):parse_semantic(json.dumps(good),3,['E2'])
    a={'scores':[4,1,2],'roles':['result','context','uncertain']};b={'scores':[4,2,2],'roles':['result','process','context']}
    q,c,z,u,why=combine([.2,.4,.6],a,b,.75)
    assert np.allclose(z,[1,.75,0]) and np.allclose(c,[1,.8125,.25])
    assert q[0]==pytest.approx(.8) and q[2]==.6
    q,c,_,_,_=combine([.2,.4,.6],None,b,.5)
    assert np.allclose(q,[.2,.4,.6]) and np.allclose(c,.25)


def test_actual_local_tokenizer_numeric_binding_and_prompt_mask():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/root/autodl-fs/unsgrpo/models/Qwen3-VL-2B-Instruct',local_files_only=True,use_fast=True)
    q=np.array([0,.01,.1,.5,1,.5,.99,.07]);c=np.array([.25,.8125,1,1,.25,1,.8125,1])
    target=sft_target(q);ids,weights,audit=completion_token_weights(tok,target,q,c)
    # Qwen's digit tokens lie wholly within one numeric span: the intended exact rule.
    for (start,end),weight in zip(audit['token_offsets'],weights):
        text=target[start:end]
        if any(ch.isdigit() for ch in text):
            assert text.isdigit()
            which=next(i for i,(a,b) in enumerate(audit['numeric_character_spans']) if a<=start<end<=b)
            assert weight==pytest.approx(c[which]*(1+q[which]))
        else:assert weight==pytest.approx(.1)
    prompt=[23,24,25];labels=torch.tensor([[-100]*len(prompt)+ids]);ww=torch.tensor([[0.]*len(prompt)+weights])
    logits=torch.randn(1,len(prompt)+len(ids),max(ids)+1,requires_grad=True)
    loss=weighted_sft_loss(logits,labels,ww);loss.backward()
    assert torch.count_nonzero(logits.grad[:,:len(prompt)-1])==0
    assert torch.count_nonzero(logits.grad[:,len(prompt)-1:])>0
