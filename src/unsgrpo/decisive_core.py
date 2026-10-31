"""Frozen four-way teacher alignment, sparse reward, and summed-token DPO."""
import hashlib
import numpy as np
from .core import parse_scores
from .reward_to_rl import group_advantages,sequence_grpo_loss

QUESTION=("Given the whole-video overview and the two clips, which clip would you retain in a short highlight reel of this video? Judge the significance and viewing value of the visible moment in this video's context. Motion, rarity, static appearance, or repetition alone does not determine value. A: the first clip is more valuable. B: the second clip is more valuable. C: their highlight value is similar. D: the visual evidence is insufficient to decide. Answer with one letter only.")


def pairs(video_id,request_id,clip_ids):
    seed=int.from_bytes(hashlib.sha256(f'2718|{video_id}|{request_id}'.encode()).digest()[:8],'big')
    order=np.random.default_rng(seed).permutation(clip_ids).tolist()
    if len(order)!=8:raise ValueError('This fixed experiment requires eight-clip windows')
    return [(order[i],order[i+1]) for i in (0,2,4,6)]


def probabilities(x):
    x=np.asarray(x,dtype=float)
    if x.shape!=(4,) or not np.isfinite(x).all():raise ValueError('Four finite log probabilities required')
    a=np.exp(x-x.max());return a/a.sum()


def unique_top(x):
    x=np.asarray(x);return int(np.argmax(x)) if np.sum(x==x.max())==1 else None


def align(first,second):
    p=probabilities(first);r=probabilities(second);q=r[[1,0,2,3]]
    bar=(p+q)/2;tv=float(np.abs(p-q).sum()/2);a=unique_top(p);b=unique_top(q)
    mass_first=float(np.exp(first).sum());mass_second=float(np.exp(second).sum())
    direction=False;reason=None;kind='unknown';margin=float(abs(bar[0]-bar[1]));w=float((1-tv)*margin)
    if a==b==2:kind='tie';reason='both_C'
    elif a==3 or b==3:reason='insufficient_D'
    elif a not in (0,1) or b not in (0,1):reason='non_directional_or_exact_top_tie'
    elif a!=b:reason='order_conflict'
    elif margin<.10:reason='margin_below_0.10'
    elif min(mass_first,mass_second)<.5:reason='candidate_mass_below_0.5'
    else:direction=True;kind='directional'
    return {'kind':kind,'kept':direction,'reason':reason,'weight':w,'margin':margin,'total_variation':tv,
        'preferred_clip_i':bool(bar[0]>bar[1]),'p':p.tolist(),'q_identity_aligned':q.tolist(),'mean_probabilities':bar.tolist(),
        'raw_mass_first':mass_first,'raw_mass_second':mass_second,'top_first':a,'top_swapped_aligned':b}


def evidence(clip_ids,edges):
    ee=[e for e in edges if e['kept']];covered={i for e in ee for i in (e['clip_i'],e['clip_j'])}
    u=float(np.mean([e['weight'] for e in ee])*len(covered)/len(clip_ids)) if ee else 0.
    return {'n_edges':len(ee),'n_covered_clips':len(covered),'sufficient':len(ee)>=2 and len(covered)>=4,'u':u}


def reward(text,clip_ids,edges):
    try:s=parse_scores(text,len(clip_ids))
    except (ValueError,TypeError):return -1.
    ee=[e for e in edges if e['kept']]
    if not ee:return .5
    ix={k:j for j,k in enumerate(clip_ids)};total=0.;mass=0.
    for e in ee:
        d=s[ix[e['clip_i']]]-s[ix[e['clip_j']]];credit=.5 if d==0 else float((d>0)==e['preferred_clip_i'])
        total+=credit*e['weight'];mass+=e['weight']
    return float(total/mass)


def dpo_loss(chosen,rejected,ref_chosen,ref_rejected,u,beta=.1):
    import torch.nn.functional as F
    if not 0<=u<=1:raise ValueError('Invalid reliability')
    if any(x.ndim!=1 or not len(x) for x in (chosen,rejected,ref_chosen,ref_rejected)):raise ValueError('Full completion logp vectors required')
    logit=beta*((chosen.float().sum()-ref_chosen.detach().float().sum())-(rejected.float().sum()-ref_rejected.detach().float().sum()))
    return -float(u)*F.logsigmoid(logit),logit.detach()


def reward_gate(new,old,coverage,positive_sources):
    tests={'at_least_36_sufficient_windows':coverage['sufficient_windows']>=36,
        'at_least_8_sources_each_with_2_windows':coverage['sources_with_two']>=8,
        'G_at_least_0.15pp':new['G']>=.0015,'G_lift_over_old_at_least_0.03pp':new['G']-old['G']>=.0003,
        'selection_not_below_greedy':new['selected_minus_greedy']>=0,'at_least_6_positive_sources':positive_sources>=6}
    return all(tests.values()),tests
