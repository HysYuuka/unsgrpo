"""Deterministic edge selection, frozen rewards and guarded update mathematics."""
import hashlib
import itertools
import numpy as np
from .core import parse_scores, reward_parts

QUESTION=("Given the whole-video overview and the two clips, which clip contributes more important information about the video's main events? A means the first clip, B the second clip, and C means similarly important or insufficient evidence. Motion alone does not imply importance; a static key result may matter. Answer one letter only.")


def uniform_indices(n,count):
    if n<1 or count<1:raise ValueError('Nonempty pool required')
    return list(range(n)) if n<count else [int(np.floor((k+.5)*n/count)) for k in range(count)]


def selected_edges(video_id,window_id,clip_ids):
    seed=int.from_bytes(hashlib.sha256(f'2718|{video_id}|{window_id}'.encode()).digest()[:8],'big')
    order=np.random.default_rng(seed).permutation(list(clip_ids)).tolist();n=len(order)
    if n==8:return [(order[i],order[(i+1)%8]) for i in range(8)]+[(order[i],order[i+4]) for i in range(4)]
    return list(itertools.combinations(order,2))[:12]


def softmax(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite conditional scores')
    e=np.exp(x-x.max());return e/e.sum()


def align_preference(first,swapped):
    a,b=np.asarray(first,dtype=float),np.asarray(swapped,dtype=float)
    if a.shape!=(3,) or b.shape!=(3,):raise ValueError('A/B/C scores required')
    aa,bb=softmax(a),softmax(b)
    p1=float(softmax(a[:2])[0]);p2=float(softmax(b[:2])[1]);p=(p1+p2)/2;t=float((aa[2]+bb[2])/2)
    w=(1-t)*(1-abs(p1-p2))*2*abs(p-.5)
    reason=None
    if aa[2]>=max(aa[:2]) or bb[2]>=max(bb[:2]):reason='C_highest'
    elif a[0]==a[1] or b[0]==b[1]:reason='A_B_tie'
    elif int(np.argmax(a[:2]))==int(np.argmax(b[:2])):reason='opposite_actual_clip_after_swap'
    elif w<.2:reason='weight_below_0.2'
    return {'p1':p1,'p2':p2,'p':p,'C_mean':t,'weight':float(w),
        'kept':reason is None,'abstention_reason':reason,'preferred_first_clip':p>.5,
        'normalized_first':aa.tolist(),'normalized_swapped':bb.tolist()}


def pair_reward(text,clip_ids,edges):
    try:s=parse_scores(text,len(clip_ids))
    except (ValueError,TypeError):return -1.
    use=[e for e in edges if e['kept']]
    if not use:return .5
    ix={k:j for j,k in enumerate(clip_ids)};num=0.;den=0.
    for e in use:
        d=s[ix[e['clip_i']]]-s[ix[e['clip_j']]]
        credit=.5 if d==0 else float((d>0)==e['preferred_first_clip'])
        num+=e['weight']*credit;den+=e['weight']
    return float(num/den)


def rewards(text,q0,clip_ids,edges):
    return {'R_old':reward_parts(text,q0,.8).total,'R_pair':pair_reward(text,clip_ids,edges)}


def reliability(reward_name,clip_ids,edges):
    if reward_name=='R_old':return 1.
    if reward_name!='R_pair':raise ValueError('Unknown selected reward')
    use=[e for e in edges if e['kept']]
    covered={c for e in use for c in (e['clip_i'],e['clip_j'])}
    return float(np.mean([e['weight'] for e in use])*len(covered)/len(clip_ids)) if use else 0.


def group_advantages(reward_values,u,guard):
    import torch
    r=torch.as_tensor(reward_values,dtype=torch.float32).detach()
    if r.ndim!=1 or len(r)!=4 or not torch.isfinite(r).all() or not 0<=u<=1:raise ValueError('Four finite rewards and bounded u required')
    std=r.std(unbiased=True);raw=(r-r.mean())/(std+1e-4)
    actual=u*(r-r.mean())/(std.clamp_min(.05)+1e-4) if guard else raw
    return actual.detach(),raw.detach(),float(std)


def sequence_grpo_loss(current,old,reference,advantage,u):
    """One completion mean; caller divides by the fixed 16 completions/update."""
    import torch
    if current.ndim!=1 or current.shape!=old.shape or current.shape!=reference.shape or not len(current):raise ValueError('Aligned completion logps required')
    if not all(bool(torch.isfinite(x).all()) for x in (current,old,reference)):raise ValueError('Nonfinite logps')
    adv=torch.as_tensor(advantage,device=current.device,dtype=torch.float32).detach()
    delta=current.float()-old.detach().float();ratio=delta.exp()
    actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
    dref=(reference.detach().float()-current.float()).clamp(-20,20)
    kl=(dref.exp()-dref-1).clamp(-10,10)
    loss=actor.mean()+.04*float(u)*kl.mean()
    return loss,{'actor':float(actor.detach().mean()),'KL':float(kl.detach().mean()),
        'actual_KL_weight':.04*float(u),'actual_loss':float(loss.detach()),
        'ratio_clipped_tokens':int((ratio!=ratio.clamp(.8,1.2)).sum())}
