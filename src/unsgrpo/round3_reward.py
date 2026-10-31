"""R1 reward and actual differentiable GRPO loss; R0 remains in core unchanged."""
from dataclasses import dataclass, asdict
import numpy as np
from .core import parse_scores, reward_parts


@dataclass(frozen=True)
class Parts:
    total: float
    pair: float | None
    anchor: float | None
    u_g: float
    valid: bool
    participating_clips: tuple
    pair_count: int


def evidence(q,c):
    q,c=np.asarray(q,dtype=float),np.asarray(c,dtype=float)
    if q.ndim!=1 or q.shape!=c.shape or not len(q) or not np.isfinite(q).all() or not np.isfinite(c).all() or np.any((q<0)|(q>1)|(c<0)|(c>1)):
        raise ValueError('Frozen q/c must be aligned finite vectors in [0,1]')
    ii,jj=np.triu_indices(len(q),1)
    use=(c[ii]>=.5)&(c[jj]>=.5)&(np.abs(q[ii]-q[jj])>=.15)
    ii,jj=ii[use],jj[use]
    weights=c[ii]*c[jj]*np.abs(q[ii]-q[jj])*(1+np.maximum(q[ii],q[jj]))
    high=np.flatnonzero((q>=.8)&(c>=.5));low=np.flatnonzero((q<=.2)&(c>=.5))
    active=np.unique(np.r_[ii,jj,high,low]).astype(int)
    u=float(c[active].mean()) if len(active) else 0.
    return q,c,ii,jj,weights,high,low,active,u


def reward_r1(completion,q,c):
    q,c,ii,jj,w,high,low,active,u=evidence(q,c)
    try: s=parse_scores(completion,len(q))
    except (ValueError,TypeError): return Parts(-1.,None,None,u,False,tuple(active.tolist()),len(ii))
    pair=None;anchor=None
    if len(ii):
        credit=((s[ii]-s[jj])*(q[ii]-q[jj])>0).astype(float)+.5*(s[ii]==s[jj])
        pair=float(np.dot(w,credit)/w.sum())
    av=[]
    if len(high): av.append(float(np.average(s[high],weights=c[high])))
    if len(low): av.append(float(np.average(1-s[low],weights=c[low])))
    if av:anchor=float(np.mean(av))
    total=(.8*pair+.2*anchor if pair is not None and anchor is not None else
           pair if pair is not None else anchor if anchor is not None else 0.)
    return Parts(float(total),pair,anchor,u,True,tuple(active.tolist()),len(ii))


def advantages(rewards,u,group_size=4,module='R1'):
    import torch
    if module not in ('R0','R1'):raise ValueError('Unknown reward/update module')
    rr=rewards.detach().float().reshape(-1,group_size)
    uu=torch.as_tensor(u,device=rr.device,dtype=torch.float32).detach().reshape(-1,1)
    if len(uu)!=len(rr) or not torch.isfinite(rr).all() or not torch.isfinite(uu).all() or torch.any((uu<0)|(uu>1)):
        raise ValueError('Invalid group rewards or reliability')
    std=rr.std(-1,unbiased=True,keepdim=True)
    centered=rr-rr.mean(-1,keepdim=True)
    raw=centered/(std+1e-4)
    actual=uu*centered/(std.clamp_min(.05)+1e-4) if module=='R1' else raw
    return actual.flatten().detach(),raw.flatten().detach(),std.flatten().detach()


def grpo_loss(current,old,reference,rewards,mask,u,*,module='R1',group_size=4,beta=.04,epsilon=.2):
    """Same masked sequence mean as audited Swift, with u applied to actor AND KL."""
    import torch
    if current.ndim!=2 or current.shape!=old.shape or current.shape!=reference.shape or current.shape!=mask.shape:
        raise ValueError('Aligned [completion, token] tensors required')
    if len(current)%group_size or rewards.numel()!=len(current):raise ValueError('Misbound reward groups')
    if not all(bool(torch.isfinite(t).all()) for t in (current,old,reference,mask)) or not torch.all((mask==0)|(mask==1)) or torch.any(mask.sum(-1)<=0):
        raise ValueError('Invalid log probabilities or completion mask')
    adv,raw,std=advantages(rewards,u,group_size,module)
    reliability=torch.as_tensor(u,device=current.device,dtype=torch.float32).detach().repeat_interleave(group_size)
    if module=='R0':reliability=torch.ones_like(reliability)
    ratio=(current.float()-old.detach().float()).exp()
    actor=-torch.minimum(ratio*adv[:,None],ratio.clamp(1-epsilon,1+epsilon)*adv[:,None])
    # Preserve the recorded Swift K3 guards; the reliability factor is new.
    delta=(reference.detach().float()-current.float()).clamp(-20,20)
    kl=(torch.exp(delta)-delta-1).clamp(-10,10)
    actor_seq=(actor*mask).sum(-1)/mask.sum(-1)
    kl_seq=(kl*mask).sum(-1)/mask.sum(-1)
    loss=(actor_seq+beta*reliability*kl_seq).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actual GRPO loss')
    return loss,{'advantages':adv,'raw_R0_advantages':raw,'sample_reward_std':std,
        'actor_sequence_loss':actor_seq.detach(),'KL_sequence':kl_seq.detach(),
        'KL_reliability':reliability,'actual_sequence_loss':(actor_seq+beta*reliability*kl_seq).detach()}
