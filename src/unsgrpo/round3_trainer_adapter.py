"""New objective adapters, without mutation of the frozen round-one controller."""
import json
import re
import numpy as np


def sft_target(q):
    arr=np.asarray(q,dtype=float)
    if arr.ndim!=1 or not len(arr) or not np.isfinite(arr).all() or np.any((arr<0)|(arr>1)):
        raise ValueError('Invalid SFT targets')
    return json.dumps({'scores':np.rint(100*arr).astype(int).tolist()},separators=(',',':'))


def completion_token_weights(tokenizer,target,q,c):
    """Character offsets, including any mixed numeric/structural token spans.

    A mixed token receives the length-weighted character weights; no digit is
    assigned by searching for its value (duplicates are legal). Prompt weights
    are supplied separately as zeros by the actual collator/loss path.
    """
    q,c=np.asarray(q,dtype=float),np.asarray(c,dtype=float)
    if q.shape!=c.shape or target!=sft_target(q) or np.any((c<0)|(c>1)) or not np.isfinite(c).all():
        raise ValueError('Target text or confidence mismatch')
    spans=list(re.finditer(r'\d+',target))
    if len(spans)!=len(q):raise ValueError('Numeric span binding failed')
    chars=np.full(len(target),.1)
    for i,m in enumerate(spans):chars[m.start():m.end()]=c[i]*(1+q[i])
    encoded=tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']; offsets=encoded['offset_mapping']; weights=[]
    for start,end in offsets:
        if not 0<=start<end<=len(target):raise ValueError('Unusable tokenizer offset')
        weights.append(float(chars[start:end].mean()))
    if tokenizer.decode(ids,skip_special_tokens=False)!=target:raise ValueError('Tokenizer target round trip differs')
    return ids,weights,{'numeric_character_spans':[(m.start(),m.end()) for m in spans],
        'token_offsets':offsets,'numeric_weights':(c*(1+q)).tolist(),
        'mixed_token_rule':'mean of per-character weights; no prompt or special token included'}


def weighted_sft_loss(logits,labels,weights):
    import torch
    import torch.nn.functional as F
    if labels.shape!=weights.shape or logits.shape[:2]!=labels.shape:raise ValueError('SFT loss shape mismatch')
    target=labels[:,1:]; ww=weights[:,1:].float()*(target!=-100)
    ce=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
    count=ww.sum(-1)
    if torch.any(count<=0):raise ValueError('No weighted completion tokens')
    return ((ce*ww).sum(-1)/count).mean()
