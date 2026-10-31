"""Audited Swift/Qwen media path, direct conditional scoring, exact rollout tokens."""
import copy
import hashlib
import json
import numpy as np
from .reward_to_rl_common import visual_environment, read, PROJECT


def load_engine(path,adapter=None):
    visual_environment()
    from verify_checkpoint import load_engine as load
    return load(path,adapter)


def clone_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,dict):return {k:clone_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [clone_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(clone_tree(v) for v in value)
    return value


def prompt_encoded(engine,row):
    t=engine.template;t.model=engine.model;t.set_mode('transformers')
    return t.encode({k:copy.deepcopy(row[k]) for k in ('messages','images','videos')})


def media_hash(batch):
    from training_telemetry import tensor_media_hash
    return tensor_media_hash(batch)


def collate(engine,encoded):
    from swift.utils import to_device
    batch=to_device(clone_tree(engine.template.data_collator([encoded])),engine.model.device)
    _,batch=engine.template.pre_forward_hook(engine.model,None,batch)
    for key in ('labels','loss_scale','_extra_kwargs'):batch.pop(key,None)
    return batch


def encoding_audit(encoded):
    grids=encoded.get('video_grid_thw');ig=encoded.get('image_grid_thw')
    vg=[] if grids is None else grids.tolist();ii=[] if ig is None else ig.tolist()
    return {'prompt_tokens':len(encoded['input_ids']),'input_ids_sha256':hashlib.sha256(np.asarray(encoded['input_ids'],dtype=np.int64).tobytes()).hexdigest(),
        'image_grid_thw':ii,'video_grid_thw':vg,'image_visual_tokens':sum(int(np.prod(g))//4 for g in ii),
        'video_visual_tokens':sum(int(np.prod(g))//4 for g in vg),'processed_video_frames':sum(g[0]*2 for g in vg),
        'media_tensor_sha256':media_hash(encoded),'no_truncation':True}


def conditional_letters(engine,row,*,audit_context=False):
    import torch
    encoded=prompt_encoded(engine,row);prefix=list(encoded['input_ids']);t=engine.template
    audit=encoding_audit(encoded)  # Swift collation consumes grid fields in-place.
    if len(prefix)+4>8192:raise ValueError('Teacher context would truncate')
    tokens=[];context_checks=[]
    for letter in ('A','B','C'):
        if audit_context:
            t.set_mode('train');payload={k:copy.deepcopy(row[k]) for k in ('messages','images','videos')}
            payload['messages'].append({'role':'assistant','content':letter});complete=t.encode(payload)
            unmasked=[i for i,x in enumerate(complete['labels']) if x!=-100]
            start=unmasked[0];suffix=t._encode_context_list(t.template_meta.suffix)[0]
            answer=[complete['labels'][i] for i in unmasked]
            if suffix:
                if answer[-len(suffix):]!=suffix:raise ValueError('Teacher completion suffix changed')
                answer=answer[:-len(suffix)]
            if complete['input_ids'][:start]!=prefix:raise ValueError('A/B/C are not scored at the same actual template position')
            context_checks.append({'letter':letter,'answer_token_ids':answer,'prediction_position':start-1,
                'prefix_sha256':hashlib.sha256(np.asarray(prefix,dtype=np.int64).tobytes()).hexdigest()})
        else:answer=t.tokenizer.encode(letter,add_special_tokens=False)
        tokens.append(answer)
    t.set_mode('transformers');single=all(len(x)==1 for x in tokens)
    if not single:
        from verify_checkpoint import prepare_answer_batch,answer_logps
        scores=[]
        for letter,answer in zip(('A','B','C'),tokens):
            payload=copy.deepcopy(row);payload['messages'].append({'role':'assistant','content':letter})
            batch,labels=prepare_answer_batch(engine,payload)
            logps,_,_=answer_logps(engine.model,batch,labels)
            scores.append(float(logps[:len(answer)].sum()))
        return scores,{'single_token_same_position':False,'answer_token_ids':tokens,'context_checks':context_checks,
            'raw_candidate_probability_mass':None,'encoding':audit,'forward_calls':3}
    batch=collate(engine,encoded)
    engine.model.eval()
    with torch.no_grad():
        logits=engine.model(**batch,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float()
        logp=logits.log_softmax(-1);scores=[float(logp[x[0]]) for x in tokens]
    return scores,{'single_token_same_position':True,'answer_token_ids':tokens,'context_checks':context_checks,
        'raw_candidate_probability_mass':float(np.exp(scores).sum()),'encoding':audit,'forward_calls':1}


def generate(engine,row,*,greedy=False):
    from swift.infer_engine import InferRequest,RequestConfig
    previous=engine.model.config.use_cache
    previous_generation=getattr(engine.generation_config,'use_cache',True)
    engine.model.config.use_cache=True;engine.generation_config.use_cache=True
    try:
        response=engine.infer([InferRequest(messages=copy.deepcopy(row['messages']),images=row['images'],videos=row['videos'])],
            RequestConfig(max_tokens=128,temperature=0 if greedy else 1,top_p=1,top_k=0,repetition_penalty=1,return_details=True))[0]
    finally:
        engine.model.config.use_cache=previous;engine.generation_config.use_cache=previous_generation
    choice=response.choices[0]
    return {'response':choice.message.content,'token_ids':choice.token_ids,'prompt_token_ids':response.prompt_token_ids,
        'finish_reason':choice.finish_reason,'usage_prompt_tokens':response.usage.prompt_tokens,
        'usage_completion_tokens':response.usage.completion_tokens}


def completion_batch(engine,row,token_ids,expected_prompt=None):
    """Append the actual generated IDs, including only actually generated EOS."""
    encoded=prompt_encoded(engine,row);prompt=list(encoded['input_ids']);audit=encoding_audit(encoded)
    if expected_prompt is not None and prompt!=expected_prompt:raise ValueError('Rollout and consumed prompt token IDs differ')
    if not token_ids or len(prompt)+len(token_ids)>8192:raise ValueError('Missing/truncated consumed completion')
    encoded['input_ids']=prompt+list(token_ids)
    if 'attention_mask' in encoded:encoded['attention_mask']=[1]*len(encoded['input_ids'])
    encoded.pop('labels',None)
    return collate(engine,encoded),audit


def token_logps(engine,batch,token_ids,*,grad=False):
    import torch
    n=len(token_ids)
    with torch.enable_grad() if grad else torch.no_grad():
        outputs=engine.model(**batch,use_cache=False,logits_to_keep=n+1,return_dict=True)
        logits=outputs.logits[0,:n].float()
        target=torch.tensor(token_ids,device=logits.device,dtype=torch.long)
        result=logits.log_softmax(-1).gather(1,target[:,None]).squeeze(1)
    if result.shape!=(n,) or not torch.isfinite(result).all():raise ValueError('Nonfinite or misaligned completion logps')
    return result
