#!/usr/bin/env python3
"""The single frozen 8B configuration, four-way conditional scores and exact cache."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from unsgrpo.decisive_common import *
from unsgrpo.decisive_core import *
from unsgrpo.reward_to_rl_model import load_engine,prompt_encoded,encoding_audit,collate
from unsgrpo.data import canonical_hash


def payload(by,k,i,j):
    r=by[k]
    return {'images':[r['overview']],'videos':[r['frame_paths'][i],r['frame_paths'][j]],
        'messages':[{'role':'user','content':'<image>\nChronological overview of the complete video.\n'+
          f'Clip A (ID {i}), original time [{r["bounds"][i][0]:.6f}, {r["bounds"][i][1]:.6f}) seconds, four sampled frames:\n<video>\n'+
          f'Clip B (ID {j}), original time [{r["bounds"][j][0]:.6f}, {r["bounds"][j][1]:.6f}) seconds, four sampled frames:\n<video>\n'+QUESTION}]}


def score(engine,row):
    import torch
    enc=prompt_encoded(engine,row);audit=encoding_audit(enc);prefix=list(enc['input_ids']);tokens=[];context=[];t=engine.template
    if audit['processed_video_frames']!=8 or len(prefix)+8>8192:raise ValueError('Invalid teacher media or truncation')
    for letter in 'ABCD':
        t.set_mode('train');r=copy.deepcopy(row);r['messages'].append({'role':'assistant','content':letter});complete=t.encode(r)
        unmasked=[i for i,x in enumerate(complete['labels']) if x!=-100];start=unmasked[0];answer=[complete['labels'][i] for i in unmasked];suffix=t._encode_context_list(t.template_meta.suffix)[0]
        if suffix:
            if answer[-len(suffix):]!=suffix:raise ValueError('Changed suffix')
            answer=answer[:-len(suffix)]
        if complete['input_ids'][:start]!=prefix:raise ValueError('Candidate answers have different scoring context')
        tokens.append(answer);context.append({'letter':letter,'token_ids':answer,'prediction_position':start-1})
    if enc['input_ids']!=prefix:raise ValueError('Context audit mutated scoring prefix')
    t.set_mode('transformers');single=all(len(x)==1 for x in tokens)
    if single:
        batch=collate(engine,enc)
        if batch['input_ids'][0].tolist()!=prefix:raise ValueError('Actual consumed scoring context differs')
        with torch.inference_mode():
            logits=engine.model(**batch,use_cache=False,logits_to_keep=1,return_dict=True).logits[0,-1].float();lp=logits.log_softmax(-1)
            values=[float(lp[x[0]]) for x in tokens]
        calls=1
    else:
        from verify_checkpoint import prepare_answer_batch,answer_logps
        values=[]
        for letter,tt in zip('ABCD',tokens):
            r=copy.deepcopy(row);r['messages'].append({'role':'assistant','content':letter});batch,labels=prepare_answer_batch(engine,r)
            logps,_,_=answer_logps(engine.model,batch,labels);values.append(float(logps[:len(tt)].sum()))
        calls=4
    return {'log_probabilities_ABCD':values,'raw_probabilities_ABCD':np.exp(values).tolist(),'raw_candidate_mass':float(np.exp(values).sum()),
        'normalized_ABCD':probabilities(values).tolist(),'single_token_same_position':single,
        'mass_scope':'same_position_single_token_full_vocabulary' if single else 'complete_answer_sequence_probability_sum',
        'tokenization_context':context,'prompt_token_ids':prefix,'encoding':audit,'actual_forward_calls':calls}


def run(scope):
    validate();plan,by,train,dev,features,pseudo=assets();out=STORE/'teacher';out.mkdir(exist_ok=True)
    cache_path=out/'view_cache.jsonl';cached={r['cache_key']:r for r in records(cache_path)} if cache_path.exists() else {}
    if scope=='technical':work=read(WORK/'technical_pairs.json')['records']
    else:
        rows=records(WORK/('dev_windows.jsonl' if scope=='dev' else 'train_windows.initial.jsonl'));choices=read(WORK/'pair_choices.json');work=[]
        for r in rows:
            work += [{'video_id':r['video_id'],'request_id':r['request_id'],'clip_i':i,'clip_j':j,'same_clip':False} for i,j in choices[r['request_id']]]
    dest=out/(scope+'_edges.jsonl')
    if dest.exists():raise FileExistsError('No hidden teacher retry')
    nominal=0;actual=0;hits=0;start=time.monotonic();results=[]
    with gpu_session('8B_'+scope,'teacher'):
        import torch
        torch.cuda.reset_peak_memory_stats();engine=load_engine(STORE/'teacher_model');engine.model.eval();engine.model.requires_grad_(False)
        if engine.model.dtype!=torch.bfloat16 or getattr(engine.model,'is_quantized',False):raise ValueError('RESOURCE_NOT_READY: teacher is not unquantized BF16')
        model_runtime={'class':type(engine.model).__name__,'dtype':str(engine.model.dtype),'device':str(engine.model.device),
            'text_hidden_size':engine.model.config.text_config.hidden_size,'text_layers':engine.model.config.text_config.num_hidden_layers,
            'attention_implementation':engine.model.config._attn_implementation,'quantized':False}
        asset_hash=sha(STORE/'teacher_model/asset_receipt.json')
        for number,r in enumerate(work):
            r=dict(r,teacher_pair_id=f'{scope}:{r["video_id"]}:{r.get("request_id","technical")}:{r["clip_i"]}:{r["clip_j"]}')
            views=[]
            for order,(a,b) in enumerate(((r['clip_i'],r['clip_j']),(r['clip_j'],r['clip_i']))):
                row=payload(by,r['video_id'],a,b);media=row['images']+[p for vv in row['videos'] for p in vv]
                refs=[reference(p) for p in media]
                if any(x['sha256']!=by[r['video_id']]['media_files_sha256'][x['path']] for x in refs):raise ValueError('Changed teacher source pixels')
                key=canonical_hash({'messages':row['messages'],'media_hashes':[x['sha256'] for x in refs],'teacher_asset':asset_hash})
                hit=key in cached;nominal+=1
                if hit:value=cached[key];hits+=1
                else:
                    t=time.monotonic();value=dict(score(engine,row),cache_key=key,actual_scoring_request_id='ABCD:'+key,messages=row['messages'],media=refs,seconds=time.monotonic()-t)
                    append(cache_path,value);cached[key]=value;actual+=value['actual_forward_calls']
                views.append(dict(value,cache_hit=hit,clip_A=a,clip_B=b,logical_view_request_id=r['teacher_pair_id']+f':order{order}'))
            result=dict(r,**align(views[0]['log_probabilities_ABCD'],views[1]['log_probabilities_ABCD']),views=views)
            results.append(result);append(dest,result)
            if (number+1)%40==0:print(f'{scope}: {number+1}/{len(work)} edges, {actual} new forwards',flush=True)
        receipt={'status':'TEACHER_OUTPUT_FROZEN','time':now(),'scope':scope,'nominal_views':nominal,'actual_forwards':actual,'cache_hits':hits,
            'actual_loaded_model':model_runtime,
            'n_pairs':len(work),'n_directional':sum(r['kept'] for r in results),'n_ties':sum(r['kind']=='tie' for r in results),'n_unknown':sum(r['kind']=='unknown' for r in results),
            'wall_seconds':time.monotonic()-start,'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),'files':{str(dest):sha(dest)},
            'append_only_cache_at_phase_end':reference(cache_path),'human_labels_read':False}
        if scope=='technical':
            same=[r for r in results if r['same_clip']];good=sum(all(unique_top(v['normalized_ABCD'])==2 for v in r['views']) for r in same)
            masses=[v['raw_candidate_mass'] for r in results for v in r['views']];fraction=float(np.mean(np.asarray(masses)>=.5))
            receipt.update(same_clip_pairs_C_in_both=good,views_mass_at_least_half_fraction=fraction,
                passed=good>=6 and fraction>=.9,status='TEACHER_INTERFACE_PASSED' if good>=6 and fraction>=.9 else 'NO_GO_TEACHER_INTERFACE')
        write(out/(scope+'_receipt.json'),receipt)
    print(f'{scope}: '+receipt['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scope',choices=['technical','dev','train']);a=p.parse_args();run(a.scope)
