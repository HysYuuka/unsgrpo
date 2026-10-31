#!/usr/bin/env python3
"""Freeze the original local semantic teacher; no human-label imports or reads."""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
from unsgrpo.round3_common import *
from unsgrpo.round3_teacher import *
from unsgrpo.round2_utils import prototypes
from unsgrpo.core import midrank_percentile


def main():
    plan,by,train,dev,feat,pseudo=assets();out=STORE/'teacher'
    if (out/'semantic.freeze.json').exists():raise FileExistsError('Teacher already frozen; no automatic regeneration')
    if not (REPORT/'metric_choice.json').exists():raise ValueError('Freeze the primary metric before teacher results')
    frozen=read(WORK/'front_end.freeze.json')
    for p,h in frozen['sources'].items():
        if sha(p)!=h:raise ValueError(f'Frozen source changed: {p}')
    clusters={k:int(pseudo[k]['cluster']) for k in by}; qexcl={};exclusion={}
    for k in by:
        if k in train:
            bank,fallback=prototypes(feat,train,clusters,exclude=k,video_equal=False)
            qexcl[k]=midrank_percentile(feat[k]@bank[clusters[k]])
            peers=[v for v in train if v!=k and clusters[v]==clusters[k]]
            exclusion[k]={'excluded_entire_video':True,'cluster':clusters[k],
                'global_other_train_fallback':bool(fallback.get(clusters[k],False)),
                'peer_ids':peers if peers else [v for v in train if v!=k]}
        else:
            qexcl[k]=np.asarray(pseudo[k]['q']);exclusion[k]={'excluded_entire_video':False,
                'cluster':clusters[k],'global_other_train_fallback':False,
                'peer_ids':[v for v in train if clusters[v]==clusters[k]]}
    np.savez_compressed(out/'reference_targets.npz',**{f'q0__{k}':pseudo[k]['q'] for k in by},
                        **{f'q_excl__{k}':qexcl[k] for k in by})
    write(out/'exclusion.json',exclusion)
    model=Path('/root/autodl-fs/unsgrpo/models/Qwen3-VL-2B-Instruct')
    asset=read(PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json')
    for item in asset['files']:
        if sha(model/item['file'])!=item['sha256']:raise ValueError('Original frozen teacher asset changed')
    visual_environment();start=time.monotonic();events={};scores={};audit_rows=[]
    precheck_ids=sorted(train)[:8];prechecks=[];early_stop=None;format_version=0
    with gpu_session('B_frozen_semantic_teacher',5400) as remaining:
        import torch
        from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
        engine=TransformersEngine(str(model),model_type='qwen3_vl',torch_dtype=torch.bfloat16,
            attn_impl='sdpa',device_map='cuda:0',use_hf=True,max_batch_size=1)
        engine.template.max_length=8192;engine.template.truncation_strategy='raise'
        engine.model.eval();engine.model.requires_grad_(False)
        def infer(kind,k,ids,messages,cap,version):
            if remaining()<5:raise TimeoutError('Teacher occupied-GPU limit')
            row=by[k];images=[row['overview']]
            videos=[[p for p in row['frame_paths'][i]] for i in ids]
            media=images+[p for group in videos for p in group]
            refs=[reference(p) for p in media]
            if any(r['sha256']!=row['media_files_sha256'][r['path']] for r in refs):raise ValueError('Changed cached teacher frames')
            engine.template.set_mode('transformers')
            enc=engine.template.encode({'messages':copy.deepcopy(messages),'images':images,'videos':videos})
            grids=enc.get('video_grid_thw'); imagegrids=enc['image_grid_thw']
            if len(enc['input_ids'])+cap>8192:raise ValueError('Teacher prompt would truncate')
            video_grids=[] if grids is None else grids.tolist()
            if len(video_grids)!=len(ids) or any(g[0]*2!=4 for g in video_grids):
                raise ValueError('Teacher must receive exactly four original frames per distinct clip')
            merge=int(engine.template.processor.image_processor.merge_size)**2
            encoding={'prompt_tokens':len(enc['input_ids']),'image_grids':imagegrids.tolist(),
                'video_grids':video_grids,'image_visual_tokens':int(sum(np.prod(g) for g in imagegrids.tolist())//merge),
                'video_visual_tokens':int(sum(np.prod(g) for g in video_grids)//merge),
                'actual_cached_frames':4*len(ids),'truncation':False}
            t=time.monotonic();response=engine.infer([InferRequest(messages=messages,images=images,videos=videos)],
                RequestConfig(max_tokens=cap,temperature=0,top_p=1,top_k=0,repetition_penalty=1,return_details=True))[0]
            torch.cuda.synchronize();text=response.choices[0].message.content
            parsed=None;error=None
            try:parsed=parse_events(text) if kind=='events' else parse_semantic(text,len(ids),[e['id'] for e in events[k]['parsed']['events']])
            except (ValueError,TypeError,KeyError) as e:error=f'{type(e).__name__}: {e}'
            result={'call_index':len(audit_rows),'kind':kind,'video_id':k,'split':'train' if k in train else 'development',
                'clip_ids':ids,'clip_bounds':[row['bounds'][i] for i in ids],
                'sampled_pts':[row['sampled_pts'][i] for i in ids],
                'format_version':version,'messages':messages,'source_media':refs,'response':text,
                'parsed':parsed,'valid':parsed is not None,'error':error,'encoding':encoding,
                'finish_reason':response.choices[0].finish_reason,'seconds':time.monotonic()-t,
                'usage_prompt_tokens':getattr(response.usage,'prompt_tokens',None),
                'usage_completion_tokens':getattr(response.usage,'completion_tokens',None)}
            audit_rows.append(result);append(out/'semantic_calls.jsonl',result)
            if len(audit_rows)%40==0:print(f'{len(audit_rows)} frozen teacher calls; GPU allowance {remaining():.0f}s',flush=True)
            return result
        def event(k):
            if k not in events:events[k]=infer('events',k,[],event_messages(),256,0)
        def packet(k,ids,version):
            event(k)
            if not events[k]['valid']:return (None,None)
            return tuple(infer(view,k,ids,semantic_messages(by[k],ids,events[k]['parsed'],view,version),384,version) for view in ('A','B'))
        # Eight IDs and their first chronological packets are fixed without labels.
        for version in (0,1):
            format_version=version
            runs=[]
            for k in precheck_ids:
                ids=list(range(min(8,len(qexcl[k]))));pair=packet(k,ids,version)
                scores[(k,0)]=pair;runs.extend(r for r in pair if r is not None)
            valid=sum(r['valid'] for r in runs);fraction=valid/16.
            prechecks.append({'format_version':version,'training_ids':precheck_ids,'expected_calls':16,
                'actual_calls':len(runs),'valid_calls':valid,'valid_fraction':fraction,
                'missing_calls_from_invalid_event_reference':16-len(runs)})
            write(out/f'format_precheck_v{version}.json',prechecks[-1])
            if fraction>=.95:format_version=version;break
            if version==1:early_stop='FORMAT_PRECHECK_FAILED_AFTER_ONE_JSON_IMPLEMENTATION_FIX'
        if early_stop is None:
            for k in sorted(by):
                event(k)
                for first in range(0,len(qexcl[k]),8):
                    if (k,first) in scores:continue
                    ids=list(range(first,min(first+8,len(qexcl[k]))))
                    scores[(k,first)]=packet(k,ids,format_version)
        resource={'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(),
            'frozen_model_requires_grad_count':sum(p.requires_grad for p in engine.model.parameters())}
    serialized=[]
    for k in sorted(by):
        for first in range(0,len(qexcl[k]),8):
            ids=list(range(first,min(first+8,len(qexcl[k]))));a,b=scores.get((k,first),(None,None))
            for j,i in enumerate(ids):
                valid_a=a is not None and a['valid'];valid_b=b is not None and b['valid']
                serialized.append({'video_id':k,'clip_id':i,'split':'train' if k in train else 'development',
                    'bounds':by[k]['bounds'][i],'sampled_pts':by[k]['sampled_pts'][i],
                    'q0':float(pseudo[k]['q'][i]),'q_excl':float(qexcl[k][i]),
                    'score_a':a['parsed']['scores'][j] if valid_a else None,
                    'score_b':b['parsed']['scores'][j] if valid_b else None,
                    'role_a':a['parsed']['roles'][j] if valid_a else None,
                    'role_b':b['parsed']['roles'][j] if valid_b else None,
                    'event_ids_a':a['parsed']['event_ids'][j] if valid_a else None,
                    'event_ids_b':b['parsed']['event_ids'][j] if valid_b else None,
                    'call_a':a['call_index'] if a else None,'call_b':b['call_index'] if b else None,
                    'event_reference_call':events.get(k,{}).get('call_index'),
                    'not_scored_reason':early_stop if (a is None or b is None) else None,
                    'source_frame_hashes':[by[k]['media_files_sha256'][p] for p in by[k]['frame_paths'][i]],
                    'prototype_self_exclusion':exclusion[k]})
    with (out/'semantic_clips.jsonl').open('x') as f:
        for r in serialized:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(out/'semantic.freeze.json',{'status':'FROZEN_BEFORE_DEVELOPMENT_QUALITY_LABEL_READS',
        'time':timestamp(),'format_prechecks':prechecks,'format_version':format_version,
        'early_stop':early_stop,'n_calls':len(audit_rows),'event_calls':sum(r['kind']=='events' for r in audit_rows),
        'semantic_calls':sum(r['kind']!='events' for r in audit_rows),
        'valid_event_calls':sum(r['kind']=='events' and r['valid'] for r in audit_rows),
        'valid_semantic_calls':sum(r['kind']!='events' and r['valid'] for r in audit_rows),
        'semantic_call_file':reference(out/'semantic_calls.jsonl'),
        'semantic_clip_file':reference(out/'semantic_clips.jsonl'),
        'reference_targets':reference(out/'reference_targets.npz'),
        'model_asset_receipt':reference(PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json'),
        'primary_metric_choice':reference(REPORT/'metric_choice.json'),
        'GPU_resources':resource,'wall_seconds':time.monotonic()-start,
        'human_annotations_read':False,'teacher_is_updating_policy':False})
    print(json.dumps({'status':'TEACHER_CACHED','early_stop':early_stop,'calls':len(audit_rows)}),flush=True)


if __name__=='__main__':main()
