#!/usr/bin/env python3
"""Fresh, complete, greedy inference and explicit invalid-window aggregation."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from unsgrpo.core import parse_scores, reward_parts
from unsgrpo.data import aggregate_windows, audit_window_record, match_prediction_rows, build_window_record, window_indices

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def model_identity(path):
    return {'path':str(path.resolve()),'files':{str(p.relative_to(path)):digest(p) for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in {'.safetensors','.bin','.json','.txt','.model'}}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',choices=['B0','B1','S0','S+','G-CAL','G-RANK'],required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--pseudo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input-audit',type=Path,default=Path(__file__).resolve().parents[1]/'reports/input_audit.json')
    p.add_argument('--common-s0',type=Path,default=Path(__file__).resolve().parents[1]/'reports/common_s0.json')
    p.add_argument('--training-receipt',type=Path)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError('Inference requires a new run directory')
    rows=[json.loads(l) for l in a.inputs.read_text().splitlines()]
    prepared=[json.loads(l) for l in a.manifest.read_text().splitlines()]
    test={r['video_id']:r for r in prepared if r['split']=='test'}
    if set(r['video_id'] for r in rows)!=set(test): raise ValueError('Input/test video coverage mismatch')
    if len({r['request_id'] for r in rows})!=len(rows): raise ValueError('Duplicate inference request ID')
    for r in rows:
        audit_window_record(r,check_media_files=True)
        if r['task']!='eval' or r['split']!='test': raise ValueError('Inference inputs must be label-free test rows')
    expected={}
    for vid,v in test.items():
        for ids in window_indices(len(v['bounds'])):
            row=build_window_record(video_id=vid,source_group=v['source_group'],split='test',duration=v['duration'],bounds=v['bounds'],frame_paths=v['frame_paths'],sampled_pts=v['sampled_pts'],overview_path=v['overview'],clip_ids=ids,task='eval')
            expected[row['request_id']]=row
    if set(expected)!={r['request_id'] for r in rows}: raise ValueError('Inference windows differ from full registered coverage')
    for row in rows:
        if any(row.get(k)!=v for k,v in expected[row['request_id']].items()): raise ValueError('Inference media/PTS/prompt differs from prepared manifest')
    audit=json.loads(a.input_audit.read_text())
    if audit.get('status')!='COMPLETED' or audit.get('input_sha256',{}).get(str(a.inputs.resolve()))!=digest(a.inputs):
        raise ValueError('Inference file is not bound to completed real input audit')
    visual=audit['visual_environment']
    if any(v is None for v in visual.values()): raise ValueError('Audited visual environment is incomplete')
    for key,value in visual.items():
        if key in os.environ and os.environ[key]!=value: raise ValueError(f'Inference visual environment differs: {key}')
        os.environ[key]=value
    pseudo_asset=json.loads(a.pseudo.read_text())
    if pseudo_asset.get('prepared_manifest_sha256')!=digest(a.manifest): raise ValueError('Pseudo scores belong to a different prepared manifest')
    pseudo=pseudo_asset['videos']
    receipt={'run_id':a.run_id,'status':'RUNNING','inputs_sha256':digest(a.inputs),
             'manifest_sha256':digest(a.manifest),'pseudo_sha256':digest(a.pseudo),
             'human_annotations_read':False,'greedy':True,'max_completion_tokens':128,
             'invalid_window_rule':'Zeros, included in overlap denominator',
             'n_windows':len(rows),'n_videos':len(test)}
    receipt.update(input_audit_sha256=digest(a.input_audit),visual_environment=visual,
                   truncation_rule='Strict JSON only, no repair; valid complete JSON at length limit remains valid',
                   inference_input_max_length=8192)
    if a.run_id=='B1':
        if a.adapter or a.model is None or a.model.resolve()!=Path(audit['model_path']).resolve():
            raise ValueError('B1 must use the original audited Qwen base without an adapter')
    if a.run_id in {'S0','S+','G-CAL','G-RANK'}:
        if a.model is None or (a.run_id=='S0' and a.adapter is not None) or (a.run_id!='S0' and a.adapter is None):
            raise ValueError('S0 requires merged S0 with no adapter; continuation methods require their final adapter')
        common=json.loads(a.common_s0.read_text())
        if common.get('status')!='COMPLETED' or Path(common['merged_model']).resolve()!=a.model.resolve():
            raise ValueError('Inference model is not the registered common merged S0')
        if a.training_receipt is None: raise ValueError('Trained endpoints require their verified training receipt')
        trained=json.loads(a.training_receipt.read_text())
        if trained.get('status')!='COMPLETED' or trained.get('mode')!='formal' or trained.get('gpu_training_verified') is not True or trained.get('method_id')!=a.run_id:
            raise ValueError('Wrong or unverified formal training endpoint')
        if a.run_id=='S0':
            if Path(trained['endpoint']['path']).resolve()!=Path(common['sft_endpoint']).resolve(): raise ValueError('S0 parent adapter mismatch')
        else:
            if trained.get('common_s0_sha256')!=digest(a.common_s0) or Path(trained['endpoint']['path']).resolve()!=a.adapter.resolve():
                raise ValueError('Continuation endpoint or common S0 identity mismatch')
            if any(digest(a.adapter/name)!=sha for name,sha in trained['endpoint']['files'].items()): raise ValueError('Final adapter files changed after training audit')
        receipt.update(training_receipt={'path':str(a.training_receipt.resolve()),'sha256':digest(a.training_receipt)},common_s0_sha256=digest(a.common_s0))
    def save(): (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    a.out.mkdir(parents=True)
    save();started=time.monotonic()
    try:
        if a.run_id=='B0':
            videos=[{'video_id':vid,'bounds':r['bounds'],'scores':pseudo[vid]['q']} for vid,r in test.items()]
            receipt.update(training_prompt_exposures=0,training_rollouts=0,
                           teacher_test_preparation_seconds=sum(r['seconds'] for r in test.values()),
                           preparation_cost_note='Includes real decode and CLIP; clustering fit/transform cost is separately recorded by make_pseudo',
                           proxy_legal_rate=None)
        else:
            if a.model is None: raise ValueError('A local model is required')
            receipt['model']=model_identity(a.model)
            receipt['adapter']=model_identity(a.adapter) if a.adapter else None
            if a.run_id=='B1':
                asset=json.loads((a.model/'asset_receipt.json').read_text())
                if any(receipt['model']['files'].get(entry['file'])!=entry['sha256'] for entry in asset['files'] if entry['file'].endswith(('.bin','.safetensors'))):
                    raise ValueError('Original Qwen base weights no longer match pinned assets')
            processor_files=audit.get('model_processor_files_sha256')
            if not processor_files: raise ValueError('Input audit is missing pinned processor identities')
            raw_differences=[name for name,sha in processor_files.items() if receipt['model']['files'].get(name)!=sha]
            if raw_differences:
                from audit_swift import model_processor_semantic_hashes
                semantic=model_processor_semantic_hashes(a.model)
                if semantic!=audit.get('model_processor_semantic_sha256'):
                    raise ValueError('Inference processor/tokenizer semantics differ from audited Qwen inputs')
            receipt['processor_raw_files_differing_only_in_registered_metadata']=raw_differences
            if a.run_id in {'S0','S+','G-CAL','G-RANK'}:
                common=json.loads(a.common_s0.read_text())
                if common.get('status')!='COMPLETED' or Path(common['merged_model']).resolve()!=a.model.resolve(): raise ValueError('Wrong common merged S0 base')
                for name,sha in common['merged_model_files'].items():
                    if receipt['model']['files'].get(name)!=sha: raise ValueError('Common S0 file hash changed')
                receipt['common_s0_sha256']=digest(a.common_s0)
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            torch.cuda.reset_peak_memory_stats()
            load_start=time.monotonic()
            engine=TransformersEngine(str(a.model),adapters=[str(a.adapter)] if a.adapter else None,
                                      model_type='qwen3_vl',torch_dtype=torch.bfloat16,attn_impl='sdpa',
                                      device_map='cuda:0',use_hf=True,max_batch_size=1)
            engine.template.max_length=8192
            engine.template.truncation_strategy='raise'
            receipt['model_load_seconds']=time.monotonic()-load_start
            cfg=RequestConfig(max_tokens=128,temperature=0,top_p=1,top_k=0,repetition_penalty=1,return_details=True)
            predictions=[];grouped=defaultdict(list);parts=[];constant=0;legal=0;prompt_tokens=0;completion_tokens=0;truncated=0
            with (a.out/'windows.jsonl').open('x') as handle:
                for row in rows:
                    t=time.monotonic()
                    response=engine.infer([InferRequest(messages=row['messages'],images=row['images'],videos=row['videos'])],cfg)[0]
                    torch.cuda.synchronize()
                    text=response.choices[0].message.content
                    truncated+=int(response.choices[0].finish_reason=='length')
                    q=[pseudo[row['video_id']]['q'][i] for i in row['clip_ids']]
                    reward=reward_parts(text,q)
                    record={'request_id':row['request_id'],'video_id':row['video_id'],
                            'media_signature':row['media_signature'],'clip_ids':row['clip_ids'],
                            'response':text,'finish_reason':response.choices[0].finish_reason,
                            'seconds':time.monotonic()-t,'reward':asdict(reward)}
                    usage=getattr(response,'usage',None)
                    if usage is not None:
                        record['prompt_tokens']=usage.prompt_tokens
                        record['completion_tokens']=usage.completion_tokens
                        prompt_tokens+=usage.prompt_tokens or 0;completion_tokens+=usage.completion_tokens or 0
                    handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush()
                    predictions.append(record);grouped[row['video_id']].append((row['clip_ids'],text));parts.append(reward)
                    if reward.valid:
                        s=parse_scores(text,len(q));legal+=1;constant+=int(np.all(s==s[0]))
                    if len(predictions)%10==0: print(json.dumps({'run':a.run_id,'windows':len(predictions),'total':len(rows)}),flush=True)
            match_prediction_rows(rows,predictions)
            videos=[];spread=[]
            for vid,video in test.items():
                scores,invalid=aggregate_windows(len(video['bounds']),grouped[vid])
                videos.append({'video_id':vid,'bounds':video['bounds'],'scores':scores.tolist(),'invalid_window_fraction':invalid})
                clip_values=defaultdict(list)
                for ids,text in grouped[vid]:
                    try: values=parse_scores(text,len(ids))
                    except (ValueError,TypeError): values=np.zeros(len(ids))
                    for i,value in zip(ids,values): clip_values[i].append(value)
                spread.extend(float(np.ptp(v)) for v in clip_values.values() if len(v)>1)
            receipt.update(proxy_legal_rate=legal/len(rows),constant_output_fraction_among_legal=constant/legal if legal else None,
                           proxy_mean_reward=float(np.mean([x.total for x in parts])),
                           proxy_mean_ranking_among_legal=float(np.mean([x.ranking for x in parts if x.valid])) if legal else None,
                           proxy_mean_calibration_among_legal=float(np.mean([x.calibration for x in parts if x.valid])) if legal else None,
                           cross_window_mean_range=float(np.mean(spread)) if spread else 0,
                           prompt_tokens=prompt_tokens,completion_tokens=completion_tokens,
                           truncated_window_fraction=truncated/len(rows),
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        output=a.out/'videos.jsonl'
        output.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in videos))
        receipt.update(status='COMPLETED',predictions_path=str(output.resolve()),predictions_sha256=digest(output))
    except BaseException as e:
        receipt.update(status='FAILED',error=f'{type(e).__name__}: {e}');raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started
        receipt['gpu_hours']=receipt['wall_seconds']/3600 if a.run_id!='B0' else 0
        save()

if __name__=='__main__': main()
