#!/usr/bin/env python3
"""One frozen S0, eight fixed windows: matched duplicate versus extra overview."""
import copy
from collections import defaultdict
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from unsgrpo.core import parse_scores,reward_parts
from unsgrpo.round2_utils import *


def additional_overview(row,dest):
    paths=[p for group in row['frame_paths'] for p in group];pts=np.asarray(row['sampled_pts']).ravel()
    old=set(np.rint(np.linspace(0,len(paths)-1,16)).astype(int).tolist());used=set(old);chosen=[]
    for target in (np.arange(16)+.5)/16*(len(paths)-1):
        i=min((i for i in range(len(paths)) if i not in used),key=lambda i:(abs(i-target),i));chosen.append(i);used.add(i)
    chosen.sort();canvas=Image.new('RGB',(896,896),'black');draw=ImageDraw.Draw(canvas)
    if any(sha(paths[i])!=row['media_files_sha256'][paths[i]] for i in chosen):
        raise ValueError('Additional cached global frame differs from original seal')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for j,i in enumerate(chosen):
        image=Image.open(paths[i]).convert('RGB');image.thumbnail((224,202));x=(j%4)*224;y=(j//4)*224
        canvas.paste(image,(x+(224-image.width)//2,y));draw.text((x+4,y+203),f'{pts[i]:.2f} s',fill='white',font=font)
    canvas.save(dest,quality=95)
    return {'original_overview_frame_indices':sorted(old),'new_global_frame_indices':chosen,
            'new_global_source_frames':[reference(paths[i]) for i in chosen],
            'new_global_pts':[float(pts[i]) for i in chosen],'overview':reference(dest)}


def tensor_sha(value):
    array=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def encode_audit(engine,row):
    template=engine.template;template.set_mode('transformers')
    encoded=template.encode({'messages':copy.deepcopy(row['messages']),'images':row['images'],'videos':row['videos']})
    ids=list(encoded['input_ids']);ig=encoded['image_grid_thw'].tolist();vg=encoded['video_grid_thw'].tolist()
    if len(ids)+128>8192 or len(ig)!=len(row['images']) or len(vg)!=1:raise ValueError('Counterfactual token/grid budget failed')
    processor=engine.template.processor;merge=int(processor.image_processor.merge_size)**2
    image_tokens=sum(int(np.prod(g))//merge for g in ig);video_tokens=int(np.prod(vg[0]))//merge
    temporal=int(getattr(processor,'video_processor',processor.image_processor).temporal_patch_size)
    if int(vg[0][0])*temporal!=32 or len(row['videos'][0])!=32:raise ValueError('Local frame resampling changed')
    if ids.count(template.image_token_id)!=image_tokens or ids.count(template.video_token_id)!=video_tokens:
        raise ValueError('Visual placeholders do not match processed grids')
    return {'prompt_tokens':len(ids),'input_ids_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
        'image_grid_thw':ig,'video_grid_thw':vg,'image_visual_tokens':image_tokens,'video_visual_tokens':video_tokens,
        'processed_local_frames':32,'video_pixel_tensor_sha256':tensor_sha(encoded['pixel_values_videos']),
        'image_pixel_tensor_sha256':tensor_sha(encoded['pixel_values']),'no_truncation':True}


def main():
    plan,by,train,dev,feat,pseudo=load_assets();out=ROOT/'input_counterfactual';out.mkdir()
    gate=read(ROOT/'cached/input_counterfactual_gate.json')
    if not gate['run']:
        write(out/'results.json',{'status':'SKIPPED_PREDEFINED_CACHE_GATE','gate':gate,'wall_seconds':0.,'new_inference_calls':0});return
    import torch
    start=time.monotonic();audit=read(PROJECT/'reports/input_audit.json')
    for key,value in audit['visual_environment'].items():os.environ[key]=value
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1')
    common=read(PROJECT/'reports/common_s0.json');model=Path(common['merged_model'])
    for name,digest in common['merged_model_files'].items():
        if sha(model/name)!=digest:raise ValueError(f'Changed frozen S0 file: {name}')
    extra={k:additional_overview(by[k],out/(k+'_extra_overview.jpg')) for k in gate['chosen_video_ids']}
    inputs={r['request_id']:r for r in records(plan['paths']['eval_pool'])}
    cached={r['request_id']:r for r in records(Path(plan['paths']['original_run'])/'inference/S0/windows.jsonl')}
    from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
    load_start=time.monotonic();engine=TransformersEngine(str(model),model_type='qwen3_vl',torch_dtype=torch.bfloat16,
        attn_impl='sdpa',device_map='cuda:0',use_hf=True,max_batch_size=1)
    engine.template.max_length=8192;engine.template.truncation_strategy='raise'
    model_load_seconds=time.monotonic()-load_start
    config=RequestConfig(max_tokens=128,temperature=0,top_p=1,top_k=0,repetition_penalty=1,return_details=True)
    rows=[];control_binding={}
    for rid in gate['selected_request_ids']:
        original=inputs[rid];vid=original['video_id']
        if any(sha(p)!=by[vid]['media_files_sha256'][p] for p in original['images']+original['videos'][0]):
            raise ValueError('Original selected media differs from sealed first-round inputs')
        for condition in ('A','B','C'):
            if time.monotonic()-start>600:raise TimeoutError('Frozen-input GPU inference budget exhausted')
            row=copy.deepcopy(original)
            if condition!='A':
                row['messages'][1]['content']=row['messages'][1]['content'].replace('<image>','<image>\n<image>',1)
                row['images'].append(row['images'][0] if condition=='B' else extra[vid]['overview']['path'])
            if row['videos']!=original['videos'] or row['clip_ids']!=original['clip_ids']:raise ValueError('Local evidence changed')
            audit_row=encode_audit(engine,row)
            if condition=='B':control_binding[rid]=(copy.deepcopy(row['messages']),audit_row)
            if condition=='C':
                messages,previous=control_binding[rid]
                if row['messages']!=messages or any(audit_row[k]!=previous[k] for k in ('input_ids_sha256','image_grid_thw','video_grid_thw','video_pixel_tensor_sha256')):
                    raise ValueError('B/C differ beyond second image contents')
            t=time.monotonic();response=engine.infer([InferRequest(messages=row['messages'],images=row['images'],videos=row['videos'])],config)[0]
            torch.cuda.synchronize();text=response.choices[0].message.content;reason=response.choices[0].finish_reason
            try:score=parse_scores(text,8);valid=True
            except (ValueError,TypeError):score=np.zeros(8);valid=False
            old_indices=extra[vid]['original_overview_frame_indices'];new_indices=extra[vid]['new_global_frame_indices']
            flat=[p for group in by[vid]['frame_paths'] for p in group]
            original_global={flat[i] for i in old_indices};extra_global={flat[i] for i in new_indices}
            local=set(original['videos'][0]);additional_unique=len(extra_global-(original_global|local)) if condition=='C' else 0
            result={'request_id':rid,'video_id':vid,'condition':condition,'clip_ids':row['clip_ids'],
                'messages':row['messages'],'images':[reference(p) for p in row['images']],
                'original_local_media_signature':original['media_signature'],'audit':audit_row,'response':text,
                'valid':valid,'scores':score.tolist(),'finish_reason':reason,'seconds':time.monotonic()-t,
                'new_unique_source_frames_beyond_original_input':additional_unique,
                'processed_extra_overview_cells':0 if condition=='A' else 16,
                'reward':asdict(reward_parts(text,np.asarray(pseudo[vid]['q'])[row['clip_ids']])),
                'cached_response_exact_match':text==cached[rid]['response'] if condition=='A' else None,
                'usage_prompt_tokens':getattr(getattr(response,'usage',None),'prompt_tokens',None)}
            rows.append(result)
            with (out/'predictions.jsonl').open('a') as handle:handle.write(__import__('json').dumps(result,allow_nan=False)+'\n')
    consistency={}
    for condition in ('A','B','C'):
        per=[]
        for k in gate['chosen_video_ids']:
            rr=[r for r in rows if r['video_id']==k and r['condition']==condition]
            per.append({'video_id':k,**overlap_statistics([r['clip_ids'] for r in rr],[np.asarray(r['scores']) for r in rr]),
                        'mean_proxy_MSE':float(np.mean([1-r['reward']['calibration'] for r in rr if r['valid']])) if any(r['valid'] for r in rr) else None})
        consistency[condition]={'per_video':per,'mean_range':float(np.mean([r['mean_range'] for r in per])),
                                'legal_fraction':float(np.mean([r['valid'] for r in rows if r['condition']==condition]))}
    write(out/'results.json',{'status':'COMPLETED','evaluation_status':'DEVELOPMENT_MECHANISM_PROBE_ONLY','gate':gate,
        'weight_updates':0,'model':str(model),'common_s0':reference(PROJECT/'reports/common_s0.json'),
        'extra_overviews':extra,'conditions':consistency,'n_inference_calls':len(rows),'model_load_seconds':model_load_seconds,
        'wall_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'A_exact_replay_count':sum(r['cached_response_exact_match'] is True for r in rows),
        'full_video_AP_computed':False,'scope':'B/C only differ in second image contents; four sources and partial coverage cannot establish an AP gain.'})


if __name__=='__main__':main()
