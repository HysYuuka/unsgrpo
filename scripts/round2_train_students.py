#!/usr/bin/env python3
"""CPU-only cheap students. This program never imports/opens human annotations."""
from collections import Counter
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from unsgrpo.data import window_indices
from unsgrpo.diagnostics import student_loss
from unsgrpo.round2_utils import ROOT,load_assets,read,records,write,reference,sha,now


class SmallStudent(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(512,64),nn.GELU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x):return self.net(x).squeeze(-1)


def parameter_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def features_for(pool,feat,stats):
    x=[];q=[]
    for row in pool:
        k=row['video_id'];ids=row['clip_ids'];f=feat[k][ids];mean=np.broadcast_to(f.mean(0),f.shape)
        position=np.linspace(0,1,len(ids))[:,None]
        full=np.broadcast_to(np.r_[stats[k][0],stats[k][1]],(len(ids),1024))
        x.append(np.c_[f,mean,position,full]);q.append(row['q'])
    return np.stack(x),np.stack(q)


def main():
    started=time.monotonic();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan,by,train,dev,feat,pseudo=load_assets();spec=plan['students'];out=ROOT/'students';out.mkdir()
    original=records(plan['paths']['train_pool']);train_pool=[]
    for row in original:
        if row['video_id'] not in train or row['split']!='train':raise ValueError('Student training pool crosses split')
        q=np.asarray(pseudo[row['video_id']]['q'])[row['clip_ids']]
        if not np.array_equal(q,row['pseudo_scores']):raise ValueError('Original continuous q metadata changed')
        train_pool.append({'request_id':row['request_id'],'video_id':row['video_id'],'clip_ids':row['clip_ids'],'q':q,
                           'pool_instance_id':row['pool_instance_id']})
    dev_pool=[]
    for k in dev:
        for ii in window_indices(len(feat[k])):
            dev_pool.append({'request_id':f'{k}__{ii[0]:06d}_{ii[-1]:06d}','video_id':k,'clip_ids':ii,
                             'q':np.asarray(pseudo[k]['q'])[ii]})
    if len(train_pool)!=1280 or len(dev_pool)!=346 or any(len(r['clip_ids'])!=8 for r in train_pool+dev_pool):
        raise ValueError('Keep original 1280/346 eight-clip pools')
    stats={k:(feat[k].mean(0),feat[k].std(0)) for k in by}
    tx,tq=features_for(train_pool,feat,stats);dx,dq=features_for(dev_pool,feat,stats)
    if tx.shape!=(1280,8,2049):raise ValueError('Feature layout changed')
    input_mean=tx.reshape(-1,2049).mean(0);input_std=np.maximum(tx.reshape(-1,2049).std(0),1e-5)
    tx=(tx-input_mean)/input_std;dx=(dx-input_mean)/input_std
    projection=np.random.default_rng(20261028).normal(0,1/np.sqrt(512),(2049,512)).astype(np.float32)
    np.save(out/'fixed_projection.npy',projection)
    np.savez_compressed(out/'raw_input_scaling.npz',mean=input_mean,std=input_std)
    views={}
    for view in spec['views']:
        a=tx.copy();b=dx.copy()
        if view=='window_only':a[:,:,1025:]=0;b[:,:,1025:]=0
        a=(a.astype(np.float32)@projection).astype(np.float32);b=(b.astype(np.float32)@projection).astype(np.float32)
        mean=a.reshape(-1,512).mean(0);std=np.maximum(a.reshape(-1,512).std(0),1e-5)
        a=(a-mean)/std;b=(b-mean)/std
        np.savez_compressed(out/f'{view}_projected_scaling.npz',mean=mean,std=std)
        views[view]=(torch.from_numpy(a),torch.from_numpy(b))
    rng=np.random.default_rng(spec['schedule_seed']);needed=spec['steps']*spec['batch_windows'];order=[]
    while len(order)<needed:order.extend(rng.permutation(len(train_pool)).tolist())
    order=np.asarray(order[:needed],dtype=np.int64).reshape(spec['steps'],spec['batch_windows'])
    np.save(out/'optimization_exposure_schedule.npy',order)
    write(out/'data_receipt.json',{'status':'FROZEN_STUDENT_INPUTS_NO_HUMAN_LABELS','source_train_pool':reference(plan['paths']['train_pool']),
        'train_video_ids':train,'development_video_ids_used_only_for_forward_diagnostics':dev,
        'source_features':{k:reference(by[k]['features_path']) for k in by},'human_annotation_files_opened':[],
        'input_dimension_before_projection':2049,'fixed_projection':reference(out/'fixed_projection.npy'),
        'model_input_dimension':512,'trainable_parameters':32897,'global_information':'Full-video CLIP mean/std only, after local clip/window inputs; no global time or identities',
        'normalization_fit_video_ids':train,'projection_seed':20261028,'exposure_schedule':reference(out/'optimization_exposure_schedule.npy'),
        'train_rows':[dict(r,q=np.asarray(r['q']).tolist()) for r in train_pool],
        'development_rows':[dict(r,q=np.asarray(r['q']).tolist()) for r in dev_pool]})
    target=torch.tensor(tq,dtype=torch.float32);devtarget=torch.tensor(dq,dtype=torch.float32)
    all_runs=[];initial_hashes={}
    for seed in spec['seeds']:
        for view in spec['views']:
            trainx,devx=views[view]
            for objective in spec['objectives']:
                if time.monotonic()-started>plan['limits']['cpu_compute_max_seconds']:raise TimeoutError('CPU student budget exhausted; no extensions')
                name=f'{view}__{objective}__seed{seed}';run=out/name;run.mkdir()
                torch.manual_seed(seed);model=SmallStudent();initial=parameter_hash(model)
                if seed in initial_hashes and initial!=initial_hashes[seed]:raise ValueError('Different objective/view initialization')
                initial_hashes[seed]=initial
                opt=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
                ledger=[];predictions={};timing={'optimizer_training_seconds':0.,'diagnostic_forward_seconds':0.}
                begin=time.monotonic();finite=True
                def snapshot(step):
                    t=time.monotonic();model.eval()
                    with torch.no_grad():
                        a=torch.cat([model(trainx[i:i+64]) for i in range(0,len(trainx),64)])
                        b=torch.cat([model(devx[i:i+64]) for i in range(0,len(devx),64)])
                    predictions[f'train_{step}']=a.numpy();predictions[f'dev_{step}']=b.numpy()
                    ledger.append({'step':step,'train_q_MSE':float((a-target).square().mean()),
                        'development_q_MSE_no_human':float((b-devtarget).square().mean()),'human_annotations_read':False})
                    timing['diagnostic_forward_seconds']+=time.monotonic()-t;model.train()
                snapshot(0)
                for step,indices in enumerate(order,1):
                    t=time.monotonic();opt.zero_grad(set_to_none=True);pred=model(trainx[indices]);q=target[indices]
                    loss=student_loss(pred,q,q,objective,temperature=spec['rank_temperature'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite student loss')
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite student gradient')
                    torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip']);opt.step()
                    timing['optimizer_training_seconds']+=time.monotonic()-t
                    if step in spec['fixed_saved_prediction_steps']:snapshot(step)
                final=parameter_hash(model)
                if final==initial:raise ValueError('No actual model parameter change')
                np.savez_compressed(run/'predictions.npz',**predictions)
                torch.save({'state_dict':model.state_dict(),'seed':seed,'view':view,'objective':objective,
                            'global_step':256,'optimizer':opt.state_dict(),'rng_state':torch.random.get_rng_state(),
                            'schedule':reference(out/'optimization_exposure_schedule.npy')},run/'final_checkpoint.pt')
                counts=Counter(train_pool[int(i)]['request_id'] for i in order.ravel())
                receipt={'status':'COMPLETED','seed':seed,'view':view,'objective':objective,'successful_optimizer_updates':256,
                    'input_window_exposures':needed,'supervised_clip_targets':needed*8,'unique_window_ids':len(counts),
                    'unique_pool_instances':len(set(order.ravel().tolist())),'window_exposure_counts':dict(counts),
                    'trainable_parameters':sum(p.numel() for p in model.parameters()),'initial_parameter_sha256':initial,'final_parameter_sha256':final,
                    'checkpoint':reference(run/'final_checkpoint.pt'),'predictions':reference(run/'predictions.npz'),
                    'human_annotation_files_opened':[],'device':'cpu','metrics_snapshots':ledger,'timing':timing,
                    'wall_seconds':time.monotonic()-begin,'final_endpoint_selected_without_human_AP':True}
                write(run/'receipt.json',receipt);all_runs.append({'name':name,'receipt':reference(run/'receipt.json')})
                print(f'{name}: 256 updates, {needed} exposures, {receipt["wall_seconds"]:.2f}s',flush=True)
    write(out/'training_complete.json',{'status':'COMPLETED_ALL_12_BEFORE_HUMAN_EVALUATION','runs':all_runs,
        'human_annotation_files_opened':[],'device':'cpu','wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
