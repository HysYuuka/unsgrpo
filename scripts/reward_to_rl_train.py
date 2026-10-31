#!/usr/bin/env python3
"""Actual Qwen LoRA GRPO/RSFT with consumed-token, loss and optimizer ledgers."""
import argparse
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.reward_to_rl import *
from unsgrpo.reward_to_rl_model import *


def fingerprint(model,lora):
    import torch
    h=hashlib.sha256();count=0
    for name,p in sorted(model.named_parameters()):
        if ('lora_' in name)!=lora:continue
        h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes());count+=p.numel()
    return {'sha256':h.hexdigest(),'parameters':count}


def independent_loss_check(current,old,ref,adv,u):
    # An explicit scalar reconstruction, separate from the differentiable helper.
    values=[]
    for c,o,r in zip(current,old,ref):
        ratio=math.exp(c-o);actor=-min(ratio*adv,min(max(ratio,.8),1.2)*adv)
        d=min(max(r-c,-20),20);kl=min(max(math.exp(d)-d-1,-10),10)
        values.append(actor+.04*u*kl)
    return sum(values)/len(values)


def train(arm,seed,technical=False):
    validate_freeze();freeze=validate_freeze('training.freeze.json')
    if arm not in ('G_plain','G_guard','RSFT'):raise ValueError('Unknown branch')
    if technical and arm!='G_guard':raise ValueError('Single technical adapter uses the guarded path')
    tag='technical4' if technical else f'seed{seed}_{arm}'
    out=STORE/'training'/tag;out.mkdir(parents=True,exist_ok=False)
    rows={r['request_id']:r for r in records(WORK/'train_windows.final.jsonl')}
    targets={r['request_id']:r for r in records(STORE/'training_pool/RSFT_targets.jsonl')}
    schedule=read(WORK/'training_schedule.json')['request_ids'];selected=freeze['selected_reward']
    edges=records(STORE/'training_pool/teacher_edges.jsonl') if selected=='R_pair' else []
    edge_map={rid:[e for e in edges if e['request_id']==rid] for rid in rows}
    steps=4 if technical else 80;receipt={'status':'RUNNING','arm':arm,'seed':seed,'technical':technical,
        'started':now(),'common_S0':reference(PROJECT/'reports/common_s0.json'),'training_freeze':reference(WORK/'training.freeze.json'),
        'selected_reward':selected,'completed_optimizer_steps':0,'prompt_exposures':0,'policy_completions_generated':0,
        'trainer':'Explicit PyTorch optimizer/actual Swift-Qwen inference and forward, completion microbatch1; fixed denominator16 for GRPO,4 for RSFT',
        'checkpoints':[],'human_labels_read':False}
    write(out/'receipt.json',receipt)
    with gpu_session('technical4_actual_GRPO' if technical else tag,cap=max(1,21600-gpu_used()-30)):
        import torch
        from verify_checkpoint import add_fresh_language_lora
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
        engine=load_engine(Path(read(PROJECT/'reports/common_s0.json')['merged_model']))
        model,targets_names,trainable=add_fresh_language_lora(engine)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
        def lr_at(k):return k/4 if k<4 else .5*(1+math.cos(math.pi*(k-4)/76))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_at)
        base_before=fingerprint(model,False);adapter_before=fingerprint(model,True)
        receipt.update(trainable_parameters=sum(p.numel() for p in params),LoRA_target_names=targets_names,
            frozen_before=base_before,initial_adapter=adapter_before)
        fixed=None;ref_before=None;loss_errors=[];nonzero_steps=0;times=[]
        def save_receipt():write(out/'receipt.json',receipt,fresh=False)
        save_receipt()
        try:
            for step in range(steps):
                t=time.monotonic();optimizer.zero_grad(set_to_none=True);step_losses=[];group_stats=[]
                for slot,rid in enumerate(schedule[step*4:step*4+4]):
                    row=rows[rid];u=targets[rid]['u']
                    if arm=='RSFT':
                        chosen=targets[rid];ids=chosen['token_ids'];batch,audit=completion_batch(engine,row,ids,chosen['prompt_token_ids'])
                        model.train();lp=token_logps(engine,batch,ids,grad=True)
                        loss=-lp.mean()*u;(loss/4).backward()
                        step_losses.append(float(loss.detach()))
                        append(out/'loss.jsonl',{'step':step+1,'slot':slot,'request_id':rid,'u':u,'CE_sequence':float(-lp.detach().mean()),
                            'actual_loss':float(loss.detach()),'batch_denominator':4,'completion_token_ids':ids,
                            'current_logps':lp.detach().cpu().tolist(),'prompt_loss_tokens':0,'encoding':audit,
                            'consumed_media_sha256':media_hash(batch)})
                        receipt['prompt_exposures']+=1
                        del batch,lp,loss
                        continue
                    model.eval();samples=[]
                    for g in range(4):
                        sample=generate(engine,row);sample.update(g=g,request_id=rid,step=step+1,slot=slot)
                        sample['rewards']=rewards(sample['response'],row['pseudo_scores'],row['clip_ids'],edge_map[rid])
                        samples.append(sample);append(out/'rollouts.jsonl',sample)
                    reward_values=[s['rewards'][selected] for s in samples]
                    adv,raw,std=group_advantages(reward_values,u,arm=='G_guard');kl_u=u if arm=='G_guard' else 1.
                    group_stats.append({'request_id':rid,'reward_values':reward_values,'u':u,'std_sample':std,
                        'raw_advantages':raw.tolist(),'actual_advantages':adv.tolist(),'effective':bool(torch.count_nonzero(adv))})
                    for g,sample in enumerate(samples):
                        ids=sample['token_ids'];batch,audit=completion_batch(engine,row,ids,sample['prompt_token_ids'])
                        model.eval();old=token_logps(engine,batch,ids).detach()
                        with model.disable_adapter():ref=token_logps(engine,batch,ids).detach()
                        if fixed is None:
                            fixed=(copy.deepcopy(row),list(ids),sample['prompt_token_ids']);ref_before=ref.cpu().tolist()
                            if not torch.equal(old,ref):raise ValueError('Fresh zero-B LoRA does not reproduce fixed S0 reference')
                        model.train();current=token_logps(engine,batch,ids,grad=True)
                        loss,parts=sequence_grpo_loss(current,old,ref,adv[g],kl_u)
                        values=[x.detach().cpu().tolist() for x in (current,old,ref)]
                        audit_loss=independent_loss_check(*values,float(adv[g]),kl_u)
                        err=abs(audit_loss-float(loss.detach()));loss_errors.append(err)
                        if err>2e-5:raise ValueError('Actual differentiable loss does not match scalar reconstruction')
                        (loss/16).backward();step_losses.append(float(loss.detach()))
                        append(out/'loss.jsonl',{'step':step+1,'slot':slot,'g':g,'request_id':rid,'u':u,
                            'actual_advantage':float(adv[g]),'raw_advantage':float(raw[g]),'reward_std_sample':std,
                            **parts,'independently_recomputed_loss':audit_loss,'reconstruction_error':err,'batch_denominator':16,
                            'current_logps':values[0],'old_logps':values[1],'reference_logps':values[2],
                            'completion_token_ids':ids,'prompt_loss_tokens':0,'encoding':audit,
                            'consumed_media_sha256':media_hash(batch),'old_ref_advantages_detached':True})
                        del batch,current,old,ref,loss
                    receipt['prompt_exposures']+=1;receipt['policy_completions_generated']+=4
                grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                if grad>0 and lr>0:nonzero_steps+=1
                duration=time.monotonic()-t;times.append(duration)
                append(out/'optimizer.jsonl',{'step':step+1,'time':now(),'lr_before_step':lr,'gradient_norm_before_clip':grad,
                    'mean_actual_sequence_loss':float(np.mean(step_losses)),'prompt_exposures_cumulative':receipt['prompt_exposures'],
                    'policy_completions_cumulative':receipt['policy_completions_generated'],'groups':group_stats,
                    'seconds':duration,'effective_group_fraction':float(np.mean([r['effective'] for r in group_stats])) if group_stats else None})
                receipt['completed_optimizer_steps']=step+1;save_receipt()
                if (step+1)%10==0:print(f'{tag}: {step+1}/{steps}; step {duration:.1f}s',flush=True)
                if not technical and (step+1) in (40,80):
                    directory=out/f'adapter_step{step+1}';model.save_pretrained(directory,safe_serialization=True)
                    receipt['checkpoints'].append(str(directory));save_receipt()
                    torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'step':step+1,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                        'numpy_rng':np.random.get_state()},out/'recovery_state.pt')
            base_after=fingerprint(model,False);adapter_after=fingerprint(model,True)
            if base_before!=base_after:raise ValueError('Frozen backbone/reference drift')
            reference_unchanged=True
            if fixed is not None:
                row,ids,prefix=fixed;batch,_=completion_batch(engine,row,ids,prefix);model.eval()
                with model.disable_adapter():after=token_logps(engine,batch,ids).cpu().tolist()
                reference_unchanged=after==ref_before
                if not reference_unchanged:raise ValueError('Disabled-adapter S0 reference changed after real updates')
            if technical and (adapter_before==adapter_after or nonzero_steps==0):raise RuntimeError('NO_ACTUAL_UPDATE_SIGNAL_IN_FOUR_TECHNICAL_STEPS')
            receipt.update(status='ACTUAL_UPDATE_CHECK_PASSED' if technical else 'COMPLETED_ACTUAL_TRAINING',ended=now(),
                frozen_after=base_after,final_adapter=adapter_after,nonzero_gradient_positive_LR_steps=nonzero_steps,
                reference_function_unchanged=reference_unchanged,max_actual_loss_reconstruction_error=max(loss_errors) if loss_errors else None,
                mean_step_seconds=float(np.mean(times)),total_step_seconds=float(sum(times)),
                peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
                technical_adapter_discarded=technical,formal_endpoint=str(out/'adapter_step80') if not technical else None)
            if not technical:receipt['endpoint_files']={str(p.relative_to(out/'adapter_step80')):sha(p) for p in (out/'adapter_step80').iterdir() if p.is_file()}
            save_receipt()
        except BaseException as exc:
            receipt.update(status='INTERRUPTED_OR_FAILED',ended=now(),error=f'{type(exc).__name__}: {exc}')
            if receipt['completed_optimizer_steps'] and not technical:
                emergency=out/'safe_interrupted_adapter';model.save_pretrained(emergency,safe_serialization=True);receipt['safe_adapter']=str(emergency)
            save_receipt();raise
    print(tag+' '+receipt['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['G_plain','G_guard','RSFT'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--technical',action='store_true');a=p.parse_args();train(a.arm,a.seed,a.technical)
