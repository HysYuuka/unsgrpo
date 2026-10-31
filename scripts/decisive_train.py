#!/usr/bin/env python3
"""Minimal new DPO objective and existing actual GRPO path in explicit directories."""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
from unsgrpo.decisive_common import *
from unsgrpo.decisive_core import *
from unsgrpo.reward_to_rl_model import load_engine,generate,completion_batch,token_logps,media_hash
from reward_to_rl_train import fingerprint,independent_loss_check


def run(arm,technical=False,resume_long=False):
    validate();frozen=validate('training.freeze.json')
    if arm not in ('P-DPO','G-plain','G-guard') or (resume_long and arm!='P-DPO'):raise ValueError('Invalid arm')
    if technical and arm=='G-plain':raise ValueError('Guard technical path covers the shared GRPO implementation')
    tag=('technical_'+arm) if technical else arm;out=STORE/'training'/tag
    if resume_long:
        before_receipt=read(out/'receipt_80.json')
        if before_receipt['status']!='COMPLETED_ACTUAL_TRAINING':raise ValueError('DPO80 incomplete')
        if (out/'receipt_320.json').exists():raise FileExistsError('No repeated long endpoint')
        start_step,stop_step=80,320
    else:
        out.mkdir(parents=True,exist_ok=False);start_step=0;stop_step=4 if technical else 80
    receipt_path=out/('technical_receipt.json' if technical else 'receipt_320.json' if resume_long else 'receipt_80.json')
    receipt={'status':'RUNNING','arm':arm,'technical':technical,'seed':42,'started':now(),'start_step':start_step,'target_step':stop_step,
        'completed_optimizer_steps':0,'prompt_exposures':0,'online_completions':0,'skipped_zero_u_batches':0,
        'common_S0':reference(PROJECT/'reports/common_s0.json'),'training_freeze':reference(WORK/'training.freeze.json'),'human_labels_read':False}
    write(receipt_path,receipt)
    rows={r['request_id']:r for r in records(WORK/'train_windows.final.jsonl')};preferences={r['request_id']:r for r in records(STORE/'pool/preferences.jsonl')}
    schedule=read(WORK/'schedule.json')['request_ids'];edges=records(STORE/'teacher/train_edges.jsonl');emap={rid:[e for e in edges if e['request_id']==rid] for rid in rows}
    with gpu_session(tag+('_long' if resume_long else ''),'student' if technical else 'training'):
        import torch
        from verify_checkpoint import add_fresh_language_lora
        from peft.utils.save_and_load import load_peft_weights,set_peft_model_state_dict
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
        engine=load_engine(Path(read(PROJECT/'reports/common_s0.json')['merged_model']));model,module_names,names=add_fresh_language_lora(engine)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min((n+1)/4,1.))
        if resume_long:
            for name,h in before_receipt['endpoint_files'].items():assert sha(out/'adapter_80'/name)==h
            state=load_peft_weights(str(out/'adapter_80'),device='cpu');set_peft_model_state_dict(model,state)
            recovery=torch.load(out/'state80.pt',map_location=engine.model.device,weights_only=False)
            optimizer.load_state_dict(recovery['optimizer']);scheduler.load_state_dict(recovery['scheduler'])
            torch.set_rng_state(recovery['torch_rng'].cpu());torch.cuda.set_rng_state(recovery['cuda_rng'].cpu());np.random.set_state(recovery['numpy_rng'])
            if fingerprint(model,True)!=before_receipt['final_adapter']:raise ValueError('DPO resume adapter not bitwise identical to80')
            if recovery['step']!=80:raise ValueError('DPO optimizer resume step mismatch')
            receipt['resume_state']=reference(out/'state80.pt');receipt['resume_from80_verified']=True
        base_before=fingerprint(model,False);adapter_before=fingerprint(model,True);fixed=None;reference_fixed=None;times=[];errors=[];effective_steps=0
        receipt.update(frozen_before=base_before,initial_adapter=adapter_before,trainable_parameters=sum(p.numel() for p in params),LoRA_targets=module_names)
        def save():write(receipt_path,receipt,fresh=False)
        save()
        try:
            for step in range(start_step,stop_step):
                begin=time.monotonic();optimizer.zero_grad(set_to_none=True);observed=[];weights=[];group_logs=[]
                batch_ids=schedule[step*4:step*4+4]
                if len(batch_ids)!=4:raise ValueError('Fixed exposure schedule incomplete')
                for slot,rid in enumerate(batch_ids):
                    row=rows[rid];pref=preferences[rid];u=pref['u'];weights.append(u if arm!='G-plain' else 1.)
                    if arm=='P-DPO':
                        current=[];reference=[];bindings=[]
                        for sample in (pref['chosen'],pref['rejected']):
                            batch,audit=completion_batch(engine,row,sample['token_ids'],sample['prompt_token_ids']);model.eval()
                            with model.disable_adapter():ref=token_logps(engine,batch,sample['token_ids']).detach()
                            if fixed is None:fixed=(copy.deepcopy(row),list(sample['token_ids']),sample['prompt_token_ids']);reference_fixed=ref.cpu().tolist()
                            model.train();cur=token_logps(engine,batch,sample['token_ids'],grad=True)
                            current.append(cur);reference.append(ref);bindings.append({'token_ids':sample['token_ids'],'encoding':audit,'consumed_media_sha256':media_hash(batch)})
                            del batch
                        loss,logit=dpo_loss(current[0],current[1],reference[0],reference[1],u)
                        arrays=[v.detach().cpu().tolist() for v in (*current,*reference)]
                        z=.1*((sum(arrays[0])-sum(arrays[2]))-(sum(arrays[1])-sum(arrays[3])))
                        independent=u*(max(-z,0)+math.log1p(math.exp(-abs(z))));error=abs(independent-float(loss.detach()));errors.append(error)
                        if error>2e-5:raise ValueError('Actual DPO loss differs from summed-token scalar reconstruction')
                        (loss/4).backward();observed.append(float(loss.detach()))
                        append(out/('loss_long.jsonl' if resume_long else 'loss.jsonl'),{'step':step+1,'slot':slot,'request_id':rid,'u':u,
                            'loss':float(loss.detach()),'logit':float(logit),'independent_loss':independent,'error':error,'batch_denominator':4,
                            'current_chosen_logps':arrays[0],'current_rejected_logps':arrays[1],'reference_chosen_logps':arrays[2],'reference_rejected_logps':arrays[3],
                            'sum_not_mean':True,'reference_detached':True,'bindings':bindings,'prompt_loss_tokens':0})
                        del current,reference,loss
                    else:
                        model.eval();samples=[]
                        for g in range(4):
                            sample=generate(engine,row);sample.update(g=g,request_id=rid,step=step+1,slot=slot,reward=reward(sample['response'],row['clip_ids'],emap[rid]));samples.append(sample);append(out/'rollouts.jsonl',sample)
                        adv,raw,std=group_advantages([r['reward'] for r in samples],u,arm=='G-guard');kl_u=u if arm=='G-guard' else 1.
                        group_logs.append({'request_id':rid,'u':u,'rewards':[r['reward'] for r in samples],'raw_advantages':raw.tolist(),'actual_advantages':adv.tolist(),'sample_std':std})
                        for g,sample in enumerate(samples):
                            batch,audit=completion_batch(engine,row,sample['token_ids'],sample['prompt_token_ids']);model.eval();old=token_logps(engine,batch,sample['token_ids']).detach()
                            with model.disable_adapter():ref=token_logps(engine,batch,sample['token_ids']).detach()
                            if fixed is None:
                                fixed=(copy.deepcopy(row),list(sample['token_ids']),sample['prompt_token_ids']);reference_fixed=ref.cpu().tolist()
                                if not torch.equal(old,ref):raise ValueError('Fresh policy differs from frozen S0')
                            model.train();current=token_logps(engine,batch,sample['token_ids'],grad=True);loss,parts=sequence_grpo_loss(current,old,ref,adv[g],kl_u)
                            arrays=[v.detach().cpu().tolist() for v in (current,old,ref)];independent=independent_loss_check(*arrays,float(adv[g]),kl_u);error=abs(independent-float(loss.detach()));errors.append(error)
                            if error>2e-5:raise ValueError('Actual GRPO loss reconstruction failed')
                            (loss/16).backward();observed.append(float(loss.detach()))
                            append(out/'loss.jsonl',{'step':step+1,'slot':slot,'g':g,'request_id':rid,'u':u,'actual_advantage':float(adv[g]),'raw_advantage':float(raw[g]),'sample_std':std,
                                **parts,'independent_loss':independent,'error':error,'batch_denominator':16,'current_logps':arrays[0],'old_logps':arrays[1],'reference_logps':arrays[2],
                                'token_ids':sample['token_ids'],'encoding':audit,'consumed_media_sha256':media_hash(batch),'old_ref_adv_detached':True,'prompt_loss_tokens':0})
                            del batch,current,old,ref,loss
                        receipt['online_completions']+=4
                    receipt['prompt_exposures']+=1
                if not any(w>0 for w in weights):
                    optimizer.zero_grad(set_to_none=True);receipt['skipped_zero_u_batches']+=1
                    append(out/'optimizer.jsonl',{'step':step+1,'optimizer_called':False,'reason':'all_zero_reliability','momentum_and_scheduler_unchanged':True});raise RuntimeError('NO_GO_TRAIN_SIGNAL: zero reliability violates fixed effective-update exposure budget')
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                if norm>0 and lr>0:effective_steps+=1
                seconds=time.monotonic()-begin;times.append(seconds);receipt['completed_optimizer_steps']+=1
                append(out/('optimizer_long.jsonl' if resume_long else 'optimizer.jsonl'),{'step':step+1,'optimizer_called':True,'lr_before_step':lr,'gradient_norm':norm,'mean_sequence_loss':float(np.mean(observed)),
                    'prompt_exposures_this_receipt':receipt['prompt_exposures'],'online_completions_this_receipt':receipt['online_completions'],'groups':group_logs,'seconds':seconds})
                save()
                if not technical and step+1 in (40,80,320):model.save_pretrained(out/f'adapter_{step+1}',safe_serialization=True)
                if not technical and (step+1)%40==0:
                    if resume_long:model.save_pretrained(out/'recovery_adapter',safe_serialization=True)
                    state={'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step+1,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'numpy_rng':np.random.get_state()}
                    torch.save(state,out/('state80.pt' if step+1==80 and arm=='P-DPO' else 'recovery_state.pt'))
                if (step+1)%10==0:print(f'{tag}: {step+1}/{stop_step}, {seconds:.1f}s/step',flush=True)
            base_after=fingerprint(model,False);adapter_after=fingerprint(model,True)
            if base_after!=base_before:raise ValueError('Frozen S0 weights changed')
            row,ids,prefix=fixed;batch,_=completion_batch(engine,row,ids,prefix);model.eval()
            with model.disable_adapter():after=token_logps(engine,batch,ids).cpu().tolist()
            if after!=reference_fixed:raise ValueError('Fixed reference functional drift')
            if technical and (adapter_after==adapter_before or effective_steps==0):raise RuntimeError('NO_GO_TRAIN_SIGNAL: no actual technical parameter update')
            receipt.update(status='ACTUAL_UPDATE_CHECK_PASSED' if technical else 'COMPLETED_ACTUAL_TRAINING',ended=now(),frozen_after=base_after,final_adapter=adapter_after,
                reference_unchanged=True,nonzero_grad_positive_lr_updates=effective_steps,max_loss_reconstruction_error=max(errors),mean_step_seconds=float(np.mean(times)),
                summed_step_seconds=sum(times),peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),technical_adapter_discarded=technical,
                endpoint=str(out/f'adapter_{stop_step}') if not technical else None)
            if not technical:receipt['endpoint_files']={str(p.relative_to(out/f'adapter_{stop_step}')):sha(p) for p in (out/f'adapter_{stop_step}').iterdir() if p.is_file()}
            save()
        except BaseException as exc:
            receipt.update(status='INCOMPLETE',error=f'{type(exc).__name__}: {exc}',ended=now());save()
            if not technical and receipt['completed_optimizer_steps']:model.save_pretrained(out/'interrupted_safe_adapter',safe_serialization=True)
            raise
    print(receipt['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['P-DPO','G-plain','G-guard'],required=True);p.add_argument('--technical',action='store_true');p.add_argument('--resume-long',action='store_true');a=p.parse_args();run(a.arm,a.technical,a.resume_long)
