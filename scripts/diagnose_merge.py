#!/usr/bin/env python3
"""CPU streaming proof of BF16 LoRA merge algebra; never modifies model files.

Checks every LoRA matrix against round_to_bf16(float32(W)+alpha/r*(B@A)),
all non-LoRA tensors against the original bit values, and reports quantization
residuals. Optional synthetic-input linear checks illustrate finite-precision
non-equivalence; they are explicitly not real-video activations or GPU results.
The previously failed full-model logp threshold remains failed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(value,reason):
    if not value:raise ValueError(reason)


class TensorStore:
    def __init__(self,root,stack):
        from safetensors import safe_open
        self.files=sorted(Path(root).glob('*.safetensors'))
        require(self.files,'No safetensors model files')
        self.mapping={}
        for path in self.files:
            handle=stack.enter_context(safe_open(str(path),framework='pt',device='cpu'))
            for key in handle.keys():
                require(key not in self.mapping,'Duplicate tensor name across shards')
                self.mapping[key]=handle
    def tensor(self,key):return self.mapping[key].get_tensor(key)
    def keys(self):return set(self.mapping)


def synthetic_linear_check(base,merged,a,b,scaling,key):
    import torch
    import torch.nn.functional as F
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn(8,base.shape[1],generator=generator,dtype=torch.float32).to(torch.bfloat16)
    # PEFT active path: BF16 base matmul, FP32 LoRA branch, final cast to BF16.
    active=(F.linear(x,base).float()+F.linear(F.linear(x.float(),a.float()),b.float())*scaling).to(torch.bfloat16)
    fused=F.linear(x,merged)
    exact32=F.linear(x.float(),base.float()+(b.float()@a.float())*scaling)
    def errors(first,second):
        delta=(first.float()-second.float()).abs()
        return {'mean_abs':float(delta.mean()),'max_abs':float(delta.max()),
                'different_fraction':float((first.float()!=second.float()).float().mean())}
    return {'source':'deterministic synthetic Gaussian BF16 inputs; CPU only; not video activations',
            'seed':seed,'shape':list(x.shape),'active_vs_bf16_merged':errors(active,fused),
            'active_vs_fp32_effective_linear':errors(active,exact32),
            'bf16_merged_vs_fp32_effective_linear':errors(fused,exact32)}


def diagnose(args,report):
    import torch
    torch.set_num_threads(args.threads)
    base_root,adapter_root,merged_root=map(lambda p:p.resolve(),(args.base,args.adapter,args.merged))
    report.update(base=str(base_root),adapter=str(adapter_root),merged=str(merged_root),
                  cpu_threads=args.threads,torch_version=torch.__version__,gpu_used=False)
    adapter_config=json.loads((adapter_root/'adapter_config.json').read_text())
    require(adapter_config['r']==16 and adapter_config['lora_alpha']==32 and adapter_config['lora_dropout']==0,
            'Adapter rank/alpha/dropout differs from fixed first-round specification')
    require(not adapter_config.get('rank_pattern') and not adapter_config.get('alpha_pattern')
            and not adapter_config.get('use_dora') and not adapter_config.get('use_rslora')
            and not adapter_config.get('fan_in_fan_out') and adapter_config.get('bias')=='none',
            'This exact-algebra diagnosis does not silently assume alternative LoRA rules')
    scaling=float(adapter_config['lora_alpha'])/adapter_config['r']
    require(scaling==2.,'Unexpected LoRA scaling')
    report['merge_rule']='float32(base_BF16)+2*(B_FP32@A_FP32), then one round to BF16'
    report['adapter_config_sha256']=sha(adapter_root/'adapter_config.json')
    config=json.loads((base_root/'config.json').read_text())
    with ExitStack() as stack:
        base,adapter,merged=(TensorStore(path,stack) for path in (base_root,adapter_root,merged_root))
        require(base.keys()==merged.keys(),'Merged model tensor names differ from original model')
        a_keys=sorted(k for k in adapter.keys() if '.lora_A.' in k)
        require(len(a_keys)==config['text_config']['num_hidden_layers']*7,
                'Not all expected language linear matrices have a LoRA A/B pair')
        tensor_mapping={}
        adapter_consumed=set()
        for akey in a_keys:
            bkey=akey.replace('.lora_A.','.lora_B.')
            require(bkey in adapter.keys(),'Missing LoRA B pair')
            stem=akey.split('.lora_A.')[0]
            matches=[key for key in base.keys() if key.endswith('.weight') and stem.endswith(key[:-7])]
            require(len(matches)==1,'Cannot uniquely bind a saved LoRA matrix to its original base tensor')
            key=matches[0]
            require(key not in tensor_mapping and '.language_model.' in key,'Duplicate or non-language LoRA target')
            tensor_mapping[key]=(akey,bkey)
            adapter_consumed.update((akey,bkey))
        require(adapter_consumed==adapter.keys(),'Unaccounted adapter tensors require separate merge rules')
        sums={'lora_elements':0,'quantized_unchanged_elements':0,'different_from_exact_bf16_elements':0,
              'delta_square_sum':0.,'quantization_error_square_sum':0.,'actual_delta_square_sum':0.,
              'max_abs_quantization_error':0.,'max_abs_delta':0.,'non_lora_elements':0}
        details=[];non_lora=[];failures=[]
        for index,key in enumerate(sorted(base.keys())):
            w0,wm=base.tensor(key),merged.tensor(key)
            require(w0.shape==wm.shape and w0.dtype==wm.dtype,f'Shape/dtype changed: {key}')
            if key not in tensor_mapping:
                equal=torch.equal(w0.reshape(-1).view(torch.uint8),wm.reshape(-1).view(torch.uint8))
                sums['non_lora_elements']+=w0.numel()
                non_lora.append({'tensor':key,'shape':list(w0.shape),'dtype':str(w0.dtype),
                                 'n_elements':w0.numel(),'bitwise_unchanged':bool(equal)})
                if not equal:failures.append({'tensor':key,'reason':'non-LoRA tensor changed'})
                continue
            require(w0.dtype==torch.bfloat16,f'Expected BF16 base/merged weights: {key}')
            akey,bkey=tensor_mapping[key]
            a,b=adapter.tensor(akey),adapter.tensor(bkey)
            require(a.shape[0]==b.shape[1]==16 and a.shape[1]==w0.shape[1] and b.shape[0]==w0.shape[0],
                    'A/B/base matrix shapes disagree')
            delta=(b.float()@a.float())*scaling
            exact32=w0.float()+delta
            expected=exact32.to(torch.bfloat16)
            unequal=wm.view(torch.int16)!=expected.view(torch.int16)
            mismatch=int(unequal.sum())
            actual_delta=wm.float()-w0.float()
            error=wm.float()-exact32
            n=w0.numel()
            unchanged=int((wm.view(torch.int16)==w0.view(torch.int16)).sum())
            d2=float(delta.square().sum(dtype=torch.float64));e2=float(error.square().sum(dtype=torch.float64))
            actual_d2=float(actual_delta.square().sum(dtype=torch.float64))
            max_error=float(error.abs().max());max_delta=float(delta.abs().max())
            sums['lora_elements']+=n;sums['quantized_unchanged_elements']+=unchanged
            sums['different_from_exact_bf16_elements']+=mismatch
            sums['delta_square_sum']+=d2;sums['quantization_error_square_sum']+=e2
            sums['actual_delta_square_sum']+=actual_d2
            sums['max_abs_quantization_error']=max(sums['max_abs_quantization_error'],max_error)
            sums['max_abs_delta']=max(sums['max_abs_delta'],max_delta)
            detail={'tensor':key,'shape':list(w0.shape),'base_dtype':str(w0.dtype),
                    'adapter_A_dtype':str(a.dtype),'adapter_B_dtype':str(b.dtype),'scaling':scaling,
                    'n_elements':n,'exact_bf16_mismatch_elements':mismatch,
                    'quantized_unchanged_fraction':unchanged/n,
                    'rms_exact_delta':math.sqrt(d2/n),'rms_quantization_error':math.sqrt(e2/n),
                    'max_abs_quantization_error':max_error}
            if mismatch:
                detail['max_abs_actual_vs_expected_bf16']=float((wm.float()-expected.float()).abs().max())
                failures.append({'tensor':key,'reason':'actual merged tensor differs from defined BF16 merge',
                                 'n_different':mismatch,'max_abs':detail['max_abs_actual_vs_expected_bf16']})
            layer=int(key.split('.layers.')[1].split('.')[0])
            if layer in {0,config['text_config']['num_hidden_layers']//2,config['text_config']['num_hidden_layers']-1} and any(
                    key.endswith('.'+name+'.weight') for name in ('q_proj','v_proj','down_proj')):
                detail['synthetic_linear_arithmetic']=synthetic_linear_check(w0,wm,a,b,scaling,key)
            details.append(detail)
            if len(details)%28==0:
                print(json.dumps({'lora_matrices_checked':len(details),'mismatch_elements':sums['different_from_exact_bf16_elements']}),flush=True)
        n=sums['lora_elements']
        report.update(n_model_tensors=len(base.keys()),n_lora_matrices=len(details),n_non_lora_tensors=len(non_lora),
                      per_lora_matrix=details,non_lora_tensors=non_lora,failures=failures,
                      aggregate={**sums,'quantized_unchanged_fraction':sums['quantized_unchanged_elements']/n,
                                 'rms_exact_delta':math.sqrt(sums['delta_square_sum']/n),
                                 'rms_quantization_error':math.sqrt(sums['quantization_error_square_sum']/n),
                                 'rms_actual_delta':math.sqrt(sums['actual_delta_square_sum']/n)},
                      merge_formula_verified_all_tensors=not failures)
        report['weight_files_sha256']={role:{path.name:sha(path) for path in store.files}
                                       for role,store in (('base',base),('adapter',adapter),('merged',merged))}
    if args.snapshot and args.compare:
        snapshot=json.loads(args.snapshot.read_text());compared=json.loads(args.compare.read_text())
        require(Path(snapshot['model_path']).resolve()==base_root and Path(snapshot['adapter_path']).resolve()==adapter_root
                and Path(compared['model_path']).resolve()==merged_root,'Diagnosis files do not bind to the actual logp comparison')
        require(compared['snapshot_sha256']==sha(args.snapshot)
                and snapshot['fixed_training_example']==compared['fixed_training_example'],
                'Logp comparison input/snapshot identities differ')
        require(snapshot['fixed_answer']['input_ids_sha256']==compared['fixed_answer']['input_ids_sha256']
                and snapshot['fixed_answer']['answer_token_ids']==compared['fixed_answer']['answer_token_ids'],
                'Fixed input/answer token identities differ')
        report['observed_full_model_logp_comparison']={'snapshot_path':str(args.snapshot.resolve()),
                'snapshot_sha256':sha(args.snapshot),'compare_path':str(args.compare.resolve()),
                'compare_sha256':sha(args.compare),'original_status_retained':compared['status'],
                'numeric':compared['merge_logp_comparison'],
                'greedy_equal':compared['greedy_equal_before_after_merge'],
                'token_identities_equal':True,'threshold_modified':False,
                'limit':'Saved forward reports bind metadata/tokens; actual pixel equality is separately proved by processor restoration and media seals.'}
    require(report['merge_formula_verified_all_tensors'],'Full-tensor merge proof failed; retain concrete failures')
    report['conclusion']='Every merged tensor follows the registered BF16 arithmetic; this does not assert functional equivalence. Pre-merge to merged probability differences remain measured quantization/operation-rounding effects.'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','adapter','merged','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--snapshot',type=Path);p.add_argument('--compare',type=Path)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    require(args.threads>=1,'CPU thread count must be positive')
    require(not args.out.exists(),'Use a new diagnosis report')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),
            'model_files_modified':False,'gpu_used':False,'gpu_training_verified':False}
    started=time.monotonic()
    try:
        diagnose(args,report);report['status']='COMPLETED'
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,ended_utc=datetime.now(timezone.utc).isoformat())
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
