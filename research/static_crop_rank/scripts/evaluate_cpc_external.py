"""Complete frozen CPC evaluation, after the controlled queue and GPU timing finish."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import gc,json,os,sys,time,signal,hashlib
ROOT=Path(__file__).resolve().parents[1];STORE=Path('/root/autodl-fs/highlight-static-rank');sys.path.insert(0,str(ROOT/'src'))

def write(p,d):
 p=Path(p);temp=p.with_suffix(p.suffix+'.partial');temp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(p)

def summarize(tables,records):
 import numpy as np
 from scipy.stats import rankdata
 byid={r['source_id']:r for r in records};names=list(tables);groups=sorted({r['source_group'] for r in records})
 arrays={};summaries={}
 for name,rows in tables.items():
  assert [r['source_id'] for r in rows]==list(byid)
  grouped=defaultdict(list)
  for r in rows:
   y=np.asarray(r['mos'],dtype=float);s=np.asarray(r['scores'],dtype=float);order=sorted(range(len(s)),key=lambda i:(-s[i],i));ranked=sorted(y,reverse=True)
   independent={'regret':float(max(y)-y[order[0]])}
   for n in (5,10):
    for k in (1,2,3,4):independent[f'acc{k}_{n}']=sum(y[j]>=ranked[min(n,len(y))-1] for j in order[:k])/min(k,len(y))
    independent[f'mean_acc_{n}']=float(np.mean([independent[f'acc{k}_{n}'] for k in (1,2,3,4)]))
   a=rankdata(s);b=rankdata(y);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b);independent['srcc']=float(a.dot(b)/den) if den else 0.
   for k,v in independent.items():assert abs(v-r['metrics'][k])<1e-10,(name,r['source_id'],k)
   grouped[r['source_group']].append(r['metrics'])
  arrays[name]={k:np.asarray([np.mean([r[k] for r in grouped[g]]) for g in groups]) for k in next(iter(grouped.values()))[0]}
  summaries[name]={k:float(v.mean()) for k,v in arrays[name].items()}
 flagged={r['source_group'] for r in records if r['overlap_with_GAIC_TEST']}
 masks={'all_frozen_external_sources':np.ones(len(groups),dtype=bool),'also_excluding_possible_GAIC_TEST_overlap':np.asarray([g not in flagged for g in groups])}
 comparisons={}
 for cohort,mask in masks.items():
  entry={'source_groups':int(mask.sum()),'comparisons':{}}
  for other in ('MATCHED','MUSIQ_FT'):
   pair={}
   for metric in ('acc1_5','srcc','regret'):
    delta=np.stack([arrays[f'FULL_s{s}'][metric]-arrays[f'{other}_s{s}'][metric] for s in (17,29,43)])[:,mask];average=delta.mean(0)
    rng=np.random.default_rng(20261030);boot=np.asarray([average[rng.integers(0,len(average),len(average))].mean() for _ in range(10000)]) if len(average) else np.asarray([])
    pair[metric]={'mean_delta':float(average.mean()) if len(average) else None,'seed_deltas':delta.mean(1).tolist() if len(average) else [],'ci95':np.percentile(boot,[2.5,97.5]).tolist() if len(boot) else None,'ci97_5':np.percentile(boot,[1.25,98.75]).tolist() if len(boot) else None}
   entry['comparisons'][other]=pair
  comparisons[cohort]=entry
 seeds={}
 for variant in ('FULL','MATCHED','MUSIQ_FT'):
  seeds[variant]={k:{'mean':float(np.mean([summaries[f'{variant}_s{s}'][k] for s in (17,29,43)])),'sample_SD':float(np.std([summaries[f'{variant}_s{s}'][k] for s in (17,29,43)],ddof=1))} for k in summaries[f'{variant}_s17']}
 return {'status':'ALL_FROZEN_CPC_PREDICTIONS_INDEPENDENTLY_RECHECKED','runs':summaries,'seed_summary':seeds,'cohorts':comparisons,'bootstrap_unit':'CPC source group, images averaged then aligned seeds averaged; seeds not resampled','regret_units':'CPC native assigned preference scores, not GAIC MOS','sample_size_not_augmented_by_candidates_or_seeds':True}

def main():
 state=json.loads((ROOT/'STATUS.json').read_text());assert state['state']=='CONTROLLED_MATRIX_COMPLETE','Do not expose external labels or use GPU before the core matrix and timing complete'
 for path,expected in json.loads((ROOT/'reports/CPC_EVALUATION_CODE_FREEZE.json').read_text())['files'].items():
  assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
 interrupted=False
 def stopping(sig,frame):
  nonlocal interrupted
  interrupted=True
 signal.signal(signal.SIGTERM,stopping)
 import subprocess
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 assert not [l for l in active.splitlines() if l.strip() and int(l.strip())!=os.getpid()],'Another GPU process is active'
 import torch,numpy as np
 from PIL import Image
 from croprank.core import sha,metrics
 from croprank.train import build_model
 from croprank.replay import predict
 from croprank.gaic import GAIC
 from croprank.musiq import MusiqRank
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 lock=json.loads((ROOT/'reports/CPC_EXTERNAL_LOCK.json').read_text());manifest=Path(lock['report']['manifest_path']);assert sha(manifest)==lock['report']['manifest_sha256']
 assert sha(ROOT/'configs/cpc_external_protocol.json')==lock['report']['protocol_sha256']
 records=[json.loads(l) for l in manifest.read_text().splitlines()];assert records
 frozen=json.loads((ROOT/'reports/FROZEN_CHECKPOINTS.json').read_text());published=json.loads((ROOT/'reports/PUBLISHED_WEIGHT_FREEZE_20261030.json').read_text());protocol=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());chunk=json.loads((ROOT/'ENGINEERING_RUNTIME.json').read_text())['candidate_chunk']
 jobs=[(f"{j['variant']}_s{j['seed']}",j) for j in protocol['matrix']]+[('GAIC_official_weights_ported',None),('MUSIQ_AVA_stock',None)]
 out=STORE/'runs/cpc_external';out.mkdir(exist_ok=True);tables={};costs={}
 for name,job in jobs:
  if __import__('shutil').disk_usage(STORE).free<10*1024**3:raise RuntimeError('Disk reserve below10GiB')
  checkpoint=STORE/'runs'/name/'best.pt' if job else Path(published[name]['path']);checkpoint_sha=sha(checkpoint)
  if job:assert checkpoint_sha==frozen[name]
  else:assert checkpoint_sha==published[name]['sha256']
  final=out/(name+'.jsonl');partial=out/(name+'.jsonl.partial');receipt=out/(name+'.receipt.json');rows=[]
  present=final if final.exists() else partial
  if present.exists():
   content=present.read_bytes()
   if not content.endswith(b'\n'):
    assert present==partial,'Completed predictions are immutable'
    boundary=content.rfind(b'\n')+1;(out/(name+'.interrupted_tail.txt')).write_bytes(content[boundary:]);content=content[:boundary];partial.write_bytes(content)
   rows=[json.loads(l) for l in content.splitlines()]
   for expected,r in zip(records,rows):assert r['source_id']==expected['source_id'] and r['manifest_sha256']==lock['report']['manifest_sha256'] and r['checkpoint_sha256']==checkpoint_sha
   assert len(rows)<=len(records)
  phase_start=time.monotonic();new_images=0
  if len(rows)<len(records):
   if job:
    model=build_model(job['variant'],job['seed']).cuda();weights=torch.load(checkpoint,map_location='cpu',weights_only=False);assert weights['protocol_sha256']==sha(ROOT/'PROTOCOL_LOCK.json');model.load_state_dict(weights['model'],strict=True);del weights
   else:model=(GAIC() if name.startswith('GAIC') else MusiqRank()).cuda()
   model.eval()
   with partial.open('a') as handle,torch.no_grad():
    for i in range(len(rows),len(records)):
     if interrupted:break
     r=records[i]
     with Image.open(r['path']) as source:im=source.convert('RGB')
     boxes=torch.tensor([c['bbox'] for c in r['candidates']],dtype=torch.float32,device='cuda')
     # No label tensor or annotation vector is passed to any model.
     values=predict(model,im,boxes,chunk) if job and job['variant']!='MUSIQ_FT' else model.predict(im,boxes)
     scores=values.float().cpu().tolist();assert len(scores)==r['n'] and np.isfinite(scores).all()
     y=[c['mos'] for c in r['candidates']]
     row={'source_id':r['source_id'],'source_group':r['source_group'],'source_sha256':r['sha256'],'crop_ids':[c['crop_id'] for c in r['candidates']],
      'scores':scores,'mos':y,'metrics':metrics(scores,y),'possible_GAIC_TEST_overlap':r['overlap_with_GAIC_TEST'],
      'manifest_sha256':lock['report']['manifest_sha256'],'checkpoint_sha256':checkpoint_sha}
     handle.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row);new_images+=1
     if (i+1)%100==0:
      handle.flush();os.fsync(handle.fileno());write(ROOT/'reports/EXTERNAL_POSTPROCESS_STATUS.json',{'phase':'CPC_INFERENCE','run':name,'originals_complete':i+1,'originals_total':len(records),'pid':os.getpid(),'time_HK':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()})
     if (i+1)%1000==0:print(name,i+1,len(records),flush=True)
    handle.flush();os.fsync(handle.fileno())
   del model;gc.collect();torch.cuda.empty_cache()
   if interrupted:raise SystemExit(143)
   partial.replace(final)
  assert len(rows)==len(records)
  if not final.exists():partial.replace(final)
  tables[name]=rows;costs[name]={'this_invocation_wall_seconds':time.monotonic()-phase_start,'this_invocation_new_images':new_images,'total_prediction_images':len(rows),'checkpoint_sha256':checkpoint_sha,'prediction_sha256':sha(final),'native_candidates':sum(r['n'] for r in records)}
  write(receipt,costs[name])
 result=summarize(tables,records);result.update(costs=costs,input_manifest_sha256=sha(manifest),frozen_core_checkpoints=frozen)
 write(ROOT/'reports/CPC_EXTERNAL_RESULTS.json',result)
 external=json.loads((ROOT/'reports/EXTERNAL_STATUS.json').read_text());external['CPC'].update(status='ALL12_CONTROLLED_AND2_PUBLISHED_MODELS_EVALUATED',result='reports/CPC_EXTERNAL_RESULTS.json');write(ROOT/'reports/EXTERNAL_STATUS.json',external)
 print('Complete frozen CPC inference and independent source-group statistics',flush=True)

if __name__=='__main__':main()
