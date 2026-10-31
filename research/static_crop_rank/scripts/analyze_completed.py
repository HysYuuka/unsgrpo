"""Post-training analysis only; does not select checkpoints or alter training."""
from pathlib import Path
import json,sys,gc,time
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from croprank.train import build_model,write_json
from croprank.core import load_records,load_image,keyed_seed,metrics
from croprank.replay import predict
RUNS=Path('/root/autodl-fs/highlight-static-rank/runs');lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());records=load_records('/root/autodl-fs/highlight-static-rank/data/gaic_manifest.jsonl','test');byid={r['source_id']:r for r in records};timing=sorted(records,key=lambda r:keyed_seed('timing20261030',r['source_id']))[:100]
if not (ROOT/'reports/FROZEN_CHECKPOINTS.json').exists():raise RuntimeError('No frozen complete matrix: test analysis cannot run')
torch.set_num_threads(4);hard={};cost={};groups={};failures={}
for job in lock['matrix']:
 name=f"{job['variant']}_s{job['seed']}";run=RUNS/name;rows=[json.loads(l) for l in (run/'test_predictions.jsonl').read_text().splitlines()]
 values=[];pairs=0
 for r in rows:
  boxes=np.array([c['bbox'] for c in byid[r['source_id']]['candidates']]);y=np.array(r['mos']);s=np.array(r['scores']);a,b=np.triu_indices(len(y),1)
  intersection=np.maximum(0,np.minimum(boxes[a,2:],boxes[b,2:])-np.maximum(boxes[a,:2],boxes[b,:2])).prod(1);area=(boxes[:,2:]-boxes[:,:2]).prod(1);iou=intersection/(area[a]+area[b]-intersection)
  mask=(iou>=.70)&(np.abs(y[a]-y[b])>=.5)
  if mask.any():
   margin=(s[a[mask]]-s[b[mask]])*(y[a[mask]]-y[b[mask]]);values.append(float(np.mean((margin>0)+.5*(margin==0))));pairs+=int(mask.sum())
 hard[name]={'originals':len(values),'pairs':pairs,'mean_per_original_pair_accuracy':float(np.mean(values)) if values else None,'tie_credit':.5}
 # Fixed error categories: selected crop too small/large relative to best MOS candidate,
 # and similar-area framing error; take highest regret, source_id tie-break.
 cases={}
 for r in rows:
  record=byid[r['source_id']];best=int(np.argmax(r['mos']));selected=int(np.argmax(r['scores']));bs=np.array(record['candidates'][best]['bbox']);bp=np.array(record['candidates'][selected]['bbox']);ratio=np.prod(bp[2:]-bp[:2])/np.prod(bs[2:]-bs[:2]);category='smaller_area' if ratio<.8 else 'larger_area' if ratio>1.25 else 'similar_area_framing'
  cases.setdefault(category,[]).append((r['metrics']['regret'],r['source_id'],selected,best))
 failures[name]={k:sorted(v,key=lambda z:(-z[0],z[1]))[:5] for k,v in cases.items()}
 # Time a single predeclared representative seed per architecture; all quality seeds retained.
 if job['seed']!=17:continue
 model=build_model(job['variant'],17).cuda();state=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);del state;model.eval();lat=[]
 with torch.no_grad():
  im,b,y=load_image(timing[0]);b=b.cuda();model.predict(im,b) if job['variant']=='MUSIQ_FT' else predict(model,im,b,chunk=json.loads((ROOT/'ENGINEERING_RUNTIME.json').read_text())['candidate_chunk'])
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  for r in timing:
   torch.cuda.synchronize();start=time.perf_counter();im,b,y=load_image(r);b=b.cuda();pre=time.perf_counter()-start
   s=model.predict(im,b) if job['variant']=='MUSIQ_FT' else predict(model,im,b,chunk=json.loads((ROOT/'ENGINEERING_RUNTIME.json').read_text())['candidate_chunk'])
   torch.cuda.synchronize();elapsed=time.perf_counter()-start;lat.append({'source_id':r['source_id'],'candidates':r['n'],'end_to_end_seconds':elapsed,'decode_and_box_seconds':pre})
 cost[name]={'median_seconds':float(np.median([x['end_to_end_seconds'] for x in lat])),'p95_seconds':float(np.percentile([x['end_to_end_seconds'] for x in lat],95)),'peak_bytes':torch.cuda.max_memory_allocated(),'rows':lat}
 del model;gc.collect();torch.cuda.empty_cache()
for v in ['FULL','MATCHED','MUSIQ_FT']:
 result=json.loads((ROOT/'reports/CONTROLLED_RESULTS.json').read_text())['runs'];group=[result[f'{v}_s{s}'] for s in [17,29,43]];groups[v]={k:{'mean':float(np.mean([x[k] for x in group])),'sample_sd':float(np.std([x[k] for x in group],ddof=1))} for k in group[0]}
write_json(ROOT/'reports/HARD_PAIRS.json',hard);write_json(ROOT/'reports/INFERENCE_TIMING.json',cost);write_json(ROOT/'reports/SEED_SUMMARY.json',groups);write_json(ROOT/'reports/FAILURE_CASES.json',failures)
# Representative transparent overlays: red prediction, green MOS best, chosen by fixed errors.
art=RUNS/'case_overlays';art.mkdir(exist_ok=True)
for category,cases in failures['FULL_s17'].items():
 for regret,sid,selected,best in cases:
  r=byid[sid];im=Image.open(r['path']).convert('RGB');d=ImageDraw.Draw(im);d.rectangle(r['candidates'][selected]['bbox'],outline='red',width=3);d.rectangle(r['candidates'][best]['bbox'],outline='green',width=3);im.save(art/f'{category}_{sid}.jpg')
print('Completed independent hard-pair, timing, seed SD and predeclared failure categories')
