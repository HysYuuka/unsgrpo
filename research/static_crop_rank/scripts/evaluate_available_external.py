from pathlib import Path
import sys,json,time,gc
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from croprank.core import load_records,load_image,metrics,keyed_seed
from croprank.train import write_json
from croprank.musiq import MusiqRank
from croprank.gaic import GAIC
if not (ROOT/'reports/FROZEN_CHECKPOINTS.json').exists():raise RuntimeError('Test outputs are locked until controlled matrix selection is finished')
records=load_records('/root/autodl-fs/highlight-static-rank/data/gaic_manifest.jsonl','test');timing={r['source_id'] for r in sorted(records,key=lambda r:keyed_seed('timing20261030',r['source_id']))[:100]};out=Path('/root/autodl-fs/highlight-static-rank/runs/external');out.mkdir(exist_ok=True);torch.set_num_threads(4)
report={}
for name,cls in [('GAIC_official_weights_ported',GAIC),('MUSIQ_AVA_stock',MusiqRank)]:
 model=cls().cuda().eval();rows=[];cost=[];torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for r in records:
   torch.cuda.synchronize();t=time.perf_counter();im,b,y=load_image(r);b=b.cuda();s=model.predict(im,b).float().cpu().tolist();torch.cuda.synchronize();seconds=time.perf_counter()-t
   rows.append({'source_id':r['source_id'],'scores':s,'mos':y.tolist(),'metrics':metrics(s,y.tolist())})
   if r['source_id'] in timing:cost.append(seconds)
 summary={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']};report[name]={'metrics':summary,'p50_seconds':float(np.median(cost)),'p95_seconds':float(np.percentile(cost,95)),'peak_bytes':torch.cuda.max_memory_allocated(),'inference':'all native annotated candidates on all 500 original test images','comparison_category':'published-model reference; not controlled same-training-data arm'}
 (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows));del model;gc.collect();torch.cuda.empty_cache()
write_json(ROOT/'reports/AVAILABLE_EXTERNAL_RESULTS.json',report)
status=json.loads((ROOT/'reports/EXTERNAL_STATUS.json').read_text());status['GAIC']['status']='EVALUATED_OFFICIAL_WEIGHTS_ARITHMETIC_PORT';status['MUSIQ_AVA_stock']['status']='EVALUATED';write_json(ROOT/'reports/EXTERNAL_STATUS.json',status)
print('Available published-model references completed; FGAesQ/GAFIC/CPC gaps remain explicit')
