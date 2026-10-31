from pathlib import Path
import json,sys,gc,math
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from croprank.train import build_model,evaluate,write_json
from croprank.core import load_records,sha
RUNS=Path('/root/autodl-fs/highlight-static-rank/runs');lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());frozen=json.loads((ROOT/'reports/FROZEN_CHECKPOINTS.json').read_text());test=load_records('/root/autodl-fs/highlight-static-rank/data/gaic_manifest.jsonl','test');torch.set_num_threads(4)
results={};tables={}
for job in lock['matrix']:
 name=f"{job['variant']}_s{job['seed']}";run=RUNS/name;assert sha(run/'best.pt')==frozen[name]
 prediction=run/'test_predictions.jsonl'
 if not prediction.exists():
  model=build_model(job['variant'],job['seed']).cuda();state=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);del state
  summary=evaluate(model,test,prediction,job['variant'],json.loads((ROOT/'ENGINEERING_RUNTIME.json').read_text())['candidate_chunk']);del model;gc.collect();torch.cuda.empty_cache()
 rows=[json.loads(l) for l in prediction.read_text().splitlines()];tables[name]=rows;results[name]={k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}
# Independent implementation without calling training metrics, applied to every saved score.
from scipy.stats import rankdata
for name,rows in tables.items():
 for r in rows:
  s=r['scores'];y=r['mos'];order=sorted(range(len(s)),key=lambda j:(-s[j],j));ranked=sorted(y,reverse=True)
  expect={'regret':max(y)-y[order[0]]}
  for n in [5,10]:
   for k in [1,2,3,4]:expect[f'acc{k}_{n}']=sum(y[i]>=ranked[min(n,len(y))-1] for i in order[:k])/min(k,len(y))
   expect[f'mean_acc_{n}']=np.mean([expect[f'acc{k}_{n}'] for k in [1,2,3,4]])
  rs=rankdata(s);ry=rankdata(y);rs-=np.mean(rs);ry-=np.mean(ry);den=np.linalg.norm(rs)*np.linalg.norm(ry);expect['srcc']=float(np.dot(rs,ry)/den) if den else 0.
  for k,v in expect.items():assert abs(r['metrics'][k]-v)<1e-10,(name,r['source_id'],k)
write_json(ROOT/'reports/INDEPENDENT_METRIC_RECHECK.json',{'status':'passed','original_prediction_rows':sum(map(len,tables.values()))})
comparisons={};rng=np.random.default_rng(20261030)
for other in ['MATCHED','MUSIQ_FT']:
 diffs=[];regrets=[]
 for seed in [17,29,43]:
  a=tables[f'FULL_s{seed}'];b=tables[f'{other}_s{seed}'];assert [x['source_id'] for x in a]==[x['source_id'] for x in b]
  diffs.append([x['metrics']['acc1_5']-y['metrics']['acc1_5'] for x,y in zip(a,b)])
  regrets.append([x['metrics']['regret']-y['metrics']['regret'] for x,y in zip(a,b)])
 d=np.array(diffs);image_delta=d.mean(0);boot=np.array([image_delta[rng.integers(0,len(image_delta),len(image_delta))].mean() for _ in range(10000)])
 comparisons[other]={'mean_acc_gain':float(d.mean()),'seed_gains':d.mean(1).tolist(),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'ci97_5':np.percentile(boot,[1.25,98.75]).tolist(),'regret_change':float(np.mean(regrets)),'fixed_seed_average_original_cluster_bootstrap':True}
q=comparisons['MATCHED'];supported=q['mean_acc_gain']>=.02 and q['ci97_5'][0]>0 and sum(x>0 for x in q['seed_gains'])>=2 and q['regret_change']<=0
status='METHOD_SUPPORTED' if supported else 'METHOD_NOT_SUPPORTED' if q['ci97_5'][1]<.02 else 'INCONCLUSIVE'
# Persist possible ongoing learning at the maximum extension; never call an undertrained model ineffective.
extension=json.loads((ROOT/'reports/EXTENSION_DECISION.json').read_text())
if extension['extend_all_to_50']:
 for j in lock['matrix'][:9]:
  h=json.loads((RUNS/f"{j['variant']}_s{j['seed']}"/'history.json').read_text())
  if len(h)>=50 and h[44]['train']['total']>0 and (h[44]['train']['total']-h[49]['train']['total'])/h[44]['train']['total']>=.05 and np.mean([x['val']['acc1_5'] for x in h[47:50]])-np.mean([x['val']['acc1_5'] for x in h[41:44]])>=.01:status='INCONCLUSIVE'
write_json(ROOT/'reports/CONTROLLED_RESULTS.json',{'status':status,'runs':results,'comparisons':comparisons,'external_unfinished_does_not_invalidate_mechanism_test':True})
lines=['# 完整受控矩阵结果',f'\n状态：**{status}**。外部对手尚未完成时不能宣称全面竞争力或 SOTA。','\n| 运行 | Acc1/5 | SRCC | MOS 分差 |','|---|---:|---:|---:|']
for k,v in results.items():lines.append(f"| {k} | {v['acc1_5']*100:.2f}% | {v['srcc']:.4f} | {v['regret']:.4f} |")
lines+=['\n比较区间及逐原图预测见 CONTROLLED_RESULTS.json 和各 run/test_predictions.jsonl。','\n完整结论还需硬对机制子集、计时和外部比较报告；本文件不把未执行的外部结果当作已完成。']
(ROOT/'reports/DECISION.md').write_text('\n'.join(lines)+'\n')

import subprocess
subprocess.run([sys.executable,str(ROOT/"scripts/analyze_completed.py")],check=True)

subprocess.run([sys.executable,str(ROOT/"scripts/evaluate_available_external.py")],check=True)
