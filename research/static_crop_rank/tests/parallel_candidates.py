from pathlib import Path
import sys,json,time,gc
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from croprank.train import build_model
from croprank.core import load_records,load_image
from croprank.replay import replay_backward
from croprank.musiq import MusiqRank
ROOT=Path(__file__).resolve().parents[1]
# Only execute after the formal worker has saved and exited, to measure exclusive GPU costs.
state=json.loads((ROOT/'STATUS.json').read_text());assert state['state']=='STOPPED_WITH_RECOVERABLE_CHECKPOINT',state['state']
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
records=load_records('/root/autodl-fs/highlight-static-rank/data/gaic_manifest.jsonl','train');record=max(records,key=lambda r:(r['width']*r['height'],r['n'],r['source_id']));im,boxes,mos=load_image(record);boxes=boxes.cuda();mos=mos.cuda()
m=build_model('FULL',17).cuda().train();report={'status':'running','source_id':record['source_id'],'size':im.size,'candidates':len(boxes),'purpose':'complete model/candidates software equivalence and throughput; no optimizer update, not a quality screening experiment','timing':{}}
def save():
 (ROOT/'reports/PARALLEL_CANDIDATES_ACCEPTANCE.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='parameter_errors'},flush=True)
for chunk in [4,16,32,64,90]:
 m.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();terms,s=replay_backward(m,im,boxes,mos,chunk=chunk,amp=True);torch.cuda.synchronize();duration=time.perf_counter()-start
 report['timing'][str(chunk)]={'seconds':duration,'peak_allocated':torch.cuda.max_memory_allocated(),'loss':terms['total']};save()
 if chunk==4:base={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters()};base_scores=s
 else:
  numerator=denominator=dot=gn=0.
  for n,p in m.named_parameters():
   x=p.grad.detach().float().cpu().double();y=base[n].double();numerator+=float((x-y).square().sum());denominator+=float(y.square().sum());dot+=float((x*y).sum());gn+=float(x.square().sum())
  error=(numerator/denominator)**.5;cos=dot/(denominator*gn)**.5
  report['timing'][str(chunk)].update(relative_gradient_l2=error,gradient_cosine=cos,score_max_abs=float((s-base_scores).abs().max()))
  assert error<.02 and cos>.999
  save()
best=min([4,16,32,64,90],key=lambda k:report['timing'][str(k)]['seconds']);report['selected_chunk']=best
# Full-resolution/all-candidate FP32 gradients, checked against the original chunk=4.
refs=None
for chunk in [4,best]:
 m.zero_grad(set_to_none=True);torch.cuda.empty_cache();terms,s=replay_backward(m,im,boxes,mos,chunk=chunk,amp=False)
 if refs is None:refs={n:p.grad.cpu().clone() for n,p in m.named_parameters()};ref_s=s;continue
 numerator=denominator=0.;worst=0.
 for n,p in m.named_parameters():
  x=p.grad.detach().cpu().double();y=refs[n].double();num=float((x-y).square().sum());den=float(y.square().sum());numerator+=num;denominator+=den
  if den>1e-14:worst=max(worst,(num/den)**.5)
 report['fp32']={'global_relative_l2':(numerator/denominator)**.5,'worst_material_tensor_relative_l2':worst,'score_max_abs':float((s-ref_s).abs().max())};assert report['fp32']['global_relative_l2']<1e-4 and worst<1e-4
report['status']='passed';save()
