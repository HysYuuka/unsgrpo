from pathlib import Path
import argparse,collections,gc,json,math,os,random,signal,time
import numpy as np
import torch
from .core import seed_all,keyed_seed,load_records,load_image,metrics,sha
from .model import CropRank
from .musiq import MusiqRank
from .replay import replay_backward,predict
ROOT=Path(__file__).resolve().parents[2];STORAGE=Path('/root/autodl-fs/highlight-static-rank')

def build_model(variant,seed):
 seed_all(seed)
 if variant=='MUSIQ_FT':return MusiqRank()
 # Construct common initial tensors identically, despite variant-specific layer sizes.
 base=CropRank('FULL'); common={k:v.clone() for k,v in base.state_dict().items()}
 if variant=='FULL':return base
 del base;seed_all(seed);model=CropRank(variant);state=model.state_dict()
 for k in state:
  if k in common and state[k].shape==common[k].shape:state[k]=common[k]
 model.load_state_dict(state,strict=True);return model

def param_groups(model,variant):
 groups={}
 for n,p in model.named_parameters():
  pre=variant=='MUSIQ_FT' or n.startswith('semantic.') or n.startswith(('detail.stem.','detail.layer1.','detail.layer2.'))
  lr=1e-5 if pre else 1e-4;decay=0 if p.ndim<2 or n.endswith('bias') else .05
  groups.setdefault((lr,decay),[]).append(p)
 return [{'params':p,'lr':lr,'initial_lr':lr,'weight_decay':decay} for (lr,decay),p in groups.items()]

def schedule(epoch_fraction,extend=False):
 if epoch_fraction<2:return max(1e-6,epoch_fraction/2)
 if epoch_fraction<=30:return .1+.9*(1+math.cos(math.pi*(epoch_fraction-2)/28))/2
 return .01+.09*(1+math.cos(math.pi*(min(epoch_fraction,50)-30)/20))/2

def save_checkpoint(path,obj):
 temp=path.with_suffix('.partial');torch.save(obj,temp);temp.replace(path)

def write_json(path,obj):
 p=path.with_suffix(path.suffix+'.partial');p.write_text(json.dumps(obj,indent=2));p.replace(path)

@torch.no_grad()
def evaluate(model,records,outfile,variant,chunk):
 model.eval();rows=[]
 for r in records:
  im,b,y=load_image(r);b=b.cuda()
  scores=model.predict(im,b) if variant=='MUSIQ_FT' else predict(model,im,b,chunk)
  s=scores.cpu().tolist();m=metrics(s,y.tolist())
  rows.append({'source_id':r['source_id'],'source_sha256':r['sha256'],'crop_ids':[c['crop_id'] for c in r['candidates']],'scores':s,'mos':y.tolist(),'metrics':m})
 outfile.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
 return {k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}

def train(variant,seed,epochs=30,chunk=4):
 if (ROOT/'STOP_REQUESTED_BY_USER.json').exists():
  raise RuntimeError('Experiments stopped by user; explicit renewed authorization is required to resume.')
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 manifest=STORAGE/'data/gaic_manifest.jsonl';lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());assert lock['manifest_sha256']==sha(manifest)
 for name in ['ACCEPTANCE_NUMERIC.json','MUSIQ_ACCEPTANCE.json','ADDITIONAL_ACCEPTANCE.json']:
  assert (ROOT/'reports'/name).is_file(),name
 assert json.loads((ROOT/'reports/ACCEPTANCE_NUMERIC.json').read_text())['status']=='passed'
 assert json.loads((ROOT/'reports/ADDITIONAL_ACCEPTANCE.json').read_text())['status']=='passed'
 train_records=load_records(manifest,'train');val_records=load_records(manifest,'val');run=STORAGE/'runs'/f'{variant}_s{seed}';run.mkdir(parents=True,exist_ok=True)
 model=build_model(variant,seed).cuda().train();opt=torch.optim.AdamW(param_groups(model,variant),betas=(.9,.999));global_step=0;start_epoch=0;next_index=0;best=(-float('inf'),-float('inf'));epoch_terms=[];history=[];wall_start=time.time();durations=[]
 if (run/'last.pt').exists():
  state=torch.load(run/'last.pt',map_location='cpu',weights_only=False);assert state['protocol_sha256']==sha(ROOT/'PROTOCOL_LOCK.json')
  model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);global_step=state['step'];start_epoch=state['epoch'];next_index=state['next_index'];best=tuple(state['best']);history=state['history'];epoch_terms=state['epoch_terms']
  torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state(state['rng_cuda']);random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);del state
 stop=False
 def stopping(sig,frame):
  nonlocal stop;stop=True
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 def ckpt(epoch,next_index):
  save_checkpoint(run/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'variant':variant,'seed':seed,'epoch':epoch,'next_index':next_index,'step':global_step,'best':best,'history':history,'epoch_terms':epoch_terms,'protocol_sha256':sha(ROOT/'PROTOCOL_LOCK.json'),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(),'rng_python':random.getstate(),'rng_numpy':np.random.get_state()})
 for epoch in range(start_epoch,epochs):
  model.train();order=sorted(train_records,key=lambda r:keyed_seed(seed,epoch,r['source_id']))
  for begin in range(next_index,len(order),8):
   batch=order[begin:begin+8];opt.zero_grad(set_to_none=True);start=time.perf_counter();factor=schedule(epoch+(begin+len(batch))/len(order),epochs>30)
   for g in opt.param_groups:g['lr']=g['initial_lr']*factor
   batch_terms=[]
   for r in batch:
    im,b,y=load_image(r,seed,epoch,True);b=b.cuda();y=y.cuda()
    if variant=='MUSIQ_FT':terms,_=model.replay(im,b,y,scale=1/len(batch))
    else:terms,_=replay_backward(model,im,b,y,chunk,1/len(batch),selection=variant not in ['GEOM_BASE','MATCHED_BASE'])
    if not all(np.isfinite(list(terms.values()))):raise FloatingPointError((r['source_id'],terms))
    batch_terms.append(terms);epoch_terms.append(terms)
   norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();global_step+=1;torch.cuda.synchronize();duration=time.perf_counter()-start;durations.append(duration)
   entry={'epoch':epoch+1,'step':global_step,'next_index':begin+len(batch),'images':len(batch),'candidates':sum(r['n'] for r in batch),'loss':{k:float(np.mean([r[k] for r in batch_terms])) for k in batch_terms[0]},'gradient_norm_before_clip':norm,'seconds':duration,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'lr_factor':factor,'candidate_chunk':chunk}
   with (run/'steps.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
   write_json(ROOT/'STATUS.json',{'state':'TRAINING','variant':variant,'seed':seed,'epoch':epoch+1,'step':global_step,'images_in_epoch':begin+len(batch),'epoch_images':len(order),'last_step':entry,'pid':os.getpid(),'updated_unix':time.time()})
   if global_step%10==0:print(json.dumps(entry),flush=True)
   if global_step%50==0 or stop:ckpt(epoch,begin+len(batch))
   if len(durations)==100:
    estimate={'variant':variant,'seed':seed,'observed_optimizer_steps':100,'candidate_chunk':chunk,'global_step_end':global_step,'all_candidates':True,'median_step_seconds':float(np.median(durations)),'mean_step_seconds':float(np.mean(durations)),'train_only_hours_one_30epoch_run':float(np.mean(durations))*math.ceil(len(order)/8)*30/3600,'validation_excluded':True,'do_not_apply_to_other_architectures':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    write_json(run/f'THROUGHPUT_100_chunk{chunk}_end{global_step}.json',estimate);write_json(run/'THROUGHPUT_ACTIVE.json',estimate);print('THROUGHPUT '+json.dumps(estimate),flush=True)
   if stop:return
  next_index=0
  val=evaluate(model,val_records,run/'last_val_predictions.jsonl',variant,chunk)
  summary={'epoch':epoch+1,'train':{k:float(np.mean([r[k] for r in epoch_terms])) for k in epoch_terms[0]},'val':val,'step':global_step,'elapsed_process_seconds':time.time()-wall_start}
  history.append(summary);epoch_terms=[];quality=(val['acc1_5'],-val['regret'])
  if quality>best:
   best=quality;save_checkpoint(run/'best.pt',{'model':model.state_dict(),'variant':variant,'seed':seed,'epoch':epoch+1,'val':val,'protocol_sha256':sha(ROOT/'PROTOCOL_LOCK.json')})
   (run/'best_val_predictions.jsonl').write_bytes((run/'last_val_predictions.jsonl').read_bytes())
  write_json(run/'history.json',history);ckpt(epoch+1,0);print('EPOCH '+json.dumps(summary),flush=True)
 write_json(run/'TRAINING_COMPLETE.json',{'epochs':epochs,'best':best,'steps':global_step,'best_sha256':sha(run/'best.pt'),'test_not_evaluated':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['FULL','MATCHED','MUSIQ_FT','GEOM_BASE','MATCHED_BASE','NO_BOUNDARY']);p.add_argument('--seed',required=True,type=int);p.add_argument('--epochs',type=int,default=30);p.add_argument('--chunk',type=int,default=4);a=p.parse_args();train(a.variant,a.seed,a.epochs,a.chunk)
