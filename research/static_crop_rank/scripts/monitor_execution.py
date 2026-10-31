"""Detached 30-minute, local-only health record; never launches or restarts training."""
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import fcntl,hashlib,json,os,shutil,signal,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
STORE=Path('/root/autodl-fs/highlight-static-rank')
CONFIG=json.loads((ROOT/'configs/execution_monitor.json').read_text())
stop=False

def read(p):
 try:return json.loads(Path(p).read_text())
 except (OSError,ValueError):return {}

def alive(pid,role):
 if not pid:return False
 try:
  p=Path('/proc')/str(pid);args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if (p/'stat').read_text().split()[2]=='Z':return False
  return str(ROOT/'scripts/run_matrix.py') in args if role=='supervisor' else '-m croprank.train' in args and str((p/'cwd').readlink())==str(ROOT)
 except OSError:return False

def snapshot():
 state=read(ROOT/'STATUS.json');pids=read(ROOT/'RUNNING_PID.json');warnings=[]
 protected=read(ROOT/CONFIG['protected_hash_record']);drift=[]
 for name,h in protected.items():
  p=ROOT/name
  if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:drift.append(name)
 if drift:warnings.append('PROTECTED_IMPLEMENTATION_CHANGED')
 supervisor=alive(pids.get('supervisor'),'supervisor');worker=alive(pids.get('worker'),'worker')
 if not supervisor and state.get('state') not in CONFIG['stop_on']:warnings.append('SUPERVISOR_NOT_RUNNING_REQUIRES_REVIEW_NO_AUTO_RESTART')
 age=time.time()-state.get('updated_unix',time.time())
 if age>CONFIG['stale_progress_warning_seconds'] and state.get('state')=='TRAINING':warnings.append('STATUS_STALE_CHECK_VALIDATION_AND_LOGS_BEFORE_ACTION')
 free={str(p):shutil.disk_usage(p).free for p in (ROOT,STORE)}
 if min(free.values())<10*1024**3:warnings.append('DISK_RESERVE_BELOW10GiB')
 jobs=read(ROOT/'configs/matrix.json');runs=[]
 for job in jobs:
  name=f"{job['variant']}_s{job['seed']}";run=STORE/'runs'/name;history=read(run/'history.json') or []
  best=max(history,key=lambda r:(r['val']['acc1_5'],-r['val']['regret'],-r['epoch'])) if history else None
  runs.append({'name':name,'completed_epochs':len(history),'completion':read(run/'TRAINING_COMPLETE.json'),
   'best_validation':{'epoch':best['epoch'],**best['val']} if best else None})
 eta=None;current=pids.get('run');history=read(STORE/'runs'/str(current)/'history.json') or []
 if history and state.get('state')=='TRAINING':
  elapsed=[b['elapsed_process_seconds']-a['elapsed_process_seconds'] for a,b in zip(history,history[1:])]
  positive=sorted(d for d in elapsed[-5:] if d>0)
  if positive:
   typical=positive[len(positive)//2];fraction=state.get('images_in_epoch',0)/max(state.get('epoch_images',1),1) if state.get('epoch')==len(history)+1 else 0
   remaining=max(0,pids.get('epochs',30)-len(history)-fraction)*typical
   eta={'scope':'Current run only; no extrapolation to MUSIQ or whole matrix','remaining_wall_seconds':remaining,
    'point_HK':(datetime.now(ZoneInfo('Asia/Hong_Kong'))+timedelta(seconds=remaining)).isoformat(),'recent_epoch_seconds':typical}
 return {'time_HK':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),'state':state,'processes':{'supervisor_alive':supervisor,'worker_alive':worker,**pids},
  'runs':runs,'warnings':warnings,'protected_hash_drift':drift,'disk_free_bytes':free,'current_run_ETA':eta,
  'external_CPC':read(ROOT/'reports/CPC_CONVERSION_AND_OVERLAP.json').get('state','CPU_PREPARATION_PENDING'),
  'remote_sync':False,'no_training_mutation_or_GPU_work':True}

def main():
 global stop
 guard=(ROOT/'logs/execution_monitor.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signal.signal(signal.SIGTERM,lambda s,f:globals().__setitem__('stop',True))
 (ROOT/'logs/execution_monitor.pid').write_text(str(os.getpid())+'\n')
 while not stop:
  result=snapshot();text=json.dumps(result,ensure_ascii=False,allow_nan=False)
  with (ROOT/'logs/execution_health.jsonl').open('a') as f:f.write(text+'\n');f.flush();os.fsync(f.fileno())
  temp=ROOT/'logs/execution_monitor_latest.json.tmp';temp.write_text(json.dumps(result,indent=2,ensure_ascii=False));temp.replace(ROOT/'logs/execution_monitor_latest.json')
  print(result['time_HK'],result['state'].get('state'),result['state'].get('epoch'),result['state'].get('step'),result['warnings'],flush=True)
  if result['state'].get('state') in CONFIG['stop_on']:break
  deadline=time.monotonic()+CONFIG['interval_seconds']
  while not stop and time.monotonic()<deadline:time.sleep(min(30,max(0,deadline-time.monotonic())))

if __name__=='__main__':main()
