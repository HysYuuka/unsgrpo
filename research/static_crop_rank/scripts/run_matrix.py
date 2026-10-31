from pathlib import Path
import subprocess,os,sys,json,time,hashlib,signal,traceback,shutil
ROOT=Path(__file__).resolve().parents[1];RUNS=Path('/root/autodl-fs/highlight-static-rank/runs');child=None;stop=False
if (ROOT/'STOP_REQUESTED_BY_USER.json').exists():
 raise SystemExit('Experiments stopped by user. Explicit renewed authorization is required before removing the stop marker.')
sys.path.insert(0,str(ROOT/'src'))
from croprank.train import write_json

def terminate(sig,frame):
 global stop
 stop=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
def execute(job,epochs):
 global child
 name=f"{job['variant']}_s{job['seed']}";done=RUNS/name/'TRAINING_COMPLETE.json'
 if done.exists() and json.loads(done.read_text())['epochs']>=epochs:return
 if shutil.disk_usage(RUNS).free<10*1024**3 or shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('Storage reserve below 10 GiB; refusing unchecked disk fill')
 cmd=[sys.executable,'-u','-m','croprank.train','--variant',job['variant'],'--seed',str(job['seed']),'--epochs',str(epochs),'--chunk',str(json.loads((ROOT/'ENGINEERING_RUNTIME.json').read_text())['candidate_chunk'])]
 with (ROOT/'logs'/f'{name}.log').open('a') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT)
  (ROOT/'RUNNING_PID.json').write_text(json.dumps({'supervisor':os.getpid(),'worker':child.pid,'run':name,'epochs':epochs},indent=2))
  code=child.wait()
 if stop:return
 if code:raise RuntimeError(f'{name} exited {code}; see logs/{name}.log; matrix stopped, no scientific rejection')

def main():
 lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text());jobs=lock['matrix']
 for job in jobs:
  execute(job,30)
  if stop:return
 triggers=[]
 for job in jobs[:9]:
  h=json.loads((RUNS/f"{job['variant']}_s{job['seed']}"/'history.json').read_text())[:30]
  decline=(h[24]['train']['total']-h[29]['train']['total'])/max(h[24]['train']['total'],1e-12)
  improvement=sum(x['val']['acc1_5'] for x in h[27:30])/3-sum(x['val']['acc1_5'] for x in h[21:24])/3
  if decline>=.05 and improvement>=.01:triggers.append({'job':job,'loss_decline':decline,'val_improvement':improvement})
 write_json(ROOT/'reports/EXTENSION_DECISION.json',{'extend_all_to_50':bool(triggers),'triggers':triggers})
 if triggers:
  for job in jobs:
   execute(job,50)
   if stop:return
 # All selected checkpoint hashes fixed before any test inference.
 from croprank.core import sha
 write_json(ROOT/'reports/FROZEN_CHECKPOINTS.json',{f"{j['variant']}_s{j['seed']}":sha(RUNS/f"{j['variant']}_s{j['seed']}"/'best.pt') for j in jobs})
 subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_matrix.py')],cwd=ROOT,check=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
 write_json(ROOT/'STATUS.json',{'state':'CONTROLLED_MATRIX_COMPLETE','external_comparison_status':'see reports/EXTERNAL_STATUS.json','updated_unix':time.time()})
try:
 main()
 if stop:write_json(ROOT/'STATUS.json',{'state':'STOPPED_WITH_RECOVERABLE_CHECKPOINT','updated_unix':time.time()})
except Exception as ex:
 write_json(ROOT/'STATUS.json',{'state':'EXECUTION_ERROR','error':str(ex),'traceback':traceback.format_exc(),'updated_unix':time.time()});raise
