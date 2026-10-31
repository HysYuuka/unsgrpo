"""Finite post-core continuation; wait locally, keep all GPU tasks serial."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];stop=False;child=None
if (ROOT/'STOP_REQUESTED_BY_USER.json').exists():
 raise SystemExit('External evaluation stopped by user; no automatic continuation.')

def stopping(sig,frame):
 global stop
 stop=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)

def write(d):
 p=ROOT/'reports/EXTERNAL_CONTINUATION_STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def main():
 global stop,child
 guard=(ROOT/'logs/external_continuation.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signal.signal(signal.SIGTERM,stopping)
 while not stop:
  state=json.loads((ROOT/'STATUS.json').read_text());phase=state.get('state')
  write({'phase':'WAITING_FOR_COMPLETE_CORE_MATRIX_AND_ISOLATED_GPU','core_state':phase,'pid':os.getpid(),'updated_unix':time.time(),'poll_seconds':1800,'does_not_start_or_restart_training':True})
  if phase in ('EXECUTION_ERROR','STOPPED_WITH_RECOVERABLE_CHECKPOINT'):
   write({'phase':'STOPPED_REQUIRES_EXECUTION_REVIEW_OR_USER_RESUME','core_state':phase,'updated_unix':time.time()});return
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if phase=='CONTROLLED_MATRIX_COMPLETE' and not active:
   if not (ROOT/'reports/CPC_EXTERNAL_LOCK.json').exists():raise RuntimeError('CPC conversion lock not ready')
   with (ROOT/'logs/cpc_external_inference.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/evaluate_cpc_external.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    code=child.wait()
   if stop:write({'phase':'CONTINUATION_STOPPED_WITH_PARTIAL_PREDICTIONS_RETAINED','updated_unix':time.time()});return
   if code:raise RuntimeError(f'CPC evaluator exited{code}; inspect cpc_external_inference.log; no automatic scientific rejection')
   write({'phase':'CPC_COMPLETE_FINAL_RESEARCH_REVIEW_PENDING','result':'reports/CPC_EXTERNAL_RESULTS.json','updated_unix':time.time(),'external_FGAesQ_and_FGAesthetics_gaps_remain_explicit':True});return
  deadline=time.monotonic()+1800
  while not stop and time.monotonic()<deadline:time.sleep(min(30,max(0,deadline-time.monotonic())))
 write({'phase':'CONTINUATION_STOPPED','updated_unix':time.time()})

if __name__=='__main__':
 try:main()
 except Exception as error:write({'phase':'EXTERNAL_EXECUTION_ERROR_NOT_A_METHOD_FAILURE','error':repr(error),'updated_unix':time.time()});raise
