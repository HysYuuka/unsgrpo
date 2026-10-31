"""Explicit new namespace and conservative per-bucket GPU accounting."""
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
from zoneinfo import ZoneInfo
from .round2_utils import sha,reference,read,records,write,load_assets

PROJECT=Path('/root/autodl-tmp/unsgrpo')
WORK=Path('/root/autodl-tmp/unsgrpo-work/runs/decisive_rl_v2')
STORE=Path('/root/autodl-fs/unsgrpo/decisive_rl_v2')
REPORT=PROJECT/'reports/decisive_rl_v2'
OLD_WORK=Path('/root/autodl-tmp/unsgrpo-work/runs/reward_to_rl_v1')
OLD_REPORT=PROJECT/'reports/reward_to_rl_v1'
R2=Path('/root/autodl-tmp/unsgrpo-work/runs/round2_diagnostic_v1')
ORIGINAL=Path('/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1_recovery1')
PYTHON=Path('/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python')
LIMITS={'teacher':5400,'student':1800,'training':18000,'evaluation':3600}


def now():
    x=datetime.now(timezone.utc);return {'utc':x.isoformat(),'hong_kong':x.astimezone(ZoneInfo('Asia/Hong_Kong')).isoformat()}


def append(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def assets():return load_assets(R2)


def used(bucket=None):
    p=WORK/'GPU_ledger.jsonl'
    return sum(r['seconds'] for r in records(p) if r['event']=='end' and (bucket is None or r['bucket']==bucket)) if p.exists() else 0.


def bytes_under(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())


def storage_check():
    import shutil
    if shutil.disk_usage(WORK).free<10*1024**3 or shutil.disk_usage(STORE).free<20*1024**3:raise RuntimeError('RESOURCE_NOT_READY: disk reserve')
    if bytes_under(WORK)>5*1024**3 or bytes_under(STORE)>40*1024**3:raise RuntimeError('RESOURCE_NOT_READY: new storage budget')


@contextmanager
def gpu_session(name,bucket):
    storage_check();lock=(WORK.parent/'.round1_gpu_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remaining=min(28800-used(),LIMITS[bucket]-used(bucket))-20
    if remaining<=0:raise TimeoutError('RESOURCE_NOT_READY: GPU budget')
    start=time.monotonic();ident=f'{name}:{os.getpid()}:{time.time_ns()}'
    append(WORK/'GPU_ledger.jsonl',{'event':'start','phase':name,'bucket':bucket,'session':ident,'time':now(),'cap':remaining,'pid':os.getpid()})
    old=signal.getsignal(signal.SIGTERM)
    def stop(sig,frame):raise TimeoutError('RESOURCE_NOT_READY: GPU deadline')
    signal.signal(signal.SIGTERM,stop);timer=threading.Timer(remaining,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.daemon=True;timer.start();ok=False
    try:yield;ok=True
    finally:
        timer.cancel();signal.signal(signal.SIGTERM,old)
        append(WORK/'GPU_ledger.jsonl',{'event':'end','phase':name,'bucket':bucket,'session':ident,'time':now(),'seconds':time.monotonic()-start,'success':ok});lock.close()


def validate(name='protocol.freeze.json'):
    f=read(WORK/name)
    for group in ('sources','inputs'):
        for p,h in f[group].items():
            if sha(p)!=h:raise ValueError(f'Frozen {group} changed: {p}')
    return f


def visual_environment():
    os.environ.update(read(PROJECT/'reports/input_audit.json')['visual_environment'])
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')
