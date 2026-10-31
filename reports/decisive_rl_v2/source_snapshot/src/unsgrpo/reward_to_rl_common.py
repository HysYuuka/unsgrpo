"""New bounded reward-to-RL job; no human annotation access in this module."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
from zoneinfo import ZoneInfo
from .round2_utils import sha, reference, read, records, write, load_assets

PROJECT=Path('/root/autodl-tmp/unsgrpo')
WORK=Path('/root/autodl-tmp/unsgrpo-work/runs/reward_to_rl_v1')
STORE=Path('/root/autodl-fs/unsgrpo/reward_to_rl_v1')
REPORT=PROJECT/'reports/reward_to_rl_v1'
R2=Path('/root/autodl-tmp/unsgrpo-work/runs/round2_diagnostic_v1')
ORIGINAL=Path('/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1_recovery1')
PYTHON=Path('/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python')


def now():
    x=datetime.now(timezone.utc)
    return {'utc':x.isoformat(),'hong_kong':x.astimezone(ZoneInfo('Asia/Hong_Kong')).isoformat()}


def append(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def assets():return load_assets(R2)


def gpu_used():
    p=WORK/'gpu_sessions.jsonl';rr=records(p) if p.exists() else []
    return sum(r['seconds'] for r in rr if r['event']=='end')


def logical(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def check_storage():
    local=__import__('shutil').disk_usage(WORK);store=__import__('shutil').disk_usage(STORE)
    if local.free<10*1024**3 or store.free<120*1024**3:raise RuntimeError('STORAGE_NOT_READY: reserve')
    if logical(WORK)>5*1024**3 or logical(STORE)>15*1024**3:raise RuntimeError('STORAGE_NOT_READY: job limit')


@contextmanager
def gpu_session(name,cap=None,screen=False):
    lock=(WORK.parent/'.round1_gpu_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    used=gpu_used();remaining=21600-used-15
    if screen:remaining=min(remaining,3600-used-15)
    if cap is not None:remaining=min(remaining,float(cap))
    if remaining<=0:raise TimeoutError('BUDGET_LIMITED')
    check_storage();start=time.monotonic();ident=f'{name}:{os.getpid()}:{time.time_ns()}'
    append(WORK/'gpu_sessions.jsonl',{'event':'start','phase':name,'session':ident,'time':now(),'pid':os.getpid(),'cap':remaining})
    old=signal.getsignal(signal.SIGTERM)
    def stop(sig,frame):raise TimeoutError('BUDGET_LIMITED: GPU deadline')
    signal.signal(signal.SIGTERM,stop)
    timer=threading.Timer(remaining,lambda:os.kill(os.getpid(),signal.SIGTERM));timer.daemon=True;timer.start();ok=False
    try:
        yield lambda:max(0.,remaining-(time.monotonic()-start));ok=True
    finally:
        timer.cancel();signal.signal(signal.SIGTERM,old)
        append(WORK/'gpu_sessions.jsonl',{'event':'end','phase':name,'session':ident,'time':now(),'seconds':time.monotonic()-start,'success':ok})
        lock.close()


def visual_environment():
    os.environ.update(read(PROJECT/'reports/input_audit.json')['visual_environment'])
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')


def validate_freeze(name='screen_protocol.freeze.json'):
    frozen=read(WORK/name)
    repair=read(WORK/'software_fix1.json') if (WORK/'software_fix1.json').exists() else None
    for group in ('sources','inputs'):
        for path,digest in frozen[group].items():
            actual=sha(path)
            if actual!=digest:
                allowed=repair.get('source_replacements',{}).get(path) if repair and group=='sources' and name=='screen_protocol.freeze.json' else None
                if not allowed or allowed['original_sha256']!=digest or allowed['fixed_sha256']!=actual:
                    raise ValueError(f'Frozen {group} changed: {path}')
    return frozen
