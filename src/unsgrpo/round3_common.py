"""Round-three paths, immutable provenance and occupied-GPU accounting.

This module and the teacher/reward modules never open human annotations.
"""
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

PROJECT = Path('/root/autodl-tmp/unsgrpo')
WORK = Path('/root/autodl-tmp/unsgrpo-work/runs/round3_rl_method_v1')
STORE = Path('/root/autodl-fs/unsgrpo/round3_rl_method_v1')
REPORT = PROJECT/'reports/round3_rl_method_v1'
R2 = Path('/root/autodl-tmp/unsgrpo-work/runs/round2_diagnostic_v1')


def timestamp():
    t = datetime.now(timezone.utc)
    return {'utc': t.isoformat(), 'hong_kong': t.astimezone(ZoneInfo('Asia/Hong_Kong')).isoformat()}


def append(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())


def assets():
    return load_assets(R2)


def occupied_seconds():
    p = WORK/'gpu_sessions.jsonl'
    rr = records(p) if p.exists() else []
    return sum(r['occupied_seconds'] for r in rr if r['event']=='end')


@contextmanager
def gpu_session(phase, limit_seconds):
    """Single-card serialized session, including load, preprocessing and failures."""
    lock = (WORK.parent/'.round1_gpu_driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remaining = 43200 - occupied_seconds()
    cap = min(float(limit_seconds), remaining)
    if cap <= 0: raise TimeoutError('BUDGET_LIMITED: occupied-GPU budget exhausted')
    start = time.monotonic(); ident = f'{phase}:{os.getpid()}:{time.time_ns()}'
    append(WORK/'gpu_sessions.jsonl', {'event':'start','session_id':ident,'phase':phase,
        'pid':os.getpid(),'time':timestamp(),'limit_seconds':cap})
    previous = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame): raise TimeoutError('BUDGET_LIMITED: GPU session deadline')
    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(cap, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True; timer.start(); success = False
    try:
        yield lambda: max(0., cap-(time.monotonic()-start))
        success = True
    finally:
        timer.cancel(); signal.signal(signal.SIGTERM, previous)
        append(WORK/'gpu_sessions.jsonl', {'event':'end','session_id':ident,'phase':phase,
            'time':timestamp(),'occupied_seconds':time.monotonic()-start,'success':success})
        lock.close()


def visual_environment():
    audit = read(PROJECT/'reports/input_audit.json')
    os.environ.update(audit['visual_environment'])
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4')


def verify_previous_project_sources():
    frozen = read(PROJECT/'reports/freeze.json')
    found = {p:h for p,h in frozen['files'].items() if p.startswith(str(PROJECT)+'/')}
    for p,h in found.items():
        if sha(p)!=h: raise ValueError(f'Previous frozen source changed: {p}')
    return found
