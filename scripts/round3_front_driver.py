#!/usr/bin/env python3
"""Connection-independent finite front end; no formal training before admission."""
import os
import signal
import subprocess
import sys
import traceback
from unsgrpo.round3_common import *


def main():
    state={'status':'RUNNING_FRONT_END','time':timestamp(),'phases':{},'active_phase':None}
    if (WORK/'front_state.json').exists():raise FileExistsError('No hidden front-end restart')
    env=os.environ.copy();env.update(PYTHONPATH=str(PROJECT/'src')+os.pathsep+str(PROJECT/'scripts'),
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_HF='1')
    def save():write(WORK/'front_state.json',state,fresh=False)
    def run(name,script,args=(),gpu=False,timeout=600):
        command=[sys.executable,str(PROJECT/'scripts'/script),*args]
        state['active_phase']=name;state['phases'][name]={'status':'RUNNING','time':timestamp(),'command':command};save()
        child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
        with (WORK/'logs'/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(command,env=child_env,cwd=PROJECT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['phases'][name]['pid']=proc.pid;save()
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                raise TimeoutError(f'{name} deadline; no auto retries')
        if code:raise RuntimeError(f'{name} exited {code}; see logs')
        state['phases'][name].update(status='COMPLETED',ended=timestamp());save()
        print(name+' COMPLETED',flush=True)
    try:
        run('B_teacher','round3_build_teacher.py',gpu=True,timeout=5460)
        run('B_targets075','round3_materialize_targets.py',('--alpha','0.75'))
        run('B_quality075','round3_teacher_quality.py',('--alpha','0.75'))
        gate=read(REPORT/'teacher_gate_alpha075.json');chosen=.75
        if not gate['passed']:
            run('B_targets050','round3_materialize_targets.py',('--alpha','0.5'))
            run('B_quality050','round3_teacher_quality.py',('--alpha','0.5'))
            gate=read(REPORT/'teacher_gate_alpha050.json');chosen=.5
        state.update(status='TEACHER_READY_FOR_C' if gate['passed'] else 'TEACHER_NOT_READY',
            chosen_alpha=chosen if gate['passed'] else None,gate=reference(REPORT/('teacher_gate_alpha075.json' if chosen==.75 else 'teacher_gate_alpha050.json')),
            active_phase=None,ended=timestamp(),formal_training_started=False,GPU_occupied_seconds=occupied_seconds());save()
        print(state['status'],flush=True)
    except BaseException as e:
        state.update(status='FRONT_END_IMPLEMENTATION_FAILURE',error=str(e),traceback=traceback.format_exc(),ended=timestamp());save();raise


if __name__=='__main__':main()
