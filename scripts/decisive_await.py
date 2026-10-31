#!/usr/bin/env python3
"""Disconnected asset wait -> CPU freeze -> completely offline finite pipeline."""
import os
import subprocess
import time
from unsgrpo.decisive_common import *


def main():
    start=datetime.fromisoformat(read(WORK/'initial_state.json')['started_utc']);deadline=start.timestamp()+5400
    write(WORK/'startup_state.json',{'status':'WAITING_FOR_PINNED_TEACHER','time':now(),'pid':os.getpid(),'GPU_seconds':0},fresh=False)
    while time.time()<deadline:
        state=read(WORK/'asset_state.json')
        if state['status']=='COMPLETED_VERIFIED_OFFICIAL_BF16_ASSET':break
        if state['status']=='RESOURCE_NOT_READY':raise RuntimeError('RESOURCE_NOT_READY: asset acquisition failed; see asset_state.json')
        time.sleep(30)
    else:raise TimeoutError('RESOURCE_NOT_READY: preparation90min elapsed deadline')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    command=[str(PYTHON),str(PROJECT/'scripts/decisive_prepare.py')]
    with (WORK/'logs/prepare.log').open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
    write(WORK/'startup_state.json',{'status':'PROTOCOL_FROZEN_STARTING_OFFLINE_PIPELINE','time':now(),'pid':os.getpid()},fresh=False)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';output=os.open(WORK/'logs/driver.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    os.dup2(output,1);os.dup2(output,2);os.close(output)
    os.execve(str(PYTHON),[str(PYTHON),'-u',str(PROJECT/'scripts/decisive_driver.py')],env)


if __name__=='__main__':main()
