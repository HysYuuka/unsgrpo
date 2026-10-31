#!/usr/bin/env python3
"""One official 8B BF16 copy, pinned hashes; CPU download time is not GPU time."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import requests

PROJECT=Path('/root/autodl-tmp/unsgrpo')
WORK=Path('/root/autodl-tmp/unsgrpo-work/runs/decisive_rl_v2')
DEST=Path('/root/autodl-fs/unsgrpo/decisive_rl_v2/teacher_model')
HF_REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
WEIGHTS={
 'model-00001-of-00004.safetensors':'d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa',
 'model-00002-of-00004.safetensors':'8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5',
 'model-00003-of-00004.safetensors':'83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192',
 'model-00004-of-00004.safetensors':'0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5'}


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return h.hexdigest()


def main():
 start=time.monotonic();metadata=json.loads((WORK/'asset_metadata/modelscope_files.json').read_text())
 items=[e for e in metadata['Data']['Files'] if e['Type']=='blob' and (e['Path'].endswith(('.json','.txt','.safetensors')) or e['Path']=='README.md')]
 assert set(WEIGHTS)=={e['Path'] for e in items if e['Path'].endswith('.safetensors')}
 for e in items:
  if e['Path'] in WEIGHTS:assert e['Sha256']==WEIGHTS[e['Path']]
 needed=sum(e['Size'] for e in items)
 if needed>40*1024**3 or shutil.disk_usage(DEST).free<needed+20*1024**3:raise RuntimeError('RESOURCE_NOT_READY: storage')
 def fetch(e):
  p=DEST/e['Path'];url='https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-8B-Instruct/repo?'+urllib.parse.urlencode({'Revision':e['Revision'],'FilePath':e['Path']})
  if p.exists():
   if p.stat().st_size!=e['Size'] or sha(p)!=e['Sha256']:raise ValueError('Existing asset checksum mismatch')
  elif e['Path'].endswith('.safetensors'):
   command=[sys.executable,str(PROJECT/'scripts/ranged_download.py'),'--url',url,'--out',str(p),'--sha256',e['Sha256'],'--workers','16','--chunk-mib','8']
   with (WORK/'logs'/('asset_'+p.name+'.log')).open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
  else:
   r=requests.get(url,timeout=60);r.raise_for_status();p.write_bytes(r.content)
  if p.stat().st_size!=e['Size'] or sha(p)!=e['Sha256']:raise ValueError(f'Asset identity mismatch: {p.name}')
  return {'file':e['Path'],'bytes':e['Size'],'sha256':e['Sha256'],'official_ModelScope_blob_revision':e['Revision']}
 receipt={'status':'DOWNLOADING','repo':'Qwen/Qwen3-VL-8B-Instruct','HF_revision':HF_REV,'files':[]}
 (WORK/'asset_state.json').write_text(json.dumps(receipt,indent=2)+'\n')
 try:
  with concurrent.futures.ThreadPoolExecutor(4) as executor:result=list(executor.map(fetch,items))
  receipt.update(status='COMPLETED_VERIFIED_OFFICIAL_BF16_ASSET',files=result,wall_seconds=time.monotonic()-start,
    total_bytes=sum(e['bytes'] for e in result),weight_hashes_match_HF_LFS=True,
    provenance='Official Qwen ModelScope mirror; four weights matched official HF Xet/LFS SHA256 at pinned revision; per-file mirror commits recorded.',
    official_HF_urls=[f'https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/{HF_REV}/{name}' for name in WEIGHTS],
    GPU_seconds=0,quantization=None)
  (DEST/'asset_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  (WORK/'asset_state.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps({'status':receipt['status'],'bytes':receipt['total_bytes'],'download_seconds':receipt['wall_seconds']}),flush=True)
 except BaseException as e:
  receipt.update(status='RESOURCE_NOT_READY',error=f'{type(e).__name__}: {e}',wall_seconds=time.monotonic()-start)
  (WORK/'asset_state.json').write_text(json.dumps(receipt,indent=2)+'\n');raise


if __name__=='__main__':main()
