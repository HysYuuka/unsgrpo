#!/usr/bin/env python3
"""Copy verified first-round endpoints and records to file storage, without deletion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import traceback

GIB=1024**3
METHODS=('S0','S+','G-CAL','G-RANK')
TEXT_EXT={'.py','.sh','.json','.jsonl','.log','.txt','.md','.toml','.yaml','.yml','.csv','.jinja'}
STATE_EXT={'.safetensors','.pt','.pth','.bin','.model'}
FORBIDDEN_EXT={'.mp4','.avi','.mov','.mkv','.webm','.mpeg','.mpg','.ts','.jpg','.jpeg','.png','.bmp','.gif','.webp','.mat','.npz','.npy','.zip','.tgz','.gz','.tar','.bz2','.7z'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as target:
        json.dump(value,target,ensure_ascii=False,indent=2,allow_nan=False);target.write('\n')
        target.flush();os.fsync(target.fileno())


def reference(entry):
    path=Path(entry['path']).resolve()
    require(path.is_file() and sha(path)==entry['sha256'],f'Reference file hash mismatch: {path}')
    return path,json.loads(path.read_text())


def safe_relative(destination):
    value=PurePosixPath(destination)
    require(not value.is_absolute() and value.parts and all(p not in {'.','..'} for p in value.parts),
            'Archive destination must be a safe relative path')
    require(str(value) not in {'archive_manifest.json','archive_receipt.json'},'Reserved archive metadata destination')
    return str(value)


def allowed_source(path,kind):
    parts={part.lower() for part in path.parts}
    require(path.suffix.lower() not in FORBIDDEN_EXT,'Raw video, decoded frame, annotation or source archive is excluded')
    require(not parts.intersection({'envs','.venv','venv','site-packages','miniconda3','__pycache__','.git','raw'}),
            'Environment/cache/raw source trees are excluded')
    require(not path.name.startswith('.env'),'Local environment/credential files are excluded')
    require('labels_eval_only' not in parts or path.name=='conversion_receipt.json',
            'Only the value-free conversion receipt may be copied from evaluation-label storage')
    require(path.suffix.lower() in (TEXT_EXT|STATE_EXT if kind in {'merged','checkpoint'} else TEXT_EXT),
            f'Unexpected archive file type: {path}')


def plan_sources(index):
    entries={}
    def add(source,destination,digest=None,kind='metadata'):
        path=Path(source).resolve();relative=safe_relative(destination)
        require(path.is_file(),f'Missing archive source: {path}');allowed_source(path,kind)
        digest=digest or sha(path)
        require(isinstance(digest,str) and len(digest)==64,'A pinned SHA256 is required')
        item={'source':str(path),'destination':relative,'sha256':digest,'bytes':path.stat().st_size}
        require(relative not in entries or entries[relative]==item,'Conflicting archive destination')
        entries[relative]=item
    def model(root,files,prefix,kind):
        root=Path(root).resolve();require(root.is_dir() and files,'Missing endpoint directory/inventory')
        for name,info in files.items():
            path=(root/name).resolve();require(path.is_relative_to(root),'Model inventory escapes endpoint directory')
            digest=info['sha256'] if isinstance(info,dict) else info
            if isinstance(info,dict) and 'bytes' in info:require(path.stat().st_size==info['bytes'],'Endpoint size changed')
            add(path,prefix+'/'+name,digest,kind)
    common_path,common=reference(index['common_s0'])
    require(common.get('status')=='COMPLETED' and common.get('merge_validation_passed') is True
            and common.get('acceptance_basis')=='verified_bf16_tensor_merge_common_s0','Common S0 has not passed actual merge acceptance')
    add(common_path,'records/common_s0.json',index['common_s0']['sha256'])
    model(common['merged_model'],common['merged_model_files'],'models/common_s0','merged')
    require(set(index['runs'])==set(METHODS),'Exactly four accepted formal training endpoints are required')
    for method in METHODS:
        path,run=reference(index['runs'][method]);slug=method.replace('+','PLUS')
        require(run.get('status')=='COMPLETED' and run.get('mode')=='formal'
                and run.get('gpu_training_verified') is True and run.get('method_id')==method,
                'Archive only completed, accepted formal endpoints')
        model(run['endpoint']['path'],run['endpoint']['files'],'checkpoints/'+slug,'checkpoint')
        for file in sorted(path.parent.iterdir()):
            if file.is_file() and file.suffix.lower() in {'.json','.jsonl','.log'}:
                add(file,'runs/'+slug+'/'+file.name)
        output=Path(run['output_dir']).resolve()
        for file in sorted(output.rglob('*')):
            relative=file.relative_to(output)
            if file.is_file() and file.suffix.lower() in {'.json','.jsonl','.log'} and not any(p.startswith('checkpoint-') for p in relative.parts):
                add(file,'runs/'+slug+'/trainer/'+relative.as_posix())
    for item in index.get('metadata',[]):add(item['path'],item['destination'],item['sha256'])
    return list(entries.values())


def storage_check(out,payload):
    extra=payload+16*1024**2
    command=[sys.executable,str(Path(__file__).with_name('storage_audit.py')),
             '--reserve-path',str(out),'--peak-extra-bytes',str(extra)]
    result=subprocess.run(command,capture_output=True,text=True)
    require(result.returncode==0,f'Storage audit rejected archive peak: {result.stdout[-3000:]} {result.stderr[-500:]}')
    audit=json.loads(result.stdout)
    require(audit.get('ok') is True,'Storage audit did not pass')
    require(audit['groups']['file_archives']['allocated_bytes']+extra<=50*GIB,'Archive payload exceeds the 50 GiB archive budget')
    require(audit['filesystems']['file']['free_bytes']-extra>=20*GIB,'Archive copy would consume the 20 GiB file-storage reserve')
    return {'command':command,'peak_extra_bytes':extra,'audit':audit}


def archive(sources_index,out,receipt):
    sources_index,out,receipt=map(lambda p:Path(p).resolve(),(sources_index,out,receipt))
    require(not out.exists() and not receipt.exists(),'Existing archive or receipt is never overwritten/retried')
    require(not receipt.is_relative_to(out),'Receipt must be external to the new archive directory')
    index=json.loads(sources_index.read_text());prefix=index['prefix']
    require(isinstance(prefix,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',prefix) is not None,'Invalid archive prefix')
    archive_parent=Path(os.environ.get('UNSGRPO_ARCHIVE_DIR','/root/autodl-fs/unsgrpo/archives')).resolve()
    require(out.parent==archive_parent and out.name==prefix,'Archive must be archives/<registered prefix>')
    report={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),
            'archive_root':str(out),'local_sources_deleted':False,'files':[],
            'sources_index':{'path':str(sources_index),'sha256':sha(sources_index)}}
    started=time.monotonic();write_json(receipt,report)
    try:
        planned=plan_sources(index);total=sum(item['bytes'] for item in planned)
        report['total_bytes']=total;report['storage_preflight']=storage_check(out,total)
        out.mkdir(parents=True,exist_ok=False)
        for item in planned:
            source=Path(item['source']);target=out/item['destination']
            require(target.resolve().is_relative_to(out),'Archive target escapes root')
            target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.partial')
            h=hashlib.sha256();copied=0
            with source.open('rb') as src,partial.open('xb') as dst:
                for block in iter(lambda:src.read(8*1024**2),b''):
                    dst.write(block);h.update(block);copied+=len(block)
                dst.flush();os.fsync(dst.fileno())
            require(copied==item['bytes'] and h.hexdigest()==item['sha256'],'Source changed or differs from its pinned archive hash')
            require(sha(partial)==item['sha256'] and sha(source)==item['sha256'],'Independent target/source SHA verification failed')
            os.replace(partial,target);report['files'].append(item)
        require(sha(sources_index)==report['sources_index']['sha256'],'Archive source index changed during copy')
        manifest=out/'archive_manifest.json'
        write_json(manifest,{'status':'COMPLETED','prefix':prefix,'total_bytes':total,
                             'local_sources_deleted':False,'files':report['files'],
                             'sources_index':report['sources_index']})
        report.update(status='COMPLETED',archive_manifest={'path':str(manifest),'sha256':sha(manifest)})
    except BaseException as error:
        report.update(status='FAILED',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-index',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();result=archive(args.sources_index,args.out,args.receipt)
    print(json.dumps({'status':result['status'],'archive_root':result['archive_root'],
                      'total_bytes':result['total_bytes'],'wall_seconds':result['wall_seconds']}))
