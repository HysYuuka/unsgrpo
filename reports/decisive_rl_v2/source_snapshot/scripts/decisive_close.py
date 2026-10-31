#!/usr/bin/env python3
"""Archive small reproducibility files, never duplicate teacher/checkpoint weights."""
import shutil
from unsgrpo.decisive_common import *


def main():
    assert read(REPORT/'independent_audit.json')['status']=='PASSED'
    archived=STORE/'reproducibility';archived.mkdir(exist_ok=False)
    # Working source snapshot, metadata, logs and frozen commands are small.
    shutil.copytree(WORK,archived/'work',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source=REPORT/'source_snapshot';source.mkdir(exist_ok=False)
    frozen=read(WORK/'protocol.freeze.json')
    for path in frozen['sources']:
        p=Path(path);dest=source/p.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(REPORT/'input_and_endpoint_hashes.json',{'inputs':frozen['inputs'],'sources':frozen['sources'],
        'teacher_asset':read(STORE/'teacher_model/asset_receipt.json'),
        'training_freeze':read(WORK/'training.freeze.json') if (WORK/'training.freeze.json').exists() else None,
        'endpoints':read(STORE/'endpoints.freeze.json') if (STORE/'endpoints.freeze.json').exists() else None})
    shutil.copytree(REPORT,archived/'report',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    entries=[]
    # Weights already verified by the downloader; use its full receipt after size check.
    teacher_files={r['file']:r for r in read(STORE/'teacher_model/asset_receipt.json')['files']}
    for label,root in [('working',WORK),('persistent',STORE),('report',REPORT)]:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
            if p in (WORK/'logs/driver.log',WORK/'logs/decisive_close.log'):continue
            relative=str(p.relative_to(STORE/'teacher_model')) if p.parent==STORE/'teacher_model' else None
            if relative in teacher_files:
                item=teacher_files[relative];assert p.stat().st_size==item['bytes'];digest=item['sha256']
            else:digest=sha(p)
            entries.append({'location':label,'path':str(p),'bytes':p.stat().st_size,'sha256':digest})
    idx=STORE/'artifact_index.json';write(idx,{'time':now(),'files':entries,'teacher_weight_hash_method':'download full SHA verification + final size identity; no second weight copy',
        'live_control_logs_excluded':['working/logs/driver.log','working/logs/decisive_close.log'],
        'control_log_evidence':'Immutable snapshots in reproducibility/work/logs; live originals can append closure messages'})
    write(REPORT/'CLOSED.json',{'status':'CLOSED','decision':read(WORK/'state.json')['status'],'time':now(),'artifact_index':reference(idx),
        'independent_audit':reference(REPORT/'independent_audit.json'),'archive':str(archived),'Git_sync':False,'subagents_used':False})
    storage_check();print('Closed finite experiment with '+str(len(entries))+' indexed artifacts',flush=True)


if __name__=='__main__':main()
