#!/usr/bin/env python3
"""Freeze six independently completed, endpoint-bound inference outputs before labels."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unsgrpo.evaluation import RUN_IDS, file_sha256, validate_six_run_freeze

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--protocol-freeze',type=Path,required=True)
    p.add_argument('--receipts-index',type=Path,required=True,help='JSON: run ID -> inference receipt path')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError('Prediction freeze is immutable; choose a new output')
    registered=json.loads(a.protocol_freeze.read_text())
    if registered.get('status')!='FROZEN': raise ValueError('Formal protocol was never frozen')
    for name,sha in registered['files'].items():
        path=Path(name)
        if not path.is_absolute(): path=Path(__file__).resolve().parents[1]/path
        if file_sha256(path)!=sha: raise ValueError(f'Frozen formal asset changed: {path}')
    index=json.loads(a.receipts_index.read_text())
    if set(index)!=set(RUN_IDS): raise ValueError('Exactly six independent inference receipts required')
    result={'status':'FROZEN','created_utc':datetime.now(timezone.utc).isoformat(),
            'require_run_provenance':True,'human_annotations_read_before_freeze':False,
            'protocol_freeze':{'path':str(a.protocol_freeze.resolve()),'sha256':file_sha256(a.protocol_freeze)},
            'manifest':{'path':str(a.manifest.resolve()),'sha256':file_sha256(a.manifest)},'runs':{}}
    for method in RUN_IDS:
        path=Path(index[method]).resolve();receipt=json.loads(path.read_text())
        if receipt.get('run_id')!=method or receipt.get('status')!='COMPLETED': raise ValueError(f'Incomplete inference: {method}')
        predictions=Path(receipt['predictions_path'])
        if file_sha256(predictions)!=receipt['predictions_sha256']: raise ValueError(f'Prediction changed: {method}')
        result['runs'][method]={'path':str(predictions),'sha256':receipt['predictions_sha256'],
                                'receipt':{'path':str(path),'sha256':file_sha256(path)}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    temp=a.out.with_suffix(a.out.suffix+'.pending')
    temp.write_text(json.dumps(result,indent=2)+'\n')
    try:
        validate_six_run_freeze(temp)
        temp.replace(a.out)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(json.dumps({'status':'FROZEN','path':str(a.out.resolve()),'sha256':file_sha256(a.out)}))

if __name__=='__main__': main()
