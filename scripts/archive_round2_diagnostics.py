#!/usr/bin/env python3
"""Archive completed round-two code/results/checkpoints, never source assets."""
import os
from pathlib import Path
import sys
from unsgrpo.round2_utils import *
from archive_round1 import storage_check


def main():
    state=read(ROOT/'state.json')
    if state['status']!='COMPLETED_NUMERICAL_DIAGNOSTICS':raise ValueError('Do not archive an incomplete round as completed')
    out=Path('/root/autodl-fs/unsgrpo/archives/round2_diagnostic_v1')
    if out.exists() or (ROOT/'archive_receipt.json').exists():raise FileExistsError('Archive is immutable')
    sources=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in {'.driver.lock','archive.log'}:
            sources.append((p,p.relative_to(ROOT)))
    for p in sorted((PROJECT/'reports/round2_diagnostic_v1').iterdir()):
        if p.is_file():sources.append((p,Path('repository_reports')/p.name))
    preflight=storage_check(out,sum(p.stat().st_size for p,_ in sources))
    if not preflight['audit']['ok']:raise ValueError('Storage reservation failed')
    out.mkdir();items=[]
    for source,name in sources:
        digest=sha(source);target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as a,target.open('xb') as b:
            for block in iter(lambda:a.read(8*1024**2),b''):b.write(block)
            b.flush();os.fsync(b.fileno())
        if sha(target)!=digest or sha(source)!=digest:raise ValueError('Source/destination SHA mismatch')
        items.append({'source':str(source),'destination':str(name),'sha256':digest,'bytes':source.stat().st_size})
    result={'status':'COMPLETED','created_utc':now(),'archive_root':str(out.resolve()),'files':items,
        'local_sources_deleted':False,'source_videos_models_or_human_annotation_arrays_copied':False,
        'scope':'New diagnostic working outputs, derived case sheet, source snapshot and tiny student endpoints only',
        'total_bytes':sum(x['bytes'] for x in items),'storage_preflight':preflight}
    write(ROOT/'archive_receipt.json',result);write(out/'archive_receipt.json',result)
    write(PROJECT/'reports/round2_diagnostic_v1/ARCHIVE_RECEIPT.json',{
        'status':'COMPLETED','archive_root':str(out.resolve()),'receipt':reference(ROOT/'archive_receipt.json'),
        'n_files':len(items),'bytes':result['total_bytes'],'local_sources_deleted':False})
    print(f'ARCHIVED {len(items)} files, {result["total_bytes"]} bytes',flush=True)


if __name__=='__main__':main()
