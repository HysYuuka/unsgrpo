#!/usr/bin/env python3
"""Extract official TVSum assets without loading human annotation values."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
from unsgrpo.data import split_by_source_group, audit_split

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.manifest.exists(): raise FileExistsError("Manifest already exists; do not silently change split")
    if shutil.disk_usage(a.archive.parent).free < 5*1024**3:
        raise RuntimeError("Insufficient dataset space for extraction")
    a.out.mkdir(parents=True,exist_ok=True)
    for d in ["raw","source","labels_eval_only/source"]: (a.out/d).mkdir(parents=True,exist_ok=True)
    extracted=[]
    with tarfile.open(a.archive,"r:gz") as tar:
        for member in tar:
            if not member.isfile(): continue
            name=Path(member.name).name
            if name.endswith("-video.zip") or name.endswith("-data.zip") or name.endswith("-matlab.zip") or name in {"README","WebscopeReadMe.txt"}:
                target=a.out/"source"/name
                with tar.extractfile(member) as src, target.open("wb") as dest: shutil.copyfileobj(src,dest)
                extracted.append({"name":name,"bytes":target.stat().st_size,"sha256":sha256(target)})
    video_zips=list((a.out/"source").glob("*-video.zip"))
    if len(video_zips)!=1: raise ValueError("Exactly one official video archive required")
    with zipfile.ZipFile(video_zips[0]) as z:
        bad=z.testzip()
        if bad: raise ValueError("Corrupt ZIP member")
        members=[i for i in z.infolist() if Path(i.filename).suffix.lower() in {".mp4",".avi",".webm",".mov"} and not i.is_dir()]
        if len(members)!=50: raise ValueError(f"Expected all 50 official videos; found {len(members)}")
        for info in members:
            target=a.out/"raw"/Path(info.filename).name
            if target.exists(): raise FileExistsError(target)
            with z.open(info) as src,target.open("wb") as dest: shutil.copyfileobj(src,dest)
    # Keep label bytes outside repository and separate from model-input files.
    for pattern in ["*-data.zip","*-matlab.zip"]:
        for path in (a.out/"source").glob(pattern):
            path.replace(a.out/"labels_eval_only/source"/path.name)
    rows=[]
    seen_hash={}
    for path in sorted((a.out/"raw").iterdir()):
        digest=sha256(path)
        group=seen_hash.setdefault(digest,path.stem)
        rows.append({"video_id":path.stem,"source_group":group,"path":str(path),"sha256":digest,
                     "source":"TVSum official v1.1 archive","source_url":f"https://www.youtube.com/watch?v={path.stem}"})
    rows=split_by_source_group(rows,train_fraction=.8,seed=a.seed)
    a.manifest.parent.mkdir(parents=True,exist_ok=True)
    a.manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
    receipt={"status":"COMPLETED","source_url":"https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz",
             "archive_sha256":sha256(a.archive),"publisher_checksum_available":False,"split_seed":a.seed,
             "manifest_sha256":sha256(a.manifest),"split":audit_split(rows),"extracted":extracted,
             "annotation_values_read":False,"near_duplicate_audit":"Exact byte duplicates grouped; other near duplicates not independently certified"}
    (a.manifest.parent/"assets.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))

if __name__=="__main__": main()
