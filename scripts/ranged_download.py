#!/usr/bin/env python3
"""Bounded parallel HTTP ranges, verified before atomic promotion. No URL logging."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sha256")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--chunk-mib", type=int, default=8)
    p.add_argument("--direct", action="store_true", help="Bypass proxy for this download only")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if a.direct else urllib.request.build_opener()
    with opener.open(urllib.request.Request(a.url, headers={"Range":"bytes=0-0"}), timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        m = re.fullmatch(r"bytes 0-0/(\d+)", cr)
        if r.status != 206 or m is None or len(r.read()) != 1:
            raise RuntimeError("Origin does not support exact HTTP ranges")
        size = int(m[1])
        resolved_url = r.geturl()
    part = a.out.with_suffix(a.out.suffix+".ranges.partial")
    progress = a.out.with_suffix(a.out.suffix+".ranges.json")
    chunk = a.chunk_mib*1024**2
    state = {"size":size,"chunk":chunk,"completed":[]}
    if part.exists() and progress.exists():
        old = json.loads(progress.read_text())
        if old["size"] == size and old["chunk"] == chunk:
            state = old
    fd = os.open(part, os.O_CREAT|os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    lock = threading.Lock()
    complete = set(state["completed"])
    start = time.monotonic()
    def run(i):
        begin, end = i*chunk, min(size, (i+1)*chunk)-1
        for attempt in range(4):
            try:
                req = urllib.request.Request(resolved_url, headers={"Range":f"bytes={begin}-{end}"})
                with opener.open(req, timeout=120) as r:
                    if r.status != 206 or r.headers.get("Content-Range") != f"bytes {begin}-{end}/{size}":
                        raise RuntimeError(f"Origin range mismatch: status={r.status}, requested={begin}-{end}/{size}, received={r.headers.get('Content-Range')}")
                    payload = r.read(end-begin+2)
                if len(payload) != end-begin+1:
                    raise RuntimeError("Short/oversized HTTP range")
                count = os.pwrite(fd, payload, begin)
                if count != len(payload):
                    raise RuntimeError("short disk write")
                with lock:
                    complete.add(i)
                    state["completed"] = sorted(complete)
                    temp = progress.with_suffix(".tmp")
                    temp.write_text(json.dumps(state))
                    temp.replace(progress)
                    print(json.dumps({"parts":len(complete),"total_parts":math.ceil(size/chunk),"elapsed":round(time.monotonic()-start,2)}),flush=True)
                return
            except Exception as e:
                # Exception URLs may contain signatures: report only type.
                print(json.dumps({"chunk":i,"attempt":attempt+1,"error_type":type(e).__name__,"detail":str(e) if type(e) is RuntimeError else None}),flush=True)
                if attempt == 3:
                    raise RuntimeError(f"Range {i} exhausted retries ({type(e).__name__})") from None
                time.sleep(1)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(run,[i for i in range(math.ceil(size/chunk)) if i not in complete]))
        os.fsync(fd)
    finally:
        os.close(fd)
    h = hashlib.sha256()
    with part.open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):
            h.update(b)
    digest=h.hexdigest()
    if a.sha256 and digest != a.sha256:
        raise RuntimeError("Final SHA256 mismatch; file not promoted")
    part.replace(a.out)
    progress.unlink()
    print(json.dumps({"status":"COMPLETED","bytes":size,"sha256":digest,"elapsed":time.monotonic()-start}),flush=True)

if __name__=="__main__":
    main()
