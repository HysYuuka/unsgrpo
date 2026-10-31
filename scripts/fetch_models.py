#!/usr/bin/env python3
"""Download only required, revision-pinned assets; verify LFS SHA256 and size."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

MODELS = {
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
    "clip": ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def fetch(repo, rev, entry, dest):
    name = entry["rfilename"]
    path = dest / name
    expected = entry.get("lfs", {}).get("sha256")
    size = entry.get("lfs", {}).get("size", entry.get("size"))
    if path.exists() and (size is None or path.stat().st_size == size):
        digest = sha256(path)
        if expected is None or digest == expected:
            return {"file": name, "bytes": path.stat().st_size, "sha256": digest}
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    part = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            # Requests are revision pinned; no alternate or quantized weights.
            with urllib.request.urlopen(url, timeout=90) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=4 * 1024 * 1024)
            digest = sha256(part)
            if size is not None and part.stat().st_size != size:
                raise ValueError(f"size mismatch: {name}")
            if expected and digest != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
            part.replace(path)
            result = {"file": name, "bytes": path.stat().st_size, "sha256": digest}
            print(json.dumps(result), flush=True)
            return result
        except Exception as e:
            print(json.dumps({"file": name, "attempt": attempt + 1, "error": str(e)}), flush=True)
            if attempt == 2:
                raise
            time.sleep(2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--root", type=Path, default=Path(os.environ.get("UNSGRPO_MODEL_DIR", "/root/autodl-fs/unsgrpo/models")))
    a = p.parse_args()
    repo, rev = MODELS[a.model]
    dest = a.root / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true", timeout=45) as r:
        meta = json.load(r)
    assert meta["sha"] == rev
    # Preserve model card/license and processor/tokenizer/template; no duplicate weight formats.
    entries = [e for e in meta["siblings"] if e["rfilename"].endswith((".json", ".txt", ".md", "LICENSE", ".safetensors")) or e["rfilename"] == "pytorch_model.bin"]
    needed = sum(e.get("lfs", {}).get("size", e.get("size", 0)) for e in entries)
    if shutil.disk_usage(dest).free < needed + 20 * 1024**3:
        raise RuntimeError("Insufficient file storage for model plus 20 GiB reserve")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda e: fetch(repo, rev, e, dest), entries))
    receipt = {"repo": repo, "resolved_revision": rev, "license": meta.get("cardData", {}).get("license"), "files": results, "status": "COMPLETED"}
    (dest / "asset_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "repo": repo, "bytes": sum(e["bytes"] for e in results)}), flush=True)

if __name__ == "__main__":
    main()
