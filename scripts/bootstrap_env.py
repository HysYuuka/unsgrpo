#!/usr/bin/env python3
"""Resolve candidate dependencies, then download verified artifacts concurrently.

This script NEVER installs packages. The caller performs one reviewed installation
from wheelhouse after the resolver/download receipts are complete. Proxy removal
is limited to this resolver subprocess and download openers.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requirements", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--phase", choices=["resolve", "download", "all"], default="all")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--chunk-mib", type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.chunk_mib <= 16:
        p.error("workers must be 1..32 and chunk size 1..16 MiB")
    root = args.workdir.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        p.error("wheelhouse and resolver artifacts must be outside the code repository")
    root.mkdir(parents=True, exist_ok=True)
    wheelhouse = root / "wheelhouse"; wheelhouse.mkdir(exist_ok=True)
    report = root / "resolved.json"
    status = {"status": "RUNNING", "installs_packages": False,
              "started_unix": time.time(), "phase": args.phase, "errors": []}
    save(root / "status.json", status)
    env = os.environ.copy()
    for key in tuple(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            del env[key]
    env["PIP_CACHE_DIR"] = str(root / "pip-cache")
    try:
        if args.phase in {"resolve", "all"}:
            versions = json.loads(subprocess.check_output([
                args.python, "-c", "import importlib.metadata as m,json;print(json.dumps({x:m.version(x) for x in ['torch','torchvision','numpy']}))"
            ], text=True))
            constraints = root / "preserve_installed.constraints.txt"
            constraints.write_text("".join(f"{k}=={v}\n" for k, v in versions.items()))
            # pip 24 normally downloads every complete wheel even for --dry-run.
            # Keep its resolver and metadata checks, but retain metadata distributions
            # and skip only that final preparation pass in this disposable subprocess.
            # No installed pip file is edited; this wrapper cannot perform installation.
            resolver_wrapper = """
import sys, pip
if pip.__version__ != '24.0' or '--dry-run' not in sys.argv:
    raise RuntimeError('metadata-only wrapper requires pip 24.0 and --dry-run')
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.cli.main import main
original = RequirementPreparer.prepare_linked_requirement
def capture(self, req, parallel_builds=False):
    dist = original(self, req, parallel_builds=parallel_builds)
    req._bootstrap_dist = dist
    req.get_dist = lambda: dist
    if req.download_info is None:
        req.download_info = direct_url_from_link(req.link, req.source_dir)
    return dist
def metadata_complete(self, reqs, parallel_builds=False):
    for req in reqs:
        if req.needs_more_preparation and not hasattr(req, '_bootstrap_dist'):
            raise RuntimeError('missing captured PEP658 distribution')
RequirementPreparer.prepare_linked_requirement = capture
RequirementPreparer.prepare_linked_requirements_more = metadata_complete
raise SystemExit(main(sys.argv[1:]))
"""
            command = [args.python, "-c", resolver_wrapper, "install", "--dry-run", "--report", str(report),
                       "--index-url", "https://pypi.org/simple", "--prefer-binary", "--progress-bar", "off",
                       "--disable-pip-version-check", "--timeout", "60", "-c", str(constraints),
                       "-r", str(args.requirements.resolve())]
            status["resolver_command"] = command
            status["preserved_versions"] = versions
            save(root / "status.json", status)
            print("Resolving with official PyPI metadata; no installation is performed", flush=True)
            with (root / "resolve.log").open("w") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f"resolver exited {process.returncode}; see {root / 'resolve.log'}")
        if args.phase == "resolve":
            status.update(status="COMPLETED", resolved_report=str(report), finished_unix=time.time())
            save(root / "status.json", status)
            return
        resolved = json.loads(report.read_text())
        jobs = []
        for package in resolved["install"]:
            name = package["metadata"]["name"]
            normal = name.lower().replace("_", "-")
            if normal in {"torch", "torchvision", "torchaudio", "triton"} or normal.startswith("nvidia-"):
                raise RuntimeError(f"refusing resolved GPU/runtime replacement: {name}")
            info = package["download_info"]
            digest = info["archive_info"]["hashes"].get("sha256")
            if not digest:
                raise RuntimeError(f"missing publisher SHA256 for {name}")
            url = info["url"]
            if urllib.parse.urlsplit(url).hostname != "files.pythonhosted.org":
                raise RuntimeError(f"unexpected package download host for {name}")
            filename = urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).name)
            target = wheelhouse / filename
            jobs.append({"name": name, "version": package["metadata"]["version"], "url": url,
                         "sha256": digest, "path": str(target), "filename": filename,
                         "cached_verified": target.exists() and sha256(target) == digest})

        def request(url, *, headers=None, method=None):
            # Each opener has a local empty proxy mapping; global env is unchanged.
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return op.open(urllib.request.Request(url, headers=headers or {}, method=method), timeout=90)

        def size_job(job):
            if job["cached_verified"]:
                job["size_bytes"] = Path(job["path"]).stat().st_size
                return job
            for attempt in range(4):
                try:
                    with request(job["url"], method="HEAD") as r:
                        job["size_bytes"] = int(r.headers["Content-Length"])
                    return job
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)

        with futures.ThreadPoolExecutor(args.workers) as executor:
            jobs = list(executor.map(size_job, jobs))
        status.update(n_packages=len(jobs), total_package_bytes=sum(j["size_bytes"] for j in jobs),
                      cached_verified_packages=sum(j["cached_verified"] for j in jobs), download_started_unix=time.time())
        # Parts and assembled archives coexist briefly; preserve the local 10 GiB reserve.
        peak_extra = 2 * sum(j["size_bytes"] for j in jobs if not j["cached_verified"])
        status["download_peak_extra_bytes_estimate"] = peak_extra
        status["free_bytes_before_download"] = shutil.disk_usage(root).free
        if status["free_bytes_before_download"] - peak_extra < 10 * 1024**3:
            raise RuntimeError("dependency download peak would consume the local 10 GiB reserve")
        save(root / "status.json", status)
        save(root / "download_manifest.json", jobs)
        print(json.dumps({k: status[k] for k in ("n_packages", "total_package_bytes", "cached_verified_packages")}), flush=True)
        chunk_size = args.chunk_mib * 1024**2
        chunks = []
        for j in jobs:
            if j["cached_verified"]:
                continue
            partdir = root / "parts" / j["sha256"]; partdir.mkdir(parents=True, exist_ok=True)
            for start in range(0, j["size_bytes"], chunk_size):
                end = min(j["size_bytes"] - 1, start + chunk_size - 1)
                chunks.append((j, start, end, partdir / f"{start:012d}.part"))

        def get_chunk(chunk):
            j, start, end, target = chunk
            expected = end - start + 1
            if target.exists() and target.stat().st_size == expected:
                return expected
            for attempt in range(5):
                try:
                    with request(j["url"], headers={"Range": f"bytes={start}-{end}"}) as r:
                        if r.status != 206:
                            raise RuntimeError(f"expected HTTP 206, got {r.status}")
                        if r.headers.get("Content-Range") != f"bytes {start}-{end}/{j['size_bytes']}":
                            raise RuntimeError("Content-Range mismatch")
                        data = r.read(expected + 1)
                    if len(data) != expected:
                        raise RuntimeError("range body size mismatch")
                    temp = target.with_suffix(".partial")
                    temp.write_bytes(data); temp.replace(target)
                    return expected
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 1)

        done_bytes = 0
        with futures.ThreadPoolExecutor(args.workers) as executor:
            fs = [executor.submit(get_chunk, c) for c in chunks]
            for n, future in enumerate(futures.as_completed(fs), 1):
                done_bytes += future.result()
                status.update(completed_chunks=n, total_chunks=len(chunks), completed_chunk_bytes=done_bytes)
                save(root / "status.json", status)
                if n == 1 or n % 10 == 0 or n == len(chunks):
                    print(f"Downloaded {n}/{len(chunks)} chunks ({done_bytes / 1024**2:.1f} MiB)", flush=True)
        for j in jobs:
            if not j["cached_verified"]:
                target = Path(j["path"]); temp = target.with_suffix(target.suffix + ".partial")
                partdir = root / "parts" / j["sha256"]
                with temp.open("wb") as f:
                    for start in range(0, j["size_bytes"], chunk_size):
                        f.write((partdir / f"{start:012d}.part").read_bytes())
                if sha256(temp) != j["sha256"]:
                    temp.unlink(missing_ok=True)
                    for part in partdir.glob("*.part"):
                        part.unlink()
                    raise RuntimeError(f"SHA256 mismatch: {j['filename']}")
                temp.replace(target)
                for part in partdir.glob("*.part"):
                    part.unlink()
                partdir.rmdir()
            j["download_verified"] = True
        save(root / "download_manifest.json", jobs)
        status.update(status="COMPLETED", download_verified_packages=len(jobs), finished_unix=time.time())
        status["installation_command"] = [args.python, "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse),
                                          "--no-build-isolation", "-c", str(root / "preserve_installed.constraints.txt"),
                                          "-r", str(args.requirements.resolve())]
        save(root / "status.json", status)
        print(json.dumps({"status": "COMPLETED", "wheelhouse": str(wheelhouse),
                          "installation_command": status["installation_command"]}), flush=True)
    except Exception as exc:
        status.update(status="FAILED", finished_unix=time.time())
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        save(root / "status.json", status)
        raise


if __name__ == "__main__":
    main()
