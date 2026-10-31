#!/usr/bin/env python3
"""PTS-based real-video decoding, overview construction and frozen CLIP features."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from unsgrpo.data import clip_intervals, sample_targets
from unsgrpo.core import unit_rows

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8*1024**2), b""):
            h.update(b)
    return h.hexdigest()

def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    meta = json.loads(p.stdout)
    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    duration = float(video.get("duration", meta["format"]["duration"]))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    return duration, meta

def seal_result(result, encoder):
    files=[f for frames in result['frame_paths'] for f in frames]+[result['overview'],result['frame_bounds_path']]
    result['media_files_sha256']={f:digest(f) for f in files}
    result['encoder_asset_receipt_sha256']=digest(encoder/'asset_receipt.json')
    result['asset_seal']={'scope':'initial asset hash capture before protocol freeze','sealed_utc':datetime.now(timezone.utc).isoformat(),
                          'frames_per_clip':4,'sampled_jpeg_quality':90,'sampled_image_max_side':640,'clip_processor_use_fast':False}
    return result

def verify_seal(result, encoder):
    if result.get('encoder_asset_receipt_sha256')!=digest(encoder/'asset_receipt.json'):
        raise ValueError('Resume encoder identity differs from sealed feature producer')
    files=[f for frames in result['frame_paths'] for f in frames]+[result['overview'],result['frame_bounds_path']]
    hashes=result.get('media_files_sha256',{})
    if set(files)!=set(hashes) or any(digest(f)!=hashes[f] for f in files):
        raise ValueError('Resume media/overview/native PTS hash mismatch or absent seal')

def decode(path, out, bounds):
    targets = np.asarray(sample_targets(bounds), dtype=float).reshape(-1)
    paths, pts, errors, all_pts, terminal_holds = [], [], [], [], []
    target_index = 0
    previous = -math.inf
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        start = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("frame has no PTS")
            t = float(frame.pts * frame.time_base) - start
            if t <= previous:
                raise ValueError("duplicate or nonmonotonic PTS")
            previous = t
            all_pts.append(t)
            # One decoded frame may serve several targets; never invent frames.
            while target_index < len(targets) and t + 1e-9 >= targets[target_index]:
                error = t - targets[target_index]
                if error > .5:
                    raise ValueError(f"sampling error {error:.6f}s exceeds 0.5s")
                p = out / f"clip{target_index//4:05d}_f{target_index%4}.jpg"
                im = frame.to_image()
                im.thumbnail((640, 640))
                im.save(p, quality=90)
                paths.append(str(p))
                pts.append(t)
                errors.append(error)
                target_index += 1
        # At the finite end of a video, a target may be inside the last frame's
        # display interval. No later real frame exists. Use that actual frame,
        # retain its earlier PTS, and explicitly record this boundary exception.
        while target_index < len(targets) and all_pts and all_pts[-1] <= targets[target_index] < bounds[-1, 1]:
            error = targets[target_index] - all_pts[-1]
            if error > .5:
                raise ValueError("terminal frame hold exceeds 0.5s error limit")
            image_path = out / f"clip{target_index//4:05d}_f{target_index%4}.jpg"
            im = frame.to_image(); im.thumbnail((640, 640)); im.save(image_path, quality=90)
            paths.append(str(image_path));pts.append(all_pts[-1]);errors.append(error)
            terminal_holds.append({"sample_index":target_index,"original_target":float(targets[target_index]),"actual_pts":all_pts[-1],"absolute_error":float(error)})
            target_index += 1
    if target_index != len(targets):
        raise ValueError(f"missing end frames: {target_index}/{len(targets)}")
    frame_bounds = np.column_stack((all_pts, list(all_pts[1:]) + [float(bounds[-1, 1])]))
    if frame_bounds[0, 0] != 0 or np.any(frame_bounds[:, 1] <= frame_bounds[:, 0]):
        raise ValueError("decoded native frame intervals do not cover the video clock")
    np.save(out / "frame_bounds.npy", frame_bounds)
    return np.asarray(paths).reshape(-1, 4).tolist(), np.asarray(pts).reshape(-1, 4).tolist(), {"mean_seconds": float(np.mean(errors)), "max_seconds": float(np.max(errors)), "start_pts_seconds": start, "decoded_native_frames": len(all_pts),"terminal_frame_holds":terminal_holds,"rule":"first-after-target, except actual final frame holding a terminal target"}

def overview(paths, pts, path):
    flat = [p for row in paths for p in row]
    clock = np.asarray(pts).reshape(-1)
    indices = np.rint(np.linspace(0, len(flat)-1, 16)).astype(int)
    canvas = Image.new("RGB", (896, 896), "black")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    for j, i in enumerate(indices):
        im = Image.open(flat[i]).convert("RGB")
        im.thumbnail((224, 202))
        x, y = (j % 4) * 224, (j // 4) * 224
        canvas.paste(im, (x+(224-im.width)//2, y))
        draw.text((x+4,y+203), f"{clock[i]:.2f} s", fill="white", font=font)
    canvas.save(path, quality=95)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true", help="Reuse verified complete video receipts only")
    p.add_argument("--seal-existing", action="store_true", help="Initial hash capture for pre-freeze complete receipts; no reuse claim")
    a = p.parse_args()
    rows = [json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.seal_existing:
        count=0
        for row in rows:
            receipt_path=a.out/row['video_id']/'receipt.json'
            if not receipt_path.exists(): continue
            result=json.loads(receipt_path.read_text())
            if result['raw_sha256']!=digest(row['path']) or result['features_sha256']!=digest(result['features_path']):
                raise ValueError('Cannot seal mismatched raw/features')
            if 'media_files_sha256' in result: verify_seal(result,a.encoder)
            else:
                result=seal_result(result,a.encoder)
                receipt_path.write_text(json.dumps(result,indent=2)+'\n')
            count+=1
        print(json.dumps({'status':'INITIAL_HASH_CAPTURE_COMPLETED','n_complete_video_receipts':count,'incomplete_video_receipts_not_promoted':len(rows)-count}),flush=True)
        return
    processor = AutoProcessor.from_pretrained(a.encoder, local_files_only=True, use_fast=False)
    encoder = CLIPVisionModelWithProjection.from_pretrained(a.encoder, local_files_only=True).to(a.device).eval()
    encoder.requires_grad_(False)
    prepared, failures = [], []
    for row in rows:
        started = time.monotonic()
        dest = a.out / row["video_id"]
        dest.mkdir(exist_ok=True)
        try:
            prior = dest / "receipt.json"
            if a.resume and prior.exists():
                result = json.loads(prior.read_text())
                if result["raw_sha256"] != digest(row["path"]) or result["features_sha256"] != digest(result["features_path"]):
                    raise ValueError("Resume asset hash mismatch")
                verify_seal(result,a.encoder)
                prepared.append(result)
                print(json.dumps({"video_id":row["video_id"],"status":"REUSED_VERIFIED_COMPLETE"}),flush=True)
                continue
            duration, metadata = probe(row["path"])
            bounds = np.asarray(clip_intervals(duration), dtype=float)
            paths, pts, error = decode(row["path"], dest, bounds)
            overview_path = dest / "overview.jpg"
            overview(paths, pts, overview_path)
            vectors = []
            flat = [p for frames in paths for p in frames]
            with torch.inference_mode():
                for start in range(0, len(flat), 64):
                    imgs = [Image.open(f).convert("RGB") for f in flat[start:start+64]]
                    inputs = processor(images=imgs, return_tensors="pt").to(a.device)
                    vectors.append(encoder(**inputs).image_embeds.float().cpu().numpy())
            frames = unit_rows(np.concatenate(vectors))
            features = unit_rows(frames.reshape(-1, 4, frames.shape[-1]).mean(axis=1))
            feature_path = dest / "features.npy"
            np.save(feature_path, features.astype(np.float32))
            result = dict(row, duration=duration, bounds=bounds.tolist(), frame_paths=paths, sampled_pts=pts,
                          overview=str(overview_path), features_path=str(feature_path),
                          frame_bounds_path=str(dest/"frame_bounds.npy"),
                          sampling_error=error, raw_sha256=digest(row["path"]), features_sha256=digest(feature_path),
                          n_frames=error["decoded_native_frames"],
                          n_clips=len(bounds), n_sampled_frames=len(flat), seconds=time.monotonic()-started)
            (dest/"ffprobe.json").write_text(json.dumps(metadata,indent=2)+"\n")
            result=seal_result(result,a.encoder)
            (dest/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
            prepared.append(result)
            print(json.dumps({"video_id":row["video_id"],"status":"COMPLETED","clips":len(bounds),"sampling_error":error,"seconds":result["seconds"]}),flush=True)
        except Exception as e:
            failures.append({"video_id":row["video_id"],"error":repr(e)})
            print(json.dumps(failures[-1]),flush=True)
    (a.out/"failures.json").write_text(json.dumps(failures,indent=2)+"\n")
    if failures:
        raise RuntimeError("Real-video preparation failed; no partial set promoted")
    temp = a.out/"prepared.jsonl.partial"
    temp.write_text("".join(json.dumps(r)+"\n" for r in prepared))
    temp.replace(a.out/"prepared.jsonl")

if __name__ == "__main__":
    main()
