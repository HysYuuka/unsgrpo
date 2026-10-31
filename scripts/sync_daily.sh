#!/usr/bin/env bash
# Manual tool only. Invoke --sync only for the user's current explicit sync request.
# Automatic daily synchronization was cancelled; do not install a scheduler.
set -euo pipefail
PROJECT=/root/autodl-tmp/unsgrpo
source "$PROJECT/storage-env.sh"
mkdir -p "$UNSGRPO_RUN_DIR/sync"
cd "$PROJECT"
MODE=${1:---check}
case "$MODE" in --check|--sync) ;; *) echo 'usage: sync_daily.sh [--check|--sync]' >&2; exit 2;; esac
export UNSGRPO_SYNC_MODE="$MODE"
python - <<'PY'
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def git(*args, check=True):
    return subprocess.run(['git', *args], check=check, text=True, capture_output=True)

expected = 'https://github.com/HysYuuka/unsgrpo.git'
identity = {k: git('config', '--get', 'user.' + k, check=False).stdout.strip()
            for k in ['name', 'email']}
origin = git('remote', 'get-url', 'origin', check=False).stdout.strip()
report = {'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
          'mode': os.environ['UNSGRPO_SYNC_MODE'], 'origin_matches': origin == expected,
          'identity_name_present': bool(identity['name']),
          'identity_email_present': bool(identity['email']),
          'sync_policy': 'ON_DEMAND_USER_REQUEST_ONLY_NO_AUTOMATION',
          'push_verified': False, 'scheduling_status': 'DISABLED_BY_USER_NO_SYNC_TIMER',
          'status': 'CHECKED', 'errors': []}
if origin != expected:
    report['errors'].append('origin does not match registered repository')
if not all(identity.values()):
    report['errors'].append('git user.name/user.email missing; do not invent commit identity')

# Only small, explicit source/document extensions enter the daily commit.
allowed_roots = {'src', 'tests', 'scripts', 'docs', 'configs', 'manifests', 'reports', 'receipts', 'research'}
allowed_root_files = {'.gitignore', 'storage-env.sh', 'README.md', 'pyproject.toml',
                      'requirements.txt', 'requirements.lock', 'uv.lock', 'pytest.ini'}
allowed_ext = {'.py', '.sh', '.md', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.txt', '.lock', '.csv', '.tsv', '.ini', '.cu'}
forbidden_parts = {'raw', 'prepared', 'labels_eval_only', 'envs', 'models', 'datasets',
                   'checkpoints', 'runs', 'outputs', 'cache', '.git', '__pycache__'}
annotation_keys = {'user_anno', 'user_annotations', 'annotations', 'labels', 'ground_truth',
                   'groundtruth', 'gt', 'ratings', 'human_labels', 'human_scores',
                   'annotator_scores', 'importance_scores'}
credential_pattern = re.compile(rb'(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,})')
# Exact hashes reviewed as path/scope metadata, not annotation payloads.
reviewed_protocol_metadata = {'configs/round2_diagnostic_plan.json': '7fac77798c1ef7def334c74edf19eeeb91a83a55deabf8c1a34ea8eb0f04f0e6', 'configs/round2_diagnostic_protocol_v1.json': '780bd1e4f68534f378c9e76d630fb68b4d0afe88fb04af15f0f1ec1be06abd4d', 'reports/decisive_rl_v2/protocol.actual.json': '123c69237dc898371fbc86fbdf9bebfc90f19028536a321308088c8f0dbd6750', 'reports/reward_to_rl_v1/protocol.json': '5228e6d91f114d159f9905199344583bb9efe6d066fd6ff7aa3fdcda8e9b301b'}

def has_annotation_payload(obj):
    if isinstance(obj, dict):
        return any(str(k).lower() in annotation_keys or has_annotation_payload(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(has_annotation_payload(v) for v in obj)
    return False

def safe(path, staged=False):
    p = Path(path)
    if p.is_absolute() or '..' in p.parts or set(p.parts) & forbidden_parts:
        return False
    if not ((len(p.parts) == 1 and path in allowed_root_files) or
            (len(p.parts) > 1 and p.parts[0] in allowed_roots and (p.suffix in allowed_ext or p.name == 'LICENSE'))):
        return False
    if p.is_symlink() or (p.exists() and p.stat().st_size > 5 * 1024**2):
        return False
    if staged:
        entry = git('ls-files', '--stage', '--', path).stdout
        if entry.startswith('120000 '):
            return False
        blob = subprocess.run(['git', 'show', ':' + path], capture_output=True)
        body = blob.stdout if blob.returncode == 0 else None  # Deletion is safe.
    else:
        body = p.read_bytes() if p.exists() else None
    if body is not None:
        if len(body) > 5 * 1024**2 or credential_pattern.search(body):
            return False
        if b'\0' in body or any(marker in body.splitlines() for marker in [b'-----BEGIN PRIVATE KEY-----',
              b'-----BEGIN RSA PRIVATE KEY-----', b'-----BEGIN OPENSSH PRIVATE KEY-----']):
            return False
        if p.suffix in {'.json', '.jsonl'}:
            try:
                objects = [json.loads(line) for line in body.decode().splitlines() if line.strip()] if p.suffix == '.jsonl' else [json.loads(body)]
            except (ValueError, UnicodeError):
                return False
            if any(has_annotation_payload(obj) for obj in objects):
                if reviewed_protocol_metadata.get(path) != hashlib.sha256(body).hexdigest():
                    return False
    return True

pending = set(git('ls-files', '--others', '--exclude-standard', '-z').stdout.split('\0'))
pending |= set(git('diff', '--name-only', '-z').stdout.split('\0'))
pending.discard('')
selected = sorted(p for p in pending if safe(p))
report['eligible_files'] = selected
report['excluded_files'] = sorted(pending - set(selected))
staged = git('diff', '--cached', '--name-only', '-z').stdout.split('\0')
unsafe_staged = [p for p in staged if p and not safe(p, staged=True)]
if unsafe_staged:
    report['errors'].append('unsafe files already staged; explicit review required')
    report['unsafe_staged_files'] = unsafe_staged

if os.environ['UNSGRPO_SYNC_MODE'] == '--sync' and not report['errors']:
    # Authentication is noninteractive. Credentials never enter code or reports.
    os.environ['GIT_TERMINAL_PROMPT'] = '0'
    if selected:
        git('add', '--', *selected)
    staged = [p for p in git('diff', '--cached', '--name-only', '-z').stdout.split('\0') if p]
    if any(not safe(p, staged=True) for p in staged):
        report['errors'].append('staged audit failed')
    else:
        if staged:
            local_date = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
            git('commit', '-m', f'Record experiment code and lightweight evidence {local_date}')
        if git('rev-parse', '--verify', 'HEAD', check=False).returncode:
            report['errors'].append('no commit available for synchronization')
        else:
            pushed = git('push', 'origin', 'HEAD:refs/heads/main', check=False)
            report['push_exit_code'] = pushed.returncode
            if pushed.returncode:
                report['errors'].append('git push failed; review external sync log and authentication')
                # Do not copy potentially sensitive transport diagnostics into a tracked report.
                Path(os.environ['UNSGRPO_RUN_DIR'], 'sync', 'last_transport_error.log').write_text(pushed.stderr)
            else:
                report['push_verified'] = True
                report['status'] = 'SYNCED'
                report['commit'] = git('rev-parse', 'HEAD').stdout.strip()
if report['errors']:
    report['status'] = 'BLOCKED'
receipt = Path(os.environ['UNSGRPO_RUN_DIR'], 'sync', 'last_sync.json')
receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
sys.exit(1 if report['errors'] else 0)
PY
