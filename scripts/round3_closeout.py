#!/usr/bin/env python3
"""Archive an actual teacher-gate stop, verify budgets, and seal the deliverables."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from unsgrpo.round3_common import *


def files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts)


def logical(root):return sum(p.stat().st_size for p in files(root))


def copy_checked(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    shutil.copy2(source,dest)
    if sha(source)!=sha(dest):raise ValueError(f'Archive copy mismatch: {source}')


def main():
    if (REPORT/'CLOSED.json').exists():raise FileExistsError('Already closed')
    state=read(WORK/'front_state.json');assert state['status']=='TEACHER_NOT_READY'
    old=verify_previous_project_sources();front=read(WORK/'front_end.freeze.json')
    for group in ('sources','inputs'):
        for p,h in front[group].items():assert sha(p)==h,p
    reused={}
    prior=read(R2/'round2.freeze.json')
    for p,h in prior['sources'].items():
        if p.startswith(str(PROJECT)+'/src/'):
            assert sha(p)==h,p;reused[p]=h
    write(WORK/'reused_source_integrity.json',{'status':'PASSED','sources':reused,
        'round2_original_freeze':reference(R2/'round2.freeze.json'),'round1_unchanged_project_files':len(old)})
    # Give unattempted slots their own reasons; do not recast missing generations
    # as actual bad answers or produce a third target version.
    calls=records(STORE/'teacher/semantic_calls.jsonl');events={r['video_id']:r for r in calls if r['kind']=='events'}
    rows=records(STORE/'teacher/semantic_clips.jsonl');reasons={};per=[]
    for k in front['train']+front['development']:
        rr=[r for r in rows if r['video_id']==k]
        labels=[]
        for r in rr:
            if k in events and not events[k]['valid']:reason='INVALID_SHARED_EVENT_REFERENCE'
            elif r['call_a'] is not None or r['call_b'] is not None:reason='INVALID_SEMANTIC_RESPONSES'
            else:reason='NOT_SCORED_EARLY_FORMAT_STOP'
            labels.append(reason);reasons[reason]=reasons.get(reason,0)+1
        per.append({'video_id':k,'split':'train' if k in front['train'] else 'development','n_clips':len(rr),
            'event_reference_attempted':k in events,'event_reference_valid':events[k]['valid'] if k in events else None,
            'semantic_clip_attempts':sum(r['call_a'] is not None or r['call_b'] is not None for r in rr),
            'valid_semantic_targets':0,'high_confidence_targets':0,
            'fallback_reason_counts':{name:labels.count(name) for name in sorted(set(labels))}})
    write(REPORT/'teacher_cache_index.json',{'status':'PARTIAL_SEMANTIC_CACHE_FULL_FALLBACK_TARGETS',
        'semantic_calls':reference(STORE/'teacher/semantic_calls.jsonl'),
        'semantic_clips':reference(STORE/'teacher/semantic_clips.jsonl'),
        'targets':[reference(STORE/'teacher'/f'targets_{tag}.npz') for tag in ('alpha075','alpha050')],
        'clip_records':[reference(STORE/'teacher'/f'clips_{tag}.jsonl') for tag in ('alpha075','alpha050')],
        'exclusion':reference(STORE/'teacher/exclusion.json'),'n_clips':len(rows),'reason_counts':reasons,
        'per_video':per,'no_synthesized_semantic_scores_for_unattempted_clips':True})
    sourcepaths=sorted(set((list((PROJECT/'src/unsgrpo').glob('round3_*.py'))+
        list((PROJECT/'scripts').glob('round3_*'))+[PROJECT/'tests/test_round3_contracts.py',
        PROJECT/'reports/ROUND3_RL_METHOD_PLAN_20261028.md',PROJECT/'reports/ROUND3_RL_METHOD_PLAN_20261028.json'])))
    versions=[]
    for p in sourcepaths:
        dest=STORE/'source_snapshot'/p.relative_to(PROJECT);copy_checked(p,dest)
        versions.append({'source':reference(p),'archived':reference(dest),
            'was_frozen_before_teacher':str(p) in front['sources'],
            'postrun_only':str(p) not in front['sources']})
    write(REPORT/'source_versions.json',{'status':'SHA256_VERSIONED_NO_GIT_MUTATION','sources':versions,
        'reused_source_integrity':reference(WORK/'reused_source_integrity.json'),
        'runtime_versions':{k:importlib.metadata.version(k) for k in ('torch','ms-swift','transformers','numpy','scipy','Pillow')},
        'scientific_source_freeze':reference(WORK/'front_end.freeze.json')})
    for p in files(WORK):
        if p.name in ('closeout.log','storage_final.json'):continue
        copy_checked(p,STORE/'provenance_work'/p.relative_to(WORK))
    replay=WORK.parent/'round3_cache_replay_verification_v1'
    copy_checked(replay/'receipt.json',STORE/'cache_replay_receipt.json')
    command=[sys.executable,str(PROJECT/'scripts/storage_audit.py'),'--output',str(WORK/'storage_final.json')]
    with (WORK/'logs/storage_final.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    storage=read(WORK/'storage_final.json');assert storage['ok']
    assert storage['filesystems']['local']['free_bytes']>=10*1024**3
    assert storage['filesystems']['file']['free_bytes']>=120*1024**3
    resources=read(REPORT/'resource_receipt.json')
    now=datetime.now(timezone.utc);start=datetime.fromisoformat(read(WORK/'initial_state.json')['started_utc'])
    aend=datetime.fromisoformat(read(REPORT/'metric_choice.json')['time']['utc'])
    resources.update(time=timestamp(),engineering_elapsed_seconds_since_recorded_start=(now-start).total_seconds(),
        metric_preparation_elapsed_upper_bound_since_recorded_start_seconds=(aend-start).total_seconds(),
        metric_preparation_under_two_hours=True,engineering_since_recorded_start_under_six_hours=True,
        storage={'audit':reference(WORK/'storage_final.json'),'code_report_logical_bytes_before_closure':logical(REPORT),
            'working_logical_bytes_before_closure':logical(WORK),'replay_logical_bytes':logical(replay),
            'persistent_logical_bytes_before_report_snapshot':logical(STORE),
            'new_R3_source_bytes':sum(p.stat().st_size for p in sourcepaths if p.suffix in ('.py','.sh')),
            'local_free_bytes':storage['filesystems']['local']['free_bytes'],
            'file_free_bytes':storage['filesystems']['file']['free_bytes'],
            'local_minimum_free_bytes':10*1024**3,'file_minimum_free_bytes':120*1024**3,
            'snapshot_scope':'After teacher, cached reproduction and large provenance/source copies; before final small report/control files.',
            'raw_video_assets_added_bytes':0,'base_model_copies_added':0,'limits_passed':True},
        training_admission_failed=True,unused_GPU_budget_not_reallocated_to_search=True)
    assert resources['engineering_elapsed_seconds_since_recorded_start']<21600
    assert resources['metric_preparation_elapsed_upper_bound_since_recorded_start_seconds']<7200
    assert occupied_seconds()<43200
    assert logical(WORK)+logical(replay)<8*1024**3
    assert logical(STORE)<30*1024**3
    assert logical(REPORT)<100*1024**2
    write(REPORT/'resource_receipt.json',resources,fresh=False)
    for p in files(REPORT):copy_checked(p,STORE/'report_snapshot'/p.relative_to(REPORT))
    copy_checked(WORK/'storage_final.json',STORE/'final_control/storage_final.json')
    copy_checked(WORK/'logs/storage_final.log',STORE/'final_control/storage_final.log')
    allfiles=[]
    for label,root in [('working',WORK),('persistent',STORE),('reports',REPORT),('cache_replay',replay)]:
        for p in files(root):
            if p.name in ('closeout.log','artifact_index.json','CLOSED.json'):continue
            allfiles.append(dict(reference(p),location=label))
    index={'status':'VERIFIED_ARTIFACT_INDEX','time':timestamp(),'files':allfiles,
        'files_count':len(allfiles),'logical_bytes_including_verified_copies':sum(r['bytes'] for r in allfiles),
        'scope_note':'Contains only round-three generated files and reproduced-cache outputs; no models/videos or original annotation arrays copied. Index and final closure control records exclude themselves.',
        'original_input_references':front['inputs'],'old_artifacts_deleted':False,
        'model_endpoints':'NONE: not trained because teacher admission failed',
        'actual_policy_rollout_and_loss_ledgers':'NONE: no policy rollout or update was executed'}
    write(REPORT/'artifact_index.json',index);copy_checked(REPORT/'artifact_index.json',STORE/'artifact_index.json')
    # Re-open every indexed file, rather than trusting successful copy calls alone.
    for r in allfiles:
        assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes'],r['path']
    closure={'status':'COMPLETED_WITH_PREDECLARED_EARLY_STOP','decision':'TEACHER_NOT_READY','time':timestamp(),
        'main_report':reference(REPORT/'DECISION.md'),'actual_spec':reference(REPORT/'method_spec.actual.json'),
        'resource_receipt':reference(REPORT/'resource_receipt.json'),'artifact_index':reference(REPORT/'artifact_index.json'),
        'storage_limits_and_reserves_passed':True,'frozen_first_round_project_files_unchanged':len(old),
        'frozen_round2_reused_sources_unchanged':True,'frozen_round3_front_sources_and_inputs_unchanged':True,
        'GPU_occupied_seconds':occupied_seconds(),'new_training_runs':0,'teacher_calls':12,
        'formal_RL_hypothesis_tested':False,'automatic_Git_sync':False,'subagents_used':False,
        'persistent_root':str(STORE),'policy_adapter_files':[],'remaining_planned_work':
            'C GPU signal/update check, D/E formal training, new F inference, G external evaluation deliberately not reached after failed teacher gate.'}
    write(REPORT/'CLOSED.json',closure);copy_checked(REPORT/'CLOSED.json',STORE/'final_control/CLOSED.json')
    print(json.dumps({'status':closure['status'],'decision':closure['decision'],
        'GPU_occupied_seconds':occupied_seconds(),'temporary_bytes':logical(WORK)+logical(replay),
        'persistent_bytes':logical(STORE),'report_bytes':logical(REPORT),'indexed_files':len(allfiles)}),flush=True)


if __name__=='__main__':main()
