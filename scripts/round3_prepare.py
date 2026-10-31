#!/usr/bin/env python3
"""Stage A: source-bound metric checks, fallback selection and front-end freeze."""
import ast
import copy
import hashlib
import json
import shutil
import time
from types import SimpleNamespace
import numpy as np
import torch
from unsgrpo.round3_common import *
from unsgrpo.round2_metrics import DevelopmentEvaluator
from unsgrpo.evaluation import evaluate_top5_umt


def check_source_component():
    path=WORK/'sources/umt_tvsum.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TVSum')
    function=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate'))
    module=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(module)
    scope={'torch':torch,'nncore':SimpleNamespace(to_dict_of_list=lambda x:x)}
    exec(compile(module,str(path),'exec'),scope)
    rng=np.random.default_rng(310);errors=[]
    for n in (5,6,8,31,100):
        for kind in ('random','constant','ties'):
            human=rng.integers(1,6,(n,20));score=rng.normal(size=n)
            if kind=='constant':score[:]=.5
            elif kind=='ties':score=np.round(score)
            obj=SimpleNamespace(label={'one':{'anno':human.tolist()}},get_video_id=lambda idx:'one')
            expected=scope['evaluate'](obj,{'saliency':[torch.tensor(score,dtype=torch.float32)[None,:]]})['mAP']
            got=evaluate_top5_umt(score,human)['mean_ap']
            assert round(got,5)==expected,(n,kind,got,expected)
            errors.append(abs(got-expected))
    return {'status':'PASSED','exact_upstream_method_AST':reference(path),'fixtures':len(errors),
        'max_difference_due_to_upstream_round5':max(errors),'torch_version':torch.__version__,
        'source_median_top5_denominator_trapezoid_and_unstable_float32_ties_preserved':True}


def main():
    if (WORK/'front_end.freeze.json').exists():raise FileExistsError('Round three preparation already frozen')
    plan,by,train,dev,feat,pseudo=assets();old=verify_previous_project_sources()
    if '7 passed' not in (WORK/'logs/cpu_contracts.log').read_text():raise ValueError('CPU contracts must pass before freeze')
    pairpool={};rng=np.random.default_rng(310)
    for k in dev:
        ii,jj=np.triu_indices(len(by[k]['bounds']),1)
        selected=rng.choice(len(ii),min(2000,len(ii)),replace=False)
        pairpool[k]=np.stack([ii[selected],jj[selected]],1).tolist()
    write(WORK/'teacher_pair_pool.json',{'seed':310,'pairs':pairpool,'uses_human_values':False})
    implementation=check_source_component()
    ev=DevelopmentEvaluator(plan,by,dev);axis=[]
    for k in dev:
        bounds,ratings=ev.labels[k];r=by[k]
        assert ratings.shape==(r['n_frames'],20)
        assert np.array_equal(bounds,np.load(r['frame_bounds_path'],allow_pickle=False))
        assert abs(bounds[0,0])<1e-6 and abs(bounds[-1,1]-r['duration'])<1e-6
        assert np.allclose(bounds[1:,0],bounds[:-1,1],rtol=0,atol=1e-6)
        axis.append({'video_id':k,'native_frames':len(bounds),'annotators':20,'prediction_clips':len(r['bounds']),
            'duration':r['duration'],'native_clock':reference(r['frame_bounds_path']),
            'converted_label':reference(Path(plan['paths']['labels'])/(k+'.npz')),
            'V0_overlap_integral_mapping':'verified inherited actual PTS and stable mapper',
            'UMT_annotation_clip_axis_verified':False})
    # The upstream released annotation archive could not be obtained. Do not invent
    # two-second rows or apply top-5 to native frames to simulate public alignment.
    metric={'status':'FROZEN_BEFORE_NEW_TEACHER_RESULTS','time':timestamp(),
        'primary':'V0-stable-mapper-round12','public_top5_status':'NOT_READY',
        'preferred_public_name':'UMT-source TVSum development top-5 mAP',
        'implementation_acceptance':implementation,'native_time_axis_acceptance':axis,
        'public_clip_axis_acceptance':'NOT_READY: released UMT annotation/feature clip-time correspondence unavailable; do not infer rows are 2 seconds',
        'failed_release_archive_attempts':[reference(WORK/'sources'/n) for n in ('additional_network_receipt.json','final_network_probes.json')],
        'scope':'10 previously analyzed development sources; no paper split or independent test claim',
        'selected_without_new_teacher_quality':True,'old_benchmark_alignment_flag_modified':False,
        'rounding':'Stable frame mapper then round 12 decimals; exact ties grouped by V0; raw stable value secondary',
        'no_alternative_AP_definition_attempted':True,'maximum_possible_final_claim':'internal method signal'}
    write(REPORT/'metric_choice.json',metric)
    baselines={}
    for method in ('B0','B1','S0','SPLUS','G-CAL','G-RANK'):
        pp=Path(plan['paths']['original_run'])/'inference'/method/'videos.jsonl'
        lines=records(pp);scores={r['video_id']:np.asarray(r['scores']) for r in lines}
        # Original videos.jsonl exposes normalized scores; bind to expected clip count.
        assert all(len(scores[k])==len(by[k]['bounds']) for k in dev)
        baselines[method]={'origin':'HISTORICAL_NOT_MATCHED_128_STEP_BASELINE','predictions':reference(pp),
            'evaluation':ev.evaluate(scores),'public_top5':'NOT_READY'}
    write(REPORT/'historical_baselines.json',baselines)
    # Official selected-domain split metadata only: no highlight annotation file.
    selected=[];queues={}
    commit=read(WORK/'sources/youtube_commit_direct.json')['sha']
    for category in ('dog','gymnastics','parkour','skating','skiing','surfing'):
        p=WORK/'sources'/(category+'_vlist_sel.json');source=read(p)
        ids=sorted({k for vv,name in source if name.startswith('Test') for k in vv},
            key=lambda k:(hashlib.sha256(k.encode()).hexdigest(),k))
        queues[category]=ids
        selected += [{'category':category,'video_id':k,'hash':hashlib.sha256(k.encode()).hexdigest(),
            'official_partition':next(name for vv,name in source if name.startswith('Test') and k in vv),
            'availability':'NOT_VERIFIED_RAW_VIDEO','source_list':reference(p)} for k in ids[:4]]
    external={'status':'CANDIDATE_IDS_FROZEN_RAW_ASSETS_NOT_READY','time':timestamp(),
        'official_repository':'https://github.com/aliensunmin/DomainSpecificHighlight','official_revision':commit,
        'list_type':'vlist_sel official domain-relevant test subsets','hash_rule':'sha256(UTF-8 video_id), ascending then ID',
        'candidate_sources':selected,'replacement_queues':queues,'available_raw_sources':0,
        'complete_available_24_source_screen_ready':False,'availability_exception':'Network/assets not ready; candidates are not claimed to be available raw videos.',
        'highlight_annotations_read':False,'raw_asset_bytes_downloaded':0,
        'evaluation':'Only after internal repeat gate and raw availability; otherwise DATA_NOT_READY or NOT_REACHED_PREVIOUS_GATE',
        'replacement_rule':'Only raw unavailability, in frozen per-category hash order; every replacement logged'}
    write(REPORT/'external_screen.json',external)
    spec=read(PROJECT/'reports/ROUND3_RL_METHOD_PLAN_20261028.json')
    spec.update(status='FRONT_END_IMPLEMENTATION_FROZEN_BEFORE_TEACHER_EXECUTION',
        execution_in_this_chat=True,actual_primary=metric['primary'],public_metric_status='NOT_READY',
        engineering_start=read(WORK/'initial_state.json')['started_utc'],GPU_ledger=str(WORK/'gpu_sessions.jsonl'),
        format_implementation={'version0':'strict JSON schema instructions','version1_only_if_precheck_fails':'one fixed JSON syntax example; event/scoring semantics unchanged'},
        precheck_video_rule='first eight lexicographically sorted training IDs; first packet per video',
        source_control='Per-file SHA256; no repository commit exists; no Git mutations',
        formal_training_status='NOT_STARTED_PENDING_TEACHER_AND_REWARD_GATES',
        external_asset_status=external['status'],subagents_used=False)
    write(REPORT/'method_spec.json',spec)
    sources=[p for p in (PROJECT/'src/unsgrpo').glob('round3_*.py')]+[p for p in (PROJECT/'scripts').glob('round3_*.py')]
    sources += [PROJECT/'tests/test_round3_contracts.py',PROJECT/'reports/ROUND3_RL_METHOD_PLAN_20261028.md',PROJECT/'reports/ROUND3_RL_METHOD_PLAN_20261028.json']
    snap=WORK/'front_end_source_snapshot'
    for p in sources:
        dest=snap/p.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    input_paths=[Path(plan['paths'][n]) for n in ('prepared','pseudo','cluster_model','train_pool','eval_pool')]
    input_paths += [PROJECT/'reports/common_s0.json',PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json',REPORT/'metric_choice.json',REPORT/'external_screen.json',WORK/'teacher_pair_pool.json']
    write(WORK/'front_end.freeze.json',{'status':'FROZEN_BEFORE_TEACHER_CALLS','time':timestamp(),
        'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in input_paths},
        'prior_frozen_project_sources':old,'source_snapshot':str(snap),'train':train,'development':dev,
        'tests':reference(WORK/'logs/cpu_contracts.log'),'method_spec':reference(REPORT/'method_spec.json')})
    print(json.dumps({'status':'STAGE_A_COMPLETED','primary':metric['primary'],'public_top5':'NOT_READY',
        'native_time_axes_verified':len(axis),'upstream_component_fixtures':implementation['fixtures'],
        'external':'24 hash-frozen candidates; raw video assets NOT_READY'}),flush=True)


if __name__=='__main__':main()
