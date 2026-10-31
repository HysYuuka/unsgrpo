#!/usr/bin/env python3
"""Independent post-run target/coverage checks; no new predictions or variants."""
import json
import time
from collections import Counter
import numpy as np
from unsgrpo.round3_common import *
from unsgrpo.core import midrank_percentile
from unsgrpo.round3_teacher import parse_events, parse_semantic


def main():
    start=time.monotonic();out=STORE/'teacher';plan,by,train,dev,feat,pseudo=assets()
    state=read(WORK/'front_state.json');assert state['status']=='TEACHER_NOT_READY'
    sealed=read(out/'semantic.freeze.json');calls=records(out/'semantic_calls.jsonl')
    assert len(calls)==12 and sum(r['kind']=='events' for r in calls)==8
    assert all(r['finish_reason']=='stop' for r in calls)
    assert sum(r['valid'] for r in calls if r['kind']=='events')==1
    assert sum(r['valid'] for r in calls if r['kind']!='events')==0
    event_sets={}
    for r in calls:
        try:
            obj=parse_events(r['response']) if r['kind']=='events' else parse_semantic(r['response'],len(r['clip_ids']),event_sets[r['video_id']])
            valid=True
        except (ValueError,KeyError,TypeError):valid=False
        assert valid==r['valid']
        if r['kind']=='events' and valid:event_sets[r['video_id']]=[e['id'] for e in obj['events']]
        if r['kind']!='events':
            assert r['encoding']['actual_cached_frames']==32 and len(r['encoding']['video_grids'])==8
            assert r['encoding']['video_visual_tokens']==sum(int(np.prod(g))//4 for g in r['encoding']['video_grids'])==1008
            assert r['encoding']['image_visual_tokens']==784
        assert not r['encoding']['truncation']
        for item in r['source_media']:assert sha(item['path'])==item['sha256']
    arrays=np.load(out/'reference_targets.npz',allow_pickle=False);qerr=[]
    for k in train:
        peers=[v for v in train if v!=k and pseudo[v]['cluster']==pseudo[k]['cluster']]
        if not peers:peers=[v for v in train if v!=k]
        raw=feat[k]@np.concatenate([feat[v] for v in peers]).mean(0)
        q=midrank_percentile(raw);qerr.append(float(np.max(np.abs(q-arrays['q_excl__'+k]))))
    assert max(qerr)==0
    nclips=sum(len(pseudo[k]['q']) for k in by)
    for tag in ('alpha075','alpha050'):
        target=np.load(out/f'targets_{tag}.npz',allow_pickle=False)
        cliprows=records(out/f'clips_{tag}.jsonl');assert len(cliprows)==nclips
        assert len({(r['video_id'],r['clip_id']) for r in cliprows})==nclips
        for k in by:
            assert np.array_equal(target['q__'+k],arrays['q_excl__'+k])
            assert np.all(target['c__'+k]==.25)
            assert np.array_equal(arrays['q0__'+k],pseudo[k]['q'])
            if k in dev:assert np.array_equal(target['q__'+k],pseudo[k]['q'])
        for row in cliprows:
            assert row['source_frame_hashes']==[by[row['video_id']]['media_files_sha256'][p] for p in by[row['video_id']]['frame_paths'][row['clip_id']]]
        gate=read(REPORT/f'teacher_gate_{tag}.json');assert not gate['passed'] and gate['quality_macro'] is None
        assert gate['high_confidence_clip_coverage_video_macro']==0 and gate['training_pool_usable_windows_fraction']==0
        assert gate['direct_primary_delta_pp']==0
    old=verify_previous_project_sources();freeze=read(WORK/'front_end.freeze.json')
    for group in ('sources','inputs'):
        for p,h in freeze[group].items():assert sha(p)==h,p
    resource={'n_calls':len(calls),'events':8,'semantic':4,'tokens_prompt':sum(r['usage_prompt_tokens'] or 0 for r in calls),
        'tokens_generated':sum(r['usage_completion_tokens'] or 0 for r in calls),
        'generation_wall_seconds':sum(r['seconds'] for r in calls),'all_finish_reason_stop':True,
        'n_valid_event_calls':1,'n_valid_semantic_calls':0,'GPU_occupied_seconds':occupied_seconds(),
        'semantic_packet_video_visual_tokens':1008,'semantic_packet_image_visual_tokens':784,
        'n_clip_records_per_alpha':nclips,'train_clips':sum(len(pseudo[k]['q']) for k in train),
        'development_clips':sum(len(pseudo[k]['q']) for k in dev)}
    write(WORK/'independent_verification.json',{'status':'PASSED','time':timestamp(),'wall_seconds':time.monotonic()-start,
        'checks':{'parser_status_reconstructed':True,'all_logged_media_hashes_unchanged':True,
            'true_4_frames_per_clip_and_32_per_packet':True,'two_alpha_arrays_identical_to_q_excl':True,
            'all_weights_fall_back_to_quarter':True,'dev_scores_identical_to_original_T0':True,
            'q_excl_recomputed_with_whole_train_video_removed':True,'max_q_excl_recomputation_error':max(qerr),
            'original_project_files_unchanged':len(old),'round3_front_frozen_sources_and_inputs_unchanged':True},
        'resources':resource,'no_training_or_inference_added':True,'training_human_annotations_read':False})
    print(json.dumps(resource),flush=True)


if __name__=='__main__':main()
