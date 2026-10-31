#!/usr/bin/env python3
"""Post-run independent checks. No generation, fitting or alternative method."""
import math
from collections import defaultdict
import json
import time
import numpy as np
from sklearn.metrics import average_precision_score
from unsgrpo.reward_to_rl_common import *
from unsgrpo.core import parse_scores
from unsgrpo.diagnostics import StableMapper


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))


def check_edges(path):
    rr=records(path);maximum=0.;count=0
    for r in rr:
        a,b=[x['log_probabilities_ABC'] for x in r['views']]
        first=sigmoid(a[0]-a[1]);second=sigmoid(b[1]-b[0]);p=(first+second)/2
        abc=[]
        for scores in (a,b):
            shifted=[math.exp(v-max(scores)) for v in scores];abc.append([v/sum(shifted) for v in shifted])
        t=(abc[0][2]+abc[1][2])/2;weight=(1-t)*(1-abs(first-second))*2*abs(p-.5)
        maximum=max(maximum,abs(weight-r['weight']),abs(first-r['p1']),abs(second-r['p2']))
        expected=(abc[0][2]<max(abc[0][:2]) and abc[1][2]<max(abc[1][:2]) and a[0]!=a[1] and b[0]!=b[1]
            and int(a[1]>a[0])!=int(b[1]>b[0]) and weight>=.2)
        assert expected==r['kept']
        if r['kept']:assert (p>.5)==r['preferred_first_clip']
        for view in r['views']:
            d=view['detail'];assert d['encoding']['processed_video_frames']==8
            if d['single_token_same_position']:
                assert all(len(x)==1 for x in d['answer_token_ids'])
                assert abs(sum(math.exp(v) for v in view['log_probabilities_ABC'])-d['raw_candidate_probability_mass'])<1e-12
            else:assert d['raw_candidate_probability_mass'] is None
            count+=d['forward_calls']
    assert maximum<1e-12
    return {'edges_or_pairs':len(rr),'forward_calls':count,'max_alignment_error':maximum}


def main():
    started=time.monotonic();validate_freeze();plan,by,train,dev,feat,pseudo=assets()
    state=read(WORK/'state.json');checks={};maximum=0.
    for name in ('interface.jsonl','teacher_edges.jsonl'):
        p=STORE/'screen'/name
        if p.exists():checks[name]=check_edges(p)
    p=STORE/'screen/candidates.jsonl'
    if p.exists():
        candidates=records(p);assert len(candidates)==240
        assert len({(r['request_id'],r['g']) for r in candidates})==240
        assert len({r['rng_before_sha256'] for r in candidates})==240
        for sample in candidates:
            try:parse_scores(sample['response'],len(sample['clip_ids']));legal=True
            except (ValueError,TypeError):legal=False
            assert legal==sample['legal'] and sample['encoding']['processed_video_frames']==32
            assert sample['sampling_seed']==31415
        checks['candidates']={'count':240,'all_rng_states_distinct':True,'same_frozen_S0_only':True}
    evaluated=REPORT/'screen_result.json'
    if evaluated.exists():
        sealed=read(STORE/'screen/candidates_and_rewards.freeze.json')
        for p,h in sealed['inputs'].items():assert sha(p)==h
        hrows=records(STORE/'screen/single_window_full_video_AP.jsonl')
        lookup={(r['request_id'],r['g']):r for r in hrows}
        cached=records(ORIGINAL/'inference/S0/windows.jsonl');clock={};binary={}
        for k in dev:
            data=np.load(Path(plan['paths']['labels'])/(k+'.npz'),allow_pickle=False)
            clock[k]=StableMapper(by[k]['bounds'],data['bounds'])
            ratings=data['ratings'];assert ratings.shape[1]==20
            binary[k]=ratings>np.median(ratings,axis=0)
        for candidate in candidates:
            k=candidate['video_id'];rid=candidate['request_id'];n=len(by[k]['bounds']);summed=[0.]*n;counts=[0]*n
            for old in cached:
                if old['video_id']!=k:continue
                text=candidate['response'] if old['request_id']==rid else old['response']
                try:values=parse_scores(text,len(old['clip_ids']))
                except (ValueError,TypeError):values=np.zeros(len(old['clip_ids']))
                for i,value in zip(old['clip_ids'],values):summed[i]+=float(value);counts[i]+=1
            assert min(counts)>0
            score=np.round(clock[k](np.array([a/b for a,b in zip(summed,counts)])),12)
            ap=[float(average_precision_score(binary[k][:,j],score)) if binary[k][:,j].any() else 0. for j in range(20)]
            actual=lookup[(rid,candidate['g'])]
            maximum=max(maximum,max(abs(x-y) for x,y in zip(ap,actual['H_by_annotator'])),abs(np.mean(ap)-actual['H']))
        assert maximum<1e-12
        result=read(evaluated);groups=result['per_window'];per=defaultdict(list)
        for r in groups:
            if r['all_four_legal']:per[r['video_id']].append(r)
        O=float(np.mean([np.mean([max(r['H'])-np.mean(r['H']) for r in values]) for values in per.values()]))
        assert abs(O-result['primary_all_four_legal']['O'])<1e-12
        for name in ('R_old','R_pair'):
            video=[]
            for k,rows in per.items():
                vv=[]
                for r in rows:
                    h=np.asarray(r['H']);reward=np.round(r[name]['reward_values'],12)
                    selected=h[reward==max(reward)].mean();vv.append(float(selected-h.mean()))
                video.append(float(np.mean(vv)))
            G=float(np.mean(video));assert abs(G-result['primary_all_four_legal'][name]['G'])<1e-12
        checks['single_window_evaluation']={'full_video_candidates_recomputed':len(candidates),'user_AP_recomputed':len(candidates)*20,
            'independent_scalar_window_accumulation':True,'sklearn_AP_max_error':maximum,'G_and_O_recomputed_by_source':True,
            'independent_native_clock_reconstruction':False}
    for p in (STORE/'training').glob('*/receipt.json') if (STORE/'training').exists() else []:
        receipt=read(p)
        if receipt.get('status') in ('ACTUAL_UPDATE_CHECK_PASSED','COMPLETED_ACTUAL_TRAINING'):
            assert receipt['frozen_before']==receipt['frozen_after']
            assert receipt['completed_optimizer_steps']==(4 if receipt['technical'] else 80)
            if receipt['technical']:assert receipt['initial_adapter']!=receipt['final_adapter'] and receipt['reference_function_unchanged']
    # The new namespace must leave the prior closed round intact.
    prior=read(PROJECT/'reports/round3_rl_method_v1/artifact_index.json')
    for r in prior['files']:assert sha(r['path'])==r['sha256']
    checks['prior_round3_indexed_artifacts_unchanged']=len(prior['files'])
    write(WORK/'independent_verification.json',{'status':'PASSED','time':now(),'wall_seconds':time.monotonic()-started,
        'checks':checks,'new_GPU_calls':0,'training_human_labels_read':False,'scientific_variants_added':0})
    print(json.dumps({'status':'PASSED','max_AP_error':maximum,'checks':checks}),flush=True)


if __name__=='__main__':main()
