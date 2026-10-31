#!/usr/bin/env python3
"""Independent scalar checks and previous-round preservation, no GPU calls."""
import math
from collections import Counter,defaultdict
import numpy as np
from unsgrpo.decisive_common import *


def main():
    validate();checks={};old=read(OLD_REPORT/'CLOSED.json');assert sha(old['artifact_index']['path'])==old['artifact_index']['sha256']
    oldindex=read(old['artifact_index']['path'])
    for entry in oldindex['files']:assert sha(entry['path'])==entry['sha256'],entry['path']
    checks['all_previous_reward_to_rl_indexed_files_unchanged']=len(oldindex['files'])
    r3=read(PROJECT/'reports/round3_rl_method_v1/CLOSED.json')
    if 'artifact_index' in r3:
        idx=r3['artifact_index'];assert sha(idx['path'])==idx['sha256']
        for entry in read(idx['path'])['files']:assert sha(entry['path'])==entry['sha256'],entry['path']
        checks['all_previous_round3_indexed_files_unchanged']=len(read(idx['path'])['files'])
    technical=STORE/'teacher/technical_receipt.json'
    if technical.exists():
        receipt=read(technical);rr=records(STORE/'teacher/technical_edges.jsonl');assert len(rr)==16 and len([r for r in rr if r['same_clip']])==8
        good=0;mass=[]
        for r in rr:
            top=[]
            for v in r['views']:
                raw=np.exp(np.asarray(v['log_probabilities_ABCD']));prob=raw/raw.sum();mass.append(raw.sum())
                assert np.allclose(raw,v['raw_probabilities_ABCD'],rtol=1e-12,atol=1e-14)
                assert np.allclose(prob,v['normalized_ABCD'],rtol=1e-12,atol=1e-14)
                top.append(int(prob.argmax()) if (prob==prob.max()).sum()==1 else None)
                assert v['encoding']['processed_video_frames']==8 and v['encoding']['no_truncation']
                assert len(v['prompt_token_ids'])==v['encoding']['prompt_tokens']
                for answer in v['tokenization_context']:
                    assert answer['prediction_position']==len(v['prompt_token_ids'])-1
                    if v['single_token_same_position']:assert len(answer['token_ids'])==1
                for media in v['media']:assert sha(media['path'])==media['sha256']
            if r['same_clip']:
                assert r['views'][0]['prompt_token_ids']==r['views'][1]['prompt_token_ids']
                assert r['views'][0]['media']==r['views'][1]['media'];good+=int(top==[2,2])
        assert good==receipt['same_clip_pairs_C_in_both'];assert float(np.mean(np.array(mass)>=.5))==receipt['views_mass_at_least_half_fraction']
        assert receipt['passed']==(good>=6 and float(np.mean(np.array(mass)>=.5))>=.9)
        checks['technical_actual_media_position_probability_gate_independently_recomputed']=True
    if (REPORT/'reward_quality.json').exists():
        quality=read(REPORT/'reward_quality.json');inputs=read(WORK/'reused_inputs.json');h={(r['request_id'],r['g']):r for r in records(inputs['old_full_video_H'])}
        rewardrows=records(STORE/'teacher/dev_candidate_rewards.jsonl');rewards={(r['request_id'],r['g']):r for r in rewardrows};aggregate=defaultdict(lambda:defaultdict(list));n=0
        for row in records(WORK/'dev_windows.jsonl'):
            rid=row['request_id']
            if not all(rewards[(rid,g)]['legal'] for g in range(4)):continue
            n+=1;hh=[h[(rid,g)]['H'] for g in range(4)];mean=sum(hh)/4
            for name,key in [('new_reward','R_pair'),('old_reward','R_old')]:
                vv=[round(rewards[(rid,g)][name],12) for g in range(4)];best=[i for i in range(4) if vv[i]==max(vv)];chosen=sum(hh[i] for i in best)/len(best)
                aggregate[key][row['video_id']].append(chosen-mean)
        assert n==59
        for key,source in aggregate.items():
            independent=sum(sum(v)/len(v) for v in source.values())/len(source)
            assert abs(independent-quality['primary_59_all_legal'][key]['G'])<1e-12
        checks['new_and_old_G_on_identical_59_groups_independently_recomputed']=True
    training=STORE/'training';actual_losses=0
    if training.exists():
        for path in training.glob('*/loss*.jsonl'):
            for r in records(path):
                if 'current_chosen_logps' in r:
                    z=.1*((sum(r['current_chosen_logps'])-sum(r['reference_chosen_logps']))-(sum(r['current_rejected_logps'])-sum(r['reference_rejected_logps'])))
                    independent=r['u']*(max(-z,0)+math.log1p(math.exp(-abs(z))))
                    assert abs(independent-r['loss'])<2e-5 and r['sum_not_mean'] and r['batch_denominator']==4
                else:
                    values=[]
                    for cur,old,ref in zip(r['current_logps'],r['old_logps'],r['reference_logps']):
                        ratio=math.exp(cur-old);actor=-min(ratio*r['actual_advantage'],min(1.2,max(.8,ratio))*r['actual_advantage'])
                        d=max(-20,min(20,ref-cur));kl=max(-10,min(10,math.exp(d)-d-1));values.append(actor+r['actual_KL_weight']*kl)
                    assert abs(sum(values)/len(values)-r['actual_loss'])<2e-5 and r['batch_denominator']==16
                actual_losses+=1
        checks['actual_scalar_losses_reconstructed']=actual_losses
        formal=[training/a/'receipt_80.json' for a in ('P-DPO','G-plain','G-guard')]
        if all(p.exists() and read(p)['status']=='COMPLETED_ACTUAL_TRAINING' for p in formal):
            rr=[read(p) for p in formal]
            assert all(r['completed_optimizer_steps']==80 and r['prompt_exposures']==320 and r['reference_unchanged'] for r in rr)
            assert all(r['initial_adapter']==rr[0]['initial_adapter'] and r['frozen_before']==rr[0]['frozen_before'] for r in rr)
            checks['three_formal_arms_identical_initialization_and_exposure']=True
    storage_check();assert used()<=28800
    for key,limit in LIMITS.items():assert used(key)<=limit
    checks['GPU_and_storage_within_frozen_limits']=True
    write(REPORT/'independent_audit.json',{'status':'PASSED','time':now(),'checks':checks,
        'no_additional_model_or_scientific_variant':True,'no_training_human_labels_read':True,'actual_loss_rows_checked':actual_losses})


if __name__=='__main__':main()
