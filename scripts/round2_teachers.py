#!/usr/bin/env python3
"""Train-only teacher sensitivity, followed by development-only diagnostics."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from unsgrpo.core import midrank_percentile, unit_rows
from unsgrpo.round2_utils import *
from unsgrpo.round2_metrics import DevelopmentEvaluator, rho


def main():
    start=time.monotonic();plan,by,train,dev,feat,pseudo=load_assets();out=ROOT/'teachers';out.mkdir()
    original=joblib.load(plan['paths']['cluster_model'])
    if original['train_ids']!=train:raise ValueError('Cached train order changed')
    z=np.asarray(original['umap'].embedding_);video=unit_rows(np.stack([feat[k].mean(0) for k in train]))
    devvideo=unit_rows(np.stack([feat[k].mean(0) for k in dev]))
    original_cluster={k:int(pseudo[k]['cluster']) for k in by}
    variants={'T0':{'cluster':original_cluster,'family':'original','settings':'Frozen original R0'}}
    # One common held-out transform: this is not re-fitted on development videos.
    zdev=original['umap'].transform(devvideo)
    reproduced_dev=original['kmeans'].predict(zdev)
    if not np.array_equal(reproduced_dev,[original_cluster[k] for k in dev]):
        raise ValueError('Cached original UMAP.transform/KMeans.predict does not reproduce development assignments')
    for seed in plan['teachers']['fixed_embedding_KMeans']['seeds']:
        model=KMeans(n_clusters=10,n_init=20,random_state=seed,algorithm='lloyd').fit(z)
        cluster=dict(zip(train,map(int,model.labels_)));cluster.update(zip(dev,map(int,model.predict(zdev))))
        variants[f'KM{seed}']={'cluster':cluster,'family':'fixed_embedding_KMeans',
            'settings':{'seed':seed,'k':10,'fixed_umap':True,'n_init':20}}
        joblib.dump(model,out/f'KM{seed}.joblib',compress=3)
    import umap
    for seed in plan['teachers']['full_pipeline']['seeds']:
        config=dict(original['umap_config']);config.update(random_state=seed,transform_seed=seed)
        reducer=umap.UMAP(**config);emb=reducer.fit_transform(video);candidates=[];best=None;best_score=-np.inf
        for k in range(4,16):
            model=KMeans(n_clusters=k,n_init=20,random_state=seed,algorithm='lloyd').fit(emb)
            value=float(silhouette_score(emb,model.labels_))
            candidates.append({'k':k,'silhouette':value})
            if value>best_score:best_score,best=value,model
        cluster=dict(zip(train,map(int,best.labels_)));cluster.update(zip(dev,map(int,best.predict(reducer.transform(devvideo)))))
        variants[f'FULL{seed}']={'cluster':cluster,'family':'full_pipeline',
            'settings':{'seed':seed,'umap':config,'selected_k':best.n_clusters,'candidates':candidates}}
        joblib.dump({'umap':reducer,'kmeans':best},out/f'FULL{seed}.joblib',compress=3)
    variants['VIDEO_EQUAL']={'cluster':original_cluster,'family':'prototype', 'video_equal':True,
                             'settings':'Average each training video mean equally inside original cluster'}
    variants['GLOBAL_VIDEO_EQUAL']={'cluster':{k:0 for k in by},'family':'prototype','video_equal':True,
                                    'settings':'Equal training-video means, one global bank, no clusters'}
    qarrays={};rawarrays={};details={}
    for name,spec in variants.items():
        banks,fallback=prototypes(feat,train,spec['cluster'],video_equal=spec.get('video_equal',False))
        predictions=teacher_scores(feat,list(by),spec['cluster'],banks)
        for k in by:qarrays[f'{name}__{k}']=predictions[k]['q'];rawarrays[f'{name}__{k}']=predictions[k]['raw']
        if name=='T0':
            for k in by:
                if not np.array_equal(predictions[k]['q'],pseudo[k]['q']) or not np.allclose(predictions[k]['raw'],pseudo[k]['raw'],rtol=0,atol=1e-14):
                    raise ValueError(f'Original q/raw not reproduced: {k}')
        ari=float(adjusted_rand_score([original_cluster[k] for k in train],[spec['cluster'][k] for k in train]))
        rows=[]
        for k in by:
            before=set(t for t in train if original_cluster[t]==original_cluster[k])
            after=set(t for t in train if spec['cluster'][t]==spec['cluster'][k])
            rows.append({'video_id':k,'split':'train' if k in train else 'dev',
                'teacher_vs_T0_spearman':rho(predictions[k]['q'],pseudo[k]['q']),
                'mean_absolute_q_change':float(np.mean(np.abs(predictions[k]['q']-pseudo[k]['q']))),
                'reference_membership_jaccard':len(before&after)/len(before|after),
                'raw_std':float(np.std(predictions[k]['raw'])),'raw_span':float(np.ptp(predictions[k]['raw']))})
        details[name]={'family':spec['family'],'settings':spec['settings'],'train_partition_ARI':ari,
                      'cluster_assignments':spec['cluster'],'per_video':rows,
                      'train_rank_stability':numerical_summary([r['teacher_vs_T0_spearman'] for r in rows if r['split']=='train']),
                      'dev_rank_stability':numerical_summary([r['teacher_vs_T0_spearman'] for r in rows if r['split']=='dev'])}
    # Self-exclusion and video weighting are isolated: no training human labels.
    loo=[]
    for k in train:
        row={'video_id':k,'n_clips':len(feat[k]),'cluster':original_cluster[k]}
        peers=[t for t in train if original_cluster[t]==original_cluster[k]]
        row['self_clip_mass']=len(feat[k])/sum(len(feat[t]) for t in peers)
        row['equal_video_self_mass']=1/len(peers)
        for equal,label in [(False,'clip_weighted'),(True,'video_equal')]:
            inc,_=prototypes(feat,train,original_cluster,video_equal=equal)
            exc,fallback=prototypes(feat,train,original_cluster,video_equal=equal,exclude=k)
            a=feat[k]@inc[original_cluster[k]];b=feat[k]@exc[original_cluster[k]]
            qa,qb=midrank_percentile(a),midrank_percentile(b)
            row[label]={'q_spearman':rho(qa,qb),'mean_absolute_q_change':float(np.mean(np.abs(qa-qb))),
                'mean_absolute_raw_change':float(np.mean(np.abs(a-b))),'global_fallback':bool(fallback.get(original_cluster[k],False))}
            qarrays[f'LOO_{label}__{k}']=qb
        loo.append(row)
    np.savez_compressed(out/'q_scores.npz',**qarrays);np.savez_compressed(out/'raw_scores.npz',**rawarrays)
    write(out/'unsupervised_teacher_receipt.json',{'status':'TEACHERS_FROZEN_BEFORE_NEW_HUMAN_COMPARISONS',
        'train_human_labels_read':False,'fit_video_ids':train,'feature_sources':{k:reference(by[k]['features_path']) for k in by},
        'q_scores':reference(out/'q_scores.npz'),'raw_scores':reference(out/'raw_scores.npz'),
        'variant_families':{k:v['family'] for k,v in details.items()}})
    # Only now load the 10 development annotation files for all frozen versions.
    ev=DevelopmentEvaluator(plan,by,dev)
    for name in details:
        details[name]['development_evaluation']=ev.evaluate({k:qarrays[f'{name}__{k}'] for k in dev})
        for row in details[name]['per_video']:
            k=row['video_id']
            if k in dev:row['teacher_vs_human_clip_spearman']=rho(qarrays[f'{name}__{k}'],ev.clip_labels[k])
    # All pairwise teacher stability, independently of cluster label permutations.
    matrix={}
    for a in details:
        matrix[a]={b:float(np.mean([rho(qarrays[f'{a}__{k}'],qarrays[f'{b}__{k}']) for k in dev])) for b in details}
    candidates=[]
    for k in dev:
        human=midrank_percentile(ev.clip_labels[k]);stack=np.stack([qarrays[f'{name}__{k}'] for name in details]);mean=stack.mean(0);std=stack.std(0)
        family={f:np.mean([qarrays[f'{n}__{k}'] for n in details if details[n]['family']==f],axis=0)
                for f in sorted({v['family'] for v in details.values()})}
        human_user=[];positive=[];human_user_raw=[]
        bounds,ratings=ev.labels[k]
        from unsgrpo.diagnostics import StableMapper
        inv=StableMapper(bounds,by[k]['bounds'])
        for j in range(20):
            raw_user=inv(ratings[:,j]);human_user_raw.append(raw_user)
            human_user.append(midranks_percentile_safe(raw_user))
            positive.append(inv(ev.binary[k][:,j].astype(float)))
        human_user=np.stack(human_user);positive=np.stack(positive);human_user_raw=np.stack(human_user_raw)
        for i in range(len(mean)):
            high=human[i]<=.25 and std[i]<=.1 and all(f[i]>=.75 for f in family.values())
            low=human[i]>=.75 and std[i]<=.1 and all(f[i]<=.25 for f in family.values())
            frame_refs=[reference(p) for p in by[k]['frame_paths'][i]]
            if any(r['sha256']!=by[k]['media_files_sha256'][r['path']] for r in frame_refs):
                raise ValueError('Cached sampled frame differs from original sealed input')
            candidates.append({'video_id':k,'clip_index':i,'bounds':by[k]['bounds'][i],
                'teacher_mean_percentile':float(mean[i]),'teacher_std':float(std[i]),'teacher_min':float(stack[:,i].min()),'teacher_max':float(stack[:,i].max()),
                'teacher_percentiles':{n:float(qarrays[f'{n}__{k}'][i]) for n in details},
                'family_mean_percentiles':{f:float(v[i]) for f,v in family.items()},
                'human_mean_importance':float(ev.clip_labels[k][i]),'human_percentile':float(human[i]),
                'annotator_percentiles':human_user[:,i].tolist(),'annotator_percentile_std':float(human_user[:,i].std()),
                'annotator_clip_importance':human_user_raw[:,i].tolist(),'annotator_importance_std':float(human_user_raw[:,i].std()),
                'annotator_fraction_above_own_median':float(positive[:,i].mean()),
                'stable_wrong_proxy_high':bool(high),'stable_wrong_human_high':bool(low),
                'gap':float(mean[i]-human[i]),'frames':frame_refs,'sampled_pts':by[k]['sampled_pts'][i]})
    cases=[]
    for sign,label in [(1,'proxy_high_human_low'),(-1,'human_high_proxy_low')]:
        used=set()
        for row in sorted(candidates,key=lambda r:(-sign*r['gap'],r['video_id'],r['clip_index'])):
            if sign*row['gap']<=0 or row['video_id'] in used:continue
            cases.append(dict(row,direction=label,frame_content_reviewed=False));used.add(row['video_id'])
            if len(used)==3:break
    counts={'stable_proxy_high_human_low_clips':sum(r['stable_wrong_proxy_high'] for r in candidates),
            'stable_human_high_proxy_low_clips':sum(r['stable_wrong_human_high'] for r in candidates),
            'stable_conflict_video_ids':sorted({r['video_id'] for r in candidates if r['stable_wrong_proxy_high'] or r['stable_wrong_human_high']})}
    write(out/'cases.json',{'selection':'Pre-fixed top3 signed gaps, distinct source per direction','cases':cases,'all_conflict_counts':counts})
    write(out/'results.json',{'status':'COMPLETED','evaluation_status':'DEVELOPMENT_ONLY','variants':details,
        'pairwise_dev_teacher_rho':matrix,'train_self_exclusion_factorial':loo,'conflict_counts':counts,
        'human_train_annotations_read':False,'wall_seconds':time.monotonic()-start})


def midranks_percentile_safe(v):
    return midrank_percentile(v)


if __name__=='__main__':main()
