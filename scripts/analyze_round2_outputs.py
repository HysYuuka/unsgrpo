#!/usr/bin/env python3
"""Post-run descriptive contrasts and scientific figures, without re-fitting."""
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from unsgrpo.round2_utils import *
from unsgrpo.round2_metrics import paired


def main():
    started=time.monotonic();plan,by,train,dev,feat,pseudo=load_assets();out=ROOT/'postrun_analysis';out.mkdir()
    t=read(ROOT/'teachers/results.json');c=read(ROOT/'cached/results.json');s=read(ROOT/'students/evaluation_results.json')
    cases=read(ROOT/'teachers/cases.json');gpu=read(ROOT/'input_counterfactual/results.json')
    post={};pooled={}
    for name,rows in c['cross_window'].items():
        w=np.array([r['n_repeated_clips'] for r in rows]);pooled[name]=float(np.average([r['mean_range'] for r in rows],weights=w))
        original=read(Path(plan['paths']['original_run'])/'inference'/name.replace('+','PLUS')/'receipt.json')
        if abs(pooled[name]-original['cross_window_mean_range'])>1e-12:raise ValueError('Original pooled overlap range not reproduced')
    with np.load(ROOT/'cached/scores.npz',allow_pickle=False) as predictions:
        changes=[]
        for k in dev:
            a,b=predictions[f'S0__{k}'],predictions[f'G-CAL__{k}'];i,j=np.triu_indices(len(a),1)
            changes.append({'video_id':k,'score_rank_correlation_GCAL_S0':float(spearmanr(a,b).statistic),
                'pair_order_difference_ties_half':float(np.mean(np.abs(np.sign(a[i]-a[j])-np.sign(b[i]-b[j]))/2)),
                'S0_std':float(a.std()),'GCAL_std':float(b.std())})
    for name,r in c['methods'].items():
        if '__' in name:
            base=name.split('__')[0];post[name]=paired(r['evaluation'],c['methods'][base]['evaluation'])
    post['B0_minus_CONSTANT50']=paired(c['methods']['B0']['evaluation'],c['methods']['CONSTANT50']['evaluation'])
    post['GCAL_minus_S0']=paired(c['methods']['G-CAL']['evaluation'],c['methods']['S0']['evaluation'])
    source_conflicts=[]
    with np.load(ROOT/'teachers/q_scores.npz',allow_pickle=False) as qs:
        # Already frozen case rules; per-source rates use video units, not 1408 trials.
        from unsgrpo.round2_metrics import DevelopmentEvaluator
        from unsgrpo.core import midrank_percentile
        ev=DevelopmentEvaluator(plan,by,dev)
        for k in dev:
            stack=np.stack([qs[f'{n}__{k}'] for n in t['variants']]);std=stack.std(0);human=midrank_percentile(ev.clip_labels[k])
            families={f:np.mean([qs[f'{n}__{k}'] for n,r in t['variants'].items() if r['family']==f],0)
                      for f in {r['family'] for r in t['variants'].values()}}
            high=(human<=.25)&(std<=.1)&np.logical_and.reduce([v>=.75 for v in families.values()])
            low=(human>=.75)&(std<=.1)&np.logical_and.reduce([v<=.25 for v in families.values()])
            source_conflicts.append({'video_id':k,'n_clips':len(human),'stable_proxy_high_human_low':int(high.sum()),
                'stable_human_high_proxy_low':int(low.sum()),'fraction':float(np.mean(high|low))})
    curves={}
    for name,group in s['seed_averaged_groups'].items():
        runs=[r for r in s['runs'].values() if r['view']+'__'+r['objective']==name]
        curves[name]={str(step):{'train_MSE':float(np.mean([r['steps'][str(step)]['window_weighted_train_q_MSE'] for r in runs])),
            'dev_window_MSE':float(np.mean([r['steps'][str(step)]['window_weighted_dev_q_MSE'] for r in runs])),
            'dev_AP':float(np.mean([r['steps'][str(step)]['evaluation']['video_macro_ap'] for r in runs])),
            'global_q_pair_agreement':float(np.mean([v['weighted_pair_agreement'] for r in runs for v in r['steps'][str(step)]['per_video_diagnostics']]))}
            for step in [0,64,128,256]}
    desc={'status':'COMPLETED_POST_RUN_DESCRIPTIVE_ANALYSIS','evaluation_status':'DEVELOPMENT_ONLY',
        'scientific_variants_added':0,'old_pooled_overlap_range_reproduced':pooled,
        'per_video_macro_differs_from_old_pooled_weighting':True,'GCAL_vs_S0_order_changes':changes,
        'postprocessing_and_teacher_contrasts':post,'stable_conflicts_per_source':source_conflicts,
        'total_development_clips':sum(r['n_clips'] for r in source_conflicts),'seed_mean_fixed_trajectories':curves,
        'scope':'All fixed outputs retained; these are descriptive source-level summaries, not new selected endpoints.'}
    write(out/'descriptive_results.json',desc)
    figures=PROJECT/'reports/round2_diagnostic_v1/figures';figures.mkdir()
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8.2));a,b,z,d=axes.ravel()
    colors={'original':'black','fixed_embedding_KMeans':'#4477aa','full_pipeline':'#cc6677','prototype':'#228833'}
    for n,r in t['variants'].items():
        a.scatter(r['train_partition_ARI'],r['dev_rank_stability']['mean'],color=colors[r['family']],s=35)
        if n not in {'KM45','KM46'}:
            a.annotate('KM44/45/46' if n=='KM44' else n,(r['train_partition_ARI'],r['dev_rank_stability']['mean']),xytext=(4,4 if n!='T0' else -12),textcoords='offset points',fontsize=7)
    a.set(xlabel='Train clustering ARI vs original',ylabel='Development score-rank rho vs original',title='A. Cluster changes need not change teacher ranks',ylim=(.79,1.025),xlim=(-.05,1.07))
    for method,color in zip(['S0','S+','G-CAL','G-RANK'],['#4477aa','#228833','#aa3377','#ccbb44']):
        base=c['methods'][method];shrink=c['methods'][method+'__affine']
        x=[np.mean([v['mse'] for v in r['per_video']]) for r in [base,shrink]];y=[100*r['evaluation']['video_macro_ap'] for r in [base,shrink]]
        b.plot(x,y,'o-',color=color,label=method);b.annotate(method,(x[0],y[0]),xytext=(2,4),textcoords='offset points',fontsize=7)
    r=c['methods']['CONSTANT50'];b.scatter(np.mean([v['mse'] for v in r['per_video']]),100*r['evaluation']['video_macro_ap'],marker='*',s=90,color='black',label='constant50')
    b.set(xlabel='Video-macro q MSE (lower is better fit)',ylabel='Development V0 AP (%)',title='B. Fixed affine shrink improves MSE, not AP');b.legend(fontsize=7,ncol=2,loc='lower right')
    for name,r in curves.items():
        short=name.replace('window_plus_video_statistics','global').replace('window_only','local').replace('__',' / ')
        points=[r[str(step)] for step in [0,64,128,256]]
        line=z.plot([p['train_MSE'] for p in points],[100*p['dev_AP'] for p in points],'o-',label=short)[0]
        z.plot(points[-1]['train_MSE'],100*points[-1]['dev_AP'],'s',color=line.get_color(),markersize=7)
    z.set(xlabel='Training q MSE; path follows 0,64,128,256 steps',ylabel='Development V0 AP (%)',title='C. All pre-fixed trajectories (3-seed means)');z.legend(fontsize=7)
    for video in gpu['gate']['chosen_video_ids']:
        yy=[next(r['mean_range'] for r in gpu['conditions'][cond]['per_video'] if r['video_id']==video)*100 for cond in 'ABC']
        d.plot(range(3),yy,'o-',alpha=.65,label=video)
    d.plot(range(3),[gpu['conditions'][cond]['mean_range']*100 for cond in 'ABC'],'ko-',linewidth=2.3,label='4-source mean')
    d.set(xticks=range(3),xticklabels=['A original','B duplicated','C complementary'],ylabel='Shared-clip range (score points)',title='D. More global content is not a consistency guarantee');d.legend(fontsize=6.7)
    fig.tight_layout();fig.savefig(figures/'diagnostic_summary.png',dpi=180);fig.savefig(figures/'diagnostic_summary.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.3));methods=['S0','S+','G-CAL','G-RANK']
    for ax,key,title,scale in zip(axes,['mean_range','scale_normalized_mean_range','overlap_pair_order_disagreement'],
        ['Raw shared-clip range','Range / within-video score std','Pair-order disagreement'],[100,1,100]):
        values=[np.mean([r[key] for r in c['cross_window'][m]])*scale for m in methods]
        ax.bar(methods,values,color=['#4477aa','#228833','#aa3377','#ccbb44']);ax.set_title(title);ax.set_ylim(0,max(values)*1.23)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
    fig.tight_layout();fig.savefig(figures/'consistency_scales.png',dpi=180);fig.savefig(figures/'consistency_scales.pdf');plt.close(fig)
    # Honest post-run reconstruction: the original state/logs remain authoritative.
    state=read(ROOT/'state.json');events=[{'event':'POST_RUN_RECONSTRUCTION_METADATA','created_utc':now(),
        'reconstructed_after_completion':True,'source_state':reference(ROOT/'state.json'),'not_a_contemporaneous_journal':True}]
    for name,r in state['phases'].items():
        events.extend([{'event':'phase_started','phase':name,'recorded_utc':r['started_utc'],'reconstructed':True},
                       {'event':'phase_completed','phase':name,'recorded_utc':r['ended_utc'],'wall_seconds':r['wall_seconds'],'reconstructed':True}])
    (ROOT/'events.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in events))
    write(out/'analysis_receipt.json',{'status':'COMPLETED','script':reference(__file__),'wall_seconds':time.monotonic()-started,
        'new_model_fits':0,'new_inference_calls':0,'plots':[reference(p) for p in figures.iterdir()]})


if __name__=='__main__':main()
