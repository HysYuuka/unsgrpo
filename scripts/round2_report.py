#!/usr/bin/env python3
"""Record all bounded diagnostics; no checkpoint/variant selection or training."""
import csv
import io
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from unsgrpo.round2_utils import *


def csv_file(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='')
    w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows(rows)
    with Path(path).open('x') as f:f.write(s.getvalue())


def f(value,percent=False):
    return 'undefined' if value is None else f'{value*(100 if percent else 1):.4f}'


def main():
    plan=read(ROOT/'protocol.json');teachers=read(ROOT/'teachers/results.json');cache=read(ROOT/'cached/results.json')
    students=read(ROOT/'students/evaluation_results.json');counter=read(ROOT/'input_counterfactual/results.json')
    cases=read(ROOT/'teachers/cases.json');out=PROJECT/'reports/round2_diagnostic_v1'
    teacher_rows=[];cached_rows=[];student_rows=[]
    for name,r in teachers['variants'].items():
        diag={v['video_id']:v for v in r['per_video']}
        for v in r['development_evaluation']['per_video']:
            row=diag[v['video_id']];teacher_rows.append({'teacher':name,'family':r['family'],'video_id':v['video_id'],
                'development_AP':v['mean_ap'],'unrounded_stable_AP':v['unrounded_stable_ap'],
                'train_partition_ARI':r['train_partition_ARI'],'q_spearman_vs_original':row['teacher_vs_T0_spearman'],
                'q_MAE_vs_original':row['mean_absolute_q_change'],'teacher_vs_human_clip_rho':row['teacher_vs_human_clip_spearman'],
                'reference_membership_jaccard':row['reference_membership_jaccard']})
    for name,r in cache['methods'].items():
        diag={v['video_id']:v for v in r['per_video']}
        for v in r['evaluation']['per_video']:
            rr=diag[v['video_id']];cached_rows.append({'method':name,'video_id':v['video_id'],'development_AP':v['mean_ap'],
                'unrounded_stable_AP':v['unrounded_stable_ap'],'q_MSE':rr['mse'],'q_pair_agreement':rr['weighted_pair_agreement'],
                'score_std':rr['std'],'score_range':rr['range'],'pair_tie_fraction':rr['pair_tie_fraction'],
                'overlap_range':rr.get('overlap',{}).get('mean_range')})
    for name,r in students['runs'].items():
        for step,x in r['steps'].items():
            dd={v['video_id']:v for v in x['per_video_diagnostics']}
            for v in x['evaluation']['per_video']:
                rr=dd[v['video_id']];student_rows.append({'run':name,'view':r['view'],'objective':r['objective'],
                    'seed':r['seed'],'step':int(step),'video_id':v['video_id'],'development_AP':v['mean_ap'],
                    'q_MSE':rr['mse'],'q_pair_agreement':rr['weighted_pair_agreement'],'window_pair_agreement':rr['window_pair_agreement'],
                    'overlap_mean_range':rr['overlap']['mean_range'],'scale_normalized_overlap_range':rr['overlap']['scale_normalized_mean_range'],
                    'overlap_pair_order_disagreement':rr['overlap']['overlap_pair_order_disagreement']})
    csv_file(out/'teachers_per_video.csv',teacher_rows);csv_file(out/'cached_per_video.csv',cached_rows)
    csv_file(out/'students_all_fixed_steps_per_video.csv',student_rows)
    csv_file(out/'students_trajectories.csv',students['all_fixed_step_trajectories'])
    # Display all six preselected cases, with the exact cached frames.
    image=Image.new('RGB',(1280,220*len(cases['cases'])),'white');draw=ImageDraw.Draw(image)
    for row,c in enumerate(cases['cases']):
        draw.text((5,row*220+3),f"{c['video_id']} {c['bounds'][0]:g}-{c['bounds'][1]:g}s {c['direction']} teacher={c['teacher_mean_percentile']:.3f}+-{c['teacher_std']:.3f} human={c['human_percentile']:.3f}",fill='black')
        for column,ref in enumerate(c['frames']):
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Selected input image changed')
            tile=ImageOps.contain(Image.open(ref['path']).convert('RGB'),(320,180))
            image.paste(tile,(column*320+(320-tile.width)//2,row*220+25+(180-tile.height)//2))
            draw.text((column*320+5,row*220+205),f"PTS {c['sampled_pts'][column]:.3f}s",fill='black')
    image.save(ROOT/'teachers/selected_cases.jpg',quality=92)
    evidence={p:reference(ROOT/p) for p in ['round2.freeze.json','protocol.json','teachers/results.json','teachers/cases.json',
        'cached/results.json','cached/input_counterfactual_gate.json','students/training_complete.json',
        'students/evaluation_results.json','input_counterfactual/results.json']}
    summary={'status':'COMPLETED_NUMERICAL_DIAGNOSTICS_VISUAL_INTERPRETATION_SEPARATE','evaluation_status':'DEVELOPMENT_ONLY',
        'n_sources':10,'no_human_annotations_for_fitting':True,'primary_metric':'V0 stable mapped and canonical round12, same for every old/new method',
        'teacher_variants':{n:{'family':r['family'],'ARI':r['train_partition_ARI'],
            'dev_rank_stability':r['dev_rank_stability'],'development_AP':r['development_evaluation']['video_macro_ap']}
            for n,r in teachers['variants'].items()},'stable_teacher_conflicts':teachers['conflict_counts'],
        'cached_methods':{n:{'development_AP':r['evaluation']['video_macro_ap'],
            'macro_q_MSE':float(np.mean([v['mse'] for v in r['per_video']]))} for n,r in cache['methods'].items()},
        'window_rewards':cache['window_weighted_rewards'],'cross_window':cache['cross_window'],
        'all_affine_canonical_AP_invariant':cache['max_affine_canonical_user_AP_difference']==0,
        'student_seed_groups':students['seed_averaged_groups'],'student_paired_source_differences':students['paired_source_bootstrap_seed_averaged'],
        'student_trajectory_changes':students['trajectory_changes'],
        'counterfactual':counter,'evidence':evidence,'human_frame_case_interpretation':'See separately signed visual review; not generated from scores',
        'scope':'Small CLIP students do not establish same-Qwen supervised equivalence to GRPO; all 10 old test sources are development.'}
    write(out/'summary.json',summary)
    lines=['# 第二轮低成本诊断：数值结果与证据索引','',
        '所有人工评价均为已有10个来源上的开发集诊断，不是独立测试。训练、聚类、原型和窗口偏移拟合未使用人工标签；小模型仅用原R0连续伪标签，固定最终端点。','',
        '|教师|构造|训练分组ARI|开发片段排序ρ（对原R0）|开发V0 AP%|','|---|---|---:|---:|---:|']
    for n,r in teachers['variants'].items():
        lines.append(f"|{n}|{r['family']}|{f(r['train_partition_ARI'])}|{f(r['dev_rank_stability']['mean'])}|{f(r['development_evaluation']['video_macro_ap'],True)}|")
    lines += ['', '固定降维的KM43–46仅改变聚类随机种子；FULL43/44重新拟合UMAP、重做K选择和KMeans并重新transform，不能把其变化归因于KMeans。原型等权对照保持原分组；GLOBAL_VIDEO_EQUAL还去掉分组。聚类ARI和伪标签排序稳定性是不同量。','',
        f"按预定阈值，稳定的代理高/人工低片段数={teachers['conflict_counts']['stable_proxy_high_human_low_clips']}，稳定的人工高/代理低片段数={teachers['conflict_counts']['stable_human_high_proxy_low_clips']}。病例按预定双向gap选择，每方向3例且不同视频；全部保留。训练自排除的2×2权重对照见teachers/results.json，未读取训练人工标签，也不等于完整交叉拟合。",'',
        '|已有输出/后处理|开发V0 AP%|视频宏平均伪标签MSE|','|---|---:|---:|']
    for n,r in cache['methods'].items():
        lines.append(f"|{n}|{f(r['evaluation']['video_macro_ap'],True)}|{f(np.mean([x['mse'] for x in r['per_video']]))}|")
    lines += ['',f"固定正比例缩放+平移在标准并列处理后的最大用户AP差={cache['max_affine_canonical_user_AP_difference']}。分数变窄本身不能解释AP改善；量化后的对照会改变并列，另行记录。offset_consensus只拟合重叠预测间的整体偏移，不读取q或人工标签，不裁剪。",'',
        '|端点|真实窗口校准奖励（合法）|恒定50校准奖励|固定收缩后的校准奖励|','|---|---:|---:|---:|']
    for n,r in cache['window_weighted_rewards'].items():
        lines.append(f"|{n}|{f(r['calibration_among_legal'])}|{f(r['constant50_calibration_all_windows'])}|{f(r['shrunk_calibration_among_legal'])}|")
    lines += ['', '一致性同时记录原分数跨度、除以分数标准差后的跨度、重叠片段对的排序分歧；整体缩放会自动缩小第一项，却不改善后两项。位置与邻近内容的自然窗口变化相互混杂，固定效应回归仅为描述性关联，不单独证明因果。','',
        '|小模型（3个初始化）|开发AP均值%|种子标准差（百分点）|种子范围%|','|---|---:|---:|---|']
    for n,r in students['seed_averaged_groups'].items():
        lines.append(f"|{n}|{f(r['video_macro_ap'],True)}|{f(r['seed_std_ddof1'],True)}|[{f(r['seed_min'],True)}, {f(r['seed_max'],True)}]|")
    lines += ['', '|预定监督对照|AP差（百分点）|95%来源组bootstrap区间|','|---|---:|---|']
    for n,r in students['paired_source_bootstrap_seed_averaged'].items():
        lines.append(f"|{n}|{f(r['mean'],True)}|[{f(r['ci_low'],True)}, {f(r['ci_high'],True)}]|")
    lines += ['', '所有12个小模型均256更新、4096窗口曝光，32897个可训练参数，使用同一训练曝光序列及同seed初始参数。固定随机投影后所有512个模型输入维度在两种信息视图均活跃；全局视图额外包含视频CLIP均值/标准差。固定步0/64/128/256的全部曲线均披露，未按人工AP选端点。种子先在同一来源内平均再bootstrap，不能把3×10写成30个独立来源。','',
        f"真实冻结S0输入对照状态：{counter['status']}。仅B（重复第二概览）与C（第二概览含互补全局帧）为相同文字/占位符/视觉token预算的内容对照；A是原单概览复现。局部32帧、权重、解码和聚合不变。该部分覆盖4来源、每来源2个窗口，不能冒充完整视频AP对照。具体视觉token、帧数、耗时和缓存重放情况见input_counterfactual/results.json与predictions.jsonl。",'',
        '本数值报告不自动把分数冲突翻译成画面原因。实际输入图及标注分歧见工作目录teachers/selected_cases.jpg和cases.json；视觉复核和最终主要建议另附。若后续依据本开发集改方法，需要新的独立最终来源。', '',
        '完整复现入口：`scripts/run_round2_diagnostics.py --prepare`、`--execute`；本次只使用现有缓存和离线资产，不自动同步Git。代码和输入SHA在round2.freeze.json，阶段命令、日志、实际时长和失败状态在工作目录state.json及events.jsonl。', '']
    with (out/'RESULTS.md').open('x') as handle:handle.write('\n'.join(lines))
    print('All bounded numeric results exported; visual interpretation remains explicit',flush=True)


if __name__=='__main__':main()
