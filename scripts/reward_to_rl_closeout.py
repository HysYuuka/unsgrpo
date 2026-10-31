#!/usr/bin/env python3
"""Archive the finite experiment, its one repair, and independent verification."""
import csv
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from unsgrpo.reward_to_rl_common import *


def files(path):return sorted(p for p in Path(path).rglob('*') if p.is_file() and '__pycache__' not in p.parts)


def copy_checked(source,dest):
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if sha(source)!=sha(dest):raise ValueError(f'Copy mismatch: {source}')


def main():
    state=read(WORK/'state.json')
    if state['status']=='RUNNING':raise ValueError('Do not close active work')
    if (REPORT/'CLOSED.json').exists():raise FileExistsError('Already closed')
    verification=read(WORK/'independent_verification.json')
    if verification['status']!='PASSED':raise ValueError('Independent check incomplete')
    frozen=validate_freeze();repair=read(WORK/'software_fix1.json') if (WORK/'software_fix1.json').exists() else None
    screen=read(REPORT/'screen_result.json') if (REPORT/'screen_result.json').exists() else None
    summary=read(REPORT/'execution_summary.json')
    recommendation=summary['recommendation']
    if state['status']=='REWARD_NOT_ALIGNED':
        recommendation='候选确有改善空间，但新旧奖励均未达到预定辨别力准入，结束当前配方、不启动训练。旧奖励存在弱正向选择信号，不能把停止决定解释为奖励完全没有信息。'
        text=(REPORT/'DECISION.md').read_text()
        text=text.replace(summary['recommendation'],recommendation,1)
        (REPORT/'DECISION.md').write_text(text)
        summary['clarified_conclusion']=recommendation
        write(REPORT/'execution_summary.json',summary,fresh=False)
    lines=['# 复核后的最终结论','',recommendation,'']
    if screen:
        primary=screen['primary_all_four_legal'];secondary=screen['secondary_all_groups']
        lines += [f"预定主分析含{screen['n_all_four_legal']}/{screen['n_groups']}组；信息量门槛{'通过' if screen['information_gate_passed'] else '未通过'}。所有数值均为10个既有开发来源上的诊断。",'',
            f"候选潜在增益O为{100*primary['O']:.5f}个百分点，来源95%区间[{100*primary['O_CI'][0]:.5f}, {100*primary['O_CI'][1]:.5f}]；人工最优相对greedy为{100*primary['oracle_minus_greedy']:.5f}个百分点。单窗替换的这些收益不能相加。",'',
            '|奖励|准确率及来源95%区间|G及来源95%区间（百分点）|G/O|选择−greedy（百分点）|','|---|---|---|---:|---:|']
        export=[]
        for name in ('R_old','R_pair'):
            d=primary[name];acc='不可测' if d['accuracy'] is None else f"{100*d['accuracy']:.2f}% [{100*d['accuracy_CI'][0]:.2f}, {100*d['accuracy_CI'][1]:.2f}]"
            ratio='不可测' if d['G_over_O'] is None else f"{100*d['G_over_O']:.2f}%"
            lines.append(f"|{name}|{acc}|{100*d['G']:.5f} [{100*d['G_CI'][0]:.5f}, {100*d['G_CI'][1]:.5f}]|{ratio}|{100*d['selected_minus_greedy']:.5f}|")
            export.append({'reward':name,'G':d['G'],'G_low':d['G_CI'][0],'G_high':d['G_CI'][1],'accuracy':d['accuracy'],
                'accuracy_low':d['accuracy_CI'][0] if d['accuracy_CI'] else None,'accuracy_high':d['accuracy_CI'][1] if d['accuracy_CI'] else None,
                'O':primary['O'],'G_over_O':d['G_over_O'],'selected_minus_greedy':d['selected_minus_greedy'],'base_admission':d['base_admission']})
        lines += ['', 'R_old的G为正、G/O超过20%、选择均值略高于greedy，但准确率53.12%未达到60%，因此仍不能进入训练。R_pair除准确率不足外，G/O也低于20%，选择结果均值略低于greedy。这里是固定预算准入未通过，不是已经统计证明奖励无用，也不是已经检验了GRPO优化器。','']
        with (REPORT/'screen_reward_summary.csv').open('x') as f:
            wr=csv.DictWriter(f,fieldnames=list(export[0]));wr.writeheader();wr.writerows(export)
        per=[]
        for r in primary['per_video']:
            per.append({'video_id':r['video_id'],'legal_groups':r['n_groups'],'informative_groups':r['n_informative_groups'],
                'O':r['O'],'oracle_minus_greedy':r['oracle_minus_greedy'],
                **{name+'_'+key:r[name][key] for name in ('R_old','R_pair') for key in ('G','accuracy','selected_minus_greedy','qualifying_candidate_pairs')}})
        with (REPORT/'screen_per_video.csv').open('x') as f:
            wr=csv.DictWriter(f,fieldnames=list(per[0]));wr.writeheader();wr.writerows(per)
        lines += ['',f"全部组的次分析仍完整保留：R_old G={100*secondary['R_old']['G']:.5f}、R_pair G={100*secondary['R_pair']['G']:.5f}个百分点；与主分析使用不同合法性分母，不能混用。选择奖励为{screen['selected_reward']}，没有在两个奖励均失败时另选第三种。",'',
            '筛查的来源区间反映视频数量有限，不是扩大后的独立检验；门槛是本项目的预算规则。原始240个H、4800个用户AP、同一批候选对和按来源计算的G/O均已独立复核。没有读取训练人工标签，也没有把人工最优候选用于训练。','']
        coverage=read(WORK/'teacher_coverage_postrun.json')
        lines += [f"新教师的直接失败证据：720条边仅保留22条（3.06%），仅10/60个窗口有边、覆盖6/10来源；其余50个窗口的合法候选均同奖0.5。1440个视图中B为最高项1342次（93.19%），C为0次；644条边因为交换后仍偏同一选项、实际片段身份翻转而弃权。8个同片段检查也从未把C列为最高项。这显示本提示和模板下有很强的选项/位置偏好，不能凭字母输出合法推断重要性判断正确。",'',
            f"单token同位置的原始A/B/C概率质量均值为{coverage['raw_ABC_probability_mass']['mean']:.8f}，接口能够集中输出允许的字母；归一化倾向仍不是校准正确率。此结果把失败从第三轮的JSON格式层进一步定位到了偏好证据覆盖与候选辨别力，尚未分离位置先验、输入组织和语义能力各自的因果贡献。没有因此追加新提示或调用。",'']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(per));width=.25
        ax.bar(x-width,[100*r['O'] for r in per],width,label='Oracle opportunity O',color='#b0b0b0')
        ax.bar(x,[100*r['R_old_G'] for r in per],width,label='R_old selection gain G',color='#4c78a8')
        ax.bar(x+width,[100*r['R_pair_G'] for r in per],width,label='R_pair selection gain G',color='#f58518')
        ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,[r['video_id'] for r in per],rotation=40,ha='right')
        ax.set_ylabel('Full-video V0 AP change (percentage points)');ax.set_title('One-window replacement; developmental sources, gains are not additive')
        ax.legend(fontsize=8);fig.tight_layout();(REPORT/'figures').mkdir(exist_ok=True)
        fig.savefig(REPORT/'figures/source_selection_gains.png',dpi=160);fig.savefig(REPORT/'figures/source_selection_gains.pdf');plt.close(fig)
    if repair:
        lines += ['唯一软件修复已完整登记：整理输入会原地移走网格字段，初版在整理后读取它们，误报零帧。真实CPU复核显示输入始终为2×4帧；修复只提前保存核对信息，不改变图像、计分位置、问题、seed、门槛或奖励。原始失败源码、状态、日志、报告及GPU时间均保留。','',
            '失败那次只执行了第一个顺序的1次教师前向，就在循环内被拦住；没有执行交换顺序，也没有保存这次原始分数。修复登记文件最初把此数写为2，独立控制流复核已另给计数更正。它不进入科学缓存，完整失败进程成本仍计入总预算。修复后24对/48次接口检查另行完整执行。','']
    lines += [f"GPU保守占用合计{gpu_used():.2f}秒（含进程首尾和失败尝试）；6小时是上限。工程等待时间不伪称主动人时。当前实际状态为{state['status']}，已完成与未启动的分支在DECISION.md及CSV明确分列。",'',
        '没有新训练端点时，不宣称已验证GRPO接入；有端点时只按固定step80比较，不选中间状态。预算阻断与科学门槛未通过分别解释。此次作业到此结束，不自动调参、补候选、扩大数据或追加长训。','']
    (REPORT/'FINAL_REVIEW.md').write_text('\n'.join(lines)+'\n')
    write(REPORT/'candidate_provenance_index.json',{'status':'ACTUAL_FROZEN_INPUT_OUTPUT_BINDINGS',
        'protocol':reference(REPORT/'protocol.json'),'screen_freeze':reference(WORK/'screen_protocol.freeze.json'),
        'common_S0':reference(PROJECT/'reports/common_s0.json'),'original_teacher':reference(PROJECT/'manifests/Qwen3-VL-2B-Instruct.asset.json'),
        'developer_windows':reference(WORK/'dev_windows.jsonl'),
        'new_candidates':reference(STORE/'screen/candidates.jsonl') if (STORE/'screen/candidates.jsonl').exists() else None,
        'old_greedy_not_in_sample_groups':reference(ORIGINAL/'inference/S0/windows.jsonl'),
        'teacher_edges':reference(STORE/'screen/teacher_edges.jsonl') if (STORE/'screen/teacher_edges.jsonl').exists() else None,
        'reward_label_boundary':reference(STORE/'screen/candidates_and_rewards.freeze.json') if (STORE/'screen/candidates_and_rewards.freeze.json').exists() else None,
        'software_repair':reference(WORK/'software_fix1.json') if repair else None,
        'failed_forward_count_correction':reference(WORK/'software_fix1_forward_count_correction.json') if repair else None})
    src=list((PROJECT/'src/unsgrpo').glob('reward_to_rl*.py'))+list((PROJECT/'scripts').glob('reward_to_rl*'))+[PROJECT/'tests/test_reward_to_rl.py']
    versions=[]
    for p in sorted(src):
        if not p.is_file():continue
        dest=STORE/'source_snapshot'/p.relative_to(PROJECT);copy_checked(p,dest)
        role='REPAIRED_PRE_SCREEN' if repair and str(p) in repair['source_replacements'] else 'SCREEN_FROZEN' if str(p) in frozen['sources'] else 'POSTRUN_VERIFICATION_OR_CLOSEOUT'
        versions.append({'source':reference(p),'archived':reference(dest),'role':role})
    write(REPORT/'source_versions.json',{'status':'SHA256_VERSIONED','sources':versions,
        'runtime_versions':{k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','scikit-learn','ms-swift','transformers','peft')},
        'original_source_snapshot':str(WORK/'screen_source_snapshot'),'software_fix_snapshot':str(WORK/'software_fix1_source_snapshot')})
    for p in files(WORK):
        if p.name in ('closeout.log','storage_final.json','storage_final.log'):continue
        copy_checked(p,STORE/'working_evidence'/p.relative_to(WORK))
    storage_command=[str(PYTHON),str(PROJECT/'scripts/storage_audit.py'),'--output',str(WORK/'storage_final.json')]
    with (WORK/'logs/storage_final.log').open('x') as f:subprocess.run(storage_command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
    audit=read(WORK/'storage_final.json');assert audit['ok'];check_storage()
    resource=read(REPORT/'resource_receipt.json');resource['independent_verification']=reference(WORK/'independent_verification.json')
    resource['software_repairs_used']=int(repair is not None)
    resource['failed_teacher_forward_calls_not_in_cache']=1 if repair else 0
    if screen:
        resource['actual_teacher_forwards_including_failed']=48+1440+(1 if repair else 0)
        resource['teacher_coverage_postrun']=reference(WORK/'teacher_coverage_postrun.json')
    resource['teacher_count_note']='Successful recorded interface/development/training forwards plus the unrecorded failed first view; no failed score fabricated.'
    resource['storage_final']={'audit':reference(WORK/'storage_final.json'),'temporary_logical_bytes_before_control':logical(WORK),
        'persistent_logical_bytes_before_report_snapshot':logical(STORE),'report_logical_bytes_before_control':logical(REPORT),
        'local_free_bytes':audit['filesystems']['local']['free_bytes'],'file_free_bytes':audit['filesystems']['file']['free_bytes'],
        'persistent_limit_bytes':15*1024**3,'temporary_limit_bytes':5*1024**3,'all_limits_and_reserves_passed':True,
        'scope':'After large source/evidence archives, before small report and index control files.'}
    write(REPORT/'resource_receipt.json',resource,fresh=False)
    for p in files(REPORT):copy_checked(p,STORE/'report_snapshot'/p.relative_to(REPORT))
    copy_checked(WORK/'storage_final.json',STORE/'final_control/storage_final.json')
    indexed=[]
    for label,root in [('working',WORK),('persistent',STORE),('reports',REPORT)]:
        for p in files(root):
            if p.name in ('artifact_index.json','CLOSED.json','closeout.log'):continue
            indexed.append(dict(reference(p),location=label))
    index={'status':'SHA_VERIFIED_FINAL_ARTIFACTS','time':now(),'files':indexed,'n_files':len(indexed),
        'logical_bytes_including_verified_copies':sum(r['bytes'] for r in indexed),
        'input_references':frozen['inputs'],'self_and_final_control_records_excluded':True,
        'old_models_videos_or_human_annotation_arrays_copied':False}
    write(REPORT/'artifact_index.json',index);copy_checked(REPORT/'artifact_index.json',STORE/'artifact_index.json')
    for r in indexed:assert sha(r['path'])==r['sha256'] and Path(r['path']).stat().st_size==r['bytes']
    write(REPORT/'CLOSED.json',{'status':'COMPLETED_FINITE_JOB','decision':state['status'],'time':now(),
        'main_report':reference(REPORT/'DECISION.md'),'independent_review':reference(REPORT/'FINAL_REVIEW.md'),
        'artifact_index':reference(REPORT/'artifact_index.json'),'resource_receipt':reference(REPORT/'resource_receipt.json'),
        'independent_verification':reference(WORK/'independent_verification.json'),'GPU_conservative_seconds':gpu_used(),
        'software_repairs_used':int(repair is not None),'old_rounds_unchanged':True,'Git_sync':False,'subagents_used':False,
        'remaining_work':'No automatic continuation, additional sampling, search or long training.'})
    copy_checked(REPORT/'CLOSED.json',STORE/'final_control/CLOSED.json')
    print(__import__('json').dumps({'status':'CLOSED','decision':state['status'],'GPU_seconds':gpu_used(),
        'temporary_bytes':logical(WORK),'persistent_bytes':logical(STORE),'report_bytes':logical(REPORT),'indexed_files':len(indexed)}),flush=True)


if __name__=='__main__':main()
