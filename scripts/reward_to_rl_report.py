#!/usr/bin/env python3
"""Report actual screening and endpoint evidence, including every stopped branch."""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from unsgrpo.reward_to_rl_common import *
from unsgrpo.evaluation import paired_bootstrap


def maybe(path):return read(path) if Path(path).exists() else None


def csvwrite(path,rows,fields):
    with path.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def main():
    state=read(WORK/'state.json');protocol=read(REPORT/'protocol.json')
    screen=maybe(REPORT/'screen_result.json');candidate=maybe(STORE/'screen/candidates_receipt.json')
    teacher=maybe(STORE/'screen/teacher_receipt.json');interface=maybe(STORE/'screen/interface_receipt.json')
    results=[];pervideo=[];perseed=[];contrasts=[];endpoint={}
    historic=read(PROJECT/'reports/round3_rl_method_v1/historical_baselines.json')
    for name in ('S0','G-CAL'):
        e=historic[name]['evaluation'];results.append({'method':name,'seed':42,'step':256,'origin':'HISTORICAL','status':'CACHED_REEVALUATED',
            'V0_AP_percent':100*e['video_macro_ap']})
        pervideo += [dict(method=name,seed=42,video_id=v['video_id'],origin='HISTORICAL',status='OBSERVED',V0_AP_percent=100*v['mean_ap']) for v in e['per_video']]
    for seed in (42,43):
        item=maybe(REPORT/f'endpoints_seed{seed}.json')
        if item:endpoint[seed]=item
        for arm in ('G_plain','G_guard','RSFT'):
            actual=item['methods'][arm] if item else None;receipt=maybe(STORE/f'training/seed{seed}_{arm}/receipt.json')
            status='EVALUATED_FINAL80' if actual else (receipt['status'] if receipt else 'NOT_RUN_'+state['status'])
            row={'method':arm,'seed':seed,'step':80,'origin':'NEW','status':status,
                'V0_AP_percent':100*actual['video_macro_ap'] if actual else None}
            results.append(row);perseed.append(dict(row,prompt_exposures=receipt['prompt_exposures'] if receipt else 0,
                generated_policy_completions=receipt['policy_completions_generated'] if receipt else 0,
                actual_optimizer_steps=receipt['completed_optimizer_steps'] if receipt else 0))
            if actual:pervideo += [dict(method=arm,seed=seed,video_id=v['video_id'],origin='NEW',status=status,V0_AP_percent=100*v['mean_ap']) for v in actual['per_video']]
        if item:
            methods=item['methods'];refs={**methods,'S0':historic['S0']['evaluation'],'G-CAL':historic['G-CAL']['evaluation']}
            for a,b in [('G_plain','S0'),('G_guard','S0'),('RSFT','S0'),('G_plain','RSFT'),('G_guard','RSFT'),('G_guard','G_plain')]:
                aa={r['video_id']:r['mean_ap'] for r in refs[a]['per_video']};bb={r['video_id']:r['mean_ap'] for r in refs[b]['per_video']};ids=sorted(aa)
                ci=paired_bootstrap([aa[k]-bb[k] for k in ids],source_groups=ids,n_boot=10000,seed=42)
                contrasts.append({'seed':seed,'contrast':a+'-'+b,**ci,'positive_sources':sum(aa[k]>bb[k] for k in ids)})
    two_seed_signal={}
    if set(endpoint)=={42,43}:
        for arm in ('G_plain','G_guard'):
            pairs={}
            for comparator in ('S0','RSFT'):
                diffs=[]
                for seed in (42,43):
                    first=endpoint[seed]['methods'][arm];second=historic['S0']['evaluation'] if comparator=='S0' else endpoint[seed]['methods']['RSFT']
                    diffs.append(first['video_macro_ap']-second['video_macro_ap'])
                pairs[comparator]={'seed_differences':diffs,'mean_difference':float(np.mean(diffs)),'std_ddof1':float(np.std(diffs,ddof=1)),
                    'both_positive_and_mean_at_least_1pp':min(diffs)>0 and np.mean(diffs)>=.01}
            two_seed_signal[arm]={'comparisons':pairs,'passed':all(x['both_positive_and_mean_at_least_1pp'] for x in pairs.values())}
    csvwrite(REPORT/'results.csv',results,['method','seed','step','origin','status','V0_AP_percent'])
    csvwrite(REPORT/'per_video.csv',pervideo,['method','seed','video_id','origin','status','V0_AP_percent'])
    csvwrite(REPORT/'per_seed.csv',perseed,['method','seed','step','origin','status','V0_AP_percent','prompt_exposures','generated_policy_completions','actual_optimizer_steps'])
    write(REPORT/'endpoint_contrasts.json',{'status':'PAIRED_SOURCE_CONTRASTS' if contrasts else 'NO_COMPLETE_NEW_ENDPOINTS',
        'contrasts':contrasts,'two_seed_fixed_branch_signal':two_seed_signal,'independent_unit':'source/video, not windows or repeated seeds'})
    res={'status':'ACTUAL_RESOURCE_ACCOUNTING','time':now(),'conservative_GPU_occupied_seconds':gpu_used(),'GPU_budget_seconds':21600,
        'screen_GPU_seconds':sum(v['seconds'] for v in records(WORK/'gpu_sessions.jsonl') if v['event']=='end' and (v['phase'].startswith('screen') or v['phase'].split('_process')[0] in ('interface','dev_candidates','dev_teacher'))),
        'sessions':reference(WORK/'gpu_sessions.jsonl') if (WORK/'gpu_sessions.jsonl').exists() else None,
        'screen_candidates':candidate,'screen_teacher':teacher,'interface':interface,
        'new_policy_training_runs':sum((STORE/f'training/seed{seed}_{a}/receipt.json').exists() for seed in (42,43) for a in ('G_plain','G_guard','RSFT')),
        'technical4':maybe(STORE/'training/technical4/receipt.json'),'initial_time':read(WORK/'initial_state.json'),
        'storage_before_archiving':{'temporary_bytes':logical(WORK),'persistent_bytes':logical(STORE),'light_report_bytes':logical(REPORT)},
        'currency_cost':None,'currency_note':'No billing/rental-rate evidence; budget is not consumption','Git_sync':False,'subagents_used':False}
    write(REPORT/'resource_receipt.json',res)
    if state['status']=='CANDIDATE_POOL_UNINFORMATIVE':recommend='当前G=4采样与单窗变更没有达到预定信息量门槛，结束本次检验；不能据此归罪教师或判定整个模型没有潜力。'
    elif state['status']=='REWARD_NOT_ALIGNED':recommend='候选池存在更好答案，但两种奖励均未证明能选出来；结束当前奖励配方，不追加训练或第三个教师prompt。'
    elif state['status']=='POLICY_OUTPUT_NOT_READY':recommend='四候选全合法组覆盖不足，当前策略输出尚不适合内容辨别力检验；本轮不将格式修复当语义收益。'
    elif state['status'].startswith('BUDGET_LIMITED'):recommend='受实际预算约束，未完成规定的完整比较；不把未完成解释为科学失败。'
    elif state['status']=='COMPLETED_PRESPECIFIED_EXPERIMENT':
        if any(v['passed'] for v in two_seed_signal.values()):recommend='某个固定GRPO分支在两次续训中同时超过S0与RSFT且平均至少1个百分点，有内部可重复RL信号；不等同于独立测试或SOTA。'
        else:recommend='奖励辨别力未转化为达到预定重复门槛的RL证据，或RSFT同样好；结束当前配方，不追加长训。'
    else:recommend='本轮被接口、运行状态或训练信号覆盖阻断；保留具体错误与部分产物，不填虚构端点。'
    lines=['# 奖励候选辨别力与短程RL：'+state['status'],'',recommend,'',
        '香港时间：'+now()['hong_kong']+'。全部10个来源均为既有开发数据。主指标为第二轮稳定V0、12位小数并列规范；公开top-5仍NOT_READY，不作论文benchmark比较。','',
        '## 实际执行与边界','',f"状态：{state['status']}。累计GPU保守占用{gpu_used():.2f}秒，上限21600秒；首次筛查上限3600秒。所有GPU加载、失败和进程首尾均记账。CPU/模型训练没有读取训练人工标签。旧的更新策略rollout未被用作新S0候选。",'']
    if state.get('error'):lines+=['具体阻断：`'+state['error']+'`。详细堆栈与原始阶段日志保留。','']
    if candidate:lines += [f"开发候选：{candidate['n_answers']}个新增S0回答，{candidate['n_windows']}个固定窗口；四个回答均合法的组为{candidate['all_four_legal_groups']}/{candidate['n_windows']}。greedy缓存只作单独部署参照，没有塞入采样组。",'']
    if teacher:lines += [f"偏好教师：{teacher['n_edges']}条双顺序边，保留{teacher['n_kept']}条，实际前向{teacher['forward_calls']}次；弃权原因完整记录为{teacher['abstention_counts']}。教师只看到概览与两段各四帧，从未看到策略回答。",'']
    if interface:lines += [f"真实接口检查：24对（含8个同片段对），{interface['forward_calls']}次计分前向；实际模板同一位置单token检查结果为{interface['single_token_in_all_contexts']}。归一化A/B/C倾向不是校准正确率；只有同位置单token时才解释原始三项概率质量。",'']
    if screen:
        primary=screen['primary_all_four_legal'];secondary=screen['secondary_all_groups']
        lines += ['## 候选是否有空间，奖励是否选得更好','',
            f"主分析：{screen['n_all_four_legal']}个四回答全合法组、{primary['n_sources']}个来源。达到H极差0.1个百分点的组为{primary['n_informative_groups']}，覆盖{primary['sources_with_informative_groups']}来源。平均人工最优空间O={primary['O']*100:.5f}个百分点，人工最优相对greedy={primary['oracle_minus_greedy']*100:.5f}个百分点。信息量准入：{screen['information_gate_passed']}。",'',
            '|奖励|辨别准确率|选择增益G（百分点）|G/O|选择−greedy（百分点）|基础准入|','|---|---:|---:|---:|---:|---|']
        for name in ('R_old','R_pair'):
            d=primary[name];acc='不可测' if d['accuracy'] is None else f"{d['accuracy']*100:.2f}%";ratio='不可测' if d['G_over_O'] is None else f"{d['G_over_O']*100:.2f}%"
            lines += [f"|{name}|{acc}|{d['G']*100:.5f}|{ratio}|{d['selected_minus_greedy']*100:.5f}|{d['base_admission']}|"]
        lines += ['',f"固定选择结果：{screen['selected_reward']}。这是开发决策，不能写成独立验证。每个H都只替换一个窗口，并保留同视频其余S0窗口、完整重叠聚合和完整视频V0。窗口收益不可相加。奖励并列按12位浮点规范取H均值，未用人工最优破并列。",'',
            '全部组（含非法零分窗口）的次分析、逐窗口四个H、逐用户AP和来源bootstrap区间见screen_result.json、screen_per_window.csv及文件存储screen/single_window_full_video_AP.jsonl。主分析与次分析均保留，没有删除不利视频。辨别准确率只用H差至少0.1个百分点的同一批候选对，奖励同分计半。每个窗口先平均，再视频平均、来源平均；区间只重采样来源。','',
            '候选人工最优只用于诊断，没有成为训练样本、教师修正或RSFT目标。RSFT只在训练视频的另一个冻结S0候选池中按选定奖励挑选。','']
    lines += ['## 固定80步端点','', '|方法|seed|实际状态|开发V0 AP|','|---|---:|---|---:|']
    for r in results:lines.append(f"|{r['method']}|{r['seed']}|{r['status']}|{'未评价' if r['V0_AP_percent'] is None else format(r['V0_AP_percent'],'.4f')+'%'}|")
    lines += ['', '历史G-CAL训练预算不同，仅为实用参照。三个分支若执行则共享最终池、320提示曝光、共同S0、新LoRA及固定端点；在线探索和计算量不同，不能写成严格等算力。G_guard在R_old路线u=1，只检验标准差下限。技术adapter与正式adapter分开；是否通过4步实际接入见实际receipt，而非CPU测试推断。','',
        '逐视频/逐seed结果和所有已完成的对照区间保留在CSV、endpoint_contrasts.json。未运行分支为空值及原因，未用零分冒充。只有seed42达到固定增益及至少6/10正向来源门槛且剩余预算足够，才重复seed43全部三分支。共同S0只有一个，重复仅衡量续训随机性。','',
        '## 直接观测、解释与未验证事项','',
        '直接观测限于真实生成的候选、条件概率、单窗替换AP和已完成的训练端点。奖励选择增益表示当前候选池中的辨别力，不保证GRPO一定学得到；没有信息量时不能归罪教师。','',
        '未完成技术接入或端点时，不能声称已检验可靠更新；RSFT同样好时，不能把监督信息收益归为RL特有收益。位置交换一致只表示稳定，不证明教师方向正确。没有新数据、全量伪标签、全视频比较图、JSON事件教师或超参数扫描。','',
        '## 复现与文件','',
        '冻结协议在protocol.json；实际源和输入SHA在工作目录screen_protocol.freeze.json，训练若执行另有training.freeze.json。所有候选/奖励先冻结，再由独立CPU评价进程读取10来源标签。最终artifact_index和资源收据记录实际路径与字节；大型预测与adapter留在工作/文件存储，不进入仓库。','',
        '原始完整执行命令：','', '```bash',
        'export PYTHONPATH=/root/autodl-tmp/unsgrpo/src:/root/autodl-tmp/unsgrpo/scripts',
        '/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python -m pytest /root/autodl-tmp/unsgrpo/tests/test_reward_to_rl.py -q',
        '/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python /root/autodl-tmp/unsgrpo/scripts/reward_to_rl_prepare.py',
        'screen -dmS unsgrpo-reward-to-rl /bin/bash /root/autodl-tmp/unsgrpo/scripts/reward_to_rl_detached.sh',
        '```','',
        '上述入口拒绝覆盖已有结果；不能删除失败证据后重跑并称同一次实验。全部实际阶段命令、采样seed、提示哈希、token IDs、损失分量、u、参考logps、梯度、优化器调用、耗时和内存保留于阶段日志与JSONL。未进入训练时相应账本明确不存在。']
    (REPORT/'DECISION.md').write_text('\n'.join(lines)+'\n')
    write(REPORT/'execution_summary.json',{'status':state['status'],'time':now(),'recommendation':recommend,
        'selected_reward':screen['selected_reward'] if screen else None,'screen_result':reference(REPORT/'screen_result.json') if screen else None,
        'evaluated_seeds':sorted(endpoint),'two_seed_fixed_branch_signal':two_seed_signal,'new_training_runs':res['new_policy_training_runs'],
        'GPU_seconds':gpu_used(),'human_train_labels_read':False,'Git_sync':False})
    print(state['status']+': report written',flush=True)


if __name__=='__main__':main()
