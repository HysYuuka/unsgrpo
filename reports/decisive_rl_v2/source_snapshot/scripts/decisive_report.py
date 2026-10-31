#!/usr/bin/env python3
"""Compact complete decision package; unrun endpoints remain explicitly unrun."""
import csv
from collections import Counter
from unsgrpo.decisive_common import *


def maybe(path):return read(path) if Path(path).exists() else None


def csvwrite(path,rows,fields):
    with path.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def main():
    state=read(WORK/'state.json');status=state['status'];final=maybe(REPORT/'final_evaluation.json');quality=maybe(REPORT/'reward_quality.json')
    historic=read(PROJECT/'reports/round3_rl_method_v1/historical_baselines.json');methods=final['methods'] if final else {k:historic[k]['evaluation'] for k in ('S0','G-CAL')}
    technical=maybe(STORE/'teacher/technical_receipt.json');dev_teacher=maybe(STORE/'teacher/dev_receipt.json');train_teacher=maybe(STORE/'teacher/train_receipt.json')
    results=[];pervideo=[];costs=records(WORK/'GPU_ledger.jsonl') if (WORK/'GPU_ledger.jsonl').exists() else []
    for method,arm,step in [('S0',None,0),('G-CAL',None,0),('P-DPO-80','P-DPO',80),('P-DPO-long','P-DPO',320),('G-plain','G-plain',80),('G-guard','G-guard',80)]:
        receipt=maybe(STORE/'training'/str(arm)/f'receipt_{step}.json') if arm else None;metric=methods.get(method)
        training_seconds=sum(r['seconds'] for r in costs if r['event']=='end' and r['bucket']=='training' and (r['phase'].startswith(arm) or r['phase'].startswith('decisive_train_--arm_'+arm))) if arm else 0.
        if method=='P-DPO-80':training_seconds=sum(r['seconds'] for r in costs if r['event']=='end' and r['bucket']=='training' and r['phase'] in ('P-DPO','decisive_train_--arm_P-DPO_parent_overhead'))
        exposures=0;updates=0;completions=0
        if receipt:
            exposures=receipt['prompt_exposures'];updates=receipt['completed_optimizer_steps'];completions=receipt['online_completions']
            if step==320:
                early=read(STORE/'training/P-DPO/receipt_80.json');exposures+=early['prompt_exposures'];updates+=early['completed_optimizer_steps']
        row={'method':method,'status':'EVALUATED_COMPLETE' if final and metric else 'HISTORICAL_SAME_V0' if not arm else receipt['status'] if receipt else 'NOT_TRAINED_'+status,
            'V0_AP_percent':100*metric['video_macro_ap'] if metric else None,'training_seed':42,'endpoint_step':step if arm else None,
            'actual_optimizer_calls':updates,'actual_prompt_exposures':exposures,'online_training_completions':completions,
            'new_training_GPU_seconds':training_seconds,'new_inference_GPU_seconds':sum(r['seconds'] for r in costs if r['event']=='end' and method in r['phase'] and r['bucket']=='evaluation'),
            'historical_cost_note':'Historical training cost excluded from new run' if not arm else 'Teacher/offline candidate/technical shared costs in resource_receipt.json'}
        inference_receipt=maybe(STORE/'predictions'/method/'receipt.json') if arm else read(ORIGINAL/'inference'/method/'receipt.json')
        row['legal_window_fraction']=inference_receipt.get('legal_fraction',inference_receipt.get('proxy_legal_rate')) if inference_receipt else None
        results.append(row)
        if metric:pervideo += [{'method':method,'video_id':v['video_id'],'V0_AP_percent':100*v['mean_ap'],'status':row['status']} for v in metric['per_video']]
        elif arm:
            pervideo += [{'method':method,'video_id':v['video_id'],'V0_AP_percent':None,'status':row['status']} for v in historic['S0']['evaluation']['per_video']]
    base={v['video_id']:100*v['mean_ap'] for v in methods['S0']['per_video']}
    dpo_name=final['stronger_whole_macro_DPO'] if final else None
    dpo_base={v['video_id']:100*v['mean_ap'] for v in methods[dpo_name]['per_video']} if final else {}
    for row in pervideo:
        value=row['V0_AP_percent'];row['delta_S0_pp']=value-base[row['video_id']] if value is not None else None
        row['delta_stronger_DPO_pp']=value-dpo_base[row['video_id']] if value is not None and final else None
    csvwrite(REPORT/'results.csv',results,list(results[0]));csvwrite(REPORT/'per_source.csv',pervideo,['method','video_id','V0_AP_percent','delta_S0_pp','delta_stronger_DPO_pp','status'])
    edges=records(STORE/'teacher/technical_edges.jsonl') if (STORE/'teacher/technical_edges.jsonl').exists() else []
    views=[v for e in edges for v in e['views']]
    tq={'status':status,'technical':technical,'dev_teacher':dev_teacher,'train_teacher':train_teacher,'development_reward_quality':quality,
        'nominal_technical_top_ABCD_counts':dict(Counter('ABCD'[int(max(range(4),key=lambda j:v['normalized_ABCD'][j]))] for v in views)),
        'training_teacher_coverage':maybe(REPORT/'train_teacher_coverage.json'),'training_signal':maybe(REPORT/'train_signal.json'),
        'not_reached_fields_are_null':True,'human_training_labels_read':False}
    write(REPORT/'teacher_quality.json',tq)
    cases=[]
    if technical and not technical['passed']:
        # Registered reporting rule: first three failed self-pairs ordered by source.
        from unsgrpo.decisive_core import unique_top
        failed=[e for e in edges if e['same_clip'] and not all(unique_top(v['normalized_ABCD'])==2 for v in e['views'])]
        for e in sorted(failed,key=lambda x:(x['video_id'],x['clip_i']))[:3]:
            cases.append({'rule':'First3 failed self-pairs sorted by source and clip; all failed pairs retained in full teacher output',
                'video_id':e['video_id'],'clip_i':e['clip_i'],'clip_j':e['clip_j'],'identical_actual_clip':True,
                'normalized_ABCD_both_views':[v['normalized_ABCD'] for v in e['views']],
                'raw_candidate_mass':[v['raw_candidate_mass'] for v in e['views']],'messages':e['views'][0]['messages'],'actual_input_media':e['views'][0]['media'],
                'interpretation':'Same pixels, timestamps and clip ID; unique C expected. No invented semantic description.'})
    elif quality and not quality['passed']:
        # No new search: first3 worst new-reward G windows, unique sources, ties request ID.
        chosen=set()
        for row in sorted(quality['per_window'],key=lambda x:(x['R_pair']['G'],x['video_id'],x['request_id'])):
            if row['video_id'] in chosen:continue
            chosen.add(row['video_id']);cases.append({'rule':'Lowest new-reward single-window G, max1/source, tie request ID, first3','window':row})
            if len(cases)==3:break
    elif final:
        d=final['stronger_whole_macro_DPO'];ref={r['video_id']:r['mean_ap'] for r in methods[d]['per_video']}
        for method in ('G-plain','G-guard'):
            rows=sorted(methods[method]['per_video'],key=lambda r:(r['mean_ap']-ref[r['video_id']],r['video_id']))
            cases += [{'rule':'Worst3 source AP differences vs preregistered stronger whole DPO; include positive differences if fewer than3 failures',
                'method':method,'video_id':r['video_id'],'delta_AP':r['mean_ap']-ref[r['video_id']]} for r in rows[:3]]
    write(REPORT/'failure_cases.json',{'status':'FIXED_RULE_COMPLETE','cases':cases,'images_are_actual_cached_inputs':True,'no_new_prompt_or_selection_uses_these_cases':True})
    asset=maybe(STORE/'teacher_model/asset_receipt.json')
    resources={'status':'ACTUAL_ACCOUNTING','time':now(),'GPU_seconds':used(),'GPU_hours':used()/3600,'GPU_bucket_seconds':{k:used(k) for k in LIMITS},
        'GPU_limits_seconds':LIMITS,'GPU_total_limit_seconds':28800,'GPU_ledger':reference(WORK/'GPU_ledger.jsonl') if costs else None,
        'download_wall_seconds_not_GPU':asset.get('wall_seconds') if asset else None,'preparation':reference(WORK/'protocol.freeze.json'),
        'new_storage_bytes':{'work':bytes_under(WORK),'store':bytes_under(STORE),'report_at_receipt':bytes_under(REPORT)},
        'currency_cost':None,'currency_note':'No billing evidence; elapsed compute and bytes reported','Git_sync':False,'subagents_used':False,
        'new_formal_training_arms':sum((STORE/'training'/a/'receipt_80.json').exists() for a in ('P-DPO','G-plain','G-guard')),
        'new_human_labeled_training_examples':0,'technical_training_receipts':[reference(p) for p in (STORE/'training').glob('technical_*/technical_receipt.json')] if (STORE/'training').exists() else []}
    write(REPORT/'resource_receipt.json',resources)
    if status=='NO_GO_TEACHER_INTERFACE':
        answers=['新奖励的实质选择收益尚未检验：固定8B教师未通过同片段/回答接口准入。','本轮未训练，不能判断RL相对同骨干DPO的收益或成本价值。','停止当前教师接口与奖励配置，不进入独立验证，不自动追加教师或提示筛查。']
    elif status=='NO_GO_REWARD':
        answers=['新奖励未同时达到预定覆盖与候选选择收益门槛。','本轮未训练，现有奖励证据不支持支付GRPO续训成本。','停止当前教师与奖励配方，不进入独立验证。']
    elif status=='NO_GO_TRAIN_SIGNAL':answers=['开发奖励通过，但固定训练池未提供足够覆盖或有效更新信号。','没有完成同骨干RL/DPO比较，不宣称RL胜出或无效。','停止当前采样与训练信号配方，不追加候选。']
    elif status in ('RESOURCE_NOT_READY','INCOMPLETE'):answers=['资源或工程阻断，未得到本轮完整科学结论。','未完成规定比较，不能据此判断RL成本是否值得。','停止自动投入；保留实际产物与具体阻断供复核。']
    else:
        answers=['新奖励通过固定候选收益与覆盖门槛。',{'GO_GUARDED_RL':'可靠性GRPO达到相对S0和较强DPO的全部资源决策门槛。','GO_STANDARD_RL_ONLY':'仅标准GRPO达到继续门槛，可靠性模块尚无继续证据。','TEACHER_ONLY_SIGNAL':'DPO显示伪监督信号，两个RL分支均未达到继续门槛。','NO_GO_RL_INCREMENT':'两个RL分支均未达到相对较强DPO的继续门槛。'}[status],
            '进入独立来源与训练随机性验证；此处不是独立测试。' if status.startswith('GO_') else '停止当前密集评分GRPO配方，不自动扩展实验。']
    lines=['# decisive_rl_v2：'+status,'']+['%d. %s'%(i+1,a) for i,a in enumerate(answers)]+['',
        '全部10个来源为开发集，训练seed固定42。主指标为稳定V0、映射后小数12位并列规范；公开top-5为NOT_READY。不作独立测试、SOTA或所有RL方法的结论。','',
        '|方法|状态|V0 AP (%)|新增训练GPU秒|','|---|---|---:|---:|']
    for r in results:lines.append(f"|{r['method']}|{r['status']}|{r['V0_AP_percent']:.4f}|{r['new_training_GPU_seconds']:.1f}|" if r['V0_AP_percent'] is not None else f"|{r['method']}|{r['status']}|未训练/未完成评价|{r['new_training_GPU_seconds']:.1f}|")
    if technical:lines+=['',f"直接观测：同片段双顺序均以C唯一最高 {technical['same_clip_pairs_C_in_both']}/8（要求≥6）；原始ABCD质量≥0.5的视图占 {technical['views_mass_at_least_half_fraction']:.2%}（要求≥90%）。名义{technical['nominal_views']}视图，实际{technical['actual_forwards']}次前向，缓存{technical['cache_hits']}视图。"]
    if quality:
        q=quality['primary_59_all_legal'];lines+=['',f"直接观测：59个全合法候选组，新G={100*q['R_pair']['G']:.6f}pp，旧G={100*q['R_old']['G']:.6f}pp；覆盖与全部来源结果见teacher_quality.json。单窗替换收益不能相加。"]
    if state.get('error'):lines+=['','具体工程阻断：`'+state['error']+'`。未得到完整科学结论。']
    lines+=['',f"保守GPU占用 {used():.2f}秒（含加载、进程首尾和失败尝试）；新增文件盘 {bytes_under(STORE)/1024**3:.3f}GiB，本地工作目录 {bytes_under(WORK)/1024**3:.4f}GiB。下载等待单独记录，不计为GPU。未提交或推送Git。",'',
        '解释边界：接口未通过只能说明此固定配置未满足操作性准入，不能推广为语义教师普遍无效；奖励未通过不能替代实际RL比较。单seed的来源bootstrap不包含训练随机性。分数压缩不能单独解释AP。','',
        '复核入口：results.csv、per_source.csv、teacher_quality.json、failure_cases.json、resource_receipt.json、protocol.actual.json。实际命令见工作目录commands.jsonl；输入/源码冻结见protocol.freeze.json；若已训练，固定调度与收据分别见schedule.json和文件盘training目录。','',
        '可复现执行：使用固定环境的Python，设置PYTHONPATH=src:scripts；先运行tests/test_decisive_rl.py，再按commands.jsonl中的有序命令运行。当前命名空间已结束，禁止覆盖重跑；重新执行须保留原始证据并使用新目录。']
    (REPORT/'DECISION.md').write_text('\n'.join(lines)+'\n')
    write(REPORT/'execution_receipt.json',dict(state,report_time=now(),actual_protocol=reference(REPORT/'protocol.actual.json'),no_automatic_followup=True))


if __name__=='__main__':main()
