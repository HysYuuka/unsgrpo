# 首轮实验断网运行保障

用户已授权继续，并要求每半小时检查一次、断开连接后自动完成。当前训练控制器采用独立会话、标准输入为 `/dev/null`、输出直接写本地日志；父进程链连接到机器的进程1，未绑定SSH终端。模型、视频、伪标签和依赖已在本机/挂载存储，实际训练启用 `HF_HUB_OFFLINE=1`。

GPU主流程保留原冻结代码，自动依次完成G-RANK、六组全覆盖推理、预测冻结、人工标签转换、原V0评价、端点归档和原审计报告。三个已完成训练分支及共同S0保持原身份；中断尝试单独保留。

补充收尾服务通过独立后台screen会话 `unsgrpo-round1-offline` 执行，后台脱离启动命令的测试已验证screen服务器的父进程为1。服务每1800秒检查主控制器；主流程完成后自动运行预登记的四版本数值复核、综合结果和数值失败候选、逐文件SHA校验的补充封存。全部步骤仅使用本地数据，不需要对话连接或外部API。出现实现故障会记录错误并停止，不静默追加训练预算。

只要机器及挂载存储继续运行，关闭聊天、SSH连接中断或外网波动不会要求人工推进下一阶段。自动收尾不等于机器关机后的精确训练续跑；本次不作该声明。

- 主控制器：`/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1_recovery1/state.json`
- 离线收尾状态：`/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1_recovery1/offline_completion_state.json`
- 离线收尾日志：`/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1_recovery1/offline_completion.log`
- 预登记数值复核：`/root/autodl-tmp/unsgrpo-work/runs/round1_seed42_v1/evaluation_sensitivity_v01/`
- 最终综合报告：`/root/autodl-tmp/unsgrpo/reports/round1_final_review_recovery1/FINAL_ROUND1_REVIEW.md`
- 最终轻量结果与CSV：同一报告目录。
- 补充归档：`/root/autodl-fs/unsgrpo/archives/round1_seed42_v1_recovery1_closeout_v1/`

自动综合报告提供可复核数值比较、成本、规则化失败候选与投入建议；画面机制须明确区分已观察证据和待核实假设。V1未对齐、单训练seed及10来源的限制须保留。代码和轻量记录仅在用户每次明确要求时同步Git。

用户再次要求暂停时，必须同时停止GPU控制器的全部训练后代和本后台收尾服务，避免遗留服务继续推进。可通过 `offline_completion.pid` 和screen会话名定位；暂停与恢复记录应保留。
