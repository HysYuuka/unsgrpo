# 同图裁剪排序：完整实验执行

**2026-10-31 已按用户指示停止实验，转向比赛内容整理。** FULL/MATCHED 三种子各完成 30 轮，验证均值为 63.5%/64.5%；这不是正式测试结果。MUSIQ seed17 在第 2 轮 step397 保存断点，其余任务不再自动执行。见[停止归档及结论边界](reports/STOPPED_REVIEW_20261031.md)。

- 方案：`configs/design_guide.md`；执行冻结：`PROTOCOL_LOCK.json`。
- 实时状态：`STATUS.json`；后台任务：`RUNNING_PID.json`；日志：`logs/`。
- 历史方案为 12 项核心任务、30 epoch 后统一决定是否延长至 50；该完整矩阵未完成，已停止。
- 所有正式训练使用全体合法候选、CLIP 512 与真实原图 1024 细节路径，完整反传，三个主要组分别三个 seed。
- 数据、共享权重和检查点：`/root/autodl-fs/highlight-static-rank/`。原始 GAIC 复用已有官方数据，不重复复制。
- 实现验收：`reports/ACCEPTANCE_NUMERIC.json`、`reports/ADDITIONAL_ACCEPTANCE.json`、`reports/MUSIQ_ACCEPTANCE.json`。
- 外部模型与数据有独立状态，缺项不能写成已完成或据此宣称 SOTA。

STOP_REQUESTED_BY_USER.json 阻止主队列、直接训练及外部评价等待器重新启动。只有用户重新明确授权后才允许移除标记并恢复。原运行命令保存在历史交接材料中，本次不恢复、不补跑测试。

停止需向 `SUPERVISOR_PID` 中的进程发送 SIGTERM；worker 会完成当前有效 batch 并保存可恢复状态。不要用 SIGKILL 正常停止。每 50 个优化步骤以及每个 epoch 保留 last 完整状态，仅另保留 best 权重。

软件环境锁定见 `configs/environment.txt`；源码与报告在本地 Git，媒体、数据、权重、环境、训练日志不提交。
