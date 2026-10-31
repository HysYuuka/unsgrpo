# 用户要求暂停

2026-10-27，用户通知即将断网并要求暂停。当前无活动训练或 GPU 实验进程，三个协作 agent 已中断，不设自动重启或 Git 同步。收到用户明确“继续”后恢复。

已完成：50 个真实视频全部预处理（40 train / 10 test），25264 帧、6316 片段；train-only UMAP/KMeans 与共同窗口池；2906 条实际 Qwen 输入审计；108 项本机测试及后续专项；四步 SFT（smoke_SFT_v2）和检查点重载、固定参考不变检查。正式六组尚未冻结或启动。

数据与证据：
- 代码：/root/autodl-tmp/unsgrpo
- 数据、模型：/root/autodl-fs/unsgrpo
- 处理帧、日志、检查点：/root/autodl-tmp/unsgrpo-work
- 输入审计：reports/input_audit.json
- SFT 技术端点：/root/autodl-tmp/unsgrpo-work/checkpoints/smoke_SFT_v2/v0-20261027-183346/checkpoint-4
- 实际量化共同技术底座：/root/autodl-tmp/unsgrpo-work/checkpoints/smoke_merge_v2
- 固定参考证据：reports/smoke_reference_v1.json
- BF16 全参数诊断：reports/smoke_merge_v2_bitwise_diagnosis.json

恢复顺序：先检查被中断 agent 的末次代码写入是否完整及现有针对性测试；完成 reward_plugin 对 HF 图片路径包装（bytes=None/path）严格规范化的修复。GRPO_v1 缺 msgspec、v2 rollout 帧列表类型、v3 reward 图片包装误判的失败记录均保留。重新跑全新 run_id 的 PROMPTS=1,G=4,steps=4 技术闭环，随后正式并发检查、保存重载与实际 loss 核验，再冻结并运行六组。

合并前后 logp 初拟容差失败保留；全625张量已逐位证明合并算术正确。冻结定义已决定共同 S0 为实际 BF16 量化合并端点，不声称与未合并函数精确等价。核对 protocol/core/ops 对新定义的最后衔接后才可冻结。

用户已撤销每日自动同步，仅在未来明确要求时手动同步；不得自行提交或推送。人工评价标签值尚未读取。
