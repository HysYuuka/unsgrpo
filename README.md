# 高光与裁剪研究归档

**2026-10-31：用户决定停止实验，重心转向比赛内容整理。** 本仓库按本次明确指示进行一次归档同步，不设置自动训练、自动恢复或每日同步。没有新跑测试来补齐未完成的研究结论。

| 研究记录 | 结论与入口 |
|---|---|
| 无监督高光首轮 | [首轮结果及失败复盘](reports/ROUND1_DECISION_AND_FAILURE_REVIEW_20261028.md)：当前 GRPO 配方未建立可靠收益 |
| 教师与奖励诊断 | [第二轮](reports/round2_diagnostic_v1/FINAL_REPORT.md)、[第三轮](reports/round3_rl_method_v1/DECISION.md)、[奖励复核](reports/reward_to_rl_v1/FINAL_REVIEW.md)、[最后奖励实验](reports/decisive_rl_v2/DECISION.md) |
| 静态同图裁剪排序 | [停止归档](research/static_crop_rank/reports/STOPPED_REVIEW_20261031.md)：FULL/MATCHED 验证均值 63.5%/64.5%，完整矩阵未完成，正式测试未执行 |
| 存储与恢复 | [本次清理收据](research/static_crop_rank/reports/STORAGE_CLEANUP_20261031.json)、[保留检查点与预测哈希](research/static_crop_rank/reports/RETAINED_RUN_ARTIFACTS_20261031.json) |

Git 保存研究源码、配置、曲线、指标和校验信息。原视频、原始人工标注、模型权重与可恢复检查点保留在服务器，不上传公共仓库。比赛源码与工作稿继续保存在独立的私有 highlight 仓库，不合并到这里。

## 历史首轮执行说明

以下记录原始执行安排，不能作为当前恢复实验的授权。

本轮回答：训练库重复性代理是否有效，学生是否受益，以及从共同 SFT 端点继续 GRPO 是否值得额外成本。原始方案见 [docs/ORIGINAL_PROPOSAL_20261027.md](docs/ORIGINAL_PROPOSAL_20261027.md)，执行条件见 [交接文档](docs/EXPERIMENT_REVIEW_AND_HANDOFF_20261027.md)。

当前真实状态以 `reports/execution_status.json` 和仓库外逐运行收据为准。`CANDIDATE`、合成测试通过和训练器正常退出都不等于正式六组实验完成。

环境位于 `/root/autodl-tmp/unsgrpo-work/envs/highlights`。先加载 `storage-env.sh`，再设置 `PYTHONPATH=/root/autodl-tmp/unsgrpo/src`。原视频、人工评价标注、权重、环境、检查点及完整日志在仓库外；仓库仅同步代码、配置、清单和轻量证据。

评价采用明确定义的 V0 内部诊断，不作跨论文数值比较；V1 尚待完整的真实时间轴和参考划分复核。六组预测全部冻结后，评价脚本才允许读取标注。首轮只有一个训练 seed，不能覆盖训练随机性。

正式执行须先完成真实预处理、全量实际输入审计、四步 SFT/GRPO、合并/重载/固定参考验证和正式 rollout 并发检查，再由 `scripts/freeze_round1.py` 冻结协议。固定矩阵为 B0、B1、S0、S+、G-CAL、G-RANK，主比较为 G-RANK − S+。

存储、每日同步及待配置项见 [docs/STORAGE_AND_SYNC.md](docs/STORAGE_AND_SYNC.md)；实施差异见 [docs/IMPLEMENTATION_DEVIATIONS.md](docs/IMPLEMENTATION_DEVIATIONS.md)。
