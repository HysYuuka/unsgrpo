# 同图裁剪排序实验执行交接

接收方：后续执行 agent。用户已说明本任务应由执行 agent 接手。本次交接只核查状态并编写文档，没有停止、重启或修改正在运行的训练。

**当前应接管已有后台任务。完整训练已经在进行，不能再启动一份相同队列。** 本轮科学目标仍是同图裁剪候选排序与最终选择；无须转向 RL、视频裁剪或新数据集研究。

快照时间为 **2026 年 10 月 30 日 18:08，香港时间 UTC+8**。训练持续推进，下面的步数会过时；接手时以 `STATUS.json`、进程命令行和各 run 的 `history.json` 为准。机器可读快照见 [HANDOFF_SNAPSHOT_20261030.json](/root/autodl-tmp/highlight-static-rank/reports/HANDOFF_SNAPSHOT_20261030.json)。

## 接手时先做什么

1. 阅读本文件、原始完整指导、冻结协议与工程运行配置。
2. 确认下面两个进程仍是本项目 supervisor 和 worker；进程正常时继续接管监控即可。
3. 保持既定 12 项矩阵、完整数据与训练日程，跟踪任务完成状态和磁盘预算。
4. 利用核心训练期间处理已下载 CPC 的标注转换及原图重叠审核，以及外部基线的获取与接口工作。与主训练竞争 GPU 的验收和计时应明确安排，不能污染吞吐测量。
5. 训练、统计和外部对照完成后，审核并补全最终报告。现有自动脚本不等于所有外部对照与论文级交付已经完成。

用户已经明确授权实验；看到初步算力时间后再次回答“允许实验”，没有给出新的 GPU 小时上限。无需重复询问是否可以训练。用户要求完整实现，反对冻结特征替身、删候选、降分辨率、单 seed 短训筛掉方案，也反对无边界追加对照。若使用子 agent，用户此前偏好 GPT-5.6，且主要用于搜集资料；不要未经说明替换成别的型号或扩大并行开销。

## 当前运行状态

工作区：`/root/autodl-tmp/highlight-static-rank`

持久资产：`/root/autodl-fs/highlight-static-rank`

| 项目 | 交接快照 |
|---|---|
| 正在执行 | `FULL_s17`，计划 30 epochs |
| 已完成 | 11 个 epoch，当前 epoch 12 |
| 当前步数 | global step 3713，本 epoch 已处理 664/2635 张原图 |
| 全矩阵 | 12 个任务，尚无一个完成全部 30 epochs；其余 11 个未启动 |
| supervisor PID | `478364` |
| worker PID | `478399` |
| supervisor | `.venv/bin/python -u scripts/run_matrix.py` |
| worker | `.venv/bin/python -u -m croprank.train --variant FULL --seed 17 --epochs 30 --chunk 90` |
| 环境 | `/root/autodl-tmp/highlight-static-rank/.venv/bin/python` |
| 硬件 | RTX 4090，24,564 MiB；核查时显存约 3,242 MiB |
| 异常 | 核查时任务正常，最近 loss、梯度范数均为有限值 |
| 正式测试 | 尚无 `test_predictions.jsonl`；未进行测试集选模 |

进程 ID 只能视为交接快照，后续会变化。最新 PID 在 [RUNNING_PID.json](/root/autodl-tmp/highlight-static-rank/RUNNING_PID.json)，最新状态在 [STATUS.json](/root/autodl-tmp/highlight-static-rank/STATUS.json)。

### 已有验证结果

这些是 **200 张验证原图上的单 seed 中间结果**，不应与论文测试分数直接比较，也不足以作方法成败判断。

| epoch | 训练总损失 | val Acc1/5 | val SRCC | val MOS regret |
|---|---:|---:|---:|---:|
| 1 | 0.39206 | 43.0% | 0.80761 | 0.41035 |
| 5 | 0.20014 | 47.5% | 0.84691 | 0.37050 |
| 8 | 0.17925 | **57.5%** | 0.84775 | **0.30455** |
| 10 | 0.16738 | 56.0% | **0.86018** | 0.31520 |
| 11 | 0.16175 | 56.0% | 0.85176 | 0.31765 |

按预定“Acc1/5 优先，其次更低 regret，再其次更早 epoch”的规则，当前历史最佳是 **epoch 8**。不能因为 epoch 10 的 SRCC 更高就换成另一套选模规则。

当前 run 位于 `/root/autodl-fs/highlight-static-rank/runs/FULL_s17/`：

- `last.pt`：约 1.094 GB，完整模型、optimizer、RNG、历史、当前 epoch 与已处理位置；每 50 步和每个 epoch 保存。
- `best.pt`：约 0.365 GB，仅用于验证选出的最佳模型及元数据。
- `history.json`：每个已完成 epoch 的训练与验证结果。
- `steps.jsonl`：每个正式优化步骤的 loss、梯度范数、候选数、时间和候选并行数。
- `best_val_predictions.jsonl`、`last_val_predictions.jsonl`：对应验证预测。
- `THROUGHPUT_ACTIVE.json`：当前有效的吞吐估算。

`reports/LAUNCH_STATUS.json` 和此前 `EXECUTION_UPDATE.md` 中的“第 1/2 个 epoch”是启动时快照，不能当实时状态。

## 方案和实现入口

以这些文件为依据，不能只根据旧聊天摘要重建方案：

- [原始完整指导](/root/autodl-tmp/highlight-research/plans/CROP_RANK_FULL_VALIDATION_GUIDE_20261030.md)
- [冻结执行协议](/root/autodl-tmp/highlight-static-rank/PROTOCOL_LOCK.json)
- [当前工程配置](/root/autodl-tmp/highlight-static-rank/ENGINEERING_RUNTIME.json)
- [方法实现说明](/root/autodl-tmp/highlight-static-rank/METHOD_IMPLEMENTATION.md)
- [实现验收汇总](/root/autodl-tmp/highlight-static-rank/IMPLEMENTATION_ACCEPTANCE.md)
- [环境锁定](/root/autodl-tmp/highlight-static-rank/configs/environment.txt)

`PROTOCOL_LOCK.json` 的 SHA256 是：

```text
4d75af7657fc42098452b8b7316a9427ea875e136b2059ddaf110d18f5b9537a
```

恢复 checkpoint 时会核对这个哈希。其中嵌入的原始设计说明仍带有最初的 `not_started` 字段，它是历史设计原文，不是实时执行状态。不要为“更新显示状态”改动冻结协议。

| 文件 | 职责 |
|---|---|
| `src/croprank/model.py` | FULL、MATCHED 及结构消融，CLIP 语义场、真实高分辨率细节场、几何编码、关系/边界/fusion |
| `src/croprank/core.py` | 数据读取、增强、损失、主指标、原图确定性顺序 |
| `src/croprank/replay.py` | 全候选梯度重放、分块推理 |
| `src/croprank/train.py` | 初始化、AdamW、学习率、恢复、训练、验证和选模 |
| `src/croprank/musiq.py` | MUSIQ 全网络、多尺度候选批处理与重放 |
| `src/croprank/musiq_vendor/` | 锁定的 pyiqa 0.1.16 网络及预处理，含适配说明和许可证 |
| `src/croprank/gaic.py` | GAIC 官方 MobileNet 任务权重及旧 RoI/RoD 前向算子的算术移植 |
| `scripts/run_matrix.py` | 12 项串行队列、统一延长规则、冻结检查点和测试入口 |
| `scripts/evaluate_matrix.py` | 完整核心测试、独立指标复算、配对 bootstrap、初步决定 |
| `scripts/analyze_completed.py` | 机制子集、种子均值/SD、100 图计时、预定失败案例 |
| `scripts/evaluate_available_external.py` | 当前已接入的 GAIC 发布权重、原始 MUSIQ-AVA 参照 |

### 固定训练队列

以下顺序来自 `configs/matrix.json`，supervisor 会自动推进：

| 顺序 | 任务 | seed | 当前状态 |
|---:|---|---:|---|
| 1 | FULL | 17 | 正在训练 |
| 2 | MATCHED | 17 | 等待 |
| 3 | FULL | 29 | 等待 |
| 4 | MATCHED | 29 | 等待 |
| 5 | FULL | 43 | 等待 |
| 6 | MATCHED | 43 | 等待 |
| 7 | MUSIQ_FT | 17 | 等待 |
| 8 | MUSIQ_FT | 29 | 等待 |
| 9 | MUSIQ_FT | 43 | 等待 |
| 10 | GEOM_BASE | 17 | 等待 |
| 11 | MATCHED_BASE | 17 | 等待 |
| 12 | NO_BOUNDARY | 17 | 等待 |

FULL/MATCHED/MUSIQ_FT 均保留三种子。GEOM_BASE、MATCHED_BASE 去掉选择损失；NO_BOUNDARY 去掉边界 token，细节场仍供布局使用。

主日程 30 epochs。全部任务完成后，如果任一主要组满足原指南的训练损失和验证改善条件，**全部 12 项统一续到 50 epochs，最多一次**。随后冻结所选 checkpoint 的 SHA256，再统一测试；当前不能提前运行测试脚本或拿测试结果修改结构和参数。

## 已落实的完整实现

- CLIP ViT-B/16 视觉分支全训练，原图长边 512、保持长宽比；blocks 4/8/12 融合为 256 维语义场。
- 原图独立生成长边 1024 的细节输入，ResNet18 stem/layer1/layer2 全训练；BN running statistics 固定，affine 训练。
- 每个候选 81 个布局 token、64 个边界 token、1 个评分 token；双坐标、真实框内/框外关系、候选相关源位置、2 层 fusion。
- 每张原图全部 52–90 个合法候选参与完整 reg、所有 pair 的 rank、全候选 selection 目标。
- AdamW：预训练参数 LR 1e-5、新参数 1e-4；有效 batch 8 张原图，末尾不足 8 的 batch 全保留并按实际数量平均。
- FULL 91,104,580 参数；MATCHED 91,039,044，差约 0.072%。同 seed 共同状态张量初值一致。
- MUSIQ 使用真实 AVA 10-bin 分布期望，原生多尺度 224/384/原分辨率，完整网络训练，没有固定特征缓存。

验收记录包括 `ACCEPTANCE_NUMERIC.json`、`ADDITIONAL_ACCEPTANCE.json`、`MUSIQ_ACCEPTANCE.json`、`MUSIQ_BATCH_ACCEPTANCE.json` 和 `PARALLEL_CANDIDATES_ACCEPTANCE.json`，均位于工作区 `reports/`。这些证明所执行的软件检查通过，不代表完整实验效果已经验证；正式统计、完整消融和全部外部评价尚未完成。

## 必须保留的工程优化与数据处理

### 候选并行优化

最初每块 4 个候选，100 个正式步骤实测平均 10.72 秒。发现显存余量很大后，已在最大 1024×1024 原图的全部 90 个候选上验证更大候选并行。

- FP32 全参数梯度相对 L2：`3.73e-7`，最差有意义参数张量约 `1.32e-6`。
- BF16 梯度相对 L2：约 `0.00353`，余弦约 `0.999994`。
- FULL_s17 的前 **124 步**使用原分块方式；从 **125 步**开始用候选块 90，并恢复已有 optimizer、RNG 和原图位置。
- 后续 100 个正式步骤平均 **1.219 秒**，FULL 单次 30 epoch **纯训练约 3.35 小时**；旧 29.5 小时估算只代表旧工程方式。
- 最近完整 epoch 加验证和保存约 7.7–8.2 分钟，30 epochs 的运行时间更接近约 4 小时；首次慢速阶段及中断另计。
- 原图有效 batch 仍为 8；没有减少候选、结构或分辨率。

`PROTOCOL_LOCK.json` 保留原分块参考值，**实际运行参数以 `ENGINEERING_RUNTIME.json` 为准**。手动运行 trainer 若漏掉 `--chunk 90`，其 CLI 默认仍是 4，会明显变慢。队列入口已经传入正确参数。

MUSIQ 的候选批处理固定为 4。只给不同长度 token 序列追加 validity=0 的 padding，保留全部原生 patch；与逐候选实现的分数及梯度检查通过。MUSIQ 尚未正式训练，也没有其 100 正式步骤吞吐；不能套用 FULL 的时间估算。

### GAIC 数据和标签

原始数据实际复用：

```text
/root/autodl-fs/highlight-aesthetic-rl/assets/gaic/GAIC
```

冻结 manifest：

```text
/root/autodl-fs/highlight-static-rank/data/gaic_manifest.jsonl
```

**不要整目录删除旧 `highlight-aesthetic-rl` 资产，本轮 manifest 仍引用其中的 GAIC 原图。**

官方记录是 2636/200/500，共 3336 个。训练中的 `220928`、`225959` 是完全相同的图像和候选框，但有两套 MOS。它们已按原图身份合为一个训练样本，分别计算两套人工标注的完整损失后平均；没有重写或丢弃第二套标签。因此实际独立原图为 **2635/200/500**。官方验证与测试原图不变。

官方 `-2` 无效标签按作者读取器过滤。另有 6 个合法发布 MOS 是 0.86/0.93，低于名义 1 分下限；已原样保留并使用固定 `(MOS−1)/4` 仿射变换，不做 clipping，不补造人工分数。以上例外已在训练前锁定，不应再当作新发现重做一轮数据实验。

manifest 的 SHA256：

```text
0098f7686d7a984b9bfd6993040998a1366c85d5f0a5fcb837a7179404d37d84
```

## 自动流程已接入的工作

supervisor 会依次完成训练、判断统一延长、冻结 best checkpoint 哈希、测试全部任务、独立指标复算、两项配对 bootstrap、机制子集、种子汇总、失败案例，以及当前可运行的两个外部参照。

主要统计按原图成组、三个 seed 对齐后求每图平均，再做 10,000 次 bootstrap；输出 95% 与 Bonferroni 对应的 97.5% 区间。种子不额外重采样，输出中已明确这个定义。

方法支持标准仍是 FULL 相对 MATCHED 的 Acc1/5 均值至少提高 2 个百分点、校正区间下界大于 0、至少 2/3 seed 正向、原生 MOS regret 不恶化。竞争力还要检查 MUSIQ_FT 与现代外部参照，不能因为 MAIN 的某个早期验证值好看就改判定线。

## 接手后仍需补齐的工作

### 外部评价

| 项目 | 实际进度 | 后续动作 |
|---|---|---|
| GAIC 官方任务权重 | 已下载、ZIP CRC 通过，strict 加载通过，旧 RoI/RoD 前向公式对独立标量版本核验通过，真实训练原图全候选 forward 通过 | 已排入冻结后的评价；报告为“官方发布权重＋经验证的算术移植”，不能声称已重建作者原 CUDA 二进制或具有严格同训练集公平性 |
| MUSIQ-AVA 原始权重 | 权重、原生预处理、真实反传已验收 | 已排入冻结后的发布模型参照 |
| CPC | **已下载并解压，下载进程已结束** | 转换原生标注，核验与 GAIC 训练/验证的原图重叠，再冻结外部候选评价协议；本轮不使用 CPC 训练 |
| FGAesQ 完整 DiffToken | 权重未取得，服务器对 HF 与镜像连接出现 EOF/连接故障；代码资料已在旧研究目录 | 有限次数处理下载，严格权重加载；完整 all-pairs 与单图版分列，不能以单图版替代完整版 |
| FGAesthetics-Cropping | 未下载，原图与候选几何映射未做 | 有数据后核验原图/候选坐标与重叠；无法覆盖全集时只报告固定映射子集及覆盖率 |
| GAFIC | 仅论文参照，源码审计发现的问题尚在 | 使用既有审计明确复现缺口；不要无限修复并拖延主矩阵，更不能把未经核验的修正版冒充原方法 |

CPC 新状态以 [CPC_ACQUISITION.json](/root/autodl-tmp/highlight-static-rank/reports/CPC_ACQUISITION.json) 为准，`EXTERNAL_STATUS.json` 里的 `DOWNLOADING` 已过时，接手后可整理一致。数据位置：

```text
/root/autodl-fs/highlight-static-rank/data/CPCDataset.tar.gz
/root/autodl-fs/highlight-static-rank/data/cpc/CPCDataset
```

压缩包 1,365,111,871 bytes，SHA256：

```text
dfa4ec73c9d9b4b525a8f79aee5670fac4797bad2eb1bd0e1f26f051ac3a7281
```

旧资料与审计：

- `/root/autodl-tmp/highlight-research/reviews/ranking_close_read_20261030/`
- `/root/autodl-tmp/highlight-research/reviews/two_tracks_baselines_20261030/`

### 最终交付并未全部自动化

1. `evaluate_matrix.py` 会先写一份初步 `DECISION.md`，然后才执行失败案例、计时和外部脚本；后续脚本不会自动把所有结果重写进这个文档。执行 agent 需要整合最终报告。
2. 自动状态包含 METHOD_SUPPORTED / METHOD_NOT_SUPPORTED / INCONCLUSIVE，但不会自动给出经过所有现代外部比较核验的 COMPETITIVE。需按完整指南审查后下结论。
3. 当前计时包含端到端与解码/框准备；完整预处理、源编码、候选评分、汇总等分项尚未全部单列。FGAesQ 的全 pair 计时也未接入。
4. CPC、FGAesQ、FGAesthetics 的完整测试入口和几何映射没有被 `evaluate_available_external.py` 自动完成。该脚本当前只执行 GAIC 与 MUSIQ 原始权重。
5. 完整测试脚本尚未在真实最终产物上运行。现有单元与数值验收不能代替这一步；遇到执行错误按日志修复，不能先宣称全流程已通过。
6. 最终预测、训练曲线、外部结果、代表失败图及检查点的完整归档哈希清单仍需在收尾时生成。
7. 若统计发现显著性或功效不足，按预定状态报告，不自动追加无止境的新结构、新损失或 seed 搜索。

## 监控和恢复操作

下面是供执行 agent 按需使用的操作说明，本次交接没有执行停止或重启。

### 只读查看

```bash
cd /root/autodl-tmp/highlight-static-rank
cat STATUS.json
cat RUNNING_PID.json
tail -n 5 logs/FULL_s17.log
cat /root/autodl-fs/highlight-static-rank/runs/FULL_s17/history.json
nvidia-smi
df -h /root/autodl-tmp /root/autodl-fs
```

`STATUS.json` 在验证和 checkpoint 保存期间可能暂时停留在最后一个训练步骤；配合 log、进程和 `history.json` 判断，不能仅凭几秒未更新就重启。

### 如确实需要安全停止

应向 **supervisor** 发送 SIGTERM，让它转发给 worker 并等待保存。先核对 PID 当前仍对应本项目：

```python
from pathlib import Path
import os, signal
root = Path('/root/autodl-tmp/highlight-static-rank')
pid = int((root / 'SUPERVISOR_PID').read_text())
cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
assert str(root / 'scripts/run_matrix.py') in cmd, cmd
os.kill(pid, signal.SIGTERM)
```

等待当前 batch/验证及保存完成，确认父子进程都退出，并检查状态为 `STOPPED_WITH_RECOVERABLE_CHECKPOINT`。不要直接 kill -9。

**不要只停 worker。** 当前 supervisor 主要依据退出码判断任务；单独让 worker 正常退出可能让 supervisor 继续启动下一个任务，而不是整体停止。接手者若改进这个健壮性检查，需要保存修改记录，但不必因此重训健康模型。

### 仅在旧 supervisor 和 worker 已退出后恢复

使用项目虚拟环境启动队列，恢复逻辑会读取对应 run 的 `last.pt`，不需要人工选择某个 best 权重续训：

```python
from pathlib import Path
import subprocess
root = Path('/root/autodl-tmp/highlight-static-rank')
with (root / 'logs/matrix_supervisor.log').open('a') as log:
    p = subprocess.Popen(
        [str(root / '.venv/bin/python'), '-u', str(root / 'scripts/run_matrix.py')],
        cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
(root / 'SUPERVISOR_PID').write_text(str(p.pid) + '\n')
print(p.pid)
```

如因真正的实现错误需要改变模型或损失，遵循原方案对受影响运行重新开始并留档；不要静默加载旧优化器继续一个已改变科学定义的模型。若只是恢复进程，则保持冻结协议和现有 checkpoint，不要重新执行 `lock_protocol.py`。

## 存储与清理

当前工作区约 365 MB；本项目数据约 2.8 GB、权重约 519 MB、运行产物约 1.4 GB。核查时 `/root/autodl-tmp` 约 45 GB 可用，`/root/autodl-fs` 约 157 GB 可用。

原方案目标仍为共享盘数据 30 GB、共享权重 10 GB、run 产物 30 GB，总计 70 GB；临时缓存约 10 GB，文件系统至少留 10 GB。checkpoint 保留 best＋last，不堆积每个 epoch 的 optimizer 副本。

已清理的是四个可由基础模型和 adapter 重建的 Qwen 合并权重文件，共 **17,020,561,248 bytes，约 17.02 GB / 15.85 GiB**。删除项的哈希、原路径和保留的重建来源见 [CLEANUP_RECEIPT.json](/root/autodl-tmp/highlight-static-rank/reports/CLEANUP_RECEIPT.json)。原始实验报告、日志、adapter、基础模型、唯一证据均保留。

旧目录里的数据和源码并不等于无用资源。尤其要保留本轮实际引用的 GAIC 数据源，避免清理导致正在运行的数据加载失败。

## Git 和交接边界

当前项目是单独的本地 Git 仓库：

- 分支：`codex/crop-rank-20261030`。
- 代码快照：`99c5ba0`，后续启动状态快照：`548a046`。
- 尚未配置 remote，尚未推送本项目。
- 交接前可见的已跟踪修改只有持续更新的 `reports/CPC_ACQUISITION.json`；本次新增交接文档和快照不改训练源码。

用户先前有每日同步到 `HysYuuka/unsgrpo` 的约定，需要执行 agent 按既有授权和实际仓库状态落实。此前 API 检查中 `unsgrpo` 是公开空仓库；不要与私有的旧比赛仓库 `HysYuuka/highlight` 混淆，也不能把模型、数据、媒体或凭据提交到 Git。

本次未向其他 agent 自动发送消息或创建新任务，由用户把本文件交给执行 agent。接手后应持续执行固定实验、处理明确缺项并形成一次完整结果，而不是重新讨论一套方向。
