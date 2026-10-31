# 首轮评价协议与来源核验

协议批次：20261027；运行收据的用户侧时间采用 Asia/Hong_Kong。本文件是冻结前协议；未声称已得到真实测试结果。训练与代理计算不能 import 评价模块或读取 `labels_eval_only`。人工标签只在六组预测冻结、哈希记录和覆盖检查通过后由独立评价步骤读取。

## 数据来源和任务

TVSum 官方将其定义为视频摘要数据，包含 50 视频，每视频 20 位用户对约 2 秒 shot 给出 1–5 分重要性标注。[官方 README](https://github.com/yalesong/tvsum) 提供原视频入口及 CC-BY 3.0 说明；入口于本机核验可达：[官方文件目录](https://people.csail.mit.edu/yalesong/tvsum/)，`tvsum50_ver_1_1.tgz` 标称 641 MB。是否包含完整原视频，以下载归档目录和 ffprobe 验证结果为准，不以网页标称代替资产验收。

[官方 MATLAB 评价脚本](https://github.com/yalesong/tvsum/blob/master/matlab/script_evaluate_result.m) 使用 `tvsum50(video).user_anno` 的第二轴作为标注者、第一轴作为视频帧；它展示的是预算约束摘要 F1，并非高光 top-5 mAP。不得把逐帧 `user_anno` 的每一行解释为两秒。

视频先按 `source_group` 固定 80/20 划分，再生成窗口；seed=42。初始分组至少按原 YouTube ID，并检查内容哈希重复。人工类别、标题、用户分数不进入模型输入、聚类或 split 选择；类别仅可在预测冻结后的结果分析使用。所有方法使用完全相同的视频集合，缺失或解码失败须在正式冻结前共同排除并列明理由。

## V0：本项目内部诊断（主可用协议）

协议 ID：`V0-strict-median-noninterpolated-AP-v1`。单位数值范围 [0,1]，报表转百分数时乘 100，不能与文献中的 top-5 mAP 横向比较。

1. 将模型所有 2 秒片段分数通过完整覆盖窗口平均汇总；非法窗口分数为 0，仍计入窗口覆盖分母。非法率单列。
2. 评价输入必须包含显式真实时间区间 `bounds[S,2]` 和 `ratings[S,20]`。有 PTS 时用逐帧 PTS 定义帧区间，最后一帧右边界用已核验视频终点；只有恒定帧率且帧数/时长经核验一致时才可登记使用 frame_index/fps。不得通过删除末段或归一化时间轴掩盖不一致。
3. 首轮 V0 评价单位固定为原生逐帧标注区间 `native_frame`，通过实际解码保存的 `frame_bounds.npy` 对齐。这会对持续时间较长的正例赋予更多帧权重；VFR 下帧等权还不等于时间等权，完整报告须披露。原 `nframes`、解码 `n_frames`、`ratings[n_frames,20]` 与 `frame_bounds[n_frames,2]` 必须严格一致；不根据标签变化推测 shot 边界或寻找更有利分段。
4. 将预测按真实时长重叠积分到每个评价区间：`sum(overlap*score)/interval_duration`。源区间必须有序且不相交；目标区间缺口、部分覆盖、外推全部报错。
5. 每位标注者独立计算中位数，仅严格大于中位数记正例；等于中位数视负例。不根据预测或方法调阈值。
6. 对该用户全部评价区间计算非插值 average precision。相同预测分数作为同一阈值组同时进入，避免任意并列顺序制造收益。与 sklearn `average_precision_score` 的二元无权重定义一致。[sklearn 原文](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
7. 无正例用户 AP=0，并报告人数；先平均 20 用户得到逐视频 AP，再对视频宏平均。Spearman 使用预测与用户平均分、并列平均秩；常数或单区间时记为 `null/undefined` 并报告数量，不伪造相关系数 0。Spearman 仅诊断。

手算验证：预测区间 `[0,2),[2,4),[4,4.5)` 分数 `10,40,100` 映射到 `[0,3),[3,4.5)` 得 `20,60`。正例位于预测降序第 1、3 位，非插值 AP=`(1+2/3)/2=5/6`。两条并列预测（一正一负）的 AP=1/2，与输入顺序无关。

## V1：来源绑定及仍需完成的部分

[重复性论文 arXiv v2](https://arxiv.org/html/2407.13933v2) 说明 TVSum top-5 mAP 和重复 80/20 划分，但正文不足以独立确定完整评价实现。其引用的 CHD 作者代码地址 `https://github.com/tkbadamdorj/CHD` 于本次核验返回 404；没有把推测写成作者协议。

已发现可核验的官方 [UMT TVSum 源码](https://github.com/TencentARC/UMT/blob/bde05e3de39049bf90d8807f6b2a4ed8b42acbb0/datasets/tvsum.py)：

| 项目 | 本次固定身份 |
|---|---|
| 仓库 | TencentARC/UMT（作者官方实现） |
| revision | `bde05e3de39049bf90d8807f6b2a4ed8b42acbb0` |
| 文件 | `datasets/tvsum.py` |
| SHA256 | `f7c962a7abb061d37844eb4baa1503e0e3dba54caa5483d97f68287776e7f347` |
| 本地实现 ID | `V1-UMT-top5-evaluator-component` |

该来源实现：20 用户分别严格高于各自 Torch 中位数二值化；按预测降序取前 5 个 clip；召回分母是前 5 内正例数；使用梯形 PR 积分；无正例为 0；平均视频再平均用户，最后 round(...,5)。源码使用 `torch.argsort(descending=True)`，默认不稳定并列排序，因此本地组件使用记录版本的 CPU Torch float32 默认排序，并在每次结果中写入后端身份。不得将这个算法命名为 sklearn AP@5。

本机已将上述固定 revision 的原 `evaluate` 方法从 AST 中直接提取执行；32 个只含合成标签的随机案例（含大量分数并列）与本地实现四舍五入到五位后全部一致，Torch 为 `2.8.0+cu128`、CPU 执行。此检验没有读取真实人工标签。

仍未通过的 V1 验收：原 UMT JSON 的 clip 时间轴与原始视频帧的可核验转换、目标对比论文的相同 split/平均层次、公开参考预测复算。单个来源的 evaluator 数值一致，不足以宣称真实 benchmark 已对齐。`evaluate_top5_umt` 因此始终输出 `benchmark_alignment_verified=false`；正式报告的 V1 主结果应标 `NOT_VALIDATED`。若在首轮冻结前未解决，可按交接执行明确标记的 V0 六组试验，禁止跨论文数值比较。

## 统计、完整性与标签隔离

共同 S0 指经过全部模型张量验证的实际 BF16 merged 端点；S0 推理和所有后续分支共用该端点的同一身份。合并前 adapter 与实际 merged 的概率差异必须保留实测，不能称为精确函数等价。初拟 `.03/.25` 仅作合并概率 gap 诊断；当前端点须有自己的完整 tensor 证明与固定参考不变证据，历史失败诊断不能替代新端点证明。技术格式预热和并发检查预算单列于 candidate 的 `technical_budget`，不计入正式 256 步各组曝光。

预测每 run 写新文件。期望 `request_id` 由冻结 eval 窗口池给出；结果 ID 必须字符串、唯一、集合完全一致，缺失/额外/重复均失败；非法回答保留记录并计入主指标。真实标签打开前应保存六组预测 SHA256、每视频片段覆盖收据、窗口格式合法率和代码 revision。

主配对差异为同一视频 `G-RANK - S+`。视频没有重复来源时，逐视频差异平均；同一 `source_group` 有多个视频时，先组内平均差异，再给每组同等权重。seed=42、10000 次组 bootstrap、95% percentile 区间；并列报告视频宏平均与来源组平均、有效来源组数。窗口和输入曝光不是统计独立样本。单训练 seed 的区间不包括训练随机性。

没有使用测试标签选择 prompt、奖励、K、checkpoint 或分支预算。如以后使用这些标签做设计决定，该集合须在记录中改称开发证据，并保留新的最终测试来源。

## 本机验收状态

实现：`src/unsgrpo/evaluation.py`；手算和失败检查：`tests/test_evaluation.py`。32 例原 UMT 评估方法合成复算已通过；pytest 的本机结果由父流程运行日志记录，不能引用原方案中的“33 项”作为本轮成功记录。尚未完成真实视频、标签时间轴和六组预测评价。

## 评价文件接口

冻结文件（输出路径建议仓库外）：

```json
{"require_run_provenance":true,"manifest":{"path":"/absolute/prepared.jsonl","sha256":"64位实际哈希"},"runs":{"B0":{"path":"/absolute/B0.jsonl","sha256":"64位实际哈希","receipt":{"path":"/absolute/B0/receipt.json","sha256":"64位实际哈希"}},"B1":{},"S0":{},"S+":{},"G-CAL":{},"G-RANK":{}}}
```

示例空对象须补为实际路径和哈希，缺一分支即失败。prepared 每个 test video 必须有 `video_id,source_group,split,duration,n_frames,frame_bounds_path,bounds`；预测每视频一行 `{video_id,bounds,scores}`，bounds 须与 prepared 的 2 秒片段逐项一致。每个分支必须完整覆盖相同 test 集合。

```bash
python scripts/convert_tvsum_labels.py --mat /file-storage/labels_eval_only/ydata-tvsum50.mat --freeze /runs/predictions/freeze.json --out /file-storage/labels_eval_only/converted
python scripts/evaluate.py --freeze /runs/predictions/freeze.json --annotations /file-storage/labels_eval_only/converted --out /runs/reports/six_run_v0.json
```

两脚本在打开 MAT 或转换 NPZ 前重新验证六组预测、manifest 的哈希、来源组隔离、全视频覆盖与共同时间轴。正式 CLI 强制 `require_run_provenance=true`：各分支必须绑定已完成推理收据的路径和哈希、run ID、预测及 manifest 哈希；训练分支必须再绑定已验收的正式训练收据及磁盘端点哈希，且成功更新/曝光/rollout 与 256/4096 或 256/1024/4096 固定预算一致。四个训练端点推理须共用同一 S0 身份。仅测试函数可显式 `allow_synthetic_fixture=True`，其转换与评价结果带 `synthetic_fixture_only=true`；CLI 不提供此绕过参数。转换收据绑定 freeze/manifest/native PTS 文件/转换标签哈希；评价重新核验该绑定。标签与原视频不提交代码仓库。单纯创建一个 freeze 文件不代表完成评价，所有值必须来自实际预测且通过校验。
