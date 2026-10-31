# 首轮实验运行手册

本手册给出当前实现的可执行顺序和命令接口。实际完成状态以带哈希的运行收据、optimizer消费账本和固定终点为准；命令存在、CPU测试通过、技术smoke或训练器退出成功均不等于首轮已完成。

当前已核验资产：TVSum官方v1.1的50个完整视频，按来源组固定40/10划分；真实预处理保留全部50个视频、6316个片段和25264个采样帧。末帧PTS之后没有可选后继帧的11个终段采样使用明确记录的末帧保持，涉及10个视频；最大绝对采样误差0.041667秒。首个预处理失败尝试保留日志，418.76秒的成功逐视频时间之和不含失败尝试和哈希封存开销。具体证据见 `reports/real_preprocessing.json`，不是本手册中的静态状态声明。

## 环境和目录

```bash
source /root/autodl-tmp/unsgrpo/storage-env.sh
export PATH="$UNSGRPO_WORK_DIR/envs/highlights/bin:$PATH"
export PYTHONPATH="$UNSGRPO_PROJECT_DIR/src:$UNSGRPO_PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HL_DATA="$UNSGRPO_DATA_DIR/trainer_inputs"
export HL_OUTPUT="$UNSGRPO_CHECKPOINT_DIR"
export IMAGE_MAX_TOKEN_NUM=784
export VIDEO_MIN_TOKEN_NUM=64 VIDEO_MAX_TOKEN_NUM=64
export FPS=2 FPS_MAX_FRAMES=32 FPS_MIN_FRAMES=4
POLICY="$UNSGRPO_MODEL_DIR/Qwen3-VL-2B-Instruct"
CLIP="$UNSGRPO_MODEL_DIR/clip-vit-base-patch32"
PREPARED="$UNSGRPO_PREPARED_DIR/prepared.jsonl"
PSEUDO="$UNSGRPO_PSEUDO_DIR/r0_visual_v1/pseudo.json"
python -m pip check
python "$UNSGRPO_PROJECT_DIR/scripts/storage_audit.py"
```

`storage-env.sh` 的 `HL_DATA`、`HL_OUTPUT` 保留外部已设置值，默认指向仓库外的输入和检查点。基础模型及固定资产用200 GiB文件盘，环境/帧缓存及活跃训练用50 GiB本地盘；预算、峰值检查和小I/O探测见 `docs/STORAGE_AND_SYNC.md`。当前锁文件为 `configs/environment.freeze.txt`。实际Swift为4.6.0；PyTorch 2.8.0+cu128继承自系统，未重新下载GPU运行库。

## 正式矩阵的顺序执行器

`scripts/execute_round1.py` 已实现完整固定顺序：S0训练与真实账本验收、概率快照、合并与processor恢复、原容差gap诊断、全张量BF16逐位验证、共同参考完整性检查、登记共同S0、三个续训分支及各自验收、六组独立推理、冻结预测、最后才解压MAT并评价、校验归档、最终磁盘快照和中文报告。GPU任务逐个子进程执行，项目共享文件锁阻止两个执行器并行调度GPU；脚本没有Git推送或定时任务。

```bash
# 默认仅输出可审计划；不创建运行状态、不启动GPU、不解压标注。
python "$UNSGRPO_PROJECT_DIR/scripts/execute_round1.py" --plan \
  --state-dir "$UNSGRPO_RUN_DIR/round1_seed42_v1"

# 审核后，必须已有真实FROZEN协议和完整GPU技术验收才能执行。
python "$UNSGRPO_PROJECT_DIR/scripts/execute_round1.py" --execute \
  --state-dir "$UNSGRPO_RUN_DIR/round1_seed42_v1"
```

执行前，协议`files`须冻结执行器和全部被调用代码、配置、数据、输入审计，以及官方原始MAT zip的SHA256。读取zip字节哈希不等于解析标注值；首次解压严格位于六组预测冻结之后。当前实际共同S0定义为已逐位验证的BF16量化合并模型，原0.03/0.25容差仅作保留失败记录的gap诊断，不能静默提高阈值。

所有阶段的原始argv、UTC/香港时间、子进程PID、退出码、墙钟、日志与逐产物SHA256保存在仓库外`state-dir/state.json`、`events.jsonl`和`logs/`。内部Python登记动作记录执行器源码哈希。`--resume --execute`只跳过状态为COMPLETED且真实字节哈希再次一致的阶段；任何FAILED或中断的RUNNING阶段均拒绝自动重启。它是控制器状态恢复，不是训练精确resume，也不会自动延长或重跑失败训练。只有真实报告脚本返回COMPLETED，控制器才记COMPLETED。

评价完成后，`archive_round1.py`把实际共同merged模型、四个最终checkpoint的恢复状态、日志/预测/评价JSON和轻量收据复制到`/root/autodl-fs/unsgrpo/archives/<prefix>`。复制前估计峰值，检查50 GiB归档预算和20 GiB文件盘余量；复制后逐文件核对SHA256。原视频、人工标注内容、帧缓存和环境不复制；本地原文件不删除不改动，既有路径绑定继续有效。归档代价单列，不计入方法训练成本。最终磁盘审计和报告在归档完成之后生成；归档失败则保留失败收据并停止，不自动重试或清理旧技术端点。

## 从资产重建输入

下列是复现命令，不应覆盖当前已封存产物。重跑时使用新的版本目录，并更新所有下游哈希；若已冻结，则先登记受影响的整组比较，不可只重跑有利分支。

```bash
python "$UNSGRPO_PROJECT_DIR/scripts/prepare_assets.py" \
  --archive "$UNSGRPO_DATA_DIR/tvsum50_ver_1_1.tgz" \
  --out "$UNSGRPO_DATA_DIR/tvsum" \
  --manifest "$UNSGRPO_MANIFEST_DIR/tvsum_seed42.jsonl" --seed 42

python "$UNSGRPO_PROJECT_DIR/scripts/prepare_visual.py" \
  --manifest "$UNSGRPO_MANIFEST_DIR/tvsum_seed42.jsonl" \
  --out "$UNSGRPO_PREPARED_DIR" --encoder "$CLIP" --device cuda

python "$UNSGRPO_PROJECT_DIR/scripts/make_pseudo.py" \
  --manifest "$PREPARED" --out "$UNSGRPO_PSEUDO_DIR/r0_visual_v1" --seed 42 \
  --report "$UNSGRPO_PROJECT_DIR/reports/proxy_real_diagnostics.json"

python "$UNSGRPO_PROJECT_DIR/scripts/make_examples.py" \
  --manifest "$PREPARED" --pseudo "$PSEUDO" --out "$HL_DATA" \
  --windows-per-video 32 --seed 42 \
  --report "$UNSGRPO_PROJECT_DIR/reports/data_audit.json"
```

UMAP只拟合训练视频，`n_neighbors=min(15,N_train-1)`、`n_components=min(5,N_train-2)`，固定seed与transform_seed。KMeans搜索4到`min(15,N_train-1)`，按训练轮廓系数选簇数；测试仅transform/predict。不提供静默PCA回退。伪分数在完整视频计算一次，训练池每视频32个有放回窗口实例；SFT/GRPO共同`dataset_request_id`表示同一窗口，`pool_instance_id`区分抽样实例。Swift生成时会覆盖`request_id`为completion UUID，消费账本必须使用稳定dataset ID。

```bash
python "$UNSGRPO_PROJECT_DIR/scripts/audit_swift.py" --model "$POLICY" \
  --inputs "$HL_DATA/train_sft.jsonl" "$HL_DATA/train_grpo.jsonl" \
           "$HL_DATA/eval.jsonl" \
  --out "$UNSGRPO_PROJECT_DIR/reports/input_audit.json" \
  --rows-out "$UNSGRPO_WORK_DIR/audit_rows_v1.jsonl" \
  --preview-dir "$UNSGRPO_WORK_DIR/previews"
```

正式训练前检查所有实际编码行、assistant监督范围、真实媒体张量哈希、帧数和grid以及`prompt+completion <=8192`。超长直接失败。`strict=true`不能代替预检；GRPO实际CLI仍显式登记左截断策略，前置长度门槛保证不触发它。至少人工查看processor输出的概览和局部帧，记录小动作/文字是否可辨。当前视觉检查已记录局部低分辨率的限制，未据测试标签改变预算。

## 技术闭环与冻结

当前预冻结技术检查采用同8个真实R0窗口的SFT **32次、batch=1** 冷启动，再运行`PROMPTS=1,G=4,steps=4`的GRPO闭环，以及至少2次`PROMPTS=4,G=4`的正式16-completion并发检查。此前全部非法回答、奖励−1、组方差0、梯度0的尝试保留为真实零信号失败证据，不能冒充有用更新。这个技术冷启动调整不改变正式矩阵的256/16与256/4/G4预算。每次用全新`HL_RUN_ID`，下列ID只是命名示例。

```bash
HL_RUN_ID=smoke_sft_new HL_STEPS=32 HL_SFT_BATCH=1 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" smoke-sft0 "$POLICY"

# SMOKE_MERGED必须是实际合并并通过processor恢复的smoke端点。
HL_RUN_ID=smoke_grpo_new HL_PROMPTS=1 HL_GROUP_SIZE=4 HL_STEPS=4 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" smoke-grpo "${SMOKE_MERGED:?填写实际smoke合并端点}"
```

`verify_checkpoint.py snapshot`记录固定训练答案逐token概率及greedy输出；`compare-merged`测量合并前后概率；`reference-update`验证新LoRA、固定参考、参数更新与保存重载。后一项使用一次CE更新做完整性检查，**不证明真实GRPO训练完成**。原平均绝对logp差0.03/最大差0.25容差保持不变，但属于gap诊断。技术检查发现它未通过，而全部625张量均满足准确BF16合并规则，因此预先把共同S0明确登记为实际量化merged，并保留原FAILED及数值差异；不声称合并前后函数等价。`verify_bf16_merge.py`独立验证全部适配权重等于BF16(base+2BA)、非LoRA张量逐位不变，另需正确processor和更新后固定参考不变。`run.sh merge`调用`restore_processor.py`恢复原始processor完整包并重验固定训练输入。

```bash
python "$UNSGRPO_PROJECT_DIR/scripts/freeze_round1.py" \
  --sft-smoke "${SFT_SMOKE_RUN:?实际SFT技术运行目录}" \
  --grpo-smoke "${GRPO_SMOKE_RUN:?实际GRPO技术运行目录}" \
  --formal-concurrency "${CONCURRENCY_RUN:?实际正式并发检查目录}" \
  --input-audit "$UNSGRPO_PROJECT_DIR/reports/input_audit.json" \
  --snapshot "${SFT_SNAPSHOT:?合并前GPU概率快照JSON}" \
  --merged-compare "${MERGED_COMPARE:?合并后GPU概率比较JSON}" \
  --reference-update "${REFERENCE_CHECK:?固定参考更新验收JSON}" \
  --grpo-reload "${GRPO_RELOAD:?GRPO保存重载验收JSON}" \
  --split "$UNSGRPO_MANIFEST_DIR/tvsum_seed42.jsonl" \
  --prepared "$PREPARED" --pseudo "$PSEUDO" \
  --extra-freeze "$UNSGRPO_DATA_DIR/tvsum/labels_eval_only/source/ydata-tvsum50-matlab.zip"
```

冻结脚本从真实产物重算门槛，缺项输出`INCOMPLETE`。不得手填布尔值跳过门槛。正式GRPO采用Swift实测样本标准差`correction=1`；数学独立参考helper采用总体标准差，二者差异已预登记，不混称完全相同。

## 正式六组与共同S0

矩阵固定：B0直接代理、B1原始Qwen零样本、S0共同SFT、S+继续SFT、G-CAL、G-RANK。S0恰256次成功更新×16输入=4096曝光；后三分支各256×4=1024曝光，GRPO各生成并消费4096条回答。实际optimizer账本才是这些数字的依据，池大小和计划值不是曝光实测值。

```bash
HL_RUN_ID=S0_seed42_v1 HL_STEPS=256 HL_SFT_BATCH=16 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" sft0 "$POLICY"

# 正式S0端点先做snapshot，合并、恢复processor，再测compare-merged的原始gap。
HL_RUN_ID=merge_S0_seed42_v1 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" merge "$POLICY" \
  "${S0_ADAPTER:?实际256步S0 adapter端点}"

# 随后必须完整bitwise验证+正式共同参考检查，再登记common_s0.json。
# 原compare仅特定容差FAILED可保留后继续独立验证；其他失败均停止。
# COMMON_S0必须来自已验收的实际BF16合并模型，三分支新建同规格LoRA。
HL_RUN_ID=Splus_seed42_v1 HL_PROMPTS=4 HL_STEPS=256 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" sft-continue \
  "${COMMON_S0:?实际共同合并S0目录}"
HL_RUN_ID=G_CAL_seed42_v1 HL_PROMPTS=4 HL_GROUP_SIZE=4 HL_STEPS=256 HL_RANK_WEIGHT=0 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" grpo "$COMMON_S0"
HL_RUN_ID=G_RANK_seed42_v1 HL_PROMPTS=4 HL_GROUP_SIZE=4 HL_STEPS=256 HL_RANK_WEIGHT=0.8 \
  bash "$UNSGRPO_PROJECT_DIR/scripts/run.sh" grpo "$COMMON_S0"
```

`run.sh`返回`TRAINER_EXITED_SUCCESS_REQUIRES_AUDIT`时不能直接改成完成。需核对真实`training_ledger.jsonl`、`training_summary.json`、GRPO奖励账本、有效token、成功更新、保存的端点及实际GPU成本，才能把该次正式训练收据登记为`COMPLETED`。后续分支的起始收据绑定`common_s0_sha256`；只给共同目录路径不够。

正式G-CAL/G-RANK的验收显式允许`--allow-formal-zero-signal`这一严格负结果分支：只有完整固定预算、正学习率、全程组奖励标准差/优势/梯度/KL均为0且初末参数完全不变的原账本证据全部通过，才登记`negative_outcome=ALL_GROUP_ADVANTAGES_ZERO_NO_POLICY_UPDATE`。任意未变LoRA或“存在奖励方差但梯度为0”仍失败；SFT和技术非零训练信号门槛不放宽。负结果保留真实成本和评价，不自动加步或重训。

所有训练端点推理使用同一已审计`eval.jsonl`、原视频全覆盖窗口、greedy和128输出token。每次`inference.py --out`必须全新目录；非法JSON不修补，用全零窗口并计入重叠分母。合法完整JSON恰在长度上限结束仍合法，长度终止比例另报。推理收据记录模型/adapter逐文件哈希、共同S0、输入审计与视觉环境。

```bash
python "$UNSGRPO_PROJECT_DIR/scripts/inference.py" --run-id B0 \
  --inputs "$HL_DATA/eval.jsonl" --manifest "$PREPARED" --pseudo "$PSEUDO" \
  --out "$UNSGRPO_RUN_DIR/inference_B0_v1"
python "$UNSGRPO_PROJECT_DIR/scripts/inference.py" --run-id B1 --model "$POLICY" \
  --inputs "$HL_DATA/eval.jsonl" --manifest "$PREPARED" --pseudo "$PSEUDO" \
  --out "$UNSGRPO_RUN_DIR/inference_B1_v1"
# S0用共同合并模型、无adapter；S+/G-CAL/G-RANK另传对应正式终点--adapter。
# 四个已训练方法均需--training-receipt指向其COMPLETED正式训练收据。
```

## 冻结预测、评价和报告

全部六组独立预测文件及SHA256写入预测冻结JSON，设置`require_run_provenance=true`，每分支绑定真实推理/正式训练收据：`{manifest:{path,sha256},runs:{方法:{path,sha256,...真实运行归因}}}`。`validate_six_run_freeze`逐视频验完整时间轴、视频集合、数值和真实六组归因后，才允许解包/读取评价标签。正式报告还要求评价结果`require_run_provenance=true`、`synthetic_fixture_only=false`；单测fixture不能充当首轮结果。训练脚本不接收标注路径。

```bash
python "$UNSGRPO_PROJECT_DIR/scripts/convert_tvsum_labels.py" \
  --mat "${TVSUM_MAT:?隔离目录中的官方MAT}" \
  --freeze "${PREDICTION_FREEZE:?六组预测冻结JSON}" \
  --out "$UNSGRPO_EVAL_LABEL_DIR/converted_v1"
python "$UNSGRPO_PROJECT_DIR/scripts/evaluate.py" \
  --freeze "$PREDICTION_FREEZE" --annotations "$UNSGRPO_EVAL_LABEL_DIR/converted_v1" \
  --out "$UNSGRPO_RUN_DIR/evaluation_V0_v1.json"
python "$UNSGRPO_PROJECT_DIR/scripts/report_round1.py" \
  --index "$UNSGRPO_PROJECT_DIR/reports/round1_artifacts.json" \
  --out-md "$UNSGRPO_PROJECT_DIR/reports/ROUND1_RESULTS_v1.md" \
  --out-json "$UNSGRPO_PROJECT_DIR/reports/round1_results_v1.json"
```

产物索引的每个文件引用均为`{path,sha256}`，相对路径相对于索引目录。必需键为`protocol_freeze`、`prediction_freeze`、`evaluation`、`common_s0`和六组`runs`。每组有`inference_receipt`；四个训练组另有`training_receipt`、`telemetry_summary`、`training_ledger`、`endpoint:{path,files:{相对文件:sha256}}`，两个GRPO组另有`rollout_ledger`。可选加入`proxy_diagnostics`、`storage_audit`、`sync_receipt`和`qualitative_failures`。报告验证真实账本、实际端点、全部冻结文件、共同S0及评价哈希；缺项写`INCOMPLETE`并退出2，不补假指标，不用smoke替代256步。

V0仅为内部诊断，报告逐视频AP、来源组等权的配对bootstrap主差值、合法率、代理奖励、跨窗差异、常数输出、组奖励方差及成本。V1虽有来源绑定的指标组件，真实benchmark时间轴和参考split尚未对齐，禁止跨论文比较。一个训练seed的测试视频bootstrap不包含训练随机性。若依本轮测试改方法，下一阶段须保留独立来源。

## 中断与按需同步

预处理的`--resume`只复用逐文件哈希验证的完整视频收据；`--seal-existing`只用于冻结前首次捕获哈希，不能宣称此前内容未变。训练目前没有验证优化器、scheduler、随机状态和数据游标的完整恢复；仅重载adapter属于权重恢复，不是精确resume。不得把两段不同采样顺序的日志合并成一个原样续训实验。故障先保留证据，恢复原配置或为受影响的完整比较重新登记。

用户已取消自动每日同步。**不设置任何同步cron或其他定时器，不主动提交/推送；仅在用户每次明确要求同步时调用手动脚本。** 当前提交身份和写权限缺失仅记录，不反复催问。`scripts/sync_daily.sh`保留旧文件名，默认`--check`为本地审查；`--sync`必须有该次用户明确要求。凭据、视频、标注内容、模型、环境和大检查点始终留在仓库外。
