# 奖励候选辨别力与短程RL：REWARD_NOT_ALIGNED

候选确有改善空间，但新旧奖励均未达到预定辨别力准入，结束当前配方、不启动训练。旧奖励存在弱正向选择信号，不能把停止决定解释为奖励完全没有信息。

香港时间：2026-10-29T11:29:09.520515+08:00。全部10个来源均为既有开发数据。主指标为第二轮稳定V0、12位小数并列规范；公开top-5仍NOT_READY，不作论文benchmark比较。

## 实际执行与边界

状态：REWARD_NOT_ALIGNED。累计GPU保守占用645.51秒，上限21600秒；首次筛查上限3600秒。所有GPU加载、失败和进程首尾均记账。CPU/模型训练没有读取训练人工标签。旧的更新策略rollout未被用作新S0候选。

开发候选：240个新增S0回答，60个固定窗口；四个回答均合法的组为59/60。greedy缓存只作单独部署参照，没有塞入采样组。

偏好教师：720条双顺序边，保留22条，实际前向1440次；弃权原因完整记录为{'A_B_tie': 42, 'opposite_actual_clip_after_swap': 644, 'weight_below_0.2': 12}。教师只看到概览与两段各四帧，从未看到策略回答。

真实接口检查：24对（含8个同片段对），48次计分前向；实际模板同一位置单token检查结果为True。归一化A/B/C倾向不是校准正确率；只有同位置单token时才解释原始三项概率质量。

## 候选是否有空间，奖励是否选得更好

主分析：59个四回答全合法组、10个来源。达到H极差0.1个百分点的组为59，覆盖10来源。平均人工最优空间O=0.40263个百分点，人工最优相对greedy=0.34853个百分点。信息量准入：True。

|奖励|辨别准确率|选择增益G（百分点）|G/O|选择−greedy（百分点）|基础准入|
|---|---:|---:|---:|---:|---|
|R_old|53.12%|0.11336|28.15%|0.05925|False|
|R_pair|53.13%|0.04952|12.30%|-0.00458|False|

固定选择结果：None。这是开发决策，不能写成独立验证。每个H都只替换一个窗口，并保留同视频其余S0窗口、完整重叠聚合和完整视频V0。窗口收益不可相加。奖励并列按12位浮点规范取H均值，未用人工最优破并列。

全部组（含非法零分窗口）的次分析、逐窗口四个H、逐用户AP和来源bootstrap区间见screen_result.json、screen_per_window.csv及文件存储screen/single_window_full_video_AP.jsonl。主分析与次分析均保留，没有删除不利视频。辨别准确率只用H差至少0.1个百分点的同一批候选对，奖励同分计半。每个窗口先平均，再视频平均、来源平均；区间只重采样来源。

候选人工最优只用于诊断，没有成为训练样本、教师修正或RSFT目标。RSFT只在训练视频的另一个冻结S0候选池中按选定奖励挑选。

## 固定80步端点

|方法|seed|实际状态|开发V0 AP|
|---|---:|---|---:|
|S0|42|CACHED_REEVALUATED|46.0970%|
|G-CAL|42|CACHED_REEVALUATED|46.6978%|
|G_plain|42|NOT_RUN_REWARD_NOT_ALIGNED|未评价|
|G_guard|42|NOT_RUN_REWARD_NOT_ALIGNED|未评价|
|RSFT|42|NOT_RUN_REWARD_NOT_ALIGNED|未评价|
|G_plain|43|NOT_RUN_REWARD_NOT_ALIGNED|未评价|
|G_guard|43|NOT_RUN_REWARD_NOT_ALIGNED|未评价|
|RSFT|43|NOT_RUN_REWARD_NOT_ALIGNED|未评价|

历史G-CAL训练预算不同，仅为实用参照。三个分支若执行则共享最终池、320提示曝光、共同S0、新LoRA及固定端点；在线探索和计算量不同，不能写成严格等算力。G_guard在R_old路线u=1，只检验标准差下限。技术adapter与正式adapter分开；是否通过4步实际接入见实际receipt，而非CPU测试推断。

逐视频/逐seed结果和所有已完成的对照区间保留在CSV、endpoint_contrasts.json。未运行分支为空值及原因，未用零分冒充。只有seed42达到固定增益及至少6/10正向来源门槛且剩余预算足够，才重复seed43全部三分支。共同S0只有一个，重复仅衡量续训随机性。

## 直接观测、解释与未验证事项

直接观测限于真实生成的候选、条件概率、单窗替换AP和已完成的训练端点。奖励选择增益表示当前候选池中的辨别力，不保证GRPO一定学得到；没有信息量时不能归罪教师。

未完成技术接入或端点时，不能声称已检验可靠更新；RSFT同样好时，不能把监督信息收益归为RL特有收益。位置交换一致只表示稳定，不证明教师方向正确。没有新数据、全量伪标签、全视频比较图、JSON事件教师或超参数扫描。

## 复现与文件

冻结协议在protocol.json；实际源和输入SHA在工作目录screen_protocol.freeze.json，训练若执行另有training.freeze.json。所有候选/奖励先冻结，再由独立CPU评价进程读取10来源标签。最终artifact_index和资源收据记录实际路径与字节；大型预测与adapter留在工作/文件存储，不进入仓库。

原始完整执行命令：

```bash
export PYTHONPATH=/root/autodl-tmp/unsgrpo/src:/root/autodl-tmp/unsgrpo/scripts
/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python -m pytest /root/autodl-tmp/unsgrpo/tests/test_reward_to_rl.py -q
/root/autodl-tmp/unsgrpo-work/envs/highlights/bin/python /root/autodl-tmp/unsgrpo/scripts/reward_to_rl_prepare.py
screen -dmS unsgrpo-reward-to-rl /bin/bash /root/autodl-tmp/unsgrpo/scripts/reward_to_rl_detached.sh
```

上述入口拒绝覆盖已有结果；不能删除失败证据后重跑并称同一次实验。全部实际阶段命令、采样seed、提示哈希、token IDs、损失分量、u、参考logps、梯度、优化器调用、耗时和内存保留于阶段日志与JSONL。未进入训练时相应账本明确不存在。
