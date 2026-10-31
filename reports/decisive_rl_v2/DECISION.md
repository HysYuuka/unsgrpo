# decisive_rl_v2：NO_GO_REWARD

1. 新奖励未同时达到预定覆盖与候选选择收益门槛。
2. 本轮未训练，现有奖励证据不支持支付GRPO续训成本。
3. 停止当前教师与奖励配方，不进入独立验证。

全部10个来源为开发集，训练seed固定42。主指标为稳定V0、映射后小数12位并列规范；公开top-5为NOT_READY。不作独立测试、SOTA或所有RL方法的结论。

|方法|状态|V0 AP (%)|新增训练GPU秒|
|---|---|---:|---:|
|S0|HISTORICAL_SAME_V0|46.0970|0.0|
|G-CAL|HISTORICAL_SAME_V0|46.6978|0.0|
|P-DPO-80|NOT_TRAINED_NO_GO_REWARD|未训练/未完成评价|0.0|
|P-DPO-long|NOT_TRAINED_NO_GO_REWARD|未训练/未完成评价|0.0|
|G-plain|NOT_TRAINED_NO_GO_REWARD|未训练/未完成评价|0.0|
|G-guard|NOT_TRAINED_NO_GO_REWARD|未训练/未完成评价|0.0|

直接观测：同片段双顺序均以C唯一最高 7/8（要求≥6）；原始ABCD质量≥0.5的视图占 100.00%（要求≥90%）。名义32视图，实际24次前向，缓存8视图。

直接观测：59个全合法候选组，新G=0.043725pp，旧G=0.113356pp；覆盖与全部来源结果见teacher_quality.json。单窗替换收益不能相加。

保守GPU占用 382.57秒（含加载、进程首尾和失败尝试）；新增文件盘 16.354GiB，本地工作目录 0.0013GiB。下载等待单独记录，不计为GPU。未提交或推送Git。

解释边界：接口未通过只能说明此固定配置未满足操作性准入，不能推广为语义教师普遍无效；奖励未通过不能替代实际RL比较。单seed的来源bootstrap不包含训练随机性。分数压缩不能单独解释AP。

复核入口：results.csv、per_source.csv、teacher_quality.json、failure_cases.json、resource_receipt.json、protocol.actual.json。实际命令见工作目录commands.jsonl；输入/源码冻结见protocol.freeze.json；若已训练，固定调度与收据分别见schedule.json和文件盘training目录。

可复现执行：使用固定环境的Python，设置PYTHONPATH=src:scripts；先运行tests/test_decisive_rl.py，再按commands.jsonl中的有序命令运行。当前命名空间已结束，禁止覆盖重跑；重新执行须保留原始证据并使用新目录。
