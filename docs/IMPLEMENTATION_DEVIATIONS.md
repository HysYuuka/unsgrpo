# 首轮实施差异（冻结前）

1. 交接所述 starter 在本机缺失，因此按原方案重建最小实现。本机测试和真实验收单独保存，未沿用原文的成功声明。
2. 仓库外 venv 使用 `--system-site-packages` 复用已安装 PyTorch 2.8.0+cu128 和基础包；未改原环境。候选依赖离线安装，`pip check` 通过；完整版本见 `configs/environment.freeze.txt`。
3. 官方 ms-swift 4.6.0 的组标准差为 `correction=1`，优势分母加 `1e-4`。正式配置采用实际框架行为，独立数学参考函数保留原文 `correction=0`，不声称二者完全相同。
4. 实际框架会为每个 completion 重写 request_id，因此另外保存稳定 `dataset_request_id` 与采样池 `pool_instance_id`。有放回窗口可有相同窗口 ID；池实例必须唯一且媒体、q、提示内容一致。
5. 真实视频预处理发现目标时刻可能落在末帧的持有区间，目标之后没有下一帧。此时使用覆盖目标的真实末帧，保留实际 PTS，记录每个 `terminal_frame_holds` 和绝对误差，仍执行 0.5 秒误差上限。其他目标保持目标之后第一帧的规则。不删视频、不制造未来时间戳；所有组共用该输入。
6. V0 固定为真实逐帧 PTS 对齐、用户分数严格 `> median` 二值化、非插值 AP，先用户再视频平均。V1 的真实片段和参考划分尚未完整对齐，保持 BLOCKED；首轮 V0 不作跨论文比较。
7. Qwen 经官方 ModelScope 镜像、CLIP 经 HF 镜像传输，完整权重均与 Hugging Face 固定 revision 的原始 LFS SHA256 一致；配置、processor、chat template 与版本哈希保留。

8. 实际 Swift4.6 K3 对 ref-current log ratio 限制为 [-20,20]，再将 exp(d)-d-1 限制为 [-10,10]；这不同于独立参考函数的无裁剪版本。配置登记该安全处理，实际 loss 审计记录其是否触发。
9. Swift4.6 的 SFT/processor 接受视频帧路径列表，但 rollout 的 dacite 转换把 `videos` 注解限制为 `List[str]`。版本限定的 `swift_compat.py` 仅扩展 rollout 字段为字符串视频或字符串帧列表，保留其余类型检查、请求内容和数学行为；修复前的真实失败日志保留。
10. 初拟 BF16 合并平均 logp 差阈值 0.03 未通过（实测均值 0.0462009，最大值 0.233134；greedy 相同）。原失败不改写。全量 625 个张量逐位诊断确认合并严格正确，采用“实际 BF16 量化合并端点”为共同 S0，S0 本身和全部续训分支均使用该端点。验收依据为完整参数合并证明、输入一致和相对实际 merged 的固定参考不变；合并前后的函数差异作为独立诊断报告，不声称精确函数等价。

本文件列出的差异均发生在正式矩阵冻结之前；技术检查不计正式曝光。
