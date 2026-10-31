# 无人工高光标注的视频高光检测：Qwen + SFT + GRPO 实验方案与数学推导

**版本：v1.0｜日期：2026-10-27｜定位：硕士研究的可执行首轮基线，而不是已完成的论文方法**

**任务边界：无文本查询、无人工高光训练标签、离线完整视频输入、逐片段高光检测。第一轮不做空间裁剪、目标跟踪、镜头重排、叙事剪辑或音频生成。**

配套代码目录：`highlight_grpo_starter/`。本文可以独立阅读；代码包用于避免手工复制长代码时出错。代码不包含原始视频、标注、模型权重、个人凭据或用户旧实验。

### 阅读导航

| 研究与数学 | 执行与验收 |
|---|---|
| [0 创新判断与可证明边界](#section-0) | [9 实验矩阵与曝光预算](#section-9) |
| [1—3 文献、任务与无监督边界](#section-1) | [10 环境、数据与单样本检查](#section-10) |
| [4—5 视频输入、聚类与伪标签](#section-4) | [11 SFT、合并和 GRPO 命令](#section-11) |
| [6—8 SFT、奖励与 GRPO 推导](#section-6) | [12—13 推理与独立评价](#section-12) |
| [14—15 结果判断、条件性定理与反例](#section-14) | [16—17 关键代码与执行交接](#section-16) |
| [18 原论文和官方来源](#section-18) | [19 已执行测试与未验证范围](#section-19) |

先执行可从第 9—11 节进入；检查研究逻辑优先读第 0、7、14—15 节。文件内的超参数均为本方案的首轮设定，不是已验证的最优值。

---

<a id="section-0"></a>

## 0. 先回答：创新是否太少？数学能够证明到哪里？

### 0.1 对创新性的直接判断

仅将“已有聚类伪标签 + Qwen + SFT + 标准 GRPO”串起来，**方法层面的创新偏弱**。这不意味着不值得做：它是一条合理的实验起点，可以先查明三个问题。

1. 聚类与跨视频重复性形成的代理信号，对当前数据是否有效？
2. 相同信号下，Qwen 的训练能否超过直接使用代理分数？
3. 从同一 SFT 端点出发，GRPO 是否优于继续做伪标签 SFT，而不只是消耗更多计算？

第一轮的目的就是得到这三项答案。不要先声称“首次无监督 GRPO 高光检测”，也不要把一次正结果等同于毕业或录用保证。本文没有完成全面的新颖性检索，不提供“无人做过”的结论。

**建议：现在可以搭流程；把首轮结果视为研究筛选依据，再从具体失败机制中形成方法贡献。**首轮不额外发明置信度 GRPO，不把跟踪、音频理解、事件图全部绑进来。

### 0.2 必须区分三种“可行”

| 层次 | 本文可以给出的证据 | 不能冒充的结论 |
|---|---|---|
| 数学定义可行 | 伪奖励可计算、有界；随机生成策略可定义；策略梯度不要求奖励可微；若干等价化简与条件性定理 | 不能据此断言人类高光指标必然提高 |
| 工程实现可行 | 数据接口、脚本、奖励插件、训练配置、单元测试和检查步骤 | 本次没有在真实 Qwen 权重和 GPU 上跑通训练，不声称实测显存或吞吐 |
| 科研效果成立 | 需要独立标注评价、多随机种子、强对照和失败分析 | 不能用代理奖励升高替代真实高光效果，不能保证超过 SFT |

**不存在一份不依赖数据假设的证明，能从“常见模式”推出“人类认为精彩”，再推出“非凸神经网络上的 GRPO 必然优于 SFT”。**第 15 节给出明确反例、可证明的条件性结论，以及这些结论在实际实验中的边界。这不是省略证明，而是避免把错误命题写成定理。

### 0.3 第一轮固定决策

| 项目 | 首轮取值 |
|---|---|
| 策略基座 | `Qwen/Qwen3-VL-2B-Instruct`，语言侧 LoRA |
| 任务 | 无查询、逐片段打分；不输出长推理链 |
| 输入 | 全视频 16 格概览图 + 当前最多 8 个连续片段的实际视频帧 |
| 时间单元 | 2 秒；最后一个片段保留真实剩余时长 |
| 每片段采样 | 4 帧，记录真实 PTS；常规完整片段约 2 fps |
| 窗口 | 8 片段，步长 4；覆盖不足 8 的短视频不伪造片段 |
| 输出 | `{"scores":[整数,...]}`，每个分数在 0—100 |
| 伪信号 | 固定视觉特征上的视频聚类与类内重复性；先使用视觉版 |
| SFT 标签 | 全视频尺度的伪分数，量化到 0—100 |
| GRPO | 每个完全相同的输入采样 4 个回答；标准组相对优势与 token 级裁剪目标 |
| 主奖励 | 0.8 × 伪排序一致性 + 0.2 × 全视频尺度校准；非法回答为 −1 |
| 主要对照 | 直接代理、SFT 端点、继续 SFT、校准奖励 GRPO、排序+校准 GRPO |
| 首轮目的 | 确认接口与机制；不能以一次小数据集结果作为完整毕业成果 |

这些取值是本方案的**预设起点，不是论文公认的最优超参数**。

---

<a id="section-1"></a>

## 1. 文献依据、借用部分与新增设计

### 1.1 借用的原方法

《Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence》的思路是：先聚类视频生成伪类别，再计算同类视频片段的视听平均余弦相似度，构造伪高光监督，训练高光网络。原文采用 UMAP、K-means、轮廓系数选择簇数，最终取每段视频前 50% 片段形成二值训练标签；它不是 GRPO 方法。[S1]

本文复用的是**“数据内部重复性提供高光代理信号”这一思路**，不是宣称完整复现其网络、特征与结果。

### 1.2 本文与原方法的差别

| 部件 | 原方法的相关做法 | 本方案首轮 |
|---|---|---|
| 模态 | 视听特征及融合 | 先仅视觉，避免奖励知道声音而策略听不到 |
| 特征 | 原文指定的视觉/音频编码器 | 提供冻结 CLIP 视觉特征的可操作实现；属于特征替换 |
| 监督 | top-50% 二值伪标签 | 保留连续信息，映射为全视频百分位分数 |
| 检测模型 | 专用视听网络 | Qwen3-VL-2B + LoRA |
| 优化 | 二元交叉熵 | 伪标签 SFT 后，再比较 SFT 续训和 GRPO |
| 伪标签测试 | 使用训练得到的簇及训练参照库 | 同样要求测试视频只做 transform/predict，不参与拟合 |
| 额外诊断 | 不据此猜测作者实现细节 | 可选整视频排除、视频等权的 LOO 版本，明确标注为改造 |

**因此，CLIP 版结果不能直接叫作“原论文复现结果”，也不能不核对特征、split 和评价脚本就与原表格数值进行同条件比较。**本次未核实到该方法完整、可直接执行的作者训练仓库；配套代码是依据公式编写的参考实现，不冒充作者代码。

### 1.3 工具选择

Qwen3-VL-2B-Instruct 的官方发布包含视觉/视频能力。训练命令以已核对的 `ms-swift v4.6.0` 文档和插件接口为依据。[S2][S3][S4][S5]

**不默认使用 TRL 的某个当前默认值就等于原始 GRPO。**本文显式设置优势估计、奖励缩放、损失类型、参考约束与采样参数。不同框架和版本的标准差修正、长度归一化、截断过滤可能不同，必须保存实际配置。[S3][S6]

---

<a id="section-2"></a>

## 2. 任务定义与统一符号

训练数据是无人工高光标签的视频集合：

$$
\mathcal D_{\mathrm{tr}}=\{V_n\}_{n=1}^{N}.
$$

视频 $V_n$ 的时长为 $D_n$，划成 $T_n=\lceil D_n/\Delta\rceil$ 个片段，$\Delta=2$ 秒：

$$
I_{n,t}=[(t-1)\Delta,\min(t\Delta,D_n)),\quad t=1,\ldots,T_n.
$$

每段均对应固定 ID、起止时间和实际采样帧。模型最终为整段视频输出分数向量：

$$
\hat{\mathbf s}_n=(\hat s_{n,1},\ldots,\hat s_{n,T_n}).
$$

| 符号 | 含义 |
|---|---|
| $z^v_{n,t}$ | 片段的单位化视觉特征 |
| $z^a_{n,t}$ | 可选音频特征，首轮不用 |
| $c_n$ | 视频伪类别 |
| $\rho_{n,t}$ | 重复性原始分数 |
| $q_{n,t}$ | 以完整视频为单位计算的伪百分位分数 |
| $X$ | 一条实际模型输入：概览、窗口帧、片段索引和提示 |
| $M$ | 当前窗口片段数，最多 8 |
| $Y$ | 一条模型生成的完整文本回答 |
| $\mathbf s(Y)$ | 解析回答得到的 $M$ 个分数 |
| $G$ | 同一输入的 rollout 数，默认 4；不是最终选中的片段数 |
| $\pi_\theta$ | 当前策略 |
| $\pi_{\mathrm{old}}$ | 本轮 rollout 时的行为策略 |
| $\pi_{\mathrm{ref}}$ | 固定的 SFT 参考策略 |
| $R$ | 不读取人工高光标签的代理奖励 |
| $U$ | 未知的人类高光效用，仅用于讨论可证明条件 |

本文优化的是**窗口级预测**，再汇总为全视频检测分数。不能将它宣传为已经优化任意长度视频的全局叙事剪辑或任意片段集合的最优组合。

---

<a id="section-3"></a>

## 3. 无监督边界与数据隔离

### 3.1 本文使用的准确设定

> 高光任务训练阶段，不读取人工高光时间戳、重要性分数或人工偏好。允许使用通用预训练模型及从训练视频自动计算的伪标签。

这不等于从随机参数训练，也不意味着预训练模型没有外部监督。CLIP、Qwen 的先验需要披露。预训练是否见过某个测试视频通常不能完全审计，也必须写为限制，而不是保证零污染。

SFT 可以使用自动伪标签；“SFT”这一训练名称不等于“使用人工高光标签”。反过来，若先用人工高光 GT 做 SFT，再无标签 GRPO，整条方法就不是本文定义的无人工高光监督。

### 3.2 数据划分

**工程起跑：TVSum。**它规模较小，适合完整覆盖原视频、检查时间对齐和建立首轮对照；但小样本结果不能单独证明大规模泛化。数据入口见 [S7]。

**正式评价：尽量同时加入 YouTube Highlights 等无查询高光协议。**采用其可核验的原始 train/test 身份；拿不到原视频就如实记录，不用仅有特征缓存假装完成 Qwen 视频实验。

原重复性论文对 TVSum 使用重复的 80%/20% 随机划分并报告平均值。[S1] 可以采取以下两个分离流程：

- **首轮**：预先固定一个来源分组的 80/20 划分，只完成一次技术与方法试验。若查看测试标签后据此设计新方法，这个集合之后就属于开发证据。
- **正式研究**：预先登记多个划分和训练 seed，并保留未用于方法选择的独立测试来源。固定训练预算，不通过测试标签挑 checkpoint。

不要把“五次随机划分”写成“五折交叉验证”；它们不相同。重复随机划分中的测试视频可能重叠，不应把所有出现次数视为独立样本。

### 3.3 必须先按来源分组，再切窗口

`source_group` 代表同一原始视频、同一视频不同版本或已确认的近重复来源。一个 source_group 只能属于一个 split。**禁止同一原视频的前半放训练、后半放测试。**

`manifest.jsonl` 示例：

```json
{"video_id":"v0001","source_group":"source0001","split":"train","path":"/data/raw/v0001.mp4"}
{"video_id":"v0002","source_group":"source0002","split":"test","path":"/data/raw/v0002.mp4"}
```

训练输入中不加入人工类别、标题、查询、GT、高光成片说明或评估分数。人工类别若仅用于最后的分组报告，需要明确与聚类输入分离。

### 3.4 文件隔离与信息流审计

```text
project/
  manifests/                  # 来源、split、路径、哈希
  raw/                        # 原视频
  prepared/                   # 采样帧、时间轴、固定视觉特征
  pseudo/                     # 训练库及自动伪分数
  data/                       # train_sft / train_grpo / eval JSONL
  labels_eval_only/           # 仅评估脚本读取；训练脚本不接收此路径
  outputs/                    # 模型与收据
  reports/                    # 独立评价结果
```

`pseudo_scores` 可以作为 GRPO 的**顶层元数据**传给奖励函数，但不得拼进 `messages`。否则模型可以直接抄奖励答案。SFT 的伪目标只放 assistant 答案；GRPO 输入不得保留这条 assistant 答案。

**可证明的信息流性质**：若数据处理、聚类、伪标签、SFT/GRPO 和 checkpoint 选择均是 $(\mathcal D_{\mathrm{tr}},\theta_0,\text{随机种子})$ 的函数，不读取高光标签，则高光任务训练没有直接使用这些人工标签。这个命题只约束本次流程，不覆盖预训练污染或人为调参时的隐性信息。

---

<a id="section-4"></a>

## 4. 时间轴、视频采样与模型输入

### 4.1 采样规则

对片段 $[a,b)$，取四个目标时刻：

$$
\tau_j=a+(b-a)\frac{2j+1}{8},\qquad j=0,1,2,3.
$$

解码时以 PTS 为准，选目标时刻之后的第一帧，记录真实时间；不把所有视频写死为某个 FPS。原视频是 VFR 时，尤其不能仅用 `frame_index/fps` 替代时间轴。

配套 `prepare_visual.py` 保存采样时间并检查误差；若最大误差超过 0.5 秒直接报错。这个阈值是工程审计界限，不是说 0.5 秒误差对高光无害。正式对齐应进一步报告平均/最大误差及异常视频清单。

帧列表接口中的视频位置编码可能仍依据统一的名义采样率构造，而不是自动读取保存的每帧原始 PTS。最后不足两秒的片段尤其要核对；原始起止时间由 prompt 表格明确提供，不能宣称 processor 已自动继承全部原始时钟。

四帧不能保证覆盖所有极短动作。应保留“未观测到事件”的失败类别，而不是一律归因于模型理解能力。需要提高采样率时，所有比较组统一变更并重登记预算。

### 4.2 全局概览与局部窗口

从完整视频已采样帧中均匀选择 16 帧，构成 4×4 概览图，标注原始时间。当前窗口含最多 8 个连续片段，每片段 4 帧，共最多 32 帧，按原始顺序作为一个视频帧列表输入。

两类输入的作用不同：概览提供视频背景，窗口提供局部动作细节。概览不是事件真值，也不是文本教师。

Qwen3-VL 的多模态数据接口可接收视频，ms-swift 文档也列出了视频帧路径列表形式。[S2][S4] 首轮必须实际检查处理后的 `video_grid_thw`、帧数和视觉 token 数，确认没有被 processor 再次异常采样。

### 4.3 输出格式

```json
{"scores":[12,73,65,18,92,40,27,31]}
```

必须满足：只含 `scores` 键，长度等于窗口片段数，全部是 0—100 的整数。拒绝 `NaN`、布尔值、重复键、额外解释、少项或超范围数值。**不使用 `eval`，不自动补缺项，不截断超范围值后装成有效回答。**

这些分数的语义是“在整段视频中的高光相对强度”，不是“在当前窗口中重新排出的百分位”。否则不同窗口均有一个 100 分，最终拼接会失去可比性。

### 4.4 首轮为何不直接输出任意时间戳？

固定时间网格让“看见什么、监督什么、评价什么”对齐，减少自由时间戳格式与边界误差对奖励判断的干扰。这是工程选择，不是自由时间戳不可行。日后研究边界精炼时再加入连续时间回归。

---

<a id="section-5"></a>

## 5. 固定特征、聚类与伪标签

### 5.1 可操作的视觉特征版本

首轮使用冻结的 `CLIPVisionModelWithProjection`，对片段四帧分别提取视觉向量，先单位化，平均后再次单位化：

$$
u_{n,t,j}=\frac{f_\phi(F_{n,t,j})}{\|f_\phi(F_{n,t,j})\|_2},\qquad
z^v_{n,t}=\frac{\frac14\sum_j u_{n,t,j}}{\|\frac14\sum_j u_{n,t,j}\|_2}.
$$

这里 $\phi$ 固定，不参与 SFT/GRPO。代码对零向量和非有限值报错。`CLIPVisionModelWithProjection` 的输出接口见 [S8]。

这不是原论文 3D 视频编码器的严格复现。若已有与片段时间轴一一对应的可核验特征，可以替换；要记录编码器身份和时间采样，不得把旧特征与新帧索引按行数随意配对。

### 5.2 视频伪类别

视频表示为：

$$
\bar z_n=\operatorname{norm}\left(\frac{1}{T_n}\sum_t z^v_{n,t}\right).
$$

仅在训练视频上拟合 UMAP，再对其低维表示进行 K-means。首轮搜索 $K=4,\ldots,\min(15,N-1)$，按轮廓系数选择；固定随机种子，保存每个 $K$ 的分数与簇大小。若训练样本不足或特征退化，停止并修复数据，不静默变成另一个算法。

测试视频只能执行 `UMAP.transform` 和 `KMeans.predict`；UMAP 支持将新数据映射到已学空间。[S9] **禁止将 train/test 拼在一起 `fit_transform`，再称作常规独立测试。**

K-means 优化的是：

$$
\min_{\{c_n\},\{\mu_k\}}\sum_n\|e_n-\mu_{c_n}\|_2^2,
$$

其中 $e_n$ 为降维表示。交替分配最近中心、更新簇均值各不增加该目标，因此目标单调不增且有下界。这保证目标值收敛；配合确定性的平局处理及终止条件，可得到局部稳定划分。但**不证明簇是正确语义类别，更不证明高光标签正确。**轮廓系数同样衡量几何聚类性质，而不是高光质量。

### 5.3 R0：原形式的类内平均重复性

设训练簇 $c$ 的片段集合为 $\mathcal B_c$，且特征已经单位化：

$$
\rho_{n,t}^{v}=\frac{1}{|\mathcal B_{c_n}|}\sum_{z\in\mathcal B_{c_n}} (z^v_{n,t})^\top z.
$$

这是原论文平均余弦形式的视觉版。[S1] 在训练集内按此公式构库时，包含当前视频自己的片段；不能把这个版本描述为严格只看其他视频。

**精确化简：**

$$
\mu_c=\frac{1}{|\mathcal B_c|}\sum_{z\in\mathcal B_c}z
\quad\Longrightarrow\quad
\rho_{n,t}^{v}=(z^v_{n,t})^\top\mu_{c_n}.
$$

证明是点积对求和的线性性。注意：**均值 $\mu_c$ 不再做 L2 归一化**，否则不再数值等价于原平均相似度。原始逐对计算需要约 $O(S^2d)$ 时间和可能的 $O(S^2)$ 矩阵；先构均值后打分只需 $O(Sd)$ 时间和 $O(Kd)$ 原型存储，不计已存的特征本身。

单位向量点积在 $[-1,1]$ 内，所以 $\rho\in[-1,1]$。这个范围有界不等于语义正确。

### 5.4 可选诊断 R-LOO：整视频排除、视频等权

为了分析自身重复与长视频权重，提供一个不作为首轮强制创新的版本：

$$
\rho^{\mathrm{LOO}}_{n,t}=
(z^v_{n,t})^\top\left[
\frac{1}{|\mathcal V_{c_n}\setminus\{n\}|}
\sum_{m\in\mathcal V_{c_n}\setminus\{n\}}
\left(\frac1{T_m}\sum_j z^v_{m,j}\right)
\right].
$$

它排除整个当前视频，并让每个参照视频权重相等。簇内没有其他视频时，固定回退到全部其他训练视频的等权均值，并记录比例。这个回退不读取测试数据。

该操作只能消除公式中的自视频参照和长度加权；**聚类本身仍可能使用了当前视频表示，因此它不是完整的交叉拟合独立性证明。**也不能预设 LOO 一定优于原形式。

### 5.5 全视频尺度的百分位伪分数

对一个视频全部 $T_n$ 个原始重复性分数，计算带并列平均秩的分数：

$$
q_{n,t}=\frac{\operatorname{midrank}(\rho_{n,t})-\frac12}{T_n}.
$$

其范围在 $(0,1)$。若所有原始分数相同，则 $q_{n,t}=0.5$；不人为打破并列。严格较大的原始分数获得严格较大的 $q$，因此它保留完整视频内的排序。

**只能对完整视频计算一次再切窗，不能对每个窗口重新计算。**这使同一片段出现在不同窗口时有相同目标。

这一变换并不产生置信度：非常小的原始差异也可能对应不同百分位。因此保存 `raw` 与 `q` 两份信息，用于分析。不要把 rank 差异冒充代理标签的可信概率。

### 5.6 音频的处理原则

首轮策略只接收画面，因此首轮奖励用视觉版。后续可以将固定音频分数作为额外训练信号，但要标注为“训练使用特权音频信号、推理仅视觉”，并单独报告。

若存在相同画面 $X_v$、不同声音 $X_a$ 而目标 $q$ 不同的情况，纯视觉预测的最优平方误差为：

$$
\inf_f\mathbb E[(f(X_v)-q)^2]=\mathbb E[\operatorname{Var}(q\mid X_v)].
$$

证明由条件期望分解：

$$
\mathbb E[(f-q)^2\mid X_v]=(f-\mathbb E[q\mid X_v])^2+\operatorname{Var}(q\mid X_v).
$$

因此，音频奖励可能含有视觉策略无法辨识的成分。GRPO 不会自动让普通 VL 模型“听到声音”。

---

<a id="section-6"></a>

## 6. SFT 冷启动、LoRA 与固定参考策略

### 6.1 伪目标

将完整视频目标 $q_t$ 量化为：

$$
g_t=\operatorname{round}(100q_t),\qquad g_t\in\{0,1,\ldots,100\}.
$$

对目标文本 $Y^*$ 使用只覆盖 assistant 答案 token 的交叉熵：

$$
\mathcal L_{\mathrm{SFT}}=-\mathbb E_X\frac1{L^*}\sum_{\ell=1}^{L^*}
\log\pi_\theta(Y^*_\ell\mid X,Y^*_{<\ell}).
$$

不对视频占位、用户提示或伪标签元数据计算目标损失。所有候选必须能从输入片段中找到对应关系。上式按回答长度先归一化；框架的 SFT 实际可能按整个 batch 的有效 token 数归一化，导致不同长度样本权重不同。需记录实际 reduction，而不能把两者无条件称为相同目标；所有 SFT 组保持一致。

量化误差满足 $|g_t/100-q_t|\le0.005$，故单点平方误差不超过 $2.5\times10^{-5}$。这是标签离散化误差界，不是训练误差界；它也不是排序误差界，两个很接近的伪分数可能被量化成并列。需报告这类并列比例，不把量化后的 SFT 答案认作排序奖励必然达到 1 的证明。

### 6.2 LoRA 参数化

对可训练的语言层线性权重：

$$
W=W_0+\frac{\alpha}{r}BA,
$$

冻结 $W_0$，首轮取 $r=16,\alpha=32$，LoRA dropout 为 0。视觉塔与对齐器冻结。LoRA 只减少更新参数，不消除原生视频前向的计算。

当 $B=0$、$A$ 为非零初始化时，$\partial\mathcal L/\partial A$ 初始可以为零，但

$$
\frac{\partial\mathcal L}{\partial B}=\frac\alpha r\frac{\partial\mathcal L}{\partial W}A^\top
$$

一般不为零，因此并不是“零初始化导致整个 LoRA 无法学习”。是否真的更新仍应检查实际梯度和参数差值。

### 6.3 参考策略必须是 SFT，而不是误回到原始基座

训练完共同 SFT 后，记录端点 $\theta_{\mathrm{SFT}}$，合并 LoRA 到完整权重得到 `merged_sft0`。在它上面分别新建同规格 LoRA，启动继续 SFT 和 GRPO。

在精确算术下合并满足：

$$
W_{\mathrm{merged}}x=(W_0+\alpha BA/r)x,
$$

与原 SFT 的线性映射一致。有限精度下要检查实际输出和 log probability 差异，不能只看文件名。

这样做的原因：LoRA 模式的参考策略常通过禁用当前 adapter 得到。若直接在原始基座上替换/继续某个 adapter，禁用后可能回到原始 Qwen，而不是 SFT 策略。**先合并共同 SFT，禁用新 adapter 才对应所需的 SFT 参考。**ms-swift v4.6.0 的 LoRA 路径不要求另传 `ref_model`，源码还会拒绝某些这样的传法。[S5]

SFT 继续训练组同样从合并底座新建 LoRA，以保持相同的起始函数和可更新参数族；不要一组继续旧 adapter、一组随机新增不同结构。

---

<a id="section-7"></a>

## 7. 奖励定义：首轮只做两种，且用相同伪监督

### 7.1 校准奖励：R-CAL

令解析后的预测 $s_i=g_i/100$，目标为 $q_i$：

$$
R_{\mathrm{cal}}(\mathbf s,\mathbf q)=1-\frac1M\sum_{i=1}^M(s_i-q_i)^2.
$$

由于二者都在 $[0,1]$，因此 $R_{\mathrm{cal}}\in[0,1]$。这一奖励故意简单：它检验“把伪分数回归改用 GRPO”是否真的有价值，不预设有价值。

### 7.2 全窗口的加权排序奖励：R-RANK

定义：

$$
w_{ij}=|q_i-q_j|,
$$

$$
h_{ij}=\begin{cases}
1,&(s_i-s_j)(q_i-q_j)>0,\\
\frac12,&s_i=s_j,\\
0,&\text{其他情况}.
\end{cases}
$$

对于 $q_i=q_j$，权重为零，不强迫区分。令 $W=\sum_{i<j}w_{ij}$：

$$
R_{\mathrm{rank}}=\begin{cases}
\dfrac{\sum_{i<j}w_{ij}h_{ij}}{W},&W>0,\\
\frac12,&W=0.
\end{cases}
$$

这是一个自定义、可解释的代理排序奖励，不是人类高光 mAP，也不是新颖性已经成立的算法。

因为每个 $h_{ij}\in[0,1]$，所以加权平均在 $[0,1]$。伪标签非并列的所有对都严格排对时为 1；全部反排时为 0；全部预测同分时为 0.5。

$w_{ij}$ 只表达伪排序差距，**不是经过证明的可信度**。不在首轮额外宣称不确定性估计。

### 7.3 主奖励：排序与校准共同使用

$$
R_{\mathrm{main}}=0.8R_{\mathrm{rank}}+0.2R_{\mathrm{cal}}.
$$

非法回答统一奖励 −1，合法回答奖励在 $[0,1]$。格式不再另给大额正奖励。分母、缺项和非有限数据错误应直接报错，而不是返回一个看似正常的奖励。

**为什么保留 0.2 校准？**单纯排序对于严格递增变换 $h(s)$ 不变；不同窗口可以用完全不同的数值尺度但得到同样排序奖励。共同的完整视频 $q$ 给出数值锚点，减少窗口间任意平移/压缩的自由度。它提供约束，但不保证神经网络一定校准成功；最终要测同片段跨窗口预测差异。

不加入多样性、低重复、叙事顺序等奖励。它们主要属于最终摘要/编辑集合目标，而检测阶段两次重播的精彩动作仍可能都是真高光。首轮不混淆这两个任务。

### 7.4 原始代理分数能否被“超过”？

如果奖励仅是固定代理分数的 top-k 求和，直接按代理排序就已经最优。GRPO 不可能超过该固定奖励的最大值。

但代理分数不等于人类评价。训练网络在未见视频上的人工标注表现，可能高于或低于直接代理。**需要检验的是代理监督的泛化，而不是宣称从同一固定奖励中凭空提炼出了额外真值。**

这里必须有三个结果列：代理奖励、独立高光评价、推理成本。缺少其中任何一项都容易误判。

---

<a id="section-8"></a>

## 8. GRPO 的完整数学对象与实现检查

### 8.1 回答概率与期望奖励

对一个实际输入 $X$，文本回答的概率为：

$$
\pi_\theta(Y\mid X)=\prod_{\ell=1}^{|Y|}
\pi_\theta(Y_\ell\mid X,Y_{<\ell}).
$$

即使最终输出是分数，优化对象仍然是这条文本序列的概率。奖励只需要能从回答算出，不需要对排序、JSON 解析或离散分数求导。

定义理想化的期望代理目标：

$$
J(\theta)=\mathbb E_{X}\mathbb E_{Y\sim\pi_\theta(\cdot\mid X)}[R(X,Y)].
$$

在可交换求和与求导等常规条件下：

$$
\nabla_\theta J=
\mathbb E[R(X,Y)\nabla_\theta\log\pi_\theta(Y\mid X)].
$$

**证明**：对有限长度词表输出，逐项求导，利用 $\nabla p=p\nabla\log p$ 即得。对于不依赖当前动作的 baseline $b(X)$，

$$
\mathbb E[b(X)\nabla\log\pi_\theta(Y\mid X)]
=b(X)\nabla\sum_Y\pi_\theta(Y\mid X)=0.
$$

这证明“不需要高光 GT、也不需要可微奖励才能建立策略梯度”，不证明该奖励符合人的精彩性判断。[S10]

### 8.2 组内采样和相对优势

冻结当前行为策略，给**同一份输入媒体和提示**采样：

$$
Y_1,\ldots,Y_G\overset{\mathrm{iid}}\sim\pi_{\mathrm{old}}(\cdot\mid X).
$$

$$
\bar R=\frac1G\sum_gR_g,\qquad
\sigma_R=\sqrt{\frac1G\sum_g(R_g-\bar R)^2},\qquad
A_g=\frac{R_g-\bar R}{\sigma_R+\varepsilon_R}.
$$

参考代码用总体标准差 `ddof=0`；框架若采用样本标准差，需要登记。不要一部分用 $G$，另一部分用 $G-1$ 却认为是完全相同配置。

不能在同一组里随机换帧、换窗口或换增强后仍说“同一输入的多个回答”。多个视频也不能混在一起做一个组均值。

### 8.3 包含自身的均值 baseline 有怎样的偏差？

以下等式只针对采样分布与求导策略相同的 on-policy 情形，不直接套用于在冻结旧样本上反复更新。先不做标准差缩放，记 $g_i=\nabla\log\pi(Y_i\mid X)$，则：

$$
\mathbb E\left[\frac1G\sum_i(R_i-\bar R)g_i\right]
=\frac{G-1}{G}\nabla J_X.
$$

证明：$\mathbb E[R_ig_i]=\nabla J_X$。当 $j\ne i$ 时，独立性与 $\mathbb E[g_i]=0$ 给出 $\mathbb E[R_jg_i]=0$，故

$$
\mathbb E[\bar Rg_i]=\frac1G\nabla J_X.
$$

逐项相减并平均即得。**加入由同一组样本决定的标准差后，分母与样本相关，不能继续把整个 GRPO 梯度称为这个简单的无偏或定比例估计。**裁剪、token 长度归一化、KL 和多轮更新又进一步改变目标。

### 8.4 token 级概率比与裁剪目标

$$
r_{g,\ell}(\theta)=
\exp\left(\log\pi_\theta(Y_{g,\ell}\mid X,Y_{g,<\ell})-
\log\pi_{\mathrm{old}}(Y_{g,\ell}\mid X,Y_{g,<\ell})\right).
$$

采用原始 GRPO 风格的局部代理目标：[S3][S10]

$$
\mathcal L_{\mathrm{GRPO}}=-\frac1G\sum_g\frac1{L_g}\sum_{\ell=1}^{L_g}
\left[\min\{r_{g,\ell}A_g,
\operatorname{clip}(r_{g,\ell},1-\epsilon,1+\epsilon)A_g\}
-\beta\widehat k_{g,\ell}\right].
$$

$L_g$ 只统计有效 completion token；prompt、padding 不计入。`old_logp`、`ref_logp`、奖励和优势必须停止梯度，当前 `new_logp` 必须有梯度。

Clipping 限制部分代理收益，不是参数变化的硬上限，也不检查奖励是否可信。首轮 `num_iterations=1`、行为策略等于开始更新的策略时，概率比最初等于 1，clip fraction 很低可以是正常现象；不能为了“让 clipping 起作用”随意重复更新并改变实验预算。

### 8.5 KL 的估计边界

常用采样项为：

$$
d=\log\pi_{\mathrm{ref}}(a\mid h)-\log\pi_\theta(a\mid h),\qquad
\widehat k=e^d-d-1\ge0.
$$

非负性来自 $e^d\ge1+d$。若 $a\sim\pi_\theta$，则：

$$
\mathbb E_{a\sim\pi_\theta}[\widehat k]
=\sum_a\pi_{\mathrm{ref}}(a)-1+
\sum_a\pi_\theta(a)\log\frac{\pi_\theta(a)}{\pi_{\mathrm{ref}}(a)}
=D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}}).
$$

实际 rollout 来自 $\pi_{\mathrm{old}}$。当 current 已经变化时，上述期望等式不能无条件套用到旧策略样本；它是实际训练中的近策略代理项，不能谎称任何离策略阶段都在精确估计当前 KL。即使某个 on-policy 点的数值期望相等，固定采样动作后直接对该项反向传播，也不能在忽略采样分布导数的情况下宣称获得了完整期望 KL 的精确梯度。本文遵循框架代理目标，不把这个区别隐藏在“无偏”二字里。

### 8.6 采样分布必须与概率计算一致

首轮设置 `temperature=1`、`top_p=1`、关闭 top-k 截断、`repetition_penalty=1`，不使用自定义 constrained decoding。这样尽量减少“行为分布被改写，但 log probability 仍按原 logits 算”的问题。

不强制修改输出词表；JSON 合法性由 SFT 学习、奖励和审计约束。以后引入分数 token 约束时，采样与新/旧策略 log probability 都必须使用**同一个 mask 后的重新归一化分布**。

### 8.7 最容易误判的两个日志

- 所有奖励相同，优势为零：奖励驱动项没有学习信号，但 KL 项不一定为零。
- 更新前平均优势为零、策略 loss 数值接近零：不等于梯度为零。$\sum_i A_i=0$ 不意味着 $\sum_iA_i\nabla\log\pi(Y_i)=0$。

这两项均应在小规模测试中检查。配套测试还验证了 old/ref 不回传梯度。

---
<a id="section-9"></a>

## 9. 首轮实验矩阵与预算：不要把探索变成无止境的小修小补

### 9.1 最少需要跑的组

| ID | 方法 | 学到什么 / 用来区分什么 |
|---|---|---|
| B0 | 固定 CLIP 重复性分数直接检测 | 代理信号本身的能力；无 Qwen 训练 |
| B1 | Qwen 通用基座直接按相同提示检测 | 预训练先验的零样本能力；不是主要因果对照 |
| S0 | 伪标签 SFT 共同端点 | 输出协议和初步高光学习 |
| S+ | 从合并 S0 底座新建 LoRA，继续 SFT | 同样伪监督、额外曝光的强对照 |
| G-CAL | 同一起点，GRPO + 校准奖励 | 单纯把伪分数拟合改成策略优化是否有用 |
| G-RANK | 同一起点，GRPO + 0.8 排序 + 0.2 校准 | 任务相关的排序反馈是否比校准反馈更有用 |

第一轮不强制跑 LOO、音频、7B、额外教师和新 GRPO。先完成上述主矩阵，再按结果决定扩展。若预算必须减少，至少保留 B0、S0、S+、G-RANK，明确没有隔离奖励形态的贡献。

### 9.2 一次有限的技术检查与一次预登记实验

**技术检查不是效果实验。**用少量训练视频检查解码、奖励、反向传播、合并和推理，不查看测试标签选配置。允许修复确定性的实现错误，但保存错误和修复记录。

技术接口通过后，锁定数据、伪标签、prompt、步数、采样参数和评价方案，执行整套首轮对照。不要每看一个坏案例就改 prompt，再把后续结果与原基线当成同一预登记实验。

### 9.3 首轮建议预算

下表只是可执行起点，**不是足够收敛的理论保证**。默认成功 optimizer update 为 256；SFT 冷启动和后续分支分别计数。

| 阶段 | 每次更新的不同输入曝光 | 每输入生成数 | 更新次数 | 总输入曝光 / 生成回答 |
|---|---:|---:|---:|---:|
| SFT 冷启动 S0 | 16 | 不适用 | 256 | 4,096 次输入曝光 |
| SFT 续训 S+ | 4 | 不适用 | 256 | 1,024 次输入曝光 |
| G-CAL | 4 | 4 | 256 | 1,024 次输入曝光，4,096 条 rollout |
| G-RANK | 4 | 4 | 256 | 1,024 次输入曝光，4,096 条 rollout |

每个训练视频预生成 32 个窗口实例，必要时有放回抽样。因此学生的训练分布近似按视频等权、再从该视频的窗口池采样，而不是按所有视频片段数加权；它与原形式重复性库的片段等权是两个不同选择，应同时记录。训练窗口索引和抽样 seed 固定；SFT 与 GRPO 使用相同索引池。由于框架采样器实现可能不同，**还需记录实际 `request_id` 曝光账本**，不能仅凭数据文件相同就声称逐步完全匹配。

“相同输入曝光”不等于“相同计算”：GRPO 需要 rollout、旧策略/参考概率和当前策略计算。记录视频解码时间、GPU 小时、输入/输出 token 数、峰值显存、成功更新数。论文中同时给数据预算和计算预算，不将额外算力收益归为纯算法收益。

### 9.4 更正式的预算安排

初轮只有一个 seed，报告为可行性实验。若有希望，再固定 3 个训练 seed，并在第二个数据集或未参与方法选择的来源上验证。S0 的随机性也应覆盖：每个 seed 训练自己的 S0，再从该 S0 配对分支；不能只改变后续 GRPO seed 就声称覆盖完整训练方差。

固定终点是主结果。可以保存中间端点画学习曲线，但不使用测试集挑最好的 checkpoint。若需要延长预算，所有比较分支共同延长，并另记为下一阶段；不要仅延长暂时落后的组。

---

<a id="section-10"></a>

## 10. 环境与代码文件：能复制运行的步骤，以及尚需实机核验的地方

### 10.1 本次交付的真实状态

配套核心数学、JSON、伪标签、数据构建、汇总和合成管线测试可在 CPU 运行。真实视频特征抽取、Qwen processor、SFT、LoRA 合并、GRPO、GPU 推理**没有在本次文档生成环境中执行**。

因此以下命令是依据官方 v4.6 文档与源码组织的目标机执行模板，不是伪装成已跑完的训练收据。当前环境没有下载 Qwen，也没有 ms-swift/Transformers/UMAP 的完整训练安装；实际验证范围见第 19 节。

### 10.2 单独创建研究环境

不要直接覆盖原比赛或其他项目的 Python 环境。使用 Python 3.12 是一个明确的环境起点；先按服务器 CUDA/驱动选择适配的 PyTorch，再安装其他依赖。不要把容器 CPU 测试使用的 torch 版本直接复制成 GPU 环境要求。

```bash
# 在新环境中执行。PyTorch CUDA wheel 需按目标机选择，不在这里猜测。
python -m venv .venv-highlights
source .venv-highlights/bin/activate
python -m pip install --upgrade pip
# 先安装目标 GPU 支持的 PyTorch；然后：
python -m pip install "ms-swift==4.6.0" \
  numpy scipy scikit-learn umap-learn joblib pillow av pytest tensorboard
python -m pip check
python -m pip freeze > environment.candidate.freeze.txt
swift sft --help > swift_sft_help.txt
swift rlhf --help > swift_rlhf_help.txt
```

`ms-swift==4.6.0` 是框架锚点，不等于所有传递依赖已经形成兼容锁。Qwen3-VL 需要支持该架构的 Transformers；用下面的导入检查和单样本实测验证，而非盲目依赖解析器得到的最老/最新版本。[S2][S11][S12]

```bash
python - <<'PY'
import torch, transformers, peft, trl, swift
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("peft", peft.__version__, "trl", trl.__version__)
print("CUDA available", torch.cuda.is_available())
PY
```

先下载到明确的本地路径，再使用相同路径训练与推理。记录模型 repo、resolved revision、所有权重文件哈希、processor/tokenizer 配置和 chat template。不要在每次运行时从不断变化的 `main` 分支重新取模型。

### 10.3 配套脚本职责

| 文件 | 负责的部分 | 验证边界 |
|---|---|---|
| `core.py` | 余弦化简、百分位、严格解析、奖励、汇总、区间对齐、bootstrap | 有 CPU 单元测试 |
| `prepare_visual.py` | PTS 解码、概览图、CLIP 特征与 prepared manifest | 语法检查；真实解码/模型抽取需目标机验证 |
| `make_pseudo.py` | train-only UMAP/K-means 与固定伪分数 | PCA 合成路径验证；UMAP 需目标机验证 |
| `make_examples.py` | 同一窗口池生成 SFT/GRPO/inference JSONL | 单元与合成管线测试 |
| `reward_plugin.py` | ms-swift 自定义 reward 接口 | 接口与官方源码核对；未运行真实 Trainer |
| `grpo_reference.py` | 独立 PyTorch 参考损失 | CPU 梯度测试；不是完整训练器 |
| `run.sh` / `infer.sh` | SFT、合并、续训、GRPO、推理命令 | Shell 语法检查；未执行 GPU 训练 |
| `normalize_inference.py` | 按显式字段抽取 request_id/response | 不猜测数据顺序或自动补结果 |
| `aggregate_predictions.py` | 全窗口汇总为全视频分数 | 合成测试 |
| `export_teacher.py` | 直接代理分数基线 | 合成测试 |
| `evaluate_v0.py` | 明确命名的自定义诊断指标 | 合成测试；不是论文官方 top-5 mAP |
| `test_core.py` / `test_pipeline.py` | 核心性质与数据泄漏检查 | 第 19 节给实际结果 |
| `validate_synthetic.py` | 无真实视频的模拟端到端检查 | 只证明文件与数值接口，不证明模型质量 |

### 10.4 从原视频到训练 JSONL

以下命令统一假设当前目录是 `highlight_grpo_starter/`；`/data/...` 和 `/models/...` 是必须替换的目标机路径。不得把示例路径的存在当作资产已经迁移。

```bash
# 0. 核心测试，不读取任何真实标注。
python -m pytest -q test_core.py test_pipeline.py

# 1. 手工/脚本整理 manifest.jsonl，必须有来源身份和固定 split。
# 在运行前确认 ffprobe 可用，视频和模型文件可读取。
ffprobe -version

# 2. 解码并抽取固定视觉特征。
python prepare_visual.py \
  --manifest /data/manifests/manifest.jsonl \
  --out /data/prepared \
  --encoder /models/clip-vit-base-patch32 \
  --device cuda

# 3. 只在训练视频上拟合聚类。
python make_pseudo.py \
  --manifest /data/prepared/prepared.jsonl \
  --out /data/pseudo/r0_visual \
  --modality visual --method original --embedding umap --seed 42

# 4. 生成共同窗口池以及无伪答案的测试输入。
python make_examples.py \
  --manifest /data/prepared/prepared.jsonl \
  --pseudo /data/pseudo/r0_visual/pseudo.json \
  --out /data/model_inputs --windows-per-video 32 --seed 42

# 5. 检查文件数量和哈希，固定数据版本。
sha256sum /data/prepared/prepared.jsonl /data/pseudo/r0_visual/pseudo.json \
  /data/model_inputs/*.jsonl > /data/model_inputs/data.sha256
```

不要因缺少 `umap-learn` 静默使用 PCA。代码中的 `--embedding pca` 只作为明确标识的调试/替代分支；它产生的是另一种伪类别，不能与 UMAP 主实验混记。

训练文件包含的 `videos` 是嵌套帧路径列表：一个视频占位对应一个完整帧列表。这个数据形式遵循 ms-swift 视频数据规范。[S4] **目标机应将一条样本经过实际 processor 后打印形状，确认仍有 4M 个帧位，而非重复采样成错误数量。**

### 10.5 单样本/单组硬门槛

在训练任何正式分支之前，固定一条样本做以下检查，失败直接修接口，不追加模型创新：

| 检查 | 必须看到的证据 |
|---|---|
| 时间轴 | 原片段、抽帧图片、真实 PTS、提示时刻一致 |
| 视觉输入 | 概览图清晰，窗口顺序正确；没有把一个视频拆成多个错位 placeholder |
| 标签隔离 | GRPO 的 `messages` 无 pseudo_scores/GT/assistant 伪答案 |
| 输出解析 | 合法/非法各构造一条；奖励有明确差异 |
| Trainer 元数据 | 每条 completion 收到的 q 与其 request_id、窗口片段完全对应 |
| 组采样 | 同组 G 条输入张量完全相同；不是 G 个不同视频凑组 |
| 反向 | 新 LoRA 梯度有限且至少部分非零；冻结视觉塔与基座无梯度 |
| 更新 | 优化器 step 后至少一个可训练参数发生改变；不能只看 loss 值 |
| Mask | prompt、padding 不计入 completion loss；EOS/截断处理明确 |
| 参考策略 | 禁用新 adapter 的 log probabilities 与合并 S0 一致，训练后仍不变 |

这个门槛不是要求先进行很多轮效果短测。它是一套确定性的运行契约。

---

<a id="section-11"></a>

## 11. SFT、合并与 GRPO 的实际执行

### 11.1 先跑 SFT 冷启动

```bash
export HL_DATA=/data/model_inputs
export HL_OUTPUT=/data/outputs/seed42
export HL_SEED=42
export HL_STEPS=256

# 例：当前只有一张可用 GPU 时用 1；有四张可用卡时才改为 4。
export NPROC_PER_NODE=1
bash run.sh sft0 /models/Qwen3-VL-2B-Instruct
```

脚本默认单卡 microbatch=1，SFT 冷启动梯度累积到 global batch=16。四卡只是另一种可配置情况，不是假定你的旧四卡机器仍在。

先检查格式合法率、伪分数误差和训练 loss；不能依据测试高光标签决定继续训练多久。若冷启动输出完全不能解析，应作为接口/训练有效性故障记录，不直接启动正式 GRPO。

### 11.2 合并共同起点，并做功能等价检查

```bash
# 将实际 checkpoint 目录替换进来，不能猜测框架生成的带时间戳目录。
bash run.sh merge /models/Qwen3-VL-2B-Instruct \
  /data/outputs/seed42/实际SFT目录/checkpoint-256
```

合并后保存完整模型与 processor。选固定样本，分别用“原底座 + SFT adapter”和“合并模型”计算同一答案的 token log probability，报告最大/平均绝对差异与 greedy 输出是否一致。不同精度/矩阵实现可能造成小差异，容差先登记；不要宣称逐位一致。

### 11.3 三个后续分支

```bash
export HL_PROMPTS=4
export HL_GROUP_SIZE=4
export HL_STEPS=256
MERGED=/data/outputs/seed42/merged_sft0

# 与 GRPO 共享起点、数据和新 LoRA 规格。
bash run.sh sft-continue "$MERGED"

# 校准奖励：rank_weight=0。
HL_RANK_WEIGHT=0 bash run.sh grpo "$MERGED"

# 主奖励：rank_weight=0.8。
HL_RANK_WEIGHT=0.8 bash run.sh grpo "$MERGED"
```

SFT 续训及 GRPO 的学习率起点均为 $10^{-5}$，同 cosine/warmup 设置。它们损失数值和有效梯度尺度不同，因此“相同学习率”不是最优性或完整公平性的证明；首轮避免任意调参，后续可以对所有方法给相同规模的无测试调参预算。

### 11.4 显式 GRPO 参数

| 配置 | 首轮值 | 目的 |
|---|---|---|
| `num_generations` | 4 | 每个输入一组四条回答 |
| `generation_batch_size` | PROMPTS × G | 以 completion 计量；实际样本展开需审计 |
| `num_iterations` | 1 | 不额外重复使用相同 rollout |
| `loss_type` | `grpo` | 显式选择原型长度归一化形式 |
| `advantage_estimator` | `grpo` | 组相对优势 |
| `scale_rewards` | `group` | 组内标准差缩放 |
| `beta` | 0.04 | 固定 SFT 参考的 KL 强度起点 |
| `epsilon` | 0.2 | token ratio clip 起点 |
| `kl_in_reward` | false | KL 单列进 loss，不先混进组奖励 |
| `temperature / top_p / top_k` | 1 / 1 / 0 | 尽量不改变采样支持；确认框架 0 表示关闭 top-k |
| `repetition_penalty` | 1 | 不改写数字生成分布 |
| `max_completion_length` | 128 | 足够短的 JSON；必须检查实际 tokenizer 长度 |
| `dynamic_sample` | false | 避免按奖励方差动态重采样改变曝光预算 |
| `overlong_filter` | false | 避免静默删除截断回答；解析失败仍计入 |
| `use_vllm` | false | 先减少跨推理引擎的 log probability 差异 |

参数语义和接口参见 [S3][S5][S11]；脚本不冒充在所有传递依赖版本上测试过。更新框架后重新核验这些设置，不能沿用默认值假设。

### 11.5 计算与显存不能从“2B”直接推出

BF16 参数本身约需要 $2P$ 字节，$P=2\times10^9$ 时仅权重数量级约 4 GB；这不是训练峰值显存。还需要视觉激活、KV cache、LoRA 梯度与优化器、临时 logits、多条 rollout 和框架缓冲。

若模型层数为 $L$、KV 头数为 $H_{kv}$、每头维度 $d_h$、有效序列长度 $S$、并行序列数 $B$，仅 KV 元素数量约为 $2LB S H_{kv}d_h$。这是容量估算，不是对所选 Qwen 具体架构的实测。

DDP 四张 24 GB 通常不是一张能任意放 96 GB 张量的显卡。遇到 OOM，优先降低同时 rollout 的输入数、序列长度或分辨率，并记录变化；不要悄悄减少某一组的视觉证据而保持其他组不变。

首次设备检查可用 `HL_PROMPTS=1`、`HL_STEPS=4` 在单卡运行；这是技术配置，不是正式结果。若改为多卡，保证 completion batch 与 world size、G 的整除关系。完整视频仍由滑窗覆盖，不因显存不足只评估最容易的前几个窗口。

---

<a id="section-12"></a>

## 12. 推理、跨窗口汇总与时间对齐

### 12.1 固定推理预算

所有训练端点使用相同 prompt、视频分辨率、窗口、greedy 解码和最大输出长度。首轮每窗口只生成一次，不进行测试时 best-of-G。任何重试次数也必须所有组一致，并单列成本；不能用真值挑候选。

```bash
# S0 合并模型本身：adapter 用 '-'。
bash infer.sh /data/outputs/seed42/merged_sft0 - \
  /data/predictions/s0_raw.jsonl

# 其他分支用实际新 adapter。
bash infer.sh /data/outputs/seed42/merged_sft0 \
  /data/outputs/seed42/实际GRPO目录/checkpoint-256 \
  /data/predictions/grpo_raw.jsonl

# 显式指定框架实际输出字段，不依靠行号猜测对应关系。
python normalize_inference.py \
  --input /data/predictions/grpo_raw.jsonl \
  --examples /data/model_inputs/eval.jsonl \
  --id-field request_id --response-field response \
  --out /data/predictions/grpo_normalized.jsonl

python aggregate_predictions.py \
  --examples /data/model_inputs/eval.jsonl \
  --predictions /data/predictions/grpo_normalized.jsonl \
  --out /data/predictions/grpo_video.json

python export_teacher.py --pseudo /data/pseudo/r0_visual/pseudo.json \
  --out /data/predictions/direct_teacher_video.json
```

某一框架版本可能把 ID 放在嵌套字段，或默认不保留输入元数据。先核对真实结果结构，再用显式字段路径或修改结果 writer 保留 ID；**若无法建立一一对应，停止汇总，不能猜测排序。**

### 12.2 汇总定义与可证明性质

设片段 $t$ 出现在窗口集合 $\mathcal W(t)$，汇总分数为：

$$
\hat s_t=\frac{1}{|\mathcal W(t)|}\sum_{w\in\mathcal W(t)}s_{w,t}.
$$

所有窗口使用同一个全视频目标 $q_t$。若一条窗口输出非法，该窗口所覆盖片段记为 0，仍进入分母，并另报非法率。这是统一的失败处理约定，不是说 0 是该片段真实质量。禁止静默删除非法窗口，让失败样本不参与评价。

由平方函数凸性：

$$
(\hat s_t-q_t)^2\le \frac{1}{|\mathcal W(t)|}\sum_w(s_{w,t}-q_t)^2.
$$

该式证明平均汇总的平方误差不大于各窗口误差的平均，**不证明比其中最好窗口好，也不证明人类 AP 改善**。窗口输出高度相关时，方差不会自动按窗口数倒数下降。全视频百分位标签提供共同尺度定义，但模型是否学会校准仍须测量。

局部输入可能缺少决定全视频百分位所需的信息。第 5.6 节的条件方差分解同样适用：概览图不是充分统计量的保证。报告同一片段跨窗口预测差异，避免把隐藏上下文差异误当成纯噪声。

### 12.3 标注时间单元不同时，用区间重叠映射

对评估区间 $J_j$，令：

$$
\tilde s_j=\frac{\sum_t|I_t\cap J_j|\hat s_t}{|J_j|}.
$$

要求预测时间轴完整覆盖 $J_j$，且预测片段之间不重叠。此时各权重非负且和为 1，因此映射结果在原始分数最小值与最大值之间。若所有相交片段分数相同，映射保留该常数。

这只是分段常数预测的时间积分，不是插值恢复了模型未观察到的动作。代码对部分覆盖直接报错，不把分母偷偷改为已覆盖时长。帧级标注要先根据原始视频时钟正确转换成时间区间，不凭猜测 FPS 对齐。

---

<a id="section-13"></a>

## 13. 评价协议：不允许“自己的 AP”冒充论文的 top-5 mAP

### 13.1 两套评价分开登记

**V0：配套自定义诊断协议。**每个标注者单独用其分数中位数二值化，计算 AP，先平均标注者再平均视频；同时报告与用户平均评分的 Spearman。没有正例的标注者记为 0 并记录数量，不静默剔除。AP 使用 sklearn 的非插值 average precision 定义。[S13]

**V1：用于正式论文比较的来源协议。**TVSum 高光的 top-5 mAP、YouTube Highlights mAP 等必须核实原方法的片段划分、用户处理、二值阈值、top-5 定义、并列处理、划分和最终平均层次。本次没有完成作者原评估脚本的数值复核，因此**不提供一个伪装成官方协议的 top-5 实现**。

V0 可用于第一轮受控方法判断，但不能将其 60 分与论文某表的 60 分横向比较。正式对外报告前，V1 对齐是必须完成的步骤，不是可省略的措辞。

### 13.2 独立标注输入格式

评估文件 `labels_eval_only/<video_id>.npz`：

```python
# bounds: [S, 2]，真实秒数；ratings: [S, U]，U 个标注者。
np.savez("labels_eval_only/v0002.npz", bounds=bounds, ratings=ratings)
```

该转换必须依据下载数据的字段说明和真实时钟。配套脚本不猜测你的 MATLAB/JSON 文件中哪个字段是时间、哪一轴是标注者，也不虚构文件已下载。

```bash
python evaluate_v0.py \
  --manifest /data/prepared/prepared.jsonl \
  --predictions /data/predictions/grpo_video.json \
  --annotations /data/labels_eval_only \
  --out /data/reports/grpo_v0.json
```

评估前冻结并保存预测文件哈希。任何利用这些标注选择 prompt、奖励权重、K、checkpoint 的行为，都使这些视频成为开发集；不能一边反复看结果一边称为一次独立测试。

### 13.3 至少报告这些信息

| 层面 | 主记录 |
|---|---|
| 数据 | 完整视频数、缺失/解码失败数、来源 split、片段数、窗口数 |
| 训练 | 输入曝光、rollout 数、成功更新、可训练参数、实际 tokens/GPU 小时 |
| 代理 | 排序奖励、校准奖励、原始重复性范围、簇大小、无区分组比例 |
| 输出 | JSON 合法率、截断率、同分率、跨窗口分差、预测分布 |
| 人类评价 | V0 与 V1 明确分列；逐视频结果和数据集宏平均 |
| 成本 | 模型大小、推理时间、相同推理预算下的质量、教师直接推理成本 |

### 13.4 统计检验不能用窗口伪装独立样本

主差异是同一视频上的配对差异：

$$
 d_n=m_n^{\mathrm{G\text{-}RANK}}-m_n^{\mathrm{S+}},\qquad \bar d=\frac1N\sum_n d_n.
$$

bootstrap 以视频/来源组为单位重采样，而非以高度相关的窗口为单位。若一个 source_group 有多个视频，先按来源组组织配对数据。配套 `paired_bootstrap` 只适用于输入条目已经是独立视频或来源组的情况。

单 seed 的视频 bootstrap 不能覆盖训练随机性。正式阶段同时给每个训练 seed 的结果及配对差异；若合并多个划分，应注意重复视频，不把重复测试出现次数当独立 N。小样本区间较宽应如实解释，不用反复换 split 获得显著性。

若同时检验多个主比较，预先声明一个主比较 `G-RANK − S+`，其他为次要/探索性。需要多重比较校正时按预登记 family 做，而不是只挑出 p 值最小的一组。

---

<a id="section-14"></a>

## 14. 如何读首轮结果，以及什么情况下值得继续

| 观察到的结果 | 可以说什么 | 下一步优先事项 |
|---|---|---|
| B0 已强，SFT/GRPO 都弱 | 当前学生没有有效利用信号；不能归因于奖励必然无效 | 查输入粒度、输出绑定、模型训练及全视频尺度可预测性 |
| S0 优于 B0，S+ 与 GRPO 接近 | 学习有价值，尚未证明 GRPO 的独特收益 | 以更简单训练作为强基线；不要硬写强化学习必要 |
| G-CAL 与 S+ 相近，G-RANK 更好 | 可能是排序奖励更匹配检测，而非“GRPO 换名” | 重复 seed、独立来源、保持计算记录 |
| 代理奖励升高，人类指标下降 | 对代理过度优化/代理错位是候选解释 | 检查重复背景、伪类别错误、窗口盲点；不只调小 KL |
| 真实指标提高，但推理开销大很多 | 有质量收益，部署价值未必成立 | 报质量—成本曲线，考虑轻量学生蒸馏 |
| 所有组大部分时间同奖励 | 探索、奖励分辨率或协议可能退化 | 检查解析、输出多样性、组元数据，不直接宣称无监督无效 |
| 只在见过的题材有效 | 迁移不足 | 按来源和题材分析，设计新来源验证 |

不建议在没有先验效应量和学校要求的情况下凭空指定“涨 1% 就能毕业”。一个可信的继续条件是：**主比较在多个 seed/来源中方向稳定，输出有效，收益不是由额外评测搜索或数据泄漏造成，且有明确机制证据。**一次没有显著改善不是证明方法无效，但也不是继续无限追加同类实验的理由。

### 14.1 之后可能形成贡献的方向

这些是未来候选，不是第一轮已经实现、也不是已证明无人做过：

**奖励区分常见背景与重要事件。**若错误集中在背景、片头或长时间准备，可研究类内重复性相对于跨类背景的对比信号，或事件完整性的反馈。必须与“只是换了更强特征”区分。

**对伪偏好错误的鲁棒学习。**若不同库/扰动导致排序翻转，研究如何识别错误偏好并减少其影响；只有真实观察到该问题才引入专门可信度机制。

**跨视频监督到单视频决策的迁移。**检验学生是否只记住伪类别和背景，还是能在新来源中利用局部动作；设置背景替换、时间打乱等明确诊断，并避免在测试上调方法。

**局部输入与全视频尺度。**若跨窗口分数失配是主要瓶颈，可研究明确的相对比较/锚点机制，而不简单把窗口 max-min 归一化。新增方法仍需证明比完整视频标准模型更值得。

硕士论文可以围绕一个问题形成完整链条：基线证明瓶颈 → 有针对性的奖励/学习机制 → 强对照 → 泛化和失败分析。不要把多个互不相关的小改动凑成所谓多个创新点。

---

<a id="section-15"></a>

## 15. 更严格的数学边界：哪些结论成立，哪些不能成立

以下为本方案的推导；除明确注明外，不是原论文提供的理论结果。定理结论都依赖列出的假设。

### 15.1 有限回答空间上的最优 KL 正则策略确实存在

对固定输入 $X$，设最大回答长度有限、词表有限，得到有限回答集合 $\mathcal Y$。固定参考分布 $p_0(y)>0$，有界奖励 $R(y)$，$\beta>0$。理想问题为：

$$
\max_p F(p)=\sum_y p(y)R(y)-\beta\sum_y p(y)\log\frac{p(y)}{p_0(y)},\quad \sum_y p(y)=1.
$$

定义：

$$
Z=\sum_y p_0(y)e^{R(y)/\beta},\qquad
p^*(y)=\frac{p_0(y)e^{R(y)/\beta}}{Z}.
$$

因为集合有限、奖励有界，$0<Z<\infty$。代回可得：

$$
F(p)=\beta\log Z-\beta D_{\mathrm{KL}}(p\|p^*).
$$

KL 非负，因此 $p^*$ 是唯一最优分布（在参考支持上）。且若 $R(y_1)>R(y_2)$，相对于参考策略，它会增加 $y_1$ 对 $y_2$ 的概率比：

$$
\frac{p^*(y_1)/p^*(y_2)}{p_0(y_1)/p_0(y_2)}
=e^{[R(y_1)-R(y_2)]/\beta}>1.
$$

**证明了什么：**代理奖励驱动的理想概率重分配在数学上有定义，并存在奖励—参考偏离的最优折中。

**没有证明什么：**LoRA 参数族能表示这个最优分布；有限样本 GRPO 会找到它；token 级裁剪、长度平均和近策略 KL 代理严格等价于上述序列级目标。不能把此闭式解当作实际 GRPO 的全局收敛证明。

### 15.2 代理奖励改善何时才能推出人类效用改善？

假设在评价输入/候选支持上存在统一误差界：

$$
|R(X,Y)-U(X,Y)|\le\varepsilon.
$$

则对任意两个策略 $\pi_1,\pi_0$：

$$
J_U(\pi_1)-J_U(\pi_0)
\ge J_R(\pi_1)-J_R(\pi_0)-2\varepsilon.
$$

证明：分别用 $J_U(\pi_1)\ge J_R(\pi_1)-\varepsilon$ 和 $J_U(\pi_0)\le J_R(\pi_0)+\varepsilon$ 相减即可。因此，只有代理收益超过 $2\varepsilon$ 时，这个界才保证真实收益为正。

进一步，若 $\hat\pi$ 在同一策略类上达到 $\delta$ 近似代理最优，则：

$$
J_U(\pi_U^*)-J_U(\hat\pi)\le 2\varepsilon+\delta.
$$

证明依次比较 $U$ 与 $R$，再使用代理最优性。**实际聚类奖励没有被证明满足小 $\varepsilon$，神经训练也没有被证明满足小 $\delta$。**这个定理解释需要怎样的条件，而不是提供不存在的实测保证。

### 15.3 有噪声的成对伪标签何时仍有正确方向？

设某一对片段存在二元真实顺序 $y\in\{-1,+1\}$；伪标签以独立且对称的概率 $\eta<1/2$ 翻转，错误概率与预测无关。对确定预测 $\hat y$：

$$
\mathbb E[\mathbf 1(\hat y=\tilde y)\mid y,\hat y]
=\eta+(1-2\eta)\mathbf1(\hat y=y).
$$

因为 $1-2\eta>0$，最大化期望伪正确率也最大化真实正确率。预测并列记为 1/2 时同样成立。

若错误具有背景相关、题材相关或幅度相关的系统偏置，上述结论不再自动成立；使用 $|q_i-q_j|$ 加权还要求噪声条件与权重相容。**本文不宣称重复性伪标签满足独立对称噪声。**因此需要实际的奖励审计和跨来源测试。

### 15.4 “平均重复性必然找到高光”是错误命题：一个反例

设同簇片段仅有两个单位特征：背景 $b=(1,0)$ 占 90%，真实精彩动作 $h=(0,1)$ 占 10%，两者正交。则簇均值为：

$$
\mu=0.9b+0.1h.
$$

重复性给出 $\rho(b)=0.9$、$\rho(h)=0.1$，把背景排在高光前面。对它做百分位映射和 GRPO 不会自动纠正这个错误定义；若精确优化代理，反而可能强化背景。

这个反例不证明原方法在真实数据必然失败，只证明不存在不加语义假设的普遍保证。配套测试中有该反例，避免把方法直觉写成定理。

### 15.5 纯无标签信息不能无条件识别人类偏好

固定同一个视频数据分布 $P(V)$，构造两种人类效用：在所有候选中，$U_1$ 偏好 A，$U_2$ 偏好 B，且两者的偏好完全相反。一个只观察同一无标签数据和相同先验的算法，其输出分布在这两个世界中相同，不能同时保证在两种效用下最优。

所以“无人工高光标签”仍需要任务假设或预训练先验。聚类重复性提供的是一种可检验的先验，不是从数据分布中唯一推出审美真值。Qwen 的预训练也属于外部先验，应披露。

### 15.6 固定独立分数的 top-k 最优解不需要 GRPO

若奖励为 $R(S)=\sum_{t\in S}q_t$ 且 $|S|=k$，设某个候选集合包含 $q_i<q_j$ 的 $i$ 却未包含 $j$。将 $i$ 换为 $j$ 会使奖励增加 $q_j-q_i>0$。重复交换后得到 top-k，因此 top-k 全局最优。

这说明 GRPO 对此奖励不能超越直接排序的代理最优值。学习模型仍可能通过预训练先验、正则和泛化，在新视频的人类评价上超过直接代理；但那是经验问题，需要 B0 和 S+ 对照。

### 15.7 为什么普通回归不需要绕路，当前实验仍可值得做？

若连续预测 $s\sim\mathcal N(\mu,\sigma^2)$，奖励是 $-(s-q)^2$：

$$
\mathbb E[(s-q)^2]=(\mu-q)^2+\sigma^2.
$$

直接平方误差已经提供明确梯度，GRPO 并没有凭空增加监督信息。这也是设置 G-CAL 的原因。

本方案尝试的是非可微文本分数解析与离散排序奖励。策略梯度可以绕过奖励不可微性，但还存在排序代理、直接比较学习等替代方法。**“允许优化”与“统计/计算上更高效”是两种不同命题。**

### 15.8 所有流程的证明清单

| 环节 | 已给出的形式论证 | 仍需实测的部分 |
|---|---|---|
| 视频切片 | 区间定义、覆盖要求、PTS 对齐契约 | 稀疏帧是否看见关键动作 |
| 特征/聚类 | 单位向量范围、K-means 单步不增目标 | 语义类别与高光关系 |
| 重复性 | 平均余弦的精确均值化简、范围与复杂度 | 代理是否偏好真实高光 |
| 伪尺度 | 单调性、并列处理、量化误差界 | 局部模型能否推断全局百分位 |
| SFT/LoRA | 目标定义、梯度路径、合并代数等价 | 泛化、收敛与有限精度一致性 |
| 奖励 | 有界性、排序语义、无人工标签信息流 | 错误伪标签与奖励投机 |
| GRPO | score-function 梯度、组均值因子、KL 条件等式 | 组标准差/裁剪/长度平均下的实际优化收益 |
| 全视频推理 | 覆盖、凸平均、区间积分性质 | 分数校准、人类 AP 与完整事件保留 |
| 独立评价 | 配对比较与正确抽样单位 | 样本量、seed 方差、跨域可迁移性 |

---
<a id="section-16"></a>

## 16. 关键代码示例：与配套文件保持一致

### 16.1 核心数值、严格解析与奖励

下面是配套 `core.py`。采用独立于训练框架的实现，便于先验证公式和数据契约。原形式重复性在正式构库脚本中还会按簇缓存均值，避免对每个查询视频重复构库。

```python
"""CPU-testable core for label-free highlight experiments.

No human highlight labels are accepted by reward or pseudo-label functions.
The module is intentionally independent of Qwen / ms-swift.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence
import numpy as np


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("features must be a nonempty finite [clips, dim] array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("zero feature: fix extraction; do not silently create a cosine")
    return x / norms


def recurrence_original(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Exact mean cosine, using a NON-renormalized mean of unit bank vectors."""
    q, b = unit_rows(query), unit_rows(bank)
    if q.shape[1] != b.shape[1]:
        raise ValueError("feature dimensions differ")
    return q @ b.mean(axis=0)


def recurrence_loo_video(
    video_id: str,
    feature_map: Mapping[str, np.ndarray],
    clusters: Mapping[str, int],
    *,
    global_fallback: bool = True,
) -> tuple[np.ndarray, dict]:
    """Optional adaptation: exclude the entire query video; weight peer videos equally.

    feature_map/clusters MUST contain training videos only. Cluster membership can
    still have been fitted using this video's embedding: this is not full cross-fitting.
    """
    if set(feature_map) != set(clusters) or video_id not in feature_map:
        raise ValueError("feature/cluster keys differ or unknown video")
    peers = [k for k in feature_map if k != video_id and clusters[k] == clusters[video_id]]
    fallback = False
    if not peers:
        if not global_fallback:
            raise ValueError("no other video in this cluster")
        peers = [k for k in feature_map if k != video_id]
        fallback = True
    if not peers:
        raise ValueError("at least two training videos are required")
    means = np.stack([unit_rows(feature_map[k]).mean(axis=0) for k in peers])
    score = unit_rows(feature_map[video_id]) @ means.mean(axis=0)
    return score, {"peers": peers, "global_fallback": fallback}


def midrank_percentile(values: Sequence[float]) -> np.ndarray:
    """Whole-video midrank ECDF. Ties receive equal scores; constant input -> 0.5."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("values must be finite nonempty 1D")
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        # Occupied 1-based ranks are i+1,...,j. (average_rank - 0.5) / T.
        result[order[i:j]] = (i + j) / (2.0 * len(x))
        i = j
    return result


def _no_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_scores(text: str, n_clips: int, max_score: int = 100) -> np.ndarray:
    """Strict JSON parser. No eval, regex repair, dropped clips, or clamping."""
    if not isinstance(text, str) or len(text) > 20000 or n_clips < 1:
        raise ValueError("invalid completion input")
    obj = json.loads(text, parse_constant=_no_constant, object_pairs_hook=_unique_object)
    if not isinstance(obj, dict) or set(obj) != {"scores"}:
        raise ValueError('expected exactly {"scores": [...]}')
    scores = obj["scores"]
    if not isinstance(scores, list) or len(scores) != n_clips:
        raise ValueError("wrong score count")
    if any(type(s) is not int or s < 0 or s > max_score for s in scores):
        raise ValueError("scores must be integers in the declared range")
    return np.asarray(scores, dtype=np.float64) / max_score


@dataclass(frozen=True)
class RewardParts:
    total: float
    ranking: float
    calibration: float
    valid: bool


def reward_parts(completion: str, pseudo_scores: Sequence[float], rank_weight: float = 0.8) -> RewardParts:
    q = np.asarray(pseudo_scores, dtype=np.float64)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        # Dataset bugs are not policy failures: fail rather than silently rewarding.
        raise ValueError("pseudo_scores must be finite in [0,1]")
    if not 0 <= rank_weight <= 1:
        raise ValueError("rank_weight outside [0,1]")
    try:
        s = parse_scores(completion, len(q))
    except (ValueError, TypeError, json.JSONDecodeError):
        return RewardParts(-1.0, 0.0, 0.0, False)
    ii, jj = np.triu_indices(len(q), k=1)
    delta_q, delta_s = q[ii] - q[jj], s[ii] - s[jj]
    w = np.abs(delta_q)
    correct = (delta_q * delta_s > 0).astype(np.float64)
    correct += 0.5 * (delta_s == 0)
    ranking = float(np.dot(w, correct) / w.sum()) if w.sum() > 0 else 0.5
    calibration = float(1.0 - np.mean((s - q) ** 2))
    total = rank_weight * ranking + (1.0 - rank_weight) * calibration
    return RewardParts(float(total), ranking, calibration, True)


def group_advantages(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Explicit population std (ddof=0). Framework std convention must be logged."""
    x = np.asarray(rewards, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("expected finite [prompts, generations>=2]")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    centered = x - x.mean(axis=1, keepdims=True)
    denominator = x.std(axis=1, keepdims=True) + eps
    return np.divide(centered, denominator, out=np.zeros_like(x), where=denominator > 0)


def window_indices(n_clips: int, width: int = 8, stride: int = 4) -> list[list[int]]:
    if n_clips < 1 or width < 1 or not 1 <= stride <= width:
        raise ValueError("invalid window parameters")
    if n_clips <= width:
        return [list(range(n_clips))]
    starts = list(range(0, n_clips - width + 1, stride))
    if starts[-1] != n_clips - width:
        starts.append(n_clips - width)
    return [list(range(s, s + width)) for s in starts]


def aggregate_windows(n_clips: int, predictions: Sequence[tuple[Sequence[int], str]]) -> tuple[np.ndarray, float]:
    """Invalid output contributes zero for its window, never silently disappears."""
    total = np.zeros(n_clips, dtype=np.float64)
    count = np.zeros(n_clips, dtype=np.int64)
    invalid = 0
    for ids, text in predictions:
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= n_clips)):
            raise ValueError("invalid clip indices")
        try:
            scores = parse_scores(text, len(ids))
        except (ValueError, TypeError, json.JSONDecodeError):
            invalid += 1
            scores = np.zeros(len(ids))
        total[ids] += scores
        count[ids] += 1
    if np.any(count == 0):
        raise ValueError("uncovered evaluation clips")
    return total / count, invalid / max(len(predictions), 1)


def interval_average(source_bounds: np.ndarray, scores: Sequence[float], target_bounds: np.ndarray) -> np.ndarray:
    """Duration-overlap mapping onto benchmark intervals, not nearest-index mapping."""
    src, dst = np.asarray(source_bounds, dtype=float), np.asarray(target_bounds, dtype=float)
    s = np.asarray(scores, dtype=float)
    if src.shape != (len(s), 2) or dst.ndim != 2 or dst.shape[1] != 2:
        raise ValueError("bad interval shapes")
    if not (np.isfinite(src).all() and np.isfinite(dst).all() and np.isfinite(s).all()):
        raise ValueError("non-finite intervals/scores")
    if np.any(src[:, 1] <= src[:, 0]) or np.any(dst[:, 1] <= dst[:, 0]):
        raise ValueError("nonpositive interval duration")
    overlap = np.maximum(0.0, np.minimum(dst[:, None, 1], src[None, :, 1]) - np.maximum(dst[:, None, 0], src[None, :, 0]))
    mass = overlap.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("an evaluation interval has no prediction coverage")
    # Full coverage check requires non-overlapping source bins.
    if np.any(src[1:, 0] < src[:-1, 1] - 1e-8):
        raise ValueError("source intervals overlap or are out of order")
    if not np.allclose(mass, dst[:, 1] - dst[:, 0], rtol=1e-5, atol=1e-6):
        raise ValueError("partial evaluation coverage; fix timestamps instead of renormalizing")
    return overlap @ s / mass


def paired_bootstrap(differences: Sequence[float], n_boot: int = 10000, seed: int = 0) -> dict:
    """Video-level paired bootstrap only; does NOT estimate training-seed uncertainty."""
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError("invalid paired differences")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        estimates[i] = rng.choice(d, size=len(d), replace=True).mean()
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_videos": len(d)}
```

### 16.2 ms-swift 奖励插件

该类仅使用 completion 与顶层 `pseudo_scores`，不读取人工标注、不调用大模型裁判。插件签名按 v4.6.0 官方示例核对，但真实 Trainer 的跨卡元数据展开仍要通过第 10.5 节测试。

```python
"""ms-swift v4.6.0 reward plugin. Framework integration requires a GPU smoke test."""
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import reward_parts
from swift.rewards import ORM, orms


class HighlightRecurrenceReward(ORM):
    def __call__(self, completions, pseudo_scores, **kwargs):
        if len(completions) != len(pseudo_scores):
            raise ValueError("rollout/metadata lengths differ; check repetition across ranks")
        weight = float(os.environ.get("HL_RANK_WEIGHT", "0.8"))
        values = []
        for completion, q in zip(completions, pseudo_scores):
            if not isinstance(completion, str):
                raise TypeError("expected decoded completion strings from ms-swift v4.6.0")
            values.append(reward_parts(completion, q, weight).total)
        return values


orms["highlight_recurrence"] = HighlightRecurrenceReward
```

### 16.3 GRPO 张量参考实现

用于验证梯度方向、mask、停止梯度、组顺序和数值有限性；分布式训练、模型生成、旧策略快照、全局组聚合由框架承担。`logp_new` 必须来自当前模型，不能把缓存 log probability 当成可训练张量。

```python
"""Small PyTorch loss reference for unit testing, NOT a replacement distributed trainer."""
from __future__ import annotations
import torch


def grpo_loss(logp_new, logp_old, logp_ref, rewards, mask, group_size=4, beta=0.04, clip=0.2, eps=1e-4):
    """Inputs: logps/mask [B*G,L], rewards [B*G]; all rows in prompt-major order.

    Uses population std explicitly. K3 is evaluated on rollout samples and is not an
    exact off-policy KL estimator. This reproduces a common local surrogate only.
    """
    if group_size < 2 or beta < 0 or clip < 0 or eps <= 0:
        raise ValueError("invalid GRPO hyperparameters")
    if logp_new.ndim != 2 or logp_new.shape != logp_old.shape or logp_new.shape != logp_ref.shape or mask.shape != logp_new.shape:
        raise ValueError("log-prob/mask shape mismatch")
    n, length = logp_new.shape
    if n % group_size or rewards.shape != (n,):
        raise ValueError("invalid grouping")
    mask = mask.to(logp_new.dtype)
    lengths = mask.sum(-1)
    if torch.any(lengths <= 0):
        raise ValueError("empty completion")
    with torch.no_grad():
        grouped = rewards.detach().float().reshape(-1, group_size)
        adv = (grouped - grouped.mean(-1, keepdim=True)) / (grouped.std(-1, keepdim=True, unbiased=False) + eps)
        adv = adv.reshape(n, 1)
    delta = logp_new.float() - logp_old.detach().float()
    if not torch.isfinite(delta).all():
        raise FloatingPointError("non-finite policy log-ratio")
    ratio = delta.exp()
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
    dref = logp_ref.detach().float() - logp_new.float()
    k3 = torch.expm1(dref) - dref
    per_seq = ((-surrogate + beta*k3) * mask).sum(-1) / lengths
    loss = per_seq.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite GRPO loss")
    return loss
```

### 16.4 完整阶段命令模板

```bash
#!/usr/bin/env bash
# Templates checked against ms-swift v4.6.0 docs; NOT GPU-executed here.
# Usage: bash run.sh sft0 /local/Qwen3-VL-2B-Instruct
#        bash run.sh merge /local/Qwen3-VL-2B-Instruct /actual/sft0/checkpoint-256
#        bash run.sh sft-continue ./outputs/merged_sft0
#        bash run.sh grpo ./outputs/merged_sft0
#        HL_RANK_WEIGHT=0 bash run.sh grpo ./outputs/merged_sft0
set -euo pipefail
cd "$(dirname "$0")"
STAGE=${1:?stage required}
MODEL=${2:?local model directory or model ID required}
WORLD=${NPROC_PER_NODE:-1}
SEED=${HL_SEED:-42}
PROMPTS=${HL_PROMPTS:-4}
G=${HL_GROUP_SIZE:-4}
STEPS=${HL_STEPS:-256}
DATA=${HL_DATA:-./data}
OUT=${HL_OUTPUT:-./outputs}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-784}
export VIDEO_MIN_TOKEN_NUM=${VIDEO_MIN_TOKEN_NUM:-64}
export VIDEO_MAX_TOKEN_NUM=${VIDEO_MAX_TOKEN_NUM:-64}
export FPS=2
export FPS_MAX_FRAMES=32
export FPS_MIN_FRAMES=4
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT"
common=(--model "$MODEL" --tuner_type lora --target_modules all-linear
        --freeze_vit true --freeze_aligner true --freeze_llm false
        --lora_rank 16 --lora_alpha 32 --lora_dropout 0
        --bf16 true --torch_dtype bfloat16 --attn_impl sdpa --gradient_checkpointing true
        --per_device_train_batch_size 1 --max_length 8192
        --packing false --padding_free false --split_dataset_ratio 0
        --dataset_num_proc 1 --dataloader_num_workers 2 --seed "$SEED" --data_seed 42
        --max_steps "$STEPS" --logging_steps 1 --save_steps 128 --save_total_limit 2
        --eval_strategy no --lr_scheduler_type cosine --warmup_ratio 0.05
        --weight_decay 0 --max_grad_norm 1 --strict true --report_to tensorboard)
case "$STAGE" in
sft0)
  (( 16 % WORLD == 0 )) || { echo "world size must divide 16" >&2; exit 2; }
  swift sft "${common[@]}" --dataset "$DATA/train_sft.jsonl" \
    --learning_rate 5e-5 --gradient_accumulation_steps "$((16/WORLD))" \
    --output_dir "$OUT/sft0_seed${SEED}"
  ;;
merge)
  ADAPTER=${3:?SFT checkpoint directory required}
  swift export --model "$MODEL" --adapters "$ADAPTER" --merge_lora true \
    --output_dir "$OUT/merged_sft0"
  ;;
sft-continue)
  (( PROMPTS % WORLD == 0 )) || { echo "prompt batch must divide world size" >&2; exit 2; }
  swift sft "${common[@]}" --dataset "$DATA/train_sft.jsonl" \
    --learning_rate 1e-5 --gradient_accumulation_steps "$((PROMPTS/WORLD))" \
    --output_dir "$OUT/sft_continue_seed${SEED}"
  ;;
grpo)
  COMPLETIONS=$((PROMPTS*G))
  (( COMPLETIONS % WORLD == 0 )) || { echo "completion batch must divide world size" >&2; exit 2; }
  # Do NOT pass --ref_model in LoRA mode. The frozen merged SFT base is the reference.
  swift rlhf "${common[@]}" --rlhf_type grpo --dataset "$DATA/train_grpo.jsonl" \
    --external_plugins "$PWD/reward_plugin.py" --reward_funcs highlight_recurrence \
    --remove_unused_columns false --learning_rate 1e-5 \
    --gradient_accumulation_steps "$((COMPLETIONS/WORLD))" \
    --generation_batch_size "$COMPLETIONS" --num_generations "$G" \
    --use_vllm false --loss_type grpo --advantage_estimator grpo \
    --scale_rewards group --kl_in_reward false --importance_sampling_level token \
    --num_iterations 1 --beta 0.04 --epsilon 0.2 --dynamic_sample false \
    --overlong_filter false --temperature 1 --top_p 1 --top_k 0 \
    --repetition_penalty 1 --max_completion_length 128 \
    --output_dir "$OUT/grpo_rank${HL_RANK_WEIGHT:-0.8}_seed${SEED}"
  ;;
*) echo "unknown stage: $STAGE" >&2; exit 2;;
esac
```

代码不把推理分数当成实际人类标签；示例路径也不是已经下载的资产。其余预处理、数据构建、推理规范化和评价脚本见配套代码包。

---

<a id="section-17"></a>

## 17. 给执行者的最终检查与结果模板

### 17.1 开始正式训练之前

确认四件事：数据来源无交叉；伪标签和模型看到的片段一一对应；共同 SFT 起点和冻结参考正确；完整评估输入覆盖所有测试片段。随后只执行预登记的矩阵，不自动启动音频、跟踪或额外大模型实验。

建议保存每个实验的独立收据，至少包含：

```json
{
  "run_id": "G_RANK_seed42",
  "status": "PLANNED",
  "data_manifest_sha256": "待实际计算",
  "pseudo_label_sha256": "待实际计算",
  "sft_parent_checkpoint": "待实际路径",
  "base_model_revision": "待核验",
  "framework": "ms-swift 4.6.0",
  "seed": 42,
  "n_successful_optimizer_updates": 0,
  "n_prompt_exposures": 0,
  "n_rollout_completions": 0,
  "used_human_highlight_labels_for_training": false,
  "gpu_training_verified": false
}
```

这是状态格式示例，不能把 `PLANNED` 修改为 `COMPLETED` 而不补端点和日志。断点续训需要优化器、scheduler、scaler、各 rank RNG、数据游标及精确父模型；只加载 adapter 不等于原样续训。[S11]

### 17.2 最终结果表

| 方法 | seed/split | V1 官方协议指标 | V0 自定义诊断 | 合法率 | 输入曝光/rollout | GPU 小时 | 推理成本 |
|---|---|---:|---:|---:|---|---:|---:|
| 直接重复性 | 固定 | 待测 | 待测 | 不适用 | 0 | 待测 | 待测 |
| Qwen 零样本 | 固定 | 待测 | 待测 | 待测 | 0 | 0 训练 | 待测 |
| S0 | 预登记 | 待测 | 待测 | 待测 | 待记 | 待测 | 待测 |
| S+ | 与 G 配对 | 待测 | 待测 | 待测 | 待记 | 待测 | 待测 |
| G-CAL | 与 S+ 配对 | 待测 | 待测 | 待测 | 待记 | 待测 | 待测 |
| G-RANK | 与 S+ 配对 | 待测 | 待测 | 待测 | 待记 | 待测 | 待测 |

不要填写臆测值。额外附逐视频差异、训练曲线、伪奖励与人类指标的背离案例，以及未纳入统计的视频名单。

### 17.3 研究论点的合格表达

首轮之前：**“我们提出并计划检验一个无人工高光标签的 SFT→GRPO 基线。”**

若首轮有正结果：**“在固定数据、共同起点及给定预算下，观察到某个受控收益，尚需跨 seed/数据验证。”**

经过正式实验后，才能根据真实证据写明：奖励是否更合理、GRPO 是否超越伪标签 SFT、能否泛化、成本是否值得。不要预先写“首次”“证明无监督解决高光检测”或“保证达到硕士毕业要求”。

---

<a id="section-18"></a>

## 18. 参考来源与本次核验边界

以下均为原论文、官方模型页、官方文档或官方源码；访问核对日期为 2026-10-27。网页的后续版本可能变化，目标机应归档本次使用版本。文中数学推导标明为本方案推导的部分，不冒称来源论文的定理。

**[S1]** Zahidul Islam, Sujoy Paul, Mrigank Rochan. *Unsupervised Video Highlight Detection by Learning from Audio and Visual Recurrence*. WACV 2025；核对 arXiv v2 的方法、实验设置及相关表格。

- https://arxiv.org/html/2407.13933v2
- https://arxiv.org/pdf/2407.13933v2

**[S2]** Qwen 官方模型：*Qwen3-VL-2B-Instruct*。用于确认模型与多模态接口，不把模型卡中可能滞后的安装提示照搬为最新环境锁。

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

**[S3]** ms-swift 4.6：GRPO 算法、实现和日志文档。

- https://swift.readthedocs.io/en/v4.6/Instruction/GRPO/GetStarted/GRPO.html

**[S4]** ms-swift 4.6：自定义数据、messages、多模态路径与额外字段。

- https://swift.readthedocs.io/en/v4.6/Customization/Custom-dataset.html

**[S5]** ms-swift v4.6.0 官方源码：奖励插件与 RLHF 参数/参考模型处理。

- https://raw.githubusercontent.com/modelscope/ms-swift/v4.6.0/examples/train/grpo/plugin/plugin.py
- https://raw.githubusercontent.com/modelscope/ms-swift/v4.6.0/swift/arguments/rlhf_args.py

**[S6]** Hugging Face TRL：GRPO Trainer 文档。用于提醒不同默认损失、标准差/长度设置及实现选择，不声称与 ms-swift 逐项相同。

- https://huggingface.co/docs/trl/main/en/grpo_trainer

**[S7]** TVSum 官方资源。配套代码没有下载或打包其原视频/标注，也没有猜测其标签字段。

- https://github.com/yalesong/tvsum

**[S8]** Transformers 官方 CLIP 文档：`CLIPVisionModelWithProjection` 与视觉投影特征。

- https://huggingface.co/docs/transformers/main/model_doc/clip

**[S9]** UMAP 官方文档：将新数据变换到训练得到的低维空间。

- https://umap-learn.readthedocs.io/en/latest/transform.html

**[S10]** Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. GRPO 的原始来源。本文自定义高光奖励、数学反例和代码不是该论文原始实验。

- https://arxiv.org/abs/2402.03300

**[S11]** ms-swift 4.6：命令行参数、LoRA、GRPO batch、视觉 token、推理和断点语义。

- https://swift.readthedocs.io/en/v4.6/Instruction/Command-line-parameters.html

**[S12]** ms-swift 官方 releases 与 v4.6.0 依赖约束。框架版本锚点不构成目标机完整环境锁。

- https://github.com/modelscope/ms-swift/releases
- https://raw.githubusercontent.com/modelscope/ms-swift/v4.6.0/requirements/framework.txt

**[S13]** scikit-learn：`average_precision_score` 的官方定义。用于 V0 自定义诊断，不代表 TVSum 的某篇论文协议。

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

---

<a id="section-19"></a>

## 19. 交付时实际执行了哪些验证？

### 19.1 已完成

本次在 CPU 环境中运行 **33 项 pytest 单元测试，全部通过**。内容包括平均余弦化简、整视频排除、并列百分位、严格 JSON 非法情况、奖励有界性和排序方向、组标准化、窗口覆盖、时间积分、GRPO 有限梯度及 old/ref 停止梯度、组均值梯度比例的有限枚举、重复性反例、bootstrap、量化误差和 prompt 无伪答案泄漏。

另运行 **6 项合成管线检查，全部通过**：

1. 12 个合成训练视频拟合 PCA/K-means，3 个合成测试视频仅做推断；簇原型打分与直接平均余弦数值一致。
2. 生成 48 对索引一致的 SFT/GRPO 输入；推理记录不包含伪标签。
3. 打乱预测行顺序后按明确 ID 恢复对应，并完成重叠窗口全覆盖汇总。
4. 在隔离的合成标注上运行明确命名的 V0 自定义评价。
5. 只改变测试特征，训练的簇选择及训练伪标签保持不变。
6. 人为制造 source_group 跨训练/测试重叠时，脚本拒绝继续。

全部 Python 脚本通过语法编译检查，两个 Shell 脚本通过 `bash -n`。代码包内保存 `validation_synthetic.json` 与总体验证收据。

### 19.2 未完成，不可冒充

没有运行真实视频解码或 CLIP/Qwen 模型，没有验证真实 UMAP 结果，没有在 GPU 上启动 SFT/GRPO，没有实测 LoRA 合并或 ms-swift 多卡元数据，没有获得任何 TVSum/YouTube Highlights 的真实新指标。

合成管线显式使用 PCA 调试后端；不能把这次检查写成 UMAP 或原论文复现成功。CPU 测试表明核心公式和文件接口在这些测试条件下成立，不能推出完整模型必然学到人类高光。

**最终结论：这是一份可以交给执行者开展首轮实验的方案与参考代码；其数学对象有定义，关键数值性质已验证，但科研有效性必须由后续真实对照决定。**
