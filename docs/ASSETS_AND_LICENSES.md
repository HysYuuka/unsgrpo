# 首轮资产与来源

本轮仅下载 TVSum 官方完整 50 视频、Qwen3-VL-2B-Instruct 和 CLIP ViT-B/32 所需文件。原视频、评价标注和权重不进入代码仓库。

| 资产 | 固定身份与来源 | 许可记录 |
|---|---|---|
| TVSum v1.1 | 官方作者 MIT 下载入口；包 SHA256 与各视频 SHA256 见 `manifests/assets.json`、`manifests/tvsum_seed42.jsonl` | 官方仓库称原视频为 CC-BY 3.0；下载包同时包含旧 Yahoo Webscope 研究用途及不再分发条款，保留原文，本项目不把整个包称为可任意再分发资产 |
| Qwen3-VL-2B-Instruct | revision `89644892e4d85e24eaac8bacfd4f463576704203` | 官方模型卡 Apache-2.0 |
| openai/clip-vit-base-patch32 | revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | 原始 CLIP 项目 MIT；保留 HF 模型卡与转换版本身份 |

TVSum 当前官方 README 于 2019 年更新为作者直接公开下载入口，旧 Webscope 申请步骤已划去；包内 2016 年 README 仍保留旧条款。二者同时归档，不把旧文档的申请步骤当作当前机器已完成的事实。本轮按用户要求做本地学术研究分析，不再分发原始数据或标注。

划分只按来源身份和文件哈希进行，40 个训练来源、10 个测试来源，seed=42。未读取人工评分选择 split、聚类 K、prompt 或奖励。原视频的画面中可能自然带有标题/字幕；这些属于真实像素，提示未额外加入标题、人工类别或评分。

Qwen 经官方 ModelScope 镜像传输，CLIP 经 HF 镜像传输；完整权重分别与固定 HF revision 的 LFS SHA256 对齐。所有配置、processor、chat template 和词表身份见对应 `manifests/*.asset.json`。

来源：[TVSum 官方说明](https://github.com/yalesong/tvsum)、[官方数据目录](https://people.csail.mit.edu/yalesong/tvsum/)、[Qwen 模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[CLIP 原始许可](https://github.com/openai/CLIP/blob/main/LICENSE)。
