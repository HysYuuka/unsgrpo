# 存储与按需同步执行记录

## 已落实的目录与预算

`/root/autodl-tmp` 是独立 XFS 项目盘，实际容量 **50 GiB**；`/root` 根目录的 30 GiB overlay 不是项目盘。`/root/autodl-fs` 指向 `/autodl-fs/data`，实际文件盘容量 **200 GiB**。使用 `df -B1 /root/autodl-tmp /root/autodl-fs` 核验，避免把 overlay 余量当作实验空间。

| 存储 | 内容与绝对路径 | 预算 |
|---|---|---:|
| 本地 | `/root/autodl-tmp/unsgrpo` 代码，加 `/root/autodl-tmp/unsgrpo-work/{envs,cache,prepared}` | 15 GiB |
| 本地 | `/root/autodl-tmp/unsgrpo-work/{runs,checkpoints}` | 25 GiB |
| 本地 | `/root/autodl-tmp/unsgrpo-work/tmp` 与可用余量 | 10 GiB |
| 文件盘 | `/root/autodl-fs/unsgrpo/models` | 80 GiB |
| 文件盘 | `/root/autodl-fs/unsgrpo/datasets`，包括原视频、伪标签与隔离评价标签 | 50 GiB |
| 文件盘 | `/root/autodl-fs/unsgrpo/archives` | 50 GiB |
| 文件盘 | 空闲安全余量 | 20 GiB |

预算单位是 GiB，属于执行约束而非文件系统硬配额。资产下载、解压、模型合并前，额外空间估计必须包括同时保留的源文件和目标文件。例如合并时同时保存基础模型、LoRA、合并模型；解压时同时保存压缩包和解压结果。不可通过删除必要端点或断点状态来掩盖超预算。

```bash
source /root/autodl-tmp/unsgrpo/storage-env.sh
python /root/autodl-tmp/unsgrpo/scripts/storage_audit.py \
  --output /root/autodl-tmp/unsgrpo/reports/storage_audit.json
# 示例：预留文件盘额外 12 GiB 的操作峰值，仍须保留20 GiB可用。
python /root/autodl-tmp/unsgrpo/scripts/storage_audit.py \
  --reserve-path /root/autodl-fs/unsgrpo/models --peak-extra-bytes 12884901888
```

`storage-env.sh` 明确导出 `HL_DATA`、`HL_OUTPUT`、原视频/评价标签/帧缓存/环境路径。训练命令仍需显式引用这些变量，训练器不会自动理解 UNSGRPO 名称。模型放文件盘、活跃环境和帧放本地；归档必须先复制、校验哈希，再决定是否清除可再生副本。

`storage_audit.py --throughput-mib 16` 生成小规模顺序写入（含 fsync）及缓存读取测试。该结果仅能核对挂载和基础吞吐，不能外推随机 I/O、冷读或持续训练吞吐。探测文件退出时删除。

## Git 和同步状态

本地 `main`，远程固定为 `https://github.com/HysYuuka/unsgrpo.git`。2026-10-31 已确认 GitHub 登录账户 HysYuuka 和仓库写权限；提交使用该已验证账户的 GitHub noreply 身份，凭据不写入源码。**用户已取消自动每日同步：仅在用户每次明确要求同步时执行提交/推送，不设置任何cron或其他定时器。** 本次用户明确要求实验停止后的归档同步，实际提交与远端结果由该次同步收据记录。

`scripts/sync_daily.sh` 保留原文件名，现作为按需手动工具。`--check` 核对远程和身份，并列出待同步轻量文件；缺失身份会退出失败。只有用户本次明确要求同步后，才调用 `--sync` 审查已有暂存文件、添加明确白名单内且不超过 5 MiB 的源码/配置/清单/报告，提交后以非交互方式推送 `main`。2026-10-31 增加 research 归档目录、CUDA 源码和第三方 LICENSE 文本，以保留静态裁剪完整实现；其余标注、密钥、二进制和大文件检查仍适用。不会提交视频、评价标注内容、环境、模型、帧缓存和检查点。

审查同时读取真正暂存的 Git blob，拒绝符号链接、二进制、大文件、私钥和可识别的真实 token 形式；JSON/JSONL 中的人工标注载荷字段（例如 `user_anno`、`ratings`、`gt`、`labels`）被排除。源码中包含字段名不受影响，指标摘要和标签文件哈希可以同步。小型合成安全检查已验证：标注/token 夹具被排除，指标摘要被接受，未将夹具提交。收据的同步策略与调度状态明确为用户取消自动同步，不因机器存在cron程序而宣称同步已启用。

```bash
bash /root/autodl-tmp/unsgrpo/scripts/sync_daily.sh --check
# 仅在用户本次明确要求同步，且身份/仓库写权限可用时执行：
bash /root/autodl-tmp/unsgrpo/scripts/sync_daily.sh --sync
```

自动每日同步计划已撤销，不安装、不恢复定时任务，也不因训练完成而自行提交或推送。按需同步的日志和收据写 `/root/autodl-tmp/unsgrpo-work/runs/sync`。未来若用户明确要求某一次同步，报告该次真实提交与推送状态即可；改变为自动同步仍须用户新的明确指令。

## 依赖下载的操作峰值

`scripts/bootstrap_env.py` 在仓库外解析并并发下载依赖，不执行安装；固定继承的 Torch、TorchVision、NumPy，拒绝下载新的 Torch/Triton/NVIDIA 运行库。下载使用官方 PyPI 文件及其 SHA256，支持分块恢复，组装与块同时存在的峰值按尚未缓存包体积的两倍预估，始终保留本地 10 GiB 余量。代理绕过只作用于该子进程与下载 opener，不修改全局代理配置。

pip 24.0 的普通 `--dry-run --report` 仍会在解析后串行下载完整 wheel。本脚本在仅用于 dry-run 的独立子进程中保存 PEP658 元数据并跳过最后的实体准备；实际依赖选择仍使用 pip 原 resolver，不编辑已安装 pip。元数据、原始报告、约束与下载清单保留在 `/root/autodl-tmp/unsgrpo-work/cache/bootstrap`。只有下载收据为 `COMPLETED` 且全部包哈希通过后，执行一次 `--no-index --find-links .../wheelhouse --no-build-isolation` 本地安装；安装成功、导入兼容性、GPU 验收是后续独立状态，不因下载完成而自动视为通过。
