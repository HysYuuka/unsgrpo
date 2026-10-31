# 同图裁剪排序实验：执行接管记录

更新时间：2026-10-30T18:31:14.856104+08:00。

已接管原 supervisor478364 / worker478399，未启动第二份训练队列。受控矩阵仍为12项，先全部30 epochs，再按原有单次统一规则决定是否延长到50；所有正式测试等完整端点冻结后执行。冻结协议、GAIC清单与既有训练源码未改变。

当前为 FULL_s17，第15/30 epoch，global step4665。已经完成14轮训练和验证；当前验证最优为epoch12，200张验证原图 Acc1/5=59.0%、SRCC=0.86114、原生MOS regret=0.29795。这是单seed验证结果，尚不能判断FULL是否胜过MATCHED/MUSIQ_FT。其余11项仍排队。

近期完整epoch含验证/保存约7.9分钟；当前首项约还需2.1小时，预计香港时间20:30–21:00左右结束30轮，实际以日志为准。FULL类单次约4小时，MUSIQ尚无正式100步吞吐；不能把该估算套用于整个矩阵。

CPC CPU转换与重叠筛查已完成：10797张有效标注原图，各24个候选；每图保留4–10份原生评分向量，按原始assigned preference score取均值，不冒充GAIC的1–5 MOS。按先定规则保守排除2个可能与GAIC训练/验证来源重叠的组，保留10795个来源、259080个候选；另有2个可能与GAIC TEST重叠的来源，会分别报告包含/排除这些来源的结果。这里的“可能”包括未核实的pHash/原始ID提名，不是已经证明的内容相同。不是对所有变形/裁切近重复的穷尽性无重叠保证。

CPC转换协议、manifest与未来评价代码均已锁定，尚未做任何模型CPC推理。CPU独立复核覆盖259080个原生分数、坐标合法性、原图分组权重、对齐seed、并列分数、指标污染检测以及测试前执行门禁。已查看两张预先哈希选出的真实原图框叠加，未作美学投票。CPC将在核心矩阵及其GPU计时完成后、GPU空闲时串行评价全部12个受控端点和2个已验收发布模型；真实GPU入口尚待该时点检验，出错必须记录为执行问题。

现有GAIC发布权重和MUSIQ原始权重评价仍由原队列接续。FGAesQ完整DiffToken权重与FGAesthetics-Cropping数据的新增3次元数据、3次直接下载探测均出现SSL EOF，保留确切失败记录；未用单图FGAesQ替代完整版，也未用裁剪图冒充原图。GAFIC仍保留原源码审计缺口。以上缺口不阻塞核心机制比较，但不能据此宣称完整外部竞争力或SOTA。

监控每1800秒写入logs/execution_health.jsonl及logs/execution_monitor_latest.json；不重复启动/自动重启训练，不改目标，不向外发送消息。supervisor、监控和外部等待器均脱离终端运行，可在聊天断开后继续。暂停/错误不会被等待器自动当作完成。主队列结束后的人工研究复核、完整分项计时及最终综合报告仍须核对，不把脚本存在当作已经交付。

继续遵守用户已取消每日自动同步的直接指示：本地版本记录，未配置或推送remote；旧交接中每日同步表述不执行。未使用子智能体，未删除共享GAIC资产。

证据：EXECUTION_TAKEOVER_20261030.json、CPC_CONVERSION_AND_OVERLAP.json、CPC_EXTERNAL_LOCK.json、CPC_EVALUATION_CODE_FREEZE.json、CPC_EXTERNAL_CPU_ACCEPTANCE.json及EXTERNAL_STATUS.json。
