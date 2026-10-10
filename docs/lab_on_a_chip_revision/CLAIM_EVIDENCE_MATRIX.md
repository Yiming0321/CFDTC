# Claim–Evidence Matrix for the Lab on a Chip Revision

> 使用规则：任何题目、摘要、图题、Discussion 或 Cover Letter 中的高强度声明，都必须在本表中有直接证据。状态未达到 `Supported` 的声明不得进入最终摘要。

| ID | 候选声明 | 当前状态 | 已有证据 | 关键缺口 | 最低补充要求 | 未补足时的安全表述 |
|---|---|---|---|---|---|---|
| C01 | 平台可在 100 nW–1 mW 范围内定量热功率 | **Unverified** | 现有电加热训练与全局模型指标 | 缺少分功率区间误差、LoQ、独立组测试；低端全局指标不足以证明定量 | 锁定测试集；按功率区间报告 bias/MAE/RMSE/CI；LoB/LoD/LoQ | `signals were detected down to ...; quantitative performance was maintained above an LoQ of ...` |
| C02 | 全量程相对误差 <3% | **Do not use** | MLP 全局 MAPE 约 2.31% | MAPE 与低功率误差不匹配；数据划分可能泄漏；组合模型未单独报告 | 重新划分并重算；每区间误差和最终模型指标 | 仅陈述经验证的具体功率区间和误差 |
| C03 | 内部薄膜加热器提供可追溯参考 | **Partially supported** | 已知焦耳热输入 | 电热源与流体内化学热源空间分布不同 | 至少一种标准化学热标定；不确定度预算；比较斜率和偏差 | `co-located electrical reference input` |
| C04 | 计算校正提高温度外推稳定性 | **Needs re-analysis** | 现有 OOD 温度映射结果 | 数据 split、阈值和测试独立性需审计；温度范围有限 | 独立日期/温度组测试；固定参数；报告绝对指标和 CI | `temperature-guided alignment reduced error over the tested range` |
| C05 | 系统可在开放环境稳定运行 | **Not defined** | 非真空，恒温块和气凝胶隔热下运行 | “open”含义模糊；缺少环境扰动矩阵 | 温度斜坡、气流、隔热变化实验 | `non-vacuum, non-hermetic, thermostated benchtop conditions` |
| C06 | 方法属于 physics-informed ML | **Overstated** | 温度—信号线性映射和数据驱动模型 | 缺少明确能量平衡与物理约束损失/残差模型 | 建立灰箱热模型及 ablation | `physics-guided calibration and data-driven reconstruction` |
| C07 | 系统实现 real-time / online 热功率输出 | **Not yet supported** | 在线推断概念 | 当前零相位滤波需要未来数据；总延迟未测 | 因果滤波、在线稳态判断、延迟测量 | `offline reconstruction` 或 `online inference`（仅在真实运行后） |
| C08 | 微流控螺旋结构增强混合 | **Unsupported experimentally** | 结构示意及推断 | 缺少混合图像、混合指数和无量纲分析 | 荧光混合实验；Re/Pe/Dean；混合完成位置 | `spiral microchannel`，不直接声称 Dean 增强 |
| C09 | 体系可通过平均停留时间表征 | **Incomplete** | V/Q 计算 | 缺少 RTD，泵切换和管路分散未量化 | 示踪剂阶跃/脉冲；平均与方差；浓度生效延迟 | `nominal mean residence time` |
| C10 | 表面活性剂 CMC 和 ΔHmic 与参考一致 | **Partially supported** | 五个体系落入文献范围 | 文献条件跨度大；缺少同条件参考法和独立重复 | 同批试剂、相同温度的 ITC/可靠参考；n≥3 | `consistent with reported ranges under comparable conditions` |
| C11 | 固定扫描 40–60 min，比 ITC 更快 | **Needs fair accounting** | 现有步骤时间描述 | 未统一计算启动、换液、清洗和实际样品量 | 真实日志；同一任务和精度标准下比较 | 报告本平台的实际时间，不做不公平跨文献比较 |
| C12 | 自适应策略减少实验时间约 58.20% | **Preliminary** | 原 6.txt 汇总数值 | 是否真实闭环不清楚；可能按点数估算；缺少重复 | 真实自动运行日志；至少 3 次；包含全部等待/置换/计算 | `in a preliminary retrospective analysis...`，正式稿不建议保留 |
| C13 | 自适应策略将拟合误差从 6.04% 降至 3.53% | **Preliminary** | 原 6.txt 汇总数值 | “拟合误差”定义不清；点数/预算是否公平未知 | 分别定义 CMC、ΔHmic 和曲线 RMSE；相同点数对照；重复 CI | `relative fitting error was lower under the tested protocol` |
| C14 | 系统实现闭环自主实验 | **Not yet demonstrated** | 架构和 Explore–Refine–Converge 概念 | 必须排除事后抽点；缺少命令日志、因果处理和自动停止 | 真实测量→决策→泵命令→稳态→继续；完整日志和视频 | `computer-guided adaptive sampling` |
| C15 | 自适应算法可泛化至多体系 | **Unsupported** | 当前仅 SDBS 草稿 | 算法可能针对已知 SDBS 曲线调节 | 在 SDBS 冻结参数后用于至少两个不同体系，其中建议包含 TX-100 | `demonstrated for SDBS` |
| C16 | 自适应策略节约试剂 | **Not quantified** | 减少点数的逻辑推断 | 总流量、每点等待和冲洗均影响实际消耗 | 称重/流量日志；分分析物、稀释液和参考液统计 | 只报告测量点数和运行时间 |
| C17 | 系统可处理放热和吸热过程 | **Partially supported** | TX-100 显示相反符号 | 电加热只提供正热输入；log10 目标对非正值处理不清 | 明确符号模型；吸热化学标定或双向热参考 | `resolved exothermic and endothermic chemical signatures under the tested conditions` |
| C18 | 模型可跨芯片/跨日迁移 | **Unsupported** | 未见系统验证 | 单设备校准可能主导性能 | 至少 3 芯片、3 日期；leave-one-chip/day-out | `within the calibrated device` |
| C19 | 内部参考降低重新标定成本 | **Potential claim** | 加热器可主动施加功率 | 尚无实验前后脉冲、自检阈值或漂移补偿 | 自动参考脉冲；漂移判据；前后校准一致性 | 不陈述 |
| C20 | 论文贡献是 Lab-on-a-chip 级系统创新 | **Depends on completion** | 芯片、传感、算法、泵均已具备 | 若无真实闭环和微流控物理验证，仍可能被视为组件组合 | 完成 P0/P1 任务；证明微流控带来的新实验能力 | 强调具体能力，不泛称 principle-level advance |

## 状态定义

- **Supported**：主文中有直接数据、独立重复、统计和方法细节；底层数据可获得。
- **Partially supported**：已有初步数据，但存在关键边界或缺少对照。
- **Needs re-analysis**：原数据可能有价值，但必须在新的独立 split 或统计框架下重算。
- **Preliminary**：仅概念、单次演示或事后分析。
- **Unsupported / Do not use**：当前证据不足，不得进入题目或摘要。

## 每次更新要求

完成一个实验或分析任务后：

1. 更新状态；
2. 填入对应图、表和数据文件路径；
3. 填入独立重复数和统计方法；
4. 记录代码 commit SHA；
5. 由未参与该分析的成员复核后，才可标记为 `Supported`。
