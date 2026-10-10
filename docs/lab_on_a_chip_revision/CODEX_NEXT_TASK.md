# Codex 下一步任务：修复数据划分并建立锁定测试集

## 任务编号

`P0-01 — leakage-free grouped dataset split`

## 为什么这是第一步

当前 `01_Data_split/data_split.py` 对全部数据行调用 `train_test_split(test_size=0.15, random_state=42)`。该流程会随机打散样本，只生成训练集和测试集，没有独立验证集。与此同时，后续 MLP 训练又在输入文件内部随机划分数据，并将该内部“test”集合用于 early stopping。

如果相邻样本来自同一次加热事件、同一连续实验或高度重叠的滑动窗口，随机按行划分会使近乎相同的信号同时进入训练和测试集合，从而高估泛化能力。Lab on a Chip 新稿中所有模型性能必须先基于独立实验组重新计算。

## 本任务边界

本任务只负责：

1. 审计原始数据是否含有可用于分组的元数据；
2. 建立 train/validation/locked-test 三分数据；
3. 生成可复核的 split manifest；
4. 增加自动测试，证明各集合之间不存在组级重叠。

本任务暂不训练或调优任何模型，也不更新论文性能数字。

## 需要检查的文件

- `01_Data_split/original_data.xlsx`
- `01_Data_split/data_split.py`
- 仓库中生成滑动窗口或稳态特征的其他脚本
- 任何可能记录实验日期、事件、功率状态、芯片编号或原始时间区间的文件

## 独立实验组的优先定义

按以下优先级确定 group key：

1. `heater_event_id` 或等价的单次加热事件编号；
2. `run_id` / `experiment_id`；
3. `chip_id + date_id + run_id` 的组合；
4. 可从连续时间戳和事件边界可靠重建的原始事件编号。

不得把功率值本身当作唯一 group key。相同功率可能来自不同独立实验，应保留为不同组。

如果原始文件没有任何足够可靠的分组信息，脚本必须停止并输出明确的缺失字段说明。不要自动退回到随机按行划分。

## 预期实现

重构 `01_Data_split/data_split.py` 为可复用、可测试的命令行程序。

建议接口：

```bash
python 01_Data_split/data_split.py \
  --input 01_Data_split/original_data.xlsx \
  --output-dir 01_Data_split/splits/v1 \
  --group-cols heater_event_id \
  --train-fraction 0.70 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 42
```

允许多个分组列：

```bash
--group-cols chip_id date_id run_id
```

### 必须生成的文件

```text
01_Data_split/splits/v1/
├── train.xlsx
├── validation.xlsx
├── locked_test.xlsx
├── split_manifest.csv
├── split_summary.json
└── split_config.json
```

### `split_manifest.csv` 最低字段

- `source_row_id`
- 全部分组列
- `group_key`
- `partition`
- 原始数据文件名
- 若存在：原始起止时间或窗口起止索引

### `split_summary.json` 最低内容

- 总行数和总组数；
- 每个 partition 的行数、组数及比例；
- 目标功率的范围和分位数；
- 环境温度范围；
- 每个芯片/实验日/功率区间在各 partition 的分布；
- group-overlap 检查结果；
- 配置和随机种子；
- 输入文件 SHA-256。

## 划分方法要求

- 划分单元必须是完整 group，而不是单行；
- 默认比例为 70/15/15；
- 同一 group 不得跨 partition；
- 如果有时间顺序，优先支持 chronologically blocked split；
- 如果目标是跨日期或跨芯片泛化，应允许显式指定留出某一日期/芯片作为 locked test；
- 所有操作必须可复现；
- 不得对 locked test 做任何重采样或参数选择。

建议支持两种模式：

1. `group-random`：在完整 group 层面按种子划分；
2. `chronological`：按组的起始时间排序后连续划分。

## 数据有效性检查

脚本应在写文件前检查：

- 分组列存在且无缺失；
- group key 非空；
- 三个比例之和为 1；
- 每个 partition 至少包含一个完整组；
- train/validation/test 的 group 集合两两不相交；
- 原始数据行没有丢失或重复；
- 目标列 `P1(uW)` 和四个输入特征存在；
- 报告 `P1(uW) <= 0` 的数量，但本任务不要删除这些数据。

## 自动测试

新增测试文件，建议：

`tests/test_grouped_split.py`

至少覆盖：

1. 同一 group 永不跨集合；
2. 相同 seed 生成完全相同结果；
3. 不同 seed 只改变 group 分配，不改变数据内容；
4. 缺少 group column 时明确失败；
5. 任何行不丢失、不重复；
6. 三分比例在组粒度限制下尽可能接近目标；
7. 非正目标只被报告，不被静默删除；
8. manifest 与三个输出文件可相互核对。

## README 更新

更新根目录 README 或增加：

`docs/lab_on_a_chip_revision/DATA_SPLIT_PROTOCOL.md`

明确说明：

- 为什么不再使用随机按行划分；
- train、validation、locked test 的用途；
- 何时可以打开 locked test；
- 如何重现论文使用的固定 split；
- 后续滑动窗口必须在 split 之后、partition 内部生成。

## 验收标准

本任务完成需同时满足：

- [ ] 旧的随机行级 split 不再是默认路径；
- [ ] 脚本在缺少可靠 group metadata 时拒绝运行；
- [ ] 生成 train/validation/locked-test 三个独立集合；
- [ ] manifest 可追溯每一行；
- [ ] 自动测试全部通过；
- [ ] 输出摘要明确显示 group overlap = 0；
- [ ] 未更改任何模型参数或论文数字；
- [ ] 提交信息中说明实际采用的 group 定义及其科学依据。

## 完成本任务后的下一步

`P0-02 — Refactor MLP/XGBoost training to consume the fixed train/validation files and evaluate the locked test exactly once.`
