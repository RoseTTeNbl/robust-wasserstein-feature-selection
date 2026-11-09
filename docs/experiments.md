# 实验工具与记录

核心算法在 `src/robusthw/algorithm/`；数据、基线、指标和运行协议在
`src/robusthw/experiments/`。

## 当前实验：20 维（study20-v3）

场景：4 个真坐标藏在 16 个干扰坐标里（干扰坐标与一切独立、两类同分布），每类 $n\in\{10,20,30,50\}$ 个训练点，
看检验能否分开两类、能否找到真坐标、证书是否成立。四个情形：`null`、`mean_shift`、`mixture`、`interaction`。
协议沿用 Xie、Gao、Xie (2021) 第 5.1 节：每类 1000 个测试点、5 折交叉验证选参数、100 次重复。
数据、方法与基线、指标公式、图和全部表格见 `results/study20-v3/summary.md`；
中文的逐步讲解见 [阅读指南](reading_guide.md) 第 6–8 节，设计依据和结论见研究方向笔记的“20 维实验”一节。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/run_study20.py accuracy    --workers 62
OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/run_study20.py computation --workers 62
PYTHONPATH=src python3 scripts/run_study20.py summarize
```

`computation` 读取 `accuracy.json` 里交叉验证选出的半径。参数在 `configs/study20.json`。
`summarize` 由 `study20_summary.py` 和 `study20_figures.py` 生成 `summary.md` 和 `figures/*.png`。

## 旧的通用实验框架

下面的 `freeze`/`run`/`summarize` 流程属于 v3 时期的通用框架，当前没有在用。

## 运行

先固定一套配置与源码，再执行和汇总：

```powershell
.venv-research/Scripts/python.exe -m robusthw freeze --suite development --output outputs/v3-development/protocol.json
.venv-research/Scripts/python.exe -m robusthw run outputs/v3-development/protocol.json
.venv-research/Scripts/python.exe -m robusthw summarize outputs/v3-development
```

`configs/*.json` 定义套件；`protocol.build_plan` 展开为具体数据与预算任务。
可选套件为 `development`、`main`、`anchor`、`public`、`shift`、`small`、
`scaling` 和 `ablation`。`small` 用于穷举核查，`scaling` 分别改变维数、样本数和预算。

合成实验直接生成数据。公开数据实验使用四组 UCI 数据，来源、标签和校验值见
[data_sources.json](data_sources.json)：

```powershell
.venv-research/Scripts/python.exe -m robusthw fetch-data
```

预处理仅使用训练数据；同一个数据设置中的各预算共享样本和预处理。
参数在验证数据上选择，测试数据仅用于固定模型后的评价。

## 方法与输出

通用套件提供外逼近、前向选择、前向加交换，以及边际 MMD、L1 logistic、mRMR、
联合 MMD 和随机选择基线。这里的 `l1_logistic` 使用 L1 选择变量后拟合鲁棒检验；
100 维设计中的普通 logistic refit 是另一项对比。

```text
outputs/<run>/
  protocol.json              配置、模型版本和源码指纹
  protocol.sources.zip       本次运行的源码快照
  data/<data_id>.npz         各方法共享的样本
  records/<data_id>.json     选集、拟合、预测、指标和求解轨迹
  execution.json             运行进度与状态
  analysis/                  汇总表格和图
```

`freeze`、`run` 和 `summarize` 分别对应冻结协议、执行和分析。
每次改动实现或配置后使用新的输出目录。保留求解状态、失败和超时，
只有上下界差距达到容差才将外逼近标为最优。
变化分布实验通过 `--reuse-root` 引用对应训练输出，使用同一评分函数重新评价。

报告工具在 `scripts/build_experiment_report.py`，图表和结果表的生成与核心算法分开。
先生成候选稿，再审阅发布；不把开发检查结果当作正式性能结论。
