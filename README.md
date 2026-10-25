# Wasserstein 鲁棒变量选择

在观测预算内选择少量变量，使两类数据的 Wasserstein 最坏情形检验风险尽量低。
代码与现有实验基于此前的理论版本。2026-10-25 的理论笔记从原论文的完整有限运输模型出发，
并由风险上界构造新的选择约束和对偶割；这部分尚未迁移到代码，现有结果不作为新版算法的实验结果。
当前报告收录
40 个完整小问题核查和 500 次混合数据实验，其余网格仍在运行。

## 入口

- [实验记录（中文）](reports/experiment_report_zh.pdf) / [English](reports/experiment_report.pdf)
- 中文版完整解释实验思路、数据、基线和指标；英文版简述设置与结果。两版共用英文图表。
- [逐条件结果](reports/experiment_results.csv) / [逐次运行数据](reports/data/instance_results.csv)
- [理论笔记（中文）](note/robust_coordinate_selection_zh.pdf) / [English](note/robust_coordinate_selection.pdf)
- [核心算法与代码对应](docs/algorithm.md)
- [实验执行与记录](docs/experiments.md)
- [重写计划与旧版去向](docs/rewrite_plan.md)
- [Overleaf 状态与上传](docs/overleaf.md)

## 安装和一个小例子

Python 3.12，依赖由 `uv.lock` 固定。核心只使用 NumPy、SciPy 和开源 HiGHS。

```powershell
./scripts/setup.ps1
.venv-research/Scripts/python.exe -m robusthw demo --budget 2
.venv-research/Scripts/python.exe -m pytest -q
```

其他平台：`UV_PROJECT_ENVIRONMENT=.venv-research uv sync --locked --extra experiments --extra dev`，
随后使用 `.venv-research/bin/python`。

## 代码分区

```text
src/robusthw/
  core/             固定选集 LP、对偶割、可行性检查、评分恢复
  search.py         全局搜索、穷举、前向、前向加交换
  experiments/      数据、基线、实验协议、调参、指标与汇总
configs/            八组明确的实验设置
tests/             独立数学核查与端到端回归
note/  reports/     当前双语理论与实验记录
```

全局方法返回可行风险上界和主问题的全局下界；达到指定差距才报告最优。
贪心方法只返回可行解，超时和数值失败均保留，采用明确标记的零评分兜底。

## 执行实验

先下载并核验四组官方公开数据；合成实验不需要下载。

```powershell
.venv-research/Scripts/python.exe -m robusthw fetch-data
.venv-research/Scripts/python.exe -m robusthw freeze --suite development --output outputs/development/protocol.json
.venv-research/Scripts/python.exe -m robusthw run outputs/development/protocol.json
.venv-research/Scripts/python.exe -m robusthw summarize outputs/development
```

正式套件为 `main`、`anchor`、`public`、`shift`、`small`、`scaling`、`ablation`。
每套先生成独立冻结清单；E2 的 `shift` 必须指定对应 E1 的 `--reuse-root`，不会重新训练。
命令、准确规模和产物见[执行说明](docs/experiments.md)。数据和运行产物默认保留在本地，
不将旧实验数字或开发检查结果写成新版研究结论。

本次冻结运行保存在 `outputs/study-20261018/`。`pipeline.json` 记录当前阶段，
各套件的 `execution.json` 持续更新进度；原始数据与记录保留不变。
每完成一套，流水线更新双语报告候选稿；正式 PDF 仅在两版审阅后一起替换。
混合数据中的当前评分存在明显类别偏置，报告如实保留负结果。

理论修改须完整同步中英文，显式记录双语审校，再运行 `note/build.ps1`。
在线理论包由 `scripts/package_overleaf.py --verify` 生成；具体云端状态见 Overleaf 说明。

开发检查与正式结果分开，前期实现核查见[验证记录](docs/validation.json)。
当前新增了实验并行、断点续跑和报告完整性检查；可用上述 pytest 命令复核。
