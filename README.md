# Wasserstein 鲁棒变量选择

在观测预算内选择少量变量，使两类数据的 Wasserstein 最坏情形检验风险尽量低。
本仓库按当前理论笔记与实验设计整体重写；正式研究实验尚未运行。

## 入口

- [实验设计（中文）](reports/experiment_report_zh.pdf) / [English](reports/experiment_report.pdf)
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
note/  reports/     当前双语理论与实验设计
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

理论修改须完整同步中英文，显式记录双语审校，再运行 `note/build.ps1`。
在线理论包由 `scripts/package_overleaf.py --verify` 生成；具体云端状态见 Overleaf 说明。

当前实现通过 147 项测试及十个开发实例的独立数学核查，见[验证记录](docs/validation.json)。
这些检查用于验证实现，不是正式研究结果。
