# Wasserstein 鲁棒变量选择

在观测预算内选择少量变量，使两类数据的 Wasserstein 最坏情形检验风险尽量低。
代码已同步 2026-10-25 理论笔记的完整有限运输模型、选择约束及直接对偶割。
当前实现为标准 hinge 损失的 LP 特例，支持不等类样本数，并提供外逼近搜索的上下界。
新版 100 维实验与历史半 hinge 实验分别保存。

## 入口

- [当前 100 维实验设计与结果](docs/study100.md)
- 历史半 hinge 实验记录：[中文](reports/experiment_report_zh.pdf) / [English](reports/experiment_report.pdf)
- 历史[逐条件结果](reports/experiment_results.csv) / [逐次运行数据](reports/data/instance_results.csv)
- [理论笔记（English）](note/robust_coordinate_selection.pdf) / [保留的中文版本](note/robust_coordinate_selection_zh.pdf)
- [核心算法与代码对应](docs/algorithm.md)
- [实验执行与记录](docs/experiments.md)
- [2026-10-18 重写记录与旧版去向](docs/rewrite_plan.md)
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
  core/             model.py / oracle.py / selection.py：模型、拟合、选集优化
  search.py         旧导入路径兼容入口
  experiments/      数据、基线、实验协议、调参、指标与汇总
configs/            study100.json 与既有实验协议
tests/             独立数学核查与端到端回归
note/              英文理论笔记及保留的中文版本
reports/           双语实验记录
```

全局方法返回可行风险上界和主问题的全局下界；达到指定差距才报告最优。
贪心方法只返回可行解，超时和数值失败均保留，采用明确标记的零评分兜底。

## 执行实验

新版 100 维实验使用一个独立开发样本选参数，再运行三个固定确认样本：

```powershell
.venv-research/Scripts/python.exe -m robusthw.experiments.study100 --output outputs/study100-20261025
```

配置、源码指纹、原始数据、拟合结果、逐点预测及求解轨迹分别保存。
同一命令可续跑已冻结的实验；修改配置或源码须使用新输出目录。
详细设置和基线含义见 [study100.md](docs/study100.md)。

既有通用实验工具仍可用于更大规模研究：
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

历史冻结运行保存在 `outputs/study-20261018/`。`pipeline.json` 记录阶段，
各套件的 `execution.json` 持续更新进度；原始数据与记录保留不变。
每完成一套，流水线更新双语报告候选稿；正式 PDF 仅在两版审阅后一起替换。
混合数据中的当前评分存在明显类别偏置，报告如实保留负结果。

理论笔记默认只维护英文。完成正文、公式和参考文献审校后：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer
./note/build.ps1
```

英文审校记录为 `note/english_review.json`。需要完整双语交付时，审校两版后给检查器加
`--bilingual`、给构建命令加 `-Bilingual`；双语记录单独保存在 `note/bilingual_sync.json`。
两种模式均检查源文件指纹和编译诊断，详见[理论稿维护](docs/bilingual_note.md)。实验报告继续按原要求双语维护。
在线理论包由 `scripts/package_overleaf.py --verify` 生成；具体云端状态见 Overleaf 说明。

开发检查与正式结果分开。历史半 hinge 实现的核查保留在[旧验证记录](docs/validation.json)；
新版的核验范围见[核心算法说明](docs/algorithm.md)和[验证记录](docs/validation_current.json)，
实际确认实验见[100 维研究](docs/study100.md)。
