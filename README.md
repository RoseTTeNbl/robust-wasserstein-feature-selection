# Wasserstein 鲁棒变量选择

在观测预算内选择少量变量，使两类数据的 Wasserstein 最坏情形检验风险尽量低。
当前模型为 `full_support_standard_hinge_cone_v3`，实现英文理论笔记中的有限运输问题、
齐次选择函数及外逼近算法。数值代码使用标准 hinge 损失，支持不等类样本数。

## 入口

- [英文理论笔记](note/robust_coordinate_selection.pdf) · [LaTeX 源码](note/main.tex) · [阅读副本](note/main.md)
- [Overleaf 源码包](note/robusthw_theory_overleaf.zip) · [编译与上传](docs/overleaf.md)
- [核心算法与代码对应](docs/algorithm.md)
- [100 维实验设计与基线](docs/study100.md) · [实验工具与数据记录](docs/experiments.md)
- [理论稿维护](docs/theory_note.md)

## 安装与检查

Python 3.12，依赖由 `uv.lock` 固定。核心依赖 NumPy、SciPy 和开源 HiGHS。

```powershell
./scripts/setup.ps1
.venv-research/Scripts/python.exe -m robusthw demo --budget 2
.venv-research/Scripts/python.exe -m pytest -q
.venv-research/Scripts/python.exe -m ruff check src tests scripts
```

其他平台：`UV_PROJECT_ENVIRONMENT=.venv-research uv sync --locked --extra experiments --extra dev`，
随后使用 `.venv-research/bin/python`。

## 代码结构

```text
src/robusthw/core/
  model.py          问题、返回值和预测规则
  oracle.py         固定选集与连续选择子问题的数学模型
  selection.py      外逼近、穷举和贪心搜索
  _*.py             线性规划求解、数值证书与评估支持
src/robusthw/experiments/
                    数据生成、基线、实验协议、指标与汇总
configs/            实验设置
note/               当前英文理论、审校记录、PDF 和 Overleaf 包
scripts/            安装、检查、构建、实验及报告工具
tests/              数学核查和端到端回归
```

外逼近返回可行风险上界与全局下界，只有差距达到容差才报告最优。
正文中的 `M(gamma)` 是两类运输的共同质量；代码的辅助变量 `alpha` 表示同一个量。
实验、基线和结果处理独立于核心算法，新实验使用独立输出目录。

## 构建理论笔记

修改后先完成正文、公式、证明和参考文献审校，再执行：

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer
./note/build.ps1
.venv-research/Scripts/python.exe scripts/package_overleaf.py --verify --output note/robusthw_theory_overleaf.zip
```

第一节问题定义按用户要求保持原样。编译中间文件放在独占系统 TEMP；
发布前检查全部 PDF 页面。Overleaf 包含英文源码和 BibTeX，本地打包不会更新云端项目。
