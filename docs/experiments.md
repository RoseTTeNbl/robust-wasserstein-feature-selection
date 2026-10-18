# 实验执行与记录

实验目的、数据生成公式和指标解释见[完整设计](../reports/experiment_report_zh.pdf)。
本文件只说明设计如何映射到可执行代码。当前没有正式研究结果。

## 套件与规模

| `--suite` | 内容 | 数据设置与预算任务 |
| --- | --- | --- |
| `development` | 独立实现检查，排除在正式统计外 | 10 套小数据 |
| `main` | E1 四种信号、三档规模、信号数与预算变化 | 600 套独立数据，1,800 个预算任务 |
| `anchor` | 原论文 100 维混合模型 | 500 次；只拟合全变量检验 |
| `public` | WDBC、Ionosphere、Sonar、HAR | 90 套表格训练设置、10 套 HAR；共 210 个预算任务 |
| `shift` | E2 固定训练结果后平移或加噪声 | 引用 E1 的 120 套中档数据，8 个变化点 |
| `small` | E3 穷举及独立数学核查 | 40 套；120 个大小为三的选集，核查全部 176 个至多三列选集 |
| `scaling` | E3 单独改变维数、样本数或预算 | 200 套；20 个中心各三次计时，共 240 个计时设置 |
| `ablation` | 半径与初始化 | 20 套新数据；120 次半径比较和 20 次随机初始化 |

实验格点保存在 `configs/*.json`，`protocol.build_plan` 展开并核对唯一身份。
固定数值精度与时间控制在 `SolveOptions`，验证候选网格在 `runner.py` 中明确列出。
三次计时不会被当成三套独立数据。表格数据不同训练样本数共享划分种子，训练集合嵌套；
同一训练设置的预算共享数据、预处理和共同半径。

## 冻结后运行

```powershell
.venv-research/Scripts/python.exe -m robusthw fetch-data
.venv-research/Scripts/python.exe -m robusthw freeze --suite main --output outputs/main/protocol.json
.venv-research/Scripts/python.exe -m robusthw run outputs/main/protocol.json
.venv-research/Scripts/python.exe -m robusthw summarize outputs/main
```

`freeze` 须在源码仓库的可编辑安装中运行，保存完整设置清单、当前源码和配置的 SHA-256，以及源码 ZIP。
`run` 拒绝源码、清单、快照或已有样本被改动的续跑。修正实现后应冻结到新的运行目录。
这与当前代码和文档直接覆盖为最新版并不冲突：冻结清单描述某一次实际运行。

E2 必须引用匹配的 E1 输出：

```powershell
.venv-research/Scripts/python.exe -m robusthw freeze --suite shift --output outputs/shift/protocol.json
.venv-research/Scripts/python.exe -m robusthw run outputs/shift/protocol.json --reuse-root outputs/main
```

它核验 E1 数据和拟合记录，用原评分计算变化后的预测，不重新选变量或调参。
`--limit` 只做明确标记的部分执行；不能把它报告成完整套件。

## 每个实例做什么

1. 按独立种子生成或划分训练、验证、测试数据。只用合并的训练样本计算均值和标准差，
   删除训练常量列，并保存原始索引。各预算共享这一步。
2. 用逐变量 MMD、mRMR、联合 MMD 三个参照选集，在验证集选共同半径。
   E3 的小问题和规模比较改用报告规定的固定四分之一尺度。
3. 各方法选变量，再用同一个鲁棒检验拟合。L1 逻辑回归的五个 $C$ 各自经过验证集评价；
   每个候选配置独立限时，全部五次时间和拟合次数计入该方法。
4. 参数固定后才访问测试集。记录概率平衡错误率、阈值错误率、各类错误及合成信号恢复率。
   HAR 先按受试者统计，再对九个人等权。
5. 保存全部失败和超时。零评分是明确标记的可行兜底，其错误率仍计入汇总分母。

统计选择每候选配置 300 秒，HAR 为 600 秒；一次固定选集拟合最多 60 秒并受剩余时间约束。
主间隔为 $10^{-3}$，内层为 $10^{-4}$，交换最小改善为 $10^{-4}$。所有求解器固定单线程。
半径选择耗时单列，并计入实例总耗时；预测评价耗时不伪装成变量搜索耗时。

## 输出目录

```text
outputs/<suite>/
  protocol.json              冻结设置与源码指纹
  protocol.sources.zip       该次运行的源码与配置
  data/<data_id>.npz         共享样本矩阵及受试者索引
  records/<data_id>.json     全方法、全预算、失败、拟合证书与轨迹
  execution.json             计划、已尝试、失败和文件摘要
  analysis/                  条件分开的表格、区间和图
```

结果包含原始选集、原列索引、评分、源点上界、运输价格、验证候选、数值残差及时间轨迹。
`summarize` 核验记录指纹，列出缺失和失败，不删除差的重复，不混合不同难度条件。
合成配对差值按数据重复重采样；表格划分只报告配对差值的均值和标准差；
HAR 先在每个人内平均训练重复，再按人重采样。计时重复先在各实例内取中位数。

## 本次全量运行与报告维护

当前运行目录为 `outputs/study-20261018/`，其中 `frozen-source/` 是实际执行的独立源码副本。
报告和日常文档可以更新，已冻结的算法、设置和原始记录保持不变。
`small` 和 `anchor` 已完整运行；其余阶段由以下命令按次序执行或续跑：

```powershell
.venv-research/Scripts/python.exe scripts/execute_study.py --study-root outputs/study-20261018 --data-root data/uci --workers 2 --report-script scripts/build_experiment_report.py
```

同一目录只允许一个流水线进程。查看 `pipeline.json` 判断当前阶段，查看各套件的
`execution.json` 判断数据任务进度；任务完成不等于所有方法取得最优性证明。
`main`、`public`、`shift`、`ablation` 依次执行，`scaling` 最后单独以一个进程运行。
请勿同时启动另一轮相同目录的任务，也不要在计时阶段另跑实验。

报告维护分为数据刷新和审阅发布：

```powershell
.venv-research/Scripts/python.exe scripts/build_experiment_report.py --study-root outputs/study-20261018
.venv-research/Scripts/python.exe scripts/build_experiment_report.py --study-root outputs/study-20261018 --publish --reviewed-by NAME
```

第一条命令核验并汇总结果，生成 `report-candidate/` 中的成对 PDF、图表与数据。
审阅两份 PDF 后再运行第二条；它发布已审阅的精确文件，不重新取正在增长的数据。
流水线只自动生成候选稿，不将未经审阅的排版替换到正式报告。

`reports/experiment_results.csv` 是逐条件均值与标准差；`reports/data/instance_results.csv`
保留每次运行的选集、状态、界、错误率和时间。协议及输入摘要保存在同目录的
`study_inputs.json`，全部原始样本、拟合和轨迹仍在运行目录。
用 `scripts/diagnose_prediction.py --study-root outputs/study-20261018` 可独立重放已完成的
500 次混合模型预测，解释当前评分的类别偏置；该诊断不重新训练或调参。
