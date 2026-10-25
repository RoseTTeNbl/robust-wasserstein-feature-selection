# 核心算法与代码对应

代码与 `note/main.tex` 的完整支撑模型、选择约束和直接对偶对应。
当前实现采用标准 hinge 生成损失 `ell(t)=max(0,1+t)`，是一般理论的 LP 特例；
模型版本为 `full_support_standard_hinge_v2`。旧实验使用半 hinge 和旧对偶，须单独解读。

## 核心接口

数学实现集中在三个文件，均不依赖实验模块或 sklearn：

- `core/model.py`：`Problem`、数值选项、拟合结果、仿射割及检测函数恢复。
- `core/oracle.py`：`fit_fixed`、`fit_relaxed`，原始 LP、对偶 LP 和可行性核查。
- `core/selection.py`：`select`，外逼近主问题与上下界更新；穷举和贪心也使用相同 oracle。

`search.py` 仅保留旧导入路径兼容。最小调用为：

```python
from robusthw.core import Problem, SolveOptions, fit_fixed, fit_relaxed
from robusthw.core.selection import select

problem = Problem(x0, x1, radii=[0.4, 0.36])
result = select(problem, budget=5, method="oa", options=SolveOptions(time_limit=120))
scores = result.fit.predict_score(x_test)
```

`x0`、`x1` 的行数可以不同。每类样本权重各自归一为一，默认分别为 `1/n0`、`1/n1`。
内部合并数组仅用于矩阵存储；运输证书按类返回 `gamma0`、`gamma1`。
评分非负判为类别 0，负值判为类别 1。

## 固定选集与连续扩展

`fit_fixed(problem, S)` 解手稿的固定选集问题：

$$
\begin{aligned}
\min_{a,g,\lambda}\quad&
\frac1{n_0}\sum_i a_i^{(0)}+\frac1{n_1}\sum_i a_i^{(1)}
+\rho_0\lambda_0+\rho_1\lambda_1,\\
1-g_y&\le a_i^{(0)}+\lambda_0\|x_i^{(0)}[S]-y[S]\|_1,\\
1+g_y&\le a_j^{(1)}+\lambda_1\|x_j^{(1)}[S]-y[S]\|_1.
\end{aligned}
$$

所有目标点均来自两类样本的并集。标准 hinge 可无损地限制 `g` 在 `[-1,1]`；
`a>=0`、`lambda>=0`，零评分风险为 2。非均匀权重时将 `1/nk` 换为对应权重。
结果的 `scores`、`source_bounds`、`prices` 分别对应 `g`、`a`、`lambda`。

`fit_relaxed(problem, z)` 解手稿的连续扩展：将距离项换为逐坐标加权和，增加

$$
0\le w_{km}\le\lambda_k,\qquad
\rho_0w_{0m}+\rho_1w_{1m}\le2z_m,\qquad z\in[0,1]^D.
$$

二元 `z` 时与固定选集同值；分数 `z` 用于检验凸扩展，不被解释为可部署的检测器。
返回的 `coordinate_weights` 对应 `w`。

## 运输对偶与割

`oracle.py` 直接实现当前手稿的运输对偶。标准 hinge 的点态最小损失给出

$$
H(\gamma)=2\sum_{y\in\mathcal X}\min\left\{
\sum_i\gamma_{i,y}^{(0)},\sum_j\gamma_{j,y}^{(1)}\right\}.
$$

辅助变量 `beta[k,m]>=0` 将正部约束写成线性形式：

$$
\begin{aligned}
\sum_y\gamma_{i,y}^{(k)}&=1/n_k,\\
\beta_{km}&\ge\sum_i\sum_y\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|-\rho_k\mu_m,\\
\sum_m\beta_{km}&\le\rho_k,\qquad \mu_m\ge0.
\end{aligned}
$$

对偶最大化 `H - 2*mu@z`，因此每个可行解提供全局下界

$$
c(z')=H(\gamma)-2\mu^\top z'.
$$

`Cut.slope=-2*mu`；截距为重新计算的 `H` 减显式数值裕量。
实现修复运输行和、预算和坐标约束，并保存残差；原始端重新计算可行的 `a` 后才报告上界。
这是带残差检查的浮点数值证书。`with_cut=False` 时直接求固定选集的有限 LFD LP，
避免基线拟合承担无用的全维割计算。

## 外逼近循环

`select(..., method="oa")`（别名 `global`）反复执行：

1. 调用 `fit_fixed` 评估当前选集，更新最好可行风险 `U` 并保存对偶割。
2. 求二元主问题 `min eta`，满足 `sum(z)=d`、`0<=eta<=2` 和所有 `eta>=c_s(z)`。
3. 只用主问题的全局 `mip_dual_bound` 更新 `L`，按主问题候选继续评估。
4. `U-L` 达到容差才报告 `optimal`；超时返回已有可行解、上下界及未完成状态。

由于增添坐标不会提高最优风险，恰好选择 `d` 个与至多选择 `d` 个的最优值一致。
主问题当前可行目标不是全局下界。所有内层和主问题共同使用剩余时间预算。
`enumeration` 用于小问题核验；`forward`、`forward_swap` 是没有全局最优证书的比较方法。

`FitResult.predict_score` 使用附录 A 的延拓：各类寻找最小化
`a_i + lambda_k*distance(x_i[S],x[S])` 的锚点，再按距离插值评分；
在已观测投影上返回一致的有限评分。预测仅访问所选坐标，且满足两类损失包络。

## 核查与实验

- `tests/test_new_approach.py` 独立组装连续扩展 LP，核对分数 `z`、原始可行性、
  对偶可行性、目标和割，并枚举小维选集检查全局下界。
- `tests/test_core.py` 核对解析值、不等样本数、非均匀权重、重复投影与检测函数包络。
- `tests/test_search.py` 核对外逼近和穷举、主问题上下界方向、超时与失败处理。
- `experiments/verification.py` 用独立有限运输 LP 核对固定选集风险。

当前 100 维设置和对比代码见 [study100.md](study100.md)。
原始实验记录保留模型版本和风险尺度；历史半 hinge 数据不会作为新版结果重用。
