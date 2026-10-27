# 核心算法与代码对应

代码与 `note/main.tex` 的完整支撑模型、齐次运输问题和仿射割对应。
当前实现采用标准 hinge 生成损失 `ell(t)=max(0,1+t)`，是一般理论的 LP 特例；
模型版本为 `full_support_standard_hinge_cone_v3`。已有 v2 及更早的半 hinge
实验保留原记录，不能当作本版算法的运行结果。

## 核心接口

讲解主线集中在下面三个文件，公式注释直接写在对应计算块旁；均不依赖实验模块或 sklearn：

- `core/model.py`：`Problem`、数值选项、拟合结果、仿射割及检测函数恢复。
- `core/oracle.py`：`_primal` 组装式 (4)，`_dual` 用附录 B 的归一化 LP 求解式 (5)，再生成式 (7) 的割；`fit_fixed` 串起一次内层求解。
- `core/selection.py`：`select`，外逼近主问题与上下界更新；穷举和贪心也使用相同 oracle。

展示代码时，按 `oracle._primal` → `oracle._dual` → `oracle.fit_fixed` →
`selection._Run.evaluate` → `selection._master` → `selection._global` 阅读；
最后看 `model.FitResult.predict_score` 如何恢复检测器。
式 (4) 计算的是固定选集风险 `V(S)`；式 (5) 定义 `F(z)`，二者通过式 (6)
的 `F(1_S)=V(S)` 联系。代码中的 `upper/lower` 是同一固定选集值的数值上下界，
全局搜索的 `L` 则必须来自主问题。

其余支持代码移至三个私有文件：`_linear_program.py` 封装稀疏矩阵和求解器接口，
`_certificates.py` 检查并修复浮点可行性、计算误差裕量，
`_evaluation.py` 收纳仅做固定集评估的 LFD 和可选连续扩展。
这些函数仍参与运行，但不需要在讲解主算法时逐行展开。
`fit_relaxed` 保留为连续扩展的核验入口；`search.py` 仅保留旧导入路径兼容。
最小调用为：

```python
from robusthw.core import Problem, SolveOptions, fit_fixed, fit_relaxed
from robusthw.core.selection import select

problem = Problem(x0, x1, radii=[0.4, 0.36])
result = select(problem, budget=5, method="oa", options=SolveOptions(time_limit=120))
scores = result.fit.predict_score(x_test)
```

`x0`、`x1` 的行数可以不同。每类样本权重各自归一为一，默认分别为 `1/n0`、`1/n1`。
内部合并数组仅用于矩阵存储；运输证书按类返回 `gamma0`、`gamma1`。
评分非负判为类别 0，负值判为类别 1。选择对偶中的 `gamma0`、`gamma1`
各有总质量 `M(gamma)`，每行质量为 `M(gamma)*q_i`，不再固定为单位总质量。
定义 `M(gamma)=0.5*(gamma0.sum()+gamma1.sum())`。两类行和约束保证
每类总质量都等于 `M(gamma)`，代码保留的 `alpha` 也等于这个共同质量；
它只是 LP 中便于组装矩阵的辅助标量，正文命题已经将其消去。

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

`fit_relaxed(problem, z)` 以 `eta` 为待最小化的目标上界。距离项换为逐坐标加权和，约束为

$$
0\le w_{km}\le\lambda_k,\qquad
\rho_0w_{0m}+\rho_1w_{1m}\le\eta z_m,\qquad
\sum_{k=0}^1\left(\sum_i q_i^{(k)}a_i^{(k)}+\rho_k\lambda_k\right)\le\eta.
$$

这里的 `eta` 是优化变量，坐标约束没有固定风险常数。对二元 `z`，
取 `w_km=lambda_k*z_m` 可从固定选集问题的任意可行上图点得到此式，
反向由 `w=0`（未选坐标）和 `w<=lambda` 恢复原约束。
对给定分数 `z`，这仍是 LP。`coordinate_weights` 返回 `w`，`upper` 返回可行 `eta`；
分数点的结果不被解释为可部署的检测器。

## 运输对偶与割

标准 hinge 的点态最小损失给出

$$
H(\gamma)=2\sum_{y\in\mathcal X}\min\left\{
\sum_i\gamma_{i,y}^{(0)},\sum_j\gamma_{j,y}^{(1)}\right\}.
$$

式 (5) 只以运输量 `gamma` 和坐标乘子 `mu` 为决策变量：

$$
\begin{aligned}
F(z)=\max_{\gamma,\mu\ge0}\quad&2\sqrt{H(\gamma)}-M(\gamma)-\mu^\top z,\\
\gamma^{(k)}\mathbf1&=M(\gamma)q^{(k)},\\
\sum_{m=1}^D\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+
&\le\rho_kM(\gamma),\qquad k=0,1.
\end{aligned}
$$

这里

$$
M(\gamma)=\frac12\sum_{k=0}^1\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)},\qquad
c_k(\gamma)[m]=\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|,
$$

且标量正部为 $t_+=\max\{t,0\}$。因此预算逐坐标计算超出 `rho_k*mu_m`
的正部分，再对所有坐标求和。两类行和约束确保各类总质量相同，均等于 `M(gamma)`。
默认权重 `q_i^(k)=1/n_k` 对应正文的均匀行质量约束。
实现先求附录 B 的等价归一化 LP，并用 `alpha=M(gamma)` 组装约束。
辅助变量 `beta[k,m]>=0` 将正部约束写成线性形式，
`h_y` 表示两类目标质量的最小值：

$$
\begin{aligned}
\max\quad &2\sum_y h_y,\\
\sum_y\gamma_{i,y}^{(k)}&=\alpha q_i^{(k)},\\
0\le h_y&\le\sum_i\gamma_{i,y}^{(k)}\quad(k=0,1),\\
\beta_{km}&\ge\sum_i\sum_y\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|-\rho_k\mu_m,\\
\sum_m\beta_{km}&\le\rho_k\alpha,\qquad
\alpha+\mu^\top z=1,\qquad \gamma,\beta,\mu,\alpha\ge0.
\end{aligned}
$$

归一化 LP 的最优值记为 `v`。由 `H` 的正齐次性，
`(v*gamma, v*mu)` 是式 (5) 的最优解：
`H(v*gamma)=v^2`、`M(v*gamma)=v*alpha`，目标为
`2*v-v*(alpha+mu@z)=v`。
这是 `_dual` 中矩阵块、等式块和后续缩放各自对应的数学步骤。
最终返回的对偶证书使用齐次变量，而不是归一化 LP 的原始变量。

任何齐次可行解都给出全局割

$$
c(z')=2\sqrt{H(\gamma)}-M(\gamma)-\mu^\top z'.
$$

`Cut.slope=-mu`；截距为 `2*sqrt(H)-M(gamma)` 减显式数值裕量。
实现中的 `alpha` 保存同一个质量 `M(gamma)`。
`_certificates.py` 修复行质量、预算和坐标约束，并根据实际质量修正和距离计算误差裕量。
原始端重新计算 `a`，再令 `eta` 覆盖目标及所有 `rho@w[:,m]/z_m`，才报告上界。
这属于带残差检查的浮点证书。`with_cut=False` 时直接求固定选集的有限 LFD LP，
避免基线拟合承担无用的全维割计算。

## 外逼近循环

`select(..., method="oa")`（别名 `global`）反复执行：

1. 调用 `fit_fixed` 评估当前选集，更新最好可行风险 `U` 并保存对偶割。
2. 求式 (8) 的二元问题 `min eta`，满足 `sum(z)=d`、`eta>=0` 和所有 `eta>=c_s(z)`。
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

100 维实验设计与基线见 [study100.md](study100.md)。
本版核验包括独立上图 LP、分数点凸性、全部小维二元选择和外逼近对穷举。
