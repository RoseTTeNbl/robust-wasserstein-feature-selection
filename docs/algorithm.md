# 核心算法与代码对应

本文说明 `src/robusthw/core/` 怎样实现 `note/main.tex`，公式编号都指该笔记。
实现采用标准 hinge 损失 `ell(t)=max(0,1+t)`，此时 `psi(a,b)=2*min(a,b)`，
笔记中的 Eq. (5)、(9)、(10) 都是 LP。模型版本为 `full_support_standard_hinge_cone_v3`；
v2 及更早的半 hinge 实验只保留原记录，不能当作本版的运行结果。

## 笔记与代码对照

| 笔记 | 数学对象 | 代码 |
| --- | --- | --- |
| Eq. (5) | 固定选集 `S` 的检测器 LP，给出 `V(S)` 的上界和检测器 | `oracle._primal`，修复用 `_certificates._repair_primal` |
| Eq. (9) | `F(z)` 的 Charnes–Cooper LP，给出下界和割 | `oracle._dual`，修复用 `_certificates._repair_dual` |
| Eq. (6) | 比值形式的选择函数 `F(z)` | 不直接求解；`_repair_dual` 输出满足它的 `(gamma, mu)` |
| Eq. (8)、(11) | 鞍点形式与切平面割 | `model.Cut(intercept, slope)` |
| Eq. (12) | 二元主问题 | `selection._master` |
| Algorithm 1 | 外逼近 | `selection._global`，每步评估用 `selection._Run.evaluate` |
| Eq. (13)–(14) | 检测器延拓 | `model.FitResult.predict_score` |
| Eq. (4) | 固定选集 LFD（只给下界、不给割） | `_evaluation._dual_fixed`（`with_cut=False`） |
| Eq. (10) | 分数 `z` 的上图原始问题 | `_evaluation._primal_relaxed`（`fit_relaxed`） |

`_linear_program.py` 只负责稀疏矩阵组装（`_matrix`）、带截止时间的 HiGHS 调用（`_solve`）
和选集距离（`_distance`）；讲算法时不需要逐行展开。`search.py` 只保留旧的导入路径。

## 数组与符号

- `problem.x`：合并样本 `x_1..x_N`，先 `n0` 行类 0，再 `n1` 行类 1。第 `r` 行既是源点，也是运输目标 `x_r`。
- `problem.y[i]`：第 `i` 行的类别 `k`；`sigma_k=-1`（类 0）、`+1`（类 1）。
- `problem.q`：经验权重，每类和为 1，默认 `1/n_k`，即笔记中的 `1/n_k`。
- `problem.delta[i,r,m]`：`Delta_ir[m]=|x_i[m]-x_r[m]|`。`_distance(problem, S)[i,r]`
  等于 `1_S^T Delta_ir`。
- `problem.mask(S)`：`z=1_S`；`problem.radii`：`(rho_0, rho_1)`。

各 LP 的变量顺序（代码中的 `*_start` 偏移即由此而来）：

| 函数 | 变量块 |
| --- | --- |
| `_primal` | `[a (N), lambda (2), g (N)]`，`g` 限制在 `[-1,1]` |
| `_dual` | `[gamma_hat (N*N), beta (2D), mu_hat (D), h (N), alpha (1)]` |
| `_primal_relaxed` | `[a (N), lambda (2), g (N), w (2D), eta (1)]` |
| `_dual_fixed` | `[gamma (N*N), h (N)]` |

`gamma` 按 `gamma[i*N + r]` 展平：第 `i` 行源点到第 `r` 个目标。

## 读代码的顺序

1. **`oracle._primal`（Eq. (5)）。** 每个源点、目标对 `(i,r)` 一行：
   `-a_i - lambda_k*distance[i,r] + sigma_k*g_r <= -1`，即
   `ell(sigma_k g_r) <= a_i + lambda_k ||x_i[S]-x_r[S]||_1`。目标是
   `q@a + rho@lambda`。`_repair_primal` 重置 `a_i=max_r[ell(sigma_k g_r)-lambda_k distance[i,r]]_+`，
   使解严格可行，所以 `upper >= V(S)` 是可靠的，`(g, a, lambda)` 就是检测器证书。
2. **`oracle._dual`（Eq. (9)）。** 三组约束分别对应 Eq. (9) 的三行：
   `_envelope_rows` 给出 `h_r <= p_k(r)`（hinge 时 `H=2*sum_r h_r`）；循环 `for k` 给出耦合行
   `c_k(gamma_hat)[m] <= beta_km + rho_k*mu_hat_m` 和预算行 `sum_m beta_km <= rho_k*alpha`；
   `equalities` 给出 `sum_r gamma_hat_ir = q_i*alpha` 和 `alpha + mu_hat^T z = 1`。
   最优值就是 `F(z)`（命题 4.1(a)）。
3. **`_certificates._repair_dual`（从 LP 解到割）。**
   - `_plan_rows` 把每行质量改回 `q_i`，相当于 `gamma = gamma_hat/alpha` 并投影到 `Pi`；`beta` 也除以 `alpha`。
   - 取满足 Eq. (6) 的最小 `mu`：`mu_m = max_k (c_k[m] + 误差 - beta_km)_+ / rho_k`。
   - `h_safe` 是 `H(gamma)` 减去行误差的下界；`v = h_safe/(1+mu^T z)` 就是比值形式的值。
   - 按 Eq. (8) 的最优尺度 `s = v/(1+mu^T z)` 缩放，返回 `H(s*gamma)`、`s*mu`、`alpha=s` 和数值裕量。
4. **割的形式。** `_dual` 用 `Cut(2*sqrt(h) - alpha - margin, -mu)` 建割。由于
   `2*sqrt(H(s*gamma)) - s - s*mu^T z' = v*(1 - mu_bar^T (z'-z))`，其中 `mu_bar = mu/(1+mu^T z)`，
   这就是笔记 Eq. (11) 的切平面：在 `z` 处取值 `F(z)`，在 `[0,1]^D` 上处处不超过 `F`。
   按定理 3.1(c)，只要 `(gamma, mu)` 可行，割就有效，不要求最优。
5. **`oracle.fit_fixed`（Algorithm 1 第 3–4 行）。** 先用一半时间求 `_primal`（上界和检测器），
   再在 `z=1_S` 求 `_dual`（下界和割）。`_finish` 由内层差距 `upper-lower` 和求解器状态给出状态。
6. **`selection._Run.evaluate`（第 3–7 行）。** 缓存每个选集的结果，收集割，
   并以 `upper` 更新现有最好解 `U`。
7. **`selection._master`（Eq. (12)）。** 变量 `[z, eta]`，约束 `sum z = d` 和 `eta >= c_s(z)`，
   `z` 二元、`eta>=0`，用 `scipy.optimize.milp` 求解；遇到 HiGHS 数值错误时关闭预处理重试一次。
8. **`selection._global`（Algorithm 1 的循环）。** `L` 只取主问题的全局对偶界
   `mip_dual_bound`（减去舍入裕量，并保持单调），不用主问题当前解的目标值；
   `U-L <= absolute_tolerance` 时报告 `optimal`。主问题若给出已评估过的选集，在精确求解下必有
   `L >= U`（命题 4.2）；此时若差距仍未闭合，返回 `accuracy_limit`。
9. **`model.FitResult.predict_score`（Eq. (13)）。** 对查询投影 `xi=x[S]`，每类取使
   `a_i + lambda_k*||x_i[S]-xi||_1` 最小的锚点，再按 `tau=min(1, r_0/b)` 在两个锚点的分数之间插值；
   在已观测投影处直接返回拟合分数。分数非负判为类 0，负值判为类 1。

## 输入与输出

```python
from robusthw.core import Problem, SolveOptions, fit_fixed, fit_relaxed
from robusthw.core.selection import select

problem = Problem(x0, x1, radii=[0.4, 0.36])          # 可选 q0、q1 权重
result = select(problem, budget=5, method="oa", options=SolveOptions(time_limit=120))
scores = result.fit.predict_score(x_test)
```

- `fit_fixed(problem, S)` 返回 `FitResult`：`scores=g`、`source_bounds=a`、`prices=lambda`，
  `lower <= V(S) <= upper`，以及 `cut`（`with_cut=False` 时改用 Eq. (4)，不返回割）。
- `fit_relaxed(problem, z)` 返回 `RelaxedResult`：`coordinate_weights=w`，
  `lower <= F(z) <= upper`；分数 `z` 的结果不是可部署的检测器。
- `select(...)` 返回 `SearchResult`：`fit`（最好选集的 `FitResult`）、`upper=U`、
  `lower=L`、`gap`、`status`、`trace`、`cuts`。`method` 可取 `global`/`oa`、`enumeration`、
  `forward`、`forward_swap`。后两种是没有全局下界的对照方法。

状态含义：`optimal` 表示 `U-L` 已达容差；`time_limit` 返回已有可行解和界；
`accuracy_limit` 表示数值精度不足以闭合差距；`numerical_failure`、`solver_error`
表示证书或求解器出错；启发式方法以 `complete`/`incomplete` 结束。

## 当前开销与未实现的改进

每个割都要解 Eq. (9)，它的 `2D` 个耦合行共有约 `D*N^2` 个非零元，远多于固定选集 LFD
（Eq. (4)，约 `N^2`），是主要耗时。笔记的注 4.3 指出，用 `_dual_fixed` 的 LFD 解
`gamma*` 和 `mu*_m = max_k c_k(gamma*)[m]/rho_k`（`m` 不在 `S` 中；`S` 内取 0）
就能得到同样紧的割 `V(S)*(1 - mu*^T z)`，有限终止性也仍然成立。当前代码尚未采用这一做法。

## 核查与实验

- `tests/test_new_approach.py` 独立组装 Eq. (10)，核对分数 `z` 的原始可行性、对偶可行性、
  目标值和割，并枚举小维选集检查全局下界。
- `tests/test_core.py` 核对解析值、不等样本数、非均匀权重、重复投影与检测器包络。
- `tests/test_search.py` 核对外逼近与穷举、主问题上下界方向、超时与失败处理。
- `experiments/verification.py` 用独立的有限运输 LP 核对固定选集风险。

100 维实验设计与基线见 [study100.md](study100.md)。
