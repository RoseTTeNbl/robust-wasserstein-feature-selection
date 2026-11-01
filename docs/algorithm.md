# 核心算法与代码对应

本文说明 `src/robusthw/core/` 怎样实现 `note/main.tex`，公式编号都指该笔记。
实现采用标准 hinge 损失 `ell(t)=max(0,1+t)`，此时 `H(gamma)=2*sum_y min(p_0(y),p_1(y))`，
笔记中的 Eq. (3)、(4)、(11)、(12) 都是 LP。模型版本为 `full_support_standard_hinge_cone_v3`；
v2 及更早的半 hinge 实验只保留原记录，不能当作本版的运行结果。

## 笔记与代码对照

| 笔记 | 数学对象 | 代码 |
| --- | --- | --- |
| Eq. (4) | 固定选集 `S` 的检测器 LP，给出 `V(S)` 的上界和检测器 | `oracle._primal`，修复用 `_certificates._repair_primal` |
| Eq. (12)（附录 B.1 的 `beta`） | 归一化对偶，最优值为 `F(z)`，给出下界和割 | `oracle._dual`，修复用 `_certificates._repair_dual` |
| Eq. (5) | 凸选择函数 `F(z)` | 不直接求解；`_repair_dual` 输出满足其约束的 `(gamma, mu)` |
| Eq. (7)、(13) | 切平面割与最优尺度 | `model.Cut(intercept, slope)` |
| Eq. (8) | 二元主问题 | `selection._master` |
| Algorithm 1 | 外逼近 | `selection._global`，每步评估用 `selection._Run.evaluate` |
| Eq. (9)–(10) | 检测器延拓（附录 A） | `model.FitResult.predict_score` |
| Eq. (3) | 固定选集 LFD（只给下界、不给割） | `_evaluation._dual_fixed`（`with_cut=False`） |
| Eq. (11) | 分数 `z` 的上图原问题 | `_evaluation._primal_relaxed`（`fit_relaxed`） |

`_linear_program.py` 只负责稀疏矩阵组装（`_matrix`）、带截止时间的 HiGHS 调用（`_solve`）
和选集距离（`_distance`）；讲算法时不需要逐行展开。`search.py` 只保留旧的导入路径。

## 数组与符号

- `problem.x`：合并样本，先 `n0` 行类 0，再 `n1` 行类 1。第 `r` 行既是源点，也是笔记中的目标 `y`（`y` 取遍 `X`）。
- `problem.y[i]`：第 `i` 行的类别 `k`；类 0 的损失是 `ell(-g)`，类 1 是 `ell(g)`。
- `problem.q`：经验权重，每类和为 1，默认 `1/n_k`。
- `problem.delta[i,r,m]`：`|x_i[m]-x_r[m]|`，即 `c_k(gamma)[m]` 中的单位成本。
  `_distance(problem, S)[i,r]` 等于 `||x_i[S]-x_r[S]||_1`。
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

1. **`oracle._primal`（Eq. (4)）。** 每个源点、目标对 `(i,r)` 一行：
   `-a_i - lambda_k*distance[i,r] + sigma_k*g_r <= -1`（`sigma_0=-1`、`sigma_1=1`），即
   `ell(sigma_k g_r) <= a_i + lambda_k ||x_i[S]-x_r[S]||_1`。目标是
   `q@a + rho@lambda`。`_repair_primal` 重置 `a_i=max_r[ell(sigma_k g_r)-lambda_k distance[i,r]]_+`，
   使解严格可行，所以 `upper >= V(S)` 是可靠的，`(g, a, lambda)` 就是检测器证书。
2. **`oracle._dual`（Eq. (12)）。** 这是附录 B.1 推出的归一化对偶，`beta` 显式保留：
   `_envelope_rows` 给出 `h_r <= p_k(r)`（hinge 时 `H=2*sum_r h_r`）；循环 `for k` 给出
   `c_k(gamma_hat)[m] <= beta_km + rho_k*mu_hat_m` 和 `sum_m beta_km <= rho_k*alpha`；
   `equalities` 给出行质量 `sum_r gamma_hat_ir = q_i*alpha` 和 `alpha + mu_hat^T z = 1`。
   这里 `alpha=M(gamma_hat)`，最优值是 `v(z)=F(z)`（附录 B.3）。
3. **`_certificates._repair_dual`（从 LP 解到割）。**
   - `_plan_rows` 把每行质量改回 `q_i`，相当于把 `gamma_hat/alpha` 缩放到单位质量；`beta` 也除以 `alpha`。
   - 取满足 Eq. (5) 约束（`M=1`）的最小 `mu`：`mu_m = max_k (c_k[m] + 误差 - beta_km)_+ / rho_k`。
   - `h_safe` 是 `H(gamma)` 减去行误差的下界；`v = h_safe/(1+mu^T z)` 是 Eq. (12) 的可行值。
   - 按 Eq. (13) 的最优尺度 `s = v/(1+mu^T z)` 缩放，返回 `H(s*gamma)`、`s*mu`、`alpha=M(s*gamma)=s` 和数值裕量。
4. **割的形式。** `_dual` 用 `Cut(2*sqrt(h) - alpha - margin, -mu)` 建割，即 Eq. (5) 的目标
   `2*sqrt(H)-M-mu^T z'` 在缩放后的可行对上取值，也就是 Eq. (7)：在 `z` 处等于 `v`，
   在 `[0,1]^D` 上处处不超过 `F`。笔记第 3 节末指出，任何可行对都给出有效割，不要求最优。
5. **`oracle.fit_fixed`（Algorithm 1 第 4–5 行）。** 先用一半时间求 `_primal`（上界和检测器），
   再在 `z=1_S` 求 `_dual`（下界和割）。`_finish` 由内层差距 `upper-lower` 和求解器状态给出状态。
6. **`selection._Run.evaluate`（第 3–8 行）。** 缓存每个选集的结果，收集割，
   并以 `upper` 更新现有最好解 `U`。
7. **`selection._master`（Eq. (8)，第 9 行）。** 变量 `[z, eta]`，约束 `sum z = d` 和 `eta >= c_s(z)`，
   `z` 二元、`eta>=0`，用 `scipy.optimize.milp` 求解；遇到 HiGHS 数值错误时关闭预处理重试一次。
8. **`selection._global`（Algorithm 1 的循环）。** `L` 只取主问题的全局对偶界
   `mip_dual_bound`（减去舍入裕量，并保持单调），不用主问题当前解的目标值；
   `U-L <= absolute_tolerance` 时报告 `optimal`。主问题若给出已评估过的选集，在精确求解下必有
   `L >= U`（第 3 节的终止论证）；此时若差距仍未闭合，返回 `accuracy_limit`。
9. **`model.FitResult.predict_score`（Eq. (9)）。** 对查询投影 `xi=x[S]`，每类取使
   `a_i + lambda_k*||x_i[S]-xi||_1` 最小的锚点，再按 `tau=min(1, r_0/b)` 在两个锚点的分数之间插值；
   在已观测投影处直接返回拟合分数。Eq. (10) 保证两类损失都被包络控制。
   分数非负判为类 0，负值判为类 1。

## 输入与输出

```python
from robusthw.core import Problem, SolveOptions, fit_fixed, fit_relaxed
from robusthw.core.selection import select

problem = Problem(x0, x1, radii=[0.4, 0.36])          # 可选 q0、q1 权重
result = select(problem, budget=5, method="oa", options=SolveOptions(time_limit=120))
scores = result.fit.predict_score(x_test)
```

- `fit_fixed(problem, S)` 返回 `FitResult`：`scores=g`、`source_bounds=a`、`prices=lambda`，
  `lower <= V(S) <= upper`，以及 `cut`（`with_cut=False` 时改用 Eq. (3)，不返回割）。
- `fit_relaxed(problem, z)` 返回 `RelaxedResult`：`coordinate_weights=w`，
  `lower <= F(z) <= upper`；分数 `z` 的结果不是可部署的检测器。
- `select(...)` 返回 `SearchResult`：`fit`（最好选集的 `FitResult`）、`upper=U`、
  `lower=L`、`gap`、`status`、`trace`、`cuts`。`method` 可取 `global`/`oa`、`enumeration`、
  `forward`、`forward_swap`。后两种是没有全局下界的对照方法。

状态含义：`optimal` 表示 `U-L` 已达容差；`time_limit` 返回已有可行解和界；
`accuracy_limit` 表示数值精度不足以闭合差距；`numerical_failure`、`solver_error`
表示证书或求解器出错；启发式方法以 `complete`/`incomplete` 结束。

## 当前开销与改进方向

每个割都要解 Eq. (12)，它的 `2D` 个耦合行共有约 `D*N^2` 个非零元，远多于固定选集 LFD
（Eq. (3)，约 `N^2`），是主要耗时。用 `_dual_fixed` 的 LFD 解 `gamma*` 和
`mu*_m = max_k c_k(gamma*)[m]/rho_k`（`m` 不在 `S` 中；`S` 内取 0）可以构造在 `1_S` 处同样紧的割
（这正是附录 B.2 的构造），当前代码尚未采用。开发检查表明 OA 在小例子上会评估全部子集，
瓶颈是松弛强度而非割的开销；hinge 损失下的匹配重构与研究方向见
[研究方向笔记](../note/directions/research_directions.pdf)。

## 核查与实验

- `tests/test_new_approach.py` 独立组装 Eq. (11)，核对分数 `z` 的原始可行性、对偶可行性、
  目标值和割，并枚举小维选集检查全局下界。
- `tests/test_core.py` 核对解析值、不等样本数、非均匀权重、重复投影与检测器包络。
- `tests/test_search.py` 核对外逼近与穷举、主问题上下界方向、超时与失败处理。
- `experiments/verification.py` 用独立的有限运输 LP 核对固定选集风险。

100 维实验设计与基线见 [study100.md](study100.md)。
