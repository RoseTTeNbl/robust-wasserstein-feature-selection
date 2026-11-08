# 核心算法与代码对应

本文说明 `src/robusthw/algorithm/` 怎样实现 `note/main.tex`，公式编号都指该笔记。
笔记的理论对一般损失 `ell` 成立；代码取标准 hinge 损失 `ell(t)=max(0,1+t)`，
此时 `psi(a,b)=2*min(a,b)`，`H(gamma)=2*sum_y min(p_0(y),p_1(y))`，Eq. (3)、(4) 都是 LP。
模型版本为 `hinge_ratio_cut_v4`；v3 及更早（`sqrt` 形式的割、单独的对偶 LP）只保留原记录。

## 笔记与代码对照

| 笔记 | 数学对象 | 代码 |
| --- | --- | --- |
| Eq. (4) | 固定选集 `S` 的检测器 LP：`V(S)`、检测器 `(g, a, lambda)` | `oracle.fit_fixed` |
| Eq. (3) | 最不利运输 `gamma`：Eq. (4) 各约束行的乘子 | `fit_fixed` 里的 `duals`，修成可行用 `oracle.feasible_plan` |
| `H(gamma)` | Eq. (3) 的目标 | `oracle.hinge_value` |
| Eq. (8) | 比值割 `1/F(z) <= c_t(z)` | `oracle.ratio_cut` 返回 `model.Cut(intercept, slope)` |
| Eq. (9) | 主问题：在割下最大化 `theta`，`L = 1/theta` | `selection.master` |
| Algorithm 1 | 外逼近 | `selection._outer_approximation`，每步评估用 `_Search.evaluate` |
| Eq. (5) | 检测器延拓（锚点插值） | `model.FitResult.predict_score` |

整个核心只有三个文件：`model.py`（数据与结果）、`oracle.py`（一次 LP）、`selection.py`（搜索）。

## 数组与符号

- `problem.x`：合并样本，先 `n0` 行类 0，再 `n1` 行类 1。第 `r` 行既是源点，也是笔记中的目标 `y`（`y` 取遍 `X`）。
- `problem.y[i]`：第 `i` 行的类别 `k`；类 0 的损失是 `ell(-g)`，类 1 是 `ell(g)`。
- `problem.q`：经验权重，每类和为 1，默认 `1/n_k`。
- `problem.delta[i,r,m]`：`|x_i[m]-x_r[m]|`，所以 `c_k(gamma)[m] = sum_{i in 类 k, r} gamma[i,r]*delta[i,r,m]`。
  `problem.distance(S)[i,r]` 等于 `||x_i[S]-x_r[S]||_1`。
- `problem.mask(S)`：`z=1_S`；`problem.radii`：`(rho_0, rho_1)`。
- 运输 `gamma` 存成 `(N, N)` 矩阵：第 `i` 行源点到第 `r` 个目标。

## 读代码的顺序

1. **`oracle.fit_fixed`（Eq. (4)，Algorithm 1 第 4 行）。** 变量 `[a (N), lambda (2), g (N)]`，
   每个源点、目标对 `(i,r)` 一行：`-a_i - lambda_k*distance[i,r] + sigma_k*g_r <= -1`
   （`sigma_0=-1`、`sigma_1=1`），即 `ell(sigma_k g_r) <= a_i + lambda_k ||x_i[S]-x_r[S]||_1`。
   目标是 `q@a + rho@lambda`。分数限制在 `[-1,1]`：把 hinge 分数裁剪到这里不会增大 `ell(g)` 或 `ell(-g)`。
2. **上界。** 拿求解器给出的 `(g, lambda)`，重新取最小可行的
   `a_i = max(0, max_r[ell(sigma_k g_r) - lambda_k distance[i,r]])`，于是每条约束精确成立，
   `upper = q@a + rho@lambda` 是 `V(S)` 的可靠上界。投影重合的目标先统一成同一个分数，Eq. (5) 才有定义。
3. **下界和最不利运输。** Eq. (4) 第 `(i,r)` 行的乘子就是 Eq. (3) 的 `gamma[i,r]`（LP 对偶）。
   `feasible_plan` 把它修成 Eq. (3) 严格可行：行质量不足的部分留在原地（单位运输 `iota`，代价为 0），
   某类代价超出 `rho_k` 就和 `iota` 混合到刚好用完预算。于是 `lower = H(gamma) <= V(S)`。
   一次 LP 同时给出检测器、上界、下界和割；内层差距 `upper - lower` 决定 `fit.status`。
4. **`oracle.ratio_cut`（Eq. (8)，第 5 行）。** 在 `S` 外取 `mu_m = max_k c_k(gamma)[m]/rho_k`，
   在 `S` 上取 0；按 `H(gamma)` 归一后返回 `Cut(1/H, mu/H)`，即
   `1/F(z) <= (1 + mu^T z)/H(gamma)` 对所有 `z in [0,1]^D` 成立，并在 `z = 1_S` 处取等（`gamma` 最优时）。
   这正是附录 A.1 的构造，不需要第二个 LP。任何可行的 `gamma` 都给有效割，只是可能不紧。
5. **`selection._Search.evaluate`（第 3–8 行）。** 缓存每个选集的结果，收集割，并以 `upper` 更新现有最好解 `U`。
6. **`selection.master`（Eq. (9)，第 9 行）。** 变量 `[z, theta]`，约束 `sum z = d`
   和 `theta <= intercept_s + slope_s @ z`，`z` 二元，最大化 `theta`。
   `L = 1/theta`，`theta` 取 HiGHS 的全局上界 `mip_dual_bound`，所以超时时 `L` 仍然有效。
7. **`selection._outer_approximation`（Algorithm 1 的循环）。** `U - L <= absolute_tolerance` 时报告 `optimal`。
   主问题若给出已评估过的选集，精确求解下必有 `L >= U`（第 3 节的终止论证）；
   此时若差距仍未闭合，返回 `accuracy_limit`。
8. **`model.FitResult.predict_score`（Eq. (5)）。** 对查询投影 `xi = x[S]`，每类取使
   `a_i + lambda_k*||x_i[S]-xi||_1` 最小的锚点，再按 `tau = r_0/(r_0 + r_1)`（到两个锚点的距离）在两个锚点的分数之间加权；
   在已观测投影处直接返回拟合分数。笔记第 2 节给出完整证明。分数非负判为类 0，负值判为类 1。

`oracle.highs` 是唯一的求解器包装，处理 HiGHS 的两个已知问题：进程内调度器已按另一线程数初始化时
返回 "Not Set"（去掉线程数重试），以及小主问题上预处理失败（关掉预处理重试）。

## 输入与输出

```python
from robusthw.algorithm import Problem, SolveOptions, fit_fixed, select

problem = Problem(x0, x1, radii=[0.4, 0.36])          # 可选 q0、q1 权重
fit = fit_fixed(problem, (0, 3))                      # 一个选集
result = select(problem, budget=5, method="global", options=SolveOptions(time_limit=120))
scores = result.fit.predict_score(x_test)
```

- `fit_fixed(problem, S)` 返回 `FitResult`：`scores=g`、`source_bounds=a`、`prices=lambda`，
  `lower <= V(S) <= upper`，以及 `cut`（`with_cut=False` 时为 `None`）。
- `select(...)` 返回 `SearchResult`：`fit`（最好选集的 `FitResult`）、`upper=U`、`lower=L`、`gap`、
  `status`、`trace`、`cuts`。`method` 可取 `global`（别名 `oa`）、`enumeration`、`forward`、`forward_swap`；
  后两种是没有全局下界的对照方法。

状态含义：`optimal` 表示 `U-L` 已达容差；`time_limit` 返回已有可行解和界；
`accuracy_limit` 表示数值精度不足以闭合差距；`solver_error` 表示求解器失败；
启发式方法以 `complete`/`incomplete` 结束。

## 当前开销与改进方向

每次评估只解一个 `N^2` 行的 LP，割由它的乘子直接得到。比值割比旧版的切平面割更强
（`1/c_t` 处处在切线之上），但外逼近在小例子上仍会评估大部分子集：瓶颈是 `F` 在 `Z_d` 上的连续松弛本身偏弱，
不是割的开销。对一般 `ell` 的精确重构（坐标价格、松弛的闭式、分支定界）见
[研究方向笔记](../note/directions/research_directions.pdf)。

## 核查与实验

- `tests/test_oracle.py` 用独立的运输 LP 核对每个子集的上下界，以及比值割在自身选集处取等、在其他二元点不超过 `V`；
  另核对解析值、重复点、不等权重、检测器包络 Eq. (5) 和锚点插值。
- `tests/test_selection.py` 核对外逼近与穷举都夹住真实最优值、空预算和满预算、贪心对照、超时，
  以及 HiGHS 调度器的线程数问题。
- `experiments/verification.py` 用独立的有限运输 LP 核对固定选集风险。

100 维实验设计与基线见 [study100.md](study100.md)。
