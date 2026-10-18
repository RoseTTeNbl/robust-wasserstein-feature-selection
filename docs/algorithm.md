# 核心算法与代码对应

一条记录有 $D$ 个数值，只能观察 $d$ 个。给定选集 $S$，先训练最好的鲁棒检验；
外层再寻找风险更小的选集。代码只实现实验报告使用的半 hinge 损失，没有旧 GP、
比例主问题或其他历史搜索分支。

## 1. 数据与风险

[`core/model.py`](../src/robusthw/core/model.py) 中的 `Problem` 保存两类训练矩阵、
两个正半径，以及各类总和为一的样本权重。预算不放进 `Problem`，所以所有预算
共享同一数据和半径。坐标差为 `delta[i,r,j] = abs(x[i,j]-x[r,j])`。

评分 $u\in[-1,1]$ 为正时倾向类别零。两类损失分别为

$$
\ell_0(u)=(1-u)/2,\qquad \ell_1(u)=(1+u)/2.
$$

接受类别零的概率为 $(1+u)/2$。风险是两类最坏损失的和；零评分风险为一。
因此这里的风险间隔与两类平均错误率相差二倍。

## 2. 一个固定选集怎样拟合

[`core/oracle.py`](../src/robusthw/core/oracle.py) 的 `fit_fixed` 先求解

$$
\begin{aligned}
\min_{a,u,\lambda}\quad &\sum_i q_i a_i+\sum_k\theta_k\lambda_k,\\
\text{s.t.}\quad &\ell_k(u_r)\le a_i+\lambda_k\sum_{j\in S}\Delta_{ir,j},
\quad i\in I_k,\ r\in[N],\\
&a\ge0,\quad\lambda\ge0,\quad -1\le u\le1.
\end{aligned}
$$

`scores` 是有限点的 $u$，`source_bounds` 是 $a$，`prices` 是 $\lambda$。
求解器容差可能留下微小违约，因此代码固定 $u,\lambda$，重新计算

$$
a_i=\max\{0,\max_r[\ell_k(u_r)-\lambda_k c_{ir}^S]\}.
$$

这样得到的可行目标是上界，不直接拿原始求解器目标当证书。

同一个函数还求归一化对偶：非负运输矩阵 $\Pi$ 的行和为 $\alpha q_i$，
坐标乘子为 $\mu$，且 $\alpha+\mu^\top z=1$，其中 $z=\mathbf1_S$。
半 hinge 的目标可精确计算为

$$
H(\Pi)=\sum_r\min\left\{\sum_{i\in I_0}\Pi_{ir},\sum_{i\in I_1}\Pi_{ir}\right\}.
$$

代码修复行和、运输预算及归一化，再计算这个 $H$，而非直接使用辅助变量的原始目标值。
按理论中的缩放，产生全域有效仿射割

$$
c(z')=H+H\mu^\top z-H\mu^\top z'.
$$

因此实际斜率是 `-H * mu`。`Cut` 保存截距和斜率，并减去明确记录的数值裕量。
坐标乘子使用附录 C 的数据界控制。证书采用带可行性修复的浮点计算，诊断项保存残差与裕量。

## 3. 新样本的评分

`FitResult.predict_score` 按附录 A 构造，不重新训练分类器。对新点 $y$，各类取使

$$
a_i+\lambda_k\|x_i^S-y^S\|_1
$$

最小的训练锚点 $i_k$。令 $r_0=\|x_{i_0}^S-y^S\|_1$，

$$
\rho=\|x_{i_0}^S-x_{i_1}^S\|_1,\qquad
g(y)=(1-\gamma)u_{i_0}+\gamma u_{i_1},\quad
\gamma=\min(1,r_0/\rho).
$$

当 $\rho=0$ 时取 $u_{i_0}$。相同最小值选最前面的存储索引。构造保证两类损失都在
对应包络下方；它不要求在每个训练点恰好返回存储的 $u_r$。未选坐标不会影响预测。

## 4. 外层怎样找变量

[`search.py`](../src/robusthw/search.py) 的 `select` 只包含四种方法：

| 方法 | 选点规则 | 结果含义 |
| --- | --- | --- |
| `global` | 保存已有割，再求二元主问题 | 可行解及全局上下界 |
| `enumeration` | 拟合所有大小为 $d$ 的选集 | 完成后给出枚举区间 |
| `forward` | 每轮比较所有加一个变量的选择 | 完成规定贪心过程 |
| `forward_swap` | 前向后，每轮接受最好的单变量交换 | 最多十次改进 |

多观察一个变量不会提高最优风险，因此搜索恰好 $d$ 列与理论中的至多 $d$ 列最优值相同。
各方法有自己的拟合缓存，没有跨方法免费复用。候选轮比较完才接受一步；超时仍保留
已核验的最好完整选集，预算未满则按预先确定的逐变量 MMD 排名补足，并标记未完成。

全局主问题是

$$
\min\eta\quad\text{s.t.}\quad
z\in\{0,1\}^D,\quad\sum_jz_j=d,\quad\eta\ge0,\quad\eta\ge c_s(z).
$$

可行检验给上界 $U$。下界 $L$ 只使用 HiGHS 返回的 `mip_dual_bound`；
主问题当前可行目标不能充当下界。仅当 $U-L$ 满足要求时返回 `optimal`。
贪心结果没有全局下界，状态 `complete` 也不表示全局最优。

## 5. 怎样核查

- `tests/test_core.py`：解析例子、重复样本、非均匀权重、预测包络和所有选集上的割。
- `tests/test_search.py`：全局与穷举一致、主问题失败、超时、预算补足和缓存。
- `experiments/verification.py`：独立运输 LP，使用另一套变量和矩阵检查小问题全部选集，
  同时检查每个方法自己的可行风险以及每条割。

默认统计精度和时间都来自实验报告。每次原问题、对偶和主问题共用剩余时间；
记录构造矩阵、初始化、交换和最终拟合的耗时。没有可核查检验时用零评分，且保留失败状态。
