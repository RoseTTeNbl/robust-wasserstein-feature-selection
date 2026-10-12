# 当前搜索：完整评估与全局主问题

本说明对应 0.7.0 的默认 `search="cuts"`，记录标签是
`certified_global_cuts_v1`。每步由全局二元主问题选出下一组变量，再完整求解
这组变量的检验问题。文件名保留供已有链接使用。旧 GP/候选池方法以
`search="adaptive"` 显式保留，用于历史比较。

## 1. 一次评估提供什么

固定满预算二元向量 \(z\)，内层最优风险记为 \(V(z)\)。完整求解 primal 与
dual 后得到实际检验风险上界 \(R\)、对偶下界 \(\ell\) 和全局切线

\[
\ell\le V(z)\le R,\qquad
L(x)=\ell+g^\top(x-z).
\]

外层绝对精度为 \(\varepsilon\)，内层必须满足
\(R-\ell\le\delta=\varepsilon/4\)。未达到时，在同一点收紧求解精度并完整
重优化。重试可保留更好的实际检验和同点证书，但最终只加入一条切线。
检验和切线都属于这次选中的支持；下一组变量仍完整求解。

同一个对偶证书还提供有效分式。代码保存修复后的原始目标 \(h\)、数值
下移余量 \(e\)、正尺度 \(\tau\) 和非负价格向量 \(\bar\mu\)，令

\[
n=[h-e]_+,\qquad F(x)=\frac{n}{\tau+\bar\mu^\top x}\le V(x).
\]

生成点归一化为 \(\tau+\bar\mu^\top z=1\)，所以分式在该点与 \(\ell\)
一致。`EvaluationHistory.add` 核验归属、分母、价格和生成点数值后，共同
保存实际检验、一条切线和这个分式。分式与切线来自同一次 oracle 返回值；
梯度保留原始 \(h\) 的梯度，余量 \(e\) 下移切线，不重新定义梯度。

## 2. 全部旧证书构成一个主问题

已有 \(t\) 次评估时，只有下面两个风险界需要维护：

\[
U_t=\min\{B,R_1,\ldots,R_t\},\qquad
a_t(x)=\max\{0,L_s(x),F_s(x):s\le t\}.
\]

\(B\) 是常数检验风险；\(U_t\) 来自真实可行检验。
\(a_t(x)\le V(x)\) 对每个二元组合都成立。每步求解

\[
\min_{x\in\{0,1\}^D,\ \mathbf1^\top x=d}a_t(x),
\]

得到全局下界 \(\mathrm{LB}_t\) 和候选 \(z_M\)。有效的旧下界仍保留，
因此 \(\mathrm{LB}\) 只上升。超时 MILP 的 `mip_dual_bound` 可提供全局
下界；可行目标 `fun` 本身不是全局下界。只有主问题已证最优时，才可以
用最优目标代替缺失的全局界。

`solver._ratio_master` 令 \(\eta\in[0,U_t]\)，并以共享变量
\(b_j=\eta x_j\) 表示所有分式。约束

\[
0\le b_j\le U_tx_j,\qquad b_j\le\eta,\qquad
b_j\ge\eta-U_t(1-x_j)
\]

在二元点精确强制这个乘积等式。每条分式只增加线性约束
\(n_s\le\tau_s\eta+\bar\mu_s^\top b\)，所有证书共用 \(D\) 个连续变量。
旧切线仍保留，因为下移后的切线与安全分式不必逐点互相支配。

## 3. 循环和停止条件

初始化取 \(\mathrm{LB}=0\)、\(U=B\)。初始支持由两类加权均值差除以
标准差之和的排序选出，或由 `warm_support` 给出。所有支持使用满预算；
增加观测变量仍能使用原检验，因此满预算限制不改变原最优风险。

此后循环只有以下步骤：

1. 完整评估当前组合，达到 \(R-\ell\le\varepsilon/4\)，保存实际检验与
   自有证书，更新 \(U\)。
2. 解全局二元主问题，更新有效 \(\mathrm{LB}\)。若
   \(U-\mathrm{LB}\le\varepsilon\)，返回已求得的最好实际检验。
3. 否则候选必须为尚未完成过的满预算二元组合，且满足
   \(a_t(z_M)-\mathrm{LB}\le\varepsilon/4\) 与
   \(a_t(z_M)\le U-\varepsilon/2\)，然后完整评估该候选。

第 3 步的改善条件由主问题精度推出：尚未停止时
\(\mathrm{LB}<U-\varepsilon\)，故
\(a_t(z_M)\le\mathrm{LB}+\varepsilon/4<U-3\varepsilon/4\)。代码同时核查
这些条件，防止数值精度不足、无效二元点或重复点被当成正常进展。

默认 cuts 仅使用绝对精度。显式给出非零 `relative_tolerance` 时，兼容
停止条件为 \(U-\mathrm{LB}\le\varepsilon+\varepsilon_{\rm rel}|U|\)；这种
`optimal` 不能解释成纯 \(\varepsilon\) 保证。内层和候选精度仍按绝对
\(\varepsilon/4\) 检查。旧 adaptive/master 默认相对精度保留为 \(10^{-5}\)。

## 4. 核心代码与失败状态

| 核心入口 | 实现的步骤 |
| --- | --- |
| `solver.solve` | 默认选择 cuts；显式保留两个旧比较路径 |
| `certified.solve_cuts` / `evaluate` | 初始化、完整重优化、同点精度重试和保存证书 |
| `history.EvaluationHistory.add` / `lower` | 保存归属明确的检验和证书，计算 \(a_t\) |
| `solver._ratio_master` | 带安全分式的全局二元主问题 |
| `solver._master` | `use_fractions=False` 时的纯切线主问题 |
| `oracle.GateOracle.solve` | primal、dual、可行修复及有效上下界；复用连续模型 |

当前 `EvaluationHistory(..., forecast=False)` 不实例化预测器。默认流程
没有候选池、随机外层选择、分数点内层求解、core-point 额外求解或检验
转移。保留和复用的是已有证书、固定数据距离张量和数值模型。

内层重试仍未达到要求时返回 `oracle_accuracy_limit`；主问题候选未达到
要求时返回 `master_accuracy_limit`；时间耗尽返回 `time_limit`；求解器
失败返回对应 `oracle_failure` 或 `master_failure`。新全局界若超过真实
检验上界，拒绝该界并保留之前的有效区间，返回 `numerical_inconsistency`。
这些状态都不宣称最优。

`diagnostics.evaluations` 逐点记录实际风险、切线、同证书分式与内层误差；
`trace` 记录主问题候选误差、评估前的 \(U\) 和改善阈值。`oracle_calls`
包括同点精度重试；`visited` 计不同已评估组合。因此两者不必相等。

## 5. 验证范围与历史边界

`test_certified_search.py` 用独立的原运输预算 LP 穷举比较半 hinge 最优
风险；logistic 与完整支持枚举比较。测试还检查全域证书、内层精度不足、
重复或无效主问题候选、被污染的全局界，以及默认确实不调用 GP/候选池。
历史路径测试现在显式选择 adaptive/master，继续核查其原行为。

0.5/0.6 的 GP 搜索与相应冻结记录保持原版本含义。新的四方法比较为当前
cuts 的纯切线/分式两种形式与历史 GP 的对应两种形式。它使用既有十六个
设置、四份生成数据和三次计时；这是回顾性流程比较，不是独立确认，也
不能把候选池与认证频率同时变化的效果归因于 GP 一个组件。

已保存的当前主问题对照中，保留分式使完整评估数从 601 降到 446
（25.8%），但按配置内三次计时中位数求和，耗时从 15.672 秒升到
18.509 秒（18.1%）。这支持减少内层评估次数，不支持运行速度提升。
全部 192 个结果及来源指纹独立保存在
`reports/verification/20261012/no_prediction/`；旧 GP 结果只作为对应流程对照。
