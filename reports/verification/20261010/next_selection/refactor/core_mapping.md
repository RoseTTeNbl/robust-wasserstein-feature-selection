# 核心代码与数学主线对应

默认算法每次选定二元 z 后完整求解内层，得到真实检验以及同一点对偶证书。预测只安排评估顺序；已有证书约束其它组合，全局主问题负责停止认证。

| 数学任务 | 核心代码 | 输入与输出 |
|---|---|---|
| 固定数据、半径和距离 | `problem.py: Problem` | 两类样本→距离张量、常数风险 B、只读价格界 M |
| 完整求解当前组合 | `oracle.py: GateOracle.solve` | 当前二元 z→真实检验 R、对偶下界 ell、梯度及完整证书 |
| 数值结果变成有效界 | `oracle.py: _repair_primal / _repair_dual` | 未核验解→修复后原始／对偶证书 |
| 保存并计算旧证书下界 | `history.py: EvaluationHistory.add / lower` | 一次完成评估→一条切线和安全分式；候选 z→a_t(z) |
| 排序下一个组合 | `history.py: proposal_pool / FixedKernelPredictor`；`adaptive.py` 候选筛选、排序 | 最多 8 个组合→筛选后按 m−sigma 选一个 |
| 全局下界和停止 | `solver.py: _ratio_master`；`adaptive.py: master / done` | 所有证书→全局 LB；U−LB 达精度则停止 |

## 1. 完整内层：上下界各有自己的来源

`_primal_lp` 构造 half-hinge 的有限 LP；`_primal_cone_model` 构造同一问题的 logistic 版本。`_dual_lp` 和 `_dual_cone_model` 构造源质量、预算分配、坐标覆盖和风险归一化约束。

两个 `*_cone_model` 集中普通和缓存路径中重复的数学建模。普通路径建模并求解；persistent 路径更新参数后仍求解。缓存复用不能把前一个 z 的解冒充当前解，失败状态后旧 value 不能提供证书。

`_repair_primal` 用实际损失和距离重新计算源截距，再计算风险 R。因此 `RobustTest` 是当前组合的真实可行检验；`interval / prob_class0` 将有限证书扩展到整个观测空间。只有这个 R 能更新实际最优上界 U。

`_repair_dual` 修复源行质量、分配预算、覆盖价格和归一化。记修复后归一化证书的目标为 h，梯度和数值下移量满足

\[
g_s=-h\bar\mu_s,\qquad \ell_s=h-e_s,\quad e_s\ge0.
\]

保存的 affine cut 为

\[
L_s(z)=\ell_s+g_s^T(z-z^s).
\]

梯度来自原始 h，然后切线整体减 e；不能把它当成降低分子后再求导。`Problem.price_bounds` 只计算一次 M，oracle 裁剪和 history 距离权重共用同一只读数组。

## 2. 同一证书保留完整分式

取 \(n_s=\max(0,h-e_s)\)，同一完成评估还提供

\[
F_s(z)=\frac{n_s}{\tau_s+\bar\mu_s^Tz}\le V(z),\qquad \tau_s>0,\ \bar\mu_s\ge0.
\]

分母为正，分子不超过原始对偶分子。代码核验生成点与 ell 一致，并绑定到被保留切线的同一次求解尝试。没有可用分式仍保留切线。

分式和下移切线不必全域互相支配，所以

\[
a_t(z)=\max\{0,L_s(z),F_s(z):s\le t\}.
\]

旧信息只在候选上代入，既不生成候选检验，也不额外生成切线。`use_fractions=False` 恢复原切线下界，预测规则保持相同，作为明确的比较路径。

## 3. 选点和全局停止

候选池、固定 Hamming 核和三次评估认证间隔保留原设计。通过 \(a_t(z)\le U_t-\varepsilon/2\) 的未评估组合，按预测均值减标准差排序。`FixedKernelPredictor.append` 仅学习准确实际评估的 R/B；均值、标准差都不贡献 UB/LB。

`evaluate` 完整重优化选定 z。精度重试留在同一支持，最终保留一条 own-point tangent 和同一证书的分式。每次更新 U 都来自实际检验。

`_ratio_master` 用共享变量 \(y_j=\eta z_j\) 表示所有分式：\(n_s\le\tau_s\eta+\bar\mu_s^Ty\)。product bounds 只使用实际 U；原切线约束保留。这个 MILP 搜索全体满预算二元组合，返回有效全局下界 LB。其可行目标不替代全局 bound，除非已证明 optimal。全局组合还须完整求解内层。

`U−LB` 达到声明精度才报告 optimal。时间限制和精度不足保留真实状态与有效区间。`enumerate_supports` 独立遍历支持，是核验参考，不参与默认选点。

## 4. 数值失败不能产生证书

`_milp_with_retry` 对 solve error 最多重试一次。同模型返回 infeasible 时，只有实际 incumbent 形成的完整 witness 严格满足全部 stored-float 模型约束，才允许关闭 presolve 重试。`_check_milp_witness` 使用精确有理数核验，不依赖求解器容差；真正不可行、没有见证或无效见证仍保留原失败。两次调用共享剩余时间，失败 x/fun/bound 清空。

## 5. 证据的版本范围

此前去重阶段的 `equivalence.json` 记录 24 组 logistic cone canonical 数据一致、关键函数 AST 比较、非法参数和 CLI 校验；`regression.log` 是当时 75 项回归。它们说明当时重构等价，不是当前分式算法的性能结果。

本次价格界去重另核验加权手算例 \(M=(21.25,13.75)\)、缓存共享与不可修改，并对 18 组真实 logistic 求解比较上下界及梯度，最大差为零。新增 witness 修复由 `test_ratio_witness_retry.py` 核验。当前算法的性能应读取 `next_selection` 下开发与独立确认结果；原 v3 和 1,640 项历史研究记录继续保持原版本标签。
