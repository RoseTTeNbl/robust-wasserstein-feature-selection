# 如何检查这套代码是否真的实现了理论

这份说明从“算法到底在最小化什么”开始，再跟着一次实际调用走到主问题。读代码时，先掌握这里的输入、输出与上下界关系，比从所有辅助函数逐个看起更容易发现数学错误。当前默认搜索保留每次完整求解得到的切线和同一对偶证书的分式下界；逐步解释在 `docs/adaptive_search.md`。每个选中的二元组合都完整求解，不转移旧检验作为新解。英文理论笔记在 `note/main.md`；实验设计在 `docs/experiment_plan.md`。

读默认算法时，先看下面五处。矩阵构造、数值修复和实验流程在后文展开。

| 要检查的步骤 | 核心代码 | 对应本文 |
|---|---|---|
| 输入如何固定风险问题 | `problem.py: Problem` | 第 2 节 |
| 完整求解，取得检验与证书 | `oracle.py: GateOracle.solve`、`_repair_primal`、`_repair_dual` | 第 3–5 节 |
| 旧证书约束未评估组合 | `history.py: EvaluationHistory.add`、`lower` | 第 6、10 节 |
| 选择下一个组合 | `certified.py: solve_cuts` 每步解全局主问题并检查候选精度 | 第 6 节 |
| 全局下界和停止条件 | `solver.py: _ratio_master`；`certified.py: done` 和主循环 | 第 6、10–11 节 |

## 1. 目录与职责

```
RubustHW/
  code/robusthw/
    problem.py       数据、权重、距离张量与结果对象
    oracle.py        固定变量的凸问题、对偶证书与可部署检验
    solver.py        新搜索入口、保留的主问题基线和独立枚举
    certified.py     当前默认：完整内层评估、全局主问题与停止
    adaptive.py      历史比较：候选池和风险预测
    history.py       已评估检验与证书；历史路径可开启预测器
    cli.py           单次求解入口
    experiments/
      data.py        数据机制、独立划分、训练阶段标准化
      baselines.py   MMD、Wang 稀疏核、L1 与随机选择
      metrics.py     置换检验、支持恢复、分类与压力测试指标
      runner.py      完整实验、记录、恢复运行
  code/tests/        独立数学校验与流程校验
  configs/           预先冻结的实验参数
  docs/              实验依据、代码解释、基线复现说明
  note/              英文理论正文、附录与 PDF
  reports/           实验报告源码与 PDF
  scripts/           环境准备、公开数据获取、结果汇总与绘图
  results/research_20261003/  本轮实际结果
  external/          本地第三方资料，仅作阅读，不上传
  data/              数据缓存，不上传，公开数据可重新获取
  _archive/          历史实现和笔记，不参与新版运行
  .venv-research/    新环境；旧 .venv 保留
```

核心代码不导入 COPT。半 hinge 的连续子问题和外层整数主问题都由开放求解器完成；logistic 是可选的指数锥扩展。实验用的 sklearn、绘图依赖与核心依赖在包配置中分组声明，不把第三方论文的整个工程塞进核心算法。

## 2. 从一个 Problem 对象开始

两类输入是 `x0`、`x1`，每行一个样本，每列一个坐标。`q0`、`q1` 是各自归一化的经验质量，默认均匀。`theta0`、`theta1` 是严格正的 Wasserstein 半径；本轮理论不覆盖零半径边界。`budget` 是最多观察的变量数。

距离采用共同尺度上的 L1 距离。历史统计实验只用 selection split 拟合标准化，然后冻结尺度；新版小实例计算验证直接使用预先声明的坐标尺度，不做标准化。不能给每个支持集重新标准化或改半径，否则外层比较的目标已经变了。

合并所有源点作为目标点池后，距离张量的元素是

\[
\Delta_{kimj}=|x_{ki,j}-t_{m,j}|.
\]

它的四个下标依次是类别、源点、目标点和坐标。所有源点必须出现在池里，精确有限表示和全空间延拓都使用这个条件。`Problem.price_bounds` 只计算一次源权重距离界 M，返回只读数组；oracle 价格裁剪和搜索距离权重共用这个值。

## 3. 固定支持集：原始问题提供上界

选择向量 z 固定后，真正的检验问题是

\[
\min_{a,u,\lambda:\,\lambda\ge0}\quad
J=\sum_{ki}q_{ki}a_{ki}+\theta_0\lambda_0+\theta_1\lambda_1,
\]

\[
a_{ki}+\lambda_k\sum_jz_j\Delta_{kimj}\ge\Phi(\sigma_ku_m).
\]

只有 lambda 必須非负，a/u 是自由变量。源点自距离为零，使 a 在可行解中自动非负。半 hinge 可用 r_m=(1+u_m)/2、0<=r_m<=1 表示，因此两类右边分别是 1-r_m 和 r_m。这是 LP；不需要神经网络或交替训练。

`GateOracle.fixed_support(z)` 完整重优化这个问题，并构造 `RobustTest`。其 `primal_objective` 是可行检验的稳健风险上界。程序会保守修复小的数值违反，然后重新计算目标，而不是把求解器未经检查的浮点目标直接当成证书。

为什么原始问题的目标是上界？约束让任意允许的移动后损失都被“源点截距 + 距离价格”控制。把这条逐点不等式对预算内的运输积分，便得到第 k 类风险不超过 q_k·a_k+theta_k lambda_k。两类相加就是 J。

## 4. 新门控做了什么

令每个坐标有一对连续价格 w_0j,w_1j。未选坐标价格为零，选中坐标价格不超过 lambda_k。常数检验风险 B 为半 hinge 的 1，或 logistic 的 2 log 2。

旧 box 门控给两类各一个独立额度 B/theta_k。triangle 门控共享额度 theta_0 w_0j+theta_1 w_1j<=B z_j。新 risk 门控使用正在最小化的风险 eta：

\[
J\le\eta,\qquad
\theta_0w_{0j}+\theta_1w_{1j}\le\eta z_j.
\]

在整数 z 上三种门控都保持原问题的最优值。变化发生在分数 z 上，所以 box/triangle/risk 消融用于比较外层求解效率。在固定 z 的子问题中这些约束是线性的；不能把 eta*z 当成联合凸约束。

`GateOracle.solve(z)` 对分数点求相应门控问题，对整数点求完整固定支持问题及相应对偶。它不会固定上一次的运输方案，也不会靠混合历史方案来替代重优化。

## 5. 对偶为什么能给所有支持集一条割

对偶质量 Gamma 把每个源点质量分到池中目标点。每个目标接收到两类质量后，其最小可分辨风险为半 hinge 的 min(p0,p1)，或 logistic 的齐次熵表达式。全部目标相加得到 H(Gamma)。

新风险门控对偶用 tau、Gamma_hat、beta_hat、mu_hat，并满足

\[
\sum_m\widehat\Gamma_{kim}=\tau q_{ki},\quad
\sum_j\widehat\beta_{kj}\le\tau\theta_k,
\]

\[
A_{kj}(\widehat\Gamma)\le\widehat\beta_{kj}+\theta_k\widehat\mu_j,
\qquad \tau+\widehat\mu^Tz=1.
\]

记修复后证书的原始目标为 h=H(Gamma_hat)。全局凸分式证书在当前点的梯度是

\[
g_j=-h\widehat\mu_j.
\]

`_repair_dual` 随后减去非负数值余量 e：`value = h - e`，`intercept = value - g·z`。因此保存的是原分式切线整体向下平移：

\[
\eta\ge\text{intercept}+g^Tz.
\]

同一个证书适用于所有 z，凸函数的切线在所有点都不超过该函数，因此这条割是全局下界。对偶数值解若只经过可行修复而没有精确取到最优，割仍合法，但可能不紧。代码要允许这种情况，不能用理论上的强对偶等式掩盖数值误差。

检查 oracle 返回值时最重要的是：`value` 是所选门控问题的下证书；整数 `z` 时，`upper` 来自原始固定支持问题的可行检验；分数 `z` 时，`upper` 只界定相应门控值，不能用于外层整数问题的上界。两者不能交换。归一化对偶的目标绝不能被用成原始可部署检验的风险上界。

## 6. 默认搜索的一次循环

`solver.solve` 默认以 `search="cuts"` 进入 `certified.solve_cuts`。先用两类加权均值差除以标准差之和排序，选出满预算的初始组合；平局按坐标原顺序处理。也可通过 `warm_support` 给出同样满足预算的起点。

接下来只重复两个优化步骤：

1. `evaluate` 完整求解当前二元组合的 primal 与 dual，要求真实检验风险与其对偶下界之差不超过 \(\delta=\varepsilon/4\)。必要时在同一点收紧精度重试；最终只保存一条该点切线和同一证书的安全分式。真实检验风险更新 \(U_t=\min(B,R_1,\ldots,R_t)\)。
2. 用全部已存证书解全局二元主问题，更新 \(\mathrm{LB}_t\)。先检查停止条件；尚未停止时，主问题候选必须达到 \(a_t(z)-\mathrm{LB}_t\le\varepsilon/4\)，通过 \(a_t(z)\le U_t-\varepsilon/2\) 检查，且未完成过评估，再回到第 1 步。

已有证书构成

\[
a_t(z)=\max\{0,L_s(z),F_s(z):s\le t\}.
\]

\(L_s\) 是切线，\(F_s\) 是同一次求解所得的分式下界。分式保存与主问题实现见第 10 节；未评估组合只代入证书，再由主问题决定下一点。`use_fractions=False` 去掉 \(F_s\)，用纯切线主问题，其余完整评估和精度要求相同。

`EvaluationHistory(..., forecast=False)` 只保存检验、切线和分式，不实例化预测器。默认路径没有候选池、随机选点、分数点评估或 core-point 额外求解。复用的是固定数据的距离张量、连续求解模型及已有证书。

`master` 用全部旧证书搜索整个组合空间。`mip_dual_bound` 可以更新全局 LB，可行目标 `fun` 则不能；只有返回 optimal 时才允许用最优目标作为下界。添加证书后历史 LB 仍有效，因此取运行最大值。主问题候选还须通过筛选并完整求解，它本身不是检验。

原问题允许 \(\sum_jz_j\le d\)，实现搜索 \(\sum_jz_j=d\)：增加观测坐标仍可使用原检验，所以总存在满预算最优解。这不表示每个变量都有独立统计意义。

停止条件为

\[
U_t-\mathrm{LB}_t\le\varepsilon+\varepsilon_{\mathrm{rel}}|U_t|.
\]

当前 cuts 默认 \(\varepsilon=10^{-6}\)、\(\varepsilon_{\mathrm{rel}}=0\)。显式设置非零相对容差时，`optimal` 只表示上面的混合精度；不能把它写成纯 \(\varepsilon\) 保证。旧 adaptive/master 路径未给参数时保留 \(\varepsilon_{\mathrm{rel}}=10^{-5}\)，配对验证统一设为零。超时或精度不足返回对应状态及有效区间。

保留的 `search="adaptive"` 是原 GP/候选池比较路径；`search="master"` 保留早期主问题流程、其它 gate、root preprocessing 和 legacy `cut_mode="ratio"`。它们的历史记录不改动。当前默认只支持 risk gate、`cut_mode="tangent"` 与 `core_point=False`；这里的分式由 `use_fractions` 控制。同点精度重试属于一个组合的一次评估，并且只保留最终一条切线。

## 7. 检验如何用于新样本

对每一类计算

\[
E_k(x)=\min_i[a_{ki}+\lambda_k d_z(x_{ki},x)].
\]

半 hinge 的可行 class-0 概率区间为

\[
\max(0,1-E_0(x))\le p_0(x)\le\min(1,E_1(x)).
\]

取中点得到一个完整的可部署检验。它只依赖选中坐标。`prob_class0(x)` 输出接受 class 0 的概率；确定性预测在 0.5 处阈值化，平局归 class 0。logistic 的区间端点为 exp(-E0) 和 1-exp(-E1)。

优化目标保证的是允许随机化/对应替代损失的风险。确定性阈值化误差是另一个实测量，不能把它与半 hinge 风险直接画成同一个证书。

## 8. 实验代码的数据流

`data.py` 为 synthetic trial 分别创建 selection、head、final test 的独立随机流。选择器只能看 selection。每个方法选完变量后，`runner.evaluate` 用完全独立的 head split 重新拟合同一个 robust head 和同一个 RBF SVM，然后评价 final test。这样比较的主要是支持集质量，而不是不同分类器的能力。

两样本 equality test 又使用新鲜的批次数据。支持、标准化与带宽在测试前冻结。`metrics.permutation_mmd` 每次用 199 个随机标签置换及 observed assignment，p 值为

\[
p=(1+\#\{T_{perm}\ge T_{obs}\})/200.
\]

零假设下的拒绝率必须和 alternative 下的 power 同时报告。个体分类 false-positive rate 不是这个 equality-test Type-I error，两者的分母和实验对象不同。

一个训练种子下有 20 次测试重复。汇总首先在该种子内取平均，再跨 30 个训练种子计算不确定性；不能把 600 个条件重复宣称为 600 次独立变量选择。

Wang baseline 保留一维 RBF 加权核的 mean-minus-variance 目标与稀疏单位范数约束。小 D 枚举支持、全局解球面 trust-region 子问题；真实数据大 D 用明确标记的 forward/swap 近似。`docs/baseline_reproduction.md` 记录源码符号差异。univariate MMD 排名是单独的弱基线，不能冒充完整 Wang 方法。

压力测试把两类新样本沿相反方向平移，报告若干预定 rho 的误差。它是有限分布压力实验，不是训练经验 Wasserstein 球的精确最坏风险。另外在训练源点上做半径内的平移，已知耦合成本不超过 theta，因此能直接检查 fitted certificate。

## 9. 建议的检查顺序

1. 先运行 `code/tests`，看独立 transport dual、手算一维例子、cross-support cuts 和主问题界限是否一致。
2. 运行 CLI 的小例子并开启枚举，比较两个数值区间而不只比支持。存在多个最优支持时支持不一致不一定是错误。
3. 查看 `results` 中一条 oracle/master trace：UB 只能由整数可行 test 更新，LB 来自主问题全局界；两者应逐渐接近。
4. 对一个 fitted head 改动所有未选坐标，概率应该完全不变；对半径内 source-atom 平移，替代损失不得超过上证书。
5. 最后检查统计实验的 split indices、标准化参数、permutation seeds 和失败行，再解释汇总图。数值算法正确不自动证明统计功效更高。

环境与运行命令见根 README。历史文件已归档，既不导入新版包也不当作本轮实验结果。

## 10. 同一证书怎样得到更强的主问题

一次完整评估已经求出了归一化对偶证书的原始目标 h、非负价格 \(\bar\mu\) 和正尺度 \(\tau\)。这些量确定一个有效分式

\[
f_s(z)=\frac{h}{\tau+\bar\mu^Tz}\le V(z).
\]

生成点满足 \(\tau+\bar\mu^Tz^s=1\)。代码从分子减去数值余量 e，保存

\[
n_s=\max(0,h-e),\qquad
F_s(z)=\frac{n_s}{\tau+\bar\mu^Tz}.
\]

所以 \(0\le F_s(z)\le f_s(z)\le V(z)\)。正 \(\tau\) 与非负 \(\bar\mu\) 保证每个二元点的分母为正；没有有效正尺度时仍可保留原切线。`EvaluationHistory.add` 核验分式在生成点与本次对偶下界一致，随后共同保存分式和切线。

切线使用原始 h 的梯度，再整体减 e；分式从分子减 e。两者全域不必互相支配，所以取最大值。这是在已有完整求解后保留更多信息，不增加内层调用，不改变实际风险目标。

`solver._ratio_master` 用共享变量 \(y_j=\eta z_j\) 表示全部分式。给定真实检验风险上界 U，添加

\[
0\le y_j\le Uz_j,\quad y_j\le\eta,\quad
y_j\ge\eta-U(1-z_j),\quad 0\le\eta\le U.
\]

当 \(z_j=0\) 时 \(y_j=0\)，当 \(z_j=1\) 时 \(y_j=\eta\)。于是 \(\eta\ge F_s(z)\) 等价于

\[
n_s\le\tau_s\eta+\bar\mu_s^Ty.
\]

所有分式共用 D 个连续变量 y，原切线仍保留。满预算还给出 \(\sum_jy_j=d\eta\)。每次实际 U 改变，全部 product bounds 同时更新；预测均值不能作为这里的 U。

新的主问题仍是 MILP，求 \(\min_{z\in\mathcal Z_d}a_t(z)\)。二元点保留完整证书的曲率；约束变强不自动保证每次 MILP 更快。因此实验同时记录完整评估数、主问题调用数、总时间和失败。

`test_ratio_master.py` 用直接二元枚举核验共享变量模型。新增分式搜索测试核验同点证书归属、跨组合下界、开关和完整评估要求。legacy `cut_mode="ratio"` 保留在原 master-only 路径；完整推导见 `docs/ratio_cut_derivation.md`。

## 11. 求解器数值失败怎样处理

`_milp_with_retry` 在两种情况下关闭 presolve 对同一模型重试一次：HiGHS solve error（status 4）；或者 infeasible（status 2），但实际 incumbent 形成的完整见证严格满足该模型。成功与超时的首次求解不触发这项重试。

分式主问题的见证取已经评估的最好组合 z、eta=U、y=Uz。`_check_milp_witness` 把存下的浮点系数当作精确有理数，逐项核验 bounds、binary integrality 和每条线性约束，不把容差内的正违反当成严格可行。没有见证或核验不通过，不允许据此重试 infeasible；真正不可行的模型仍报告不可行。

两次调用共享剩余时间预算。首次失败的 `x`、`fun`、`mip_dual_bound` 清空，不能贡献证书；重试仍失败则保留真实状态。`test_master_retry.py`、`test_ratio_witness_retry.py` 覆盖污染的失败字段、无效见证、真实不可行和预算耗尽。

旧失败记录的重算保留原文件与 trace，只获得原时间预算扣除原尝试耗时的余额；累计时间包含失败代价，并保留两次尝试中最好的有效界。原研究重算政策及结果见 `docs/numerical_remediation.md`。新增见证防护属于本次版本，不改变旧实验结果。

## 12. 想核查矩阵符号时，具体看哪几段

这部分用于对照 `oracle.py`，不需要先把所有矩阵构造代码读完。记 `N=n0+n1`，即合并点池的大小，`D` 是坐标数。代码中的 `p.n` 是 N，不是某一类的样本数。

### 12.1 `_primal_lp` 把风险问题怎样交给 LP

固定支持时，变量依次排列为 `a` 的 N 项、`lambda` 的两项、`r` 的 N 项。`r=(1+u)/2` 是接受 class 0 的概率。目标向量 `c` 前 N 项是各类样本权重，接下来的两项是 theta，概率项的目标系数为零。因此 `c @ x` 就是 J。

SciPy 需要的是 `A_ub @ x <= b_ub`，而数学公式写的是大于等于。把两类约束移项，就得到

\[
\begin{array}{ll}
\text{class 0:}&-a_{0i}-\lambda_0d_z(x_{0i},t_m)-r_m\le-1,\\
\text{class 1:}&-a_{1i}-\lambda_1d_z(x_{1i},t_m)+r_m\le0.
\end{array}
\]

所以在 `data.extend` 中 class 0 的 score 系数为 -1、class 1 为 +1，右边分别是 -1 和 0。检查这两个符号即可核查最容易写反的类别方向。

分数点 gate primal 另外增加两行坐标价格 `w`，每行 D 项。样本约束中的 `lambda*d_z` 改成 `sum_j Delta*w`。随后添加 `w_kj-lambda_k<=0`。风险门控最后 D 行是

\[
\theta_0w_{0j}+\theta_1w_{1j}-z_j(c^Tx)\le0.
\]

代码用 `-z[:,None] * c[nz][None,:]` 把完整 J 的非零目标系数复制进这些行。因此这确实是 `theta*w <= z*J`，不能把 J 的样本截距部分漏掉。由于 eta 只承担风险上界，固定其余变量时可取 eta=J；附录 B 的二元等价推导说明该表示。

小例子：n0=2、n1=3、D=4 时，N=5。固定支持 LP 有 12 个变量、25 条样本约束；risk gate primal 有 20 个变量、37 条约束，其中另外 8 条控制 w<=lambda，4 条控制风险门控。分数点的问题规模仍然是一个普通 LP。

### 12.2 `_dual_lp` 怎样把 overlap 写成线性目标

risk dual 的变量按 `Gamma`、`beta`、`mu`、`tau`、`h` 排列。Gamma 共 N² 项；beta 共 2D 项；mu 共 D 项；tau 一项；h 共 N 项。在上面的例子中共有 43 个变量。

`h_m` 是点池中第 m 个目标上的 overlap。添加

\[
h_m\le\sum_i\Gamma_{0im},\qquad
h_m\le\sum_i\Gamma_{1im},\qquad h_m\ge0
\]

并最大化 `sum(h)`，就等于最大化 `sum_m min(p0_m,p1_m)`。由于 LP 接口默认最小化，代码把 h 的目标系数设成 -1。这里没有近似 minimum。

等式矩阵先放 N 条 source conservation：`sum_m Gamma_kim - tau*q_ki = 0`；再放一条 `tau + mu@z = 1`。不等式矩阵放 `A_kj-beta_kj-theta_k*mu_j<=0`、两条 `sum(beta_k)-tau*theta_k<=0`，以及两组 overlap 约束。这与归一化对偶公式逐项对应。

LP 求出的原始 mu 没有直接作为最终割使用。`_repair_dual` 先修正质量行和 beta 预算，再由实际成本计算足够覆盖这些成本的非负 mu，最后把整份证书一起归一化，重新计算 overlap、梯度和数值余量。这样割来自修复后证书自身的目标，而不是求解器显示的未经核查目标。

### 12.3 两次修复分别维护什么

`_repair_primal` 先把 lambda 放回非负范围，半 hinge 概率放回 [0,1]，并把 w 放回它的门控域；随后令

\[
a_{ki}=\max\{0,\max_m[\ell_{km}-\sum_j\Delta_{kimj}w_{kj}]\}+10^{-10}.
\]

直接重新生成满足全部样本约束的截距。对于风险门控，必要时缩小 w 后再次生成截距。缩小非负 w 会提高所需截距、提高 J，所以这一遍缩放足以使 `theta*w<=z*J` 成立。这段修复维护的是上界。

`_repair_dual` 维护的是下界。两者可以给出一个小的数值区间，不能为了显示相等而强行把其中一个替换成另一个。

logistic 的 `_primal_cone` / `_dual_cone` 使用对应凸损失与熵表达式，流程仍然是“求解、修复、从自身证书计算界”。本轮大批实验使用 half hinge；logistic 主要由小型独立检验验证。

完整 trace 侧重搜索阶段、上下界、支持和耗时，不保存每一步的大型质量矩阵。若要检查一条完整对偶证书，可直接调用 `GateOracle(problem).solve(z)`，查看 `certificate` 的 gamma/beta/mu/tau 与残差；这比把紧凑 trace 误认为完整证书更准确。
