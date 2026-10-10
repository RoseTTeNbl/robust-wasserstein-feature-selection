# 如何检查这套代码是否真的实现了理论

这份说明从“算法到底在最小化什么”开始，再跟着一次实际调用走到主问题。读代码时，先掌握这里的输入、输出与上下界关系，比从所有辅助函数逐个看起更容易发现数学错误。当前 0.5.0 搜索的逐步解释在 `docs/adaptive_search.md`；每个选中的二元组合都完整求解，不转移旧检验作为新解。英文理论笔记在 `note/main.md`；实验设计在 `docs/experiment_plan.md`。

## 1. 目录与职责

```
RubustHW/
  code/robusthw/
    problem.py       数据、权重、距离张量与结果对象
    oracle.py        固定变量的凸问题、对偶证书与可部署检验
    solver.py        新搜索入口、保留的主问题基线和独立枚举
    adaptive.py      不确定性选点、完整内层求解与全局认证
    history.py       已评估点、切线、固定核风险预测和距离权重
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

距离采用共同尺度上的 L1 距离。实验中只用 selection split 拟合标准化，然后冻结尺度。不能给每个支持集重新标准化或改半径，否则外层比较的目标已经变了。

合并所有源点作为目标点池后，距离张量的元素是

\[
\Delta_{kimj}=|x_{ki,j}-t_{m,j}|.
\]

它的四个下标依次是类别、源点、目标点和坐标。所有源点必须出现在池里，这不是可随意省略的实现细节：精确有限表示和全空间延拓都使用这个条件。

## 3. 固定支持集：原始问题提供上界

选择向量 z 固定后，真正的检验问题是

\[
\min_{a,u,\lambda\ge0}\quad
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

在当前点取到值 v=H(Gamma_hat) 后，对应的全局凸分式证书的梯度是

\[
g_j=-v\widehat\mu_j.
\]

因此 `gradient` 是 g，`intercept` 是 v-g·z，切平面写成

\[
\eta\ge\text{intercept}+g^Tz.
\]

同一个证书适用于所有 z，凸函数的切线在所有点都不超过该函数，因此这条割是全局下界。对偶数值解若只经过可行修复而没有精确取到最优，割仍合法，但可能不紧。代码要允许这种情况，不能用理论上的强对偶等式掩盖数值误差。

检查 oracle 返回值时最重要的是：`value` 是所选门控问题的下证书；整数 `z` 时，`upper` 来自原始固定支持问题的可行检验；分数 `z` 时，`upper` 只界定相应门控值，不能用于外层整数问题的上界。两者不能交换。归一化对偶的目标绝不能被用成原始可部署检验的风险上界。

## 6. 新搜索的一次循环

`solver.solve` 默认进入 `adaptive.py`。历史中只保存已完整评估的二元组合、各自的实际风险上界和该点切线。候选池最多保留少量组合，来源是近期切线建议的交换、坐标排序和多样性提议。

`history.py` 先给候选计算已有切线下界 \(a_t(z)\)。只保留 \(a_t(z)\le UB-\varepsilon/2\) 的候选；这一步使用的是真正有效的下界。

通过筛选后，固定 Hamming 核的 Gaussian 预测器利用已完整评估的 \((z^s,R_s)\) 预测风险均值 \(m_t(z)\) 和标准差 \(\sigma_t(z)\)，再按 \(m_t(z)-\sigma_t(z)\) 从小到大选一个。这里 \(R_s\) 是该组合实际求解所得可行检验的风险上界；预测器只读取达到内层精度要求的完成评估。核函数是 \(k(z,z')=\tfrac14\exp[-\|z-z'\|_1/\max(1,d)]\)，因此相近组合更容易共享已知风险的信息。低均值偏向有希望的组合，高标准差偏向值得探索的组合。

这个分数只决定求解顺序，不创建候选上的检验或切线，也不更新 UB 或 LB。标准差是预测模型的量，不是经过校准的真实风险置信界。`FixedKernelPredictor` 保存并追加 Cholesky 因子，批量预测复用该因子；核参数固定，不在循环中重新拟合。当前路径不调用几何 `scalar_upper` 或 `bounds` 排序。完整预测公式与增量更新在理论笔记附录 E，逐函数说明在 `docs/adaptive_search.md`。

选中的组合调用完整 `GateOracle.solve`。修复后实际检验上界与该点切线值之差必须不超过 epsilon/4；必要时在同一组合上收紧精度重试。完成后只保留一条该点切线，且只有这次实际求解得到的检验能够更新 UB。

`_master` 包含二元选择变量、一个 eta 和全部已生成割。它周期性求解，并在没有合格候选时立即求解。所得有效全局下界用于停止判断；所得二元候选占候选池中的一个位置，仍要完整求解内层，不继承历史检验作为新解。池中至多 8 个组合；认证已排期时先完成主问题，再进行一次候选批量预测。

主问题未完全求完时，`master.fun` 只是一个可行主问题目标，不代表全局下界。代码使用 `mip_dual_bound` 更新 LB；只有主问题返回 optimal 时才允许用其最优目标代替。专门的测试模拟了“incumbent 很好但全局下界很低”的情形，以避免一个常见的虚假最优性错误。

原问题写 sum z<=d。实现可限制 sum z=d，因为多观察一个坐标可以继续使用原来的检验，最优风险不会增加。于是任意不足预算的最优支持都可补齐，仍有一个满预算最优解。这不代表每个选中变量都有唯一的统计意义。

停止条件是

\[
UB-LB\le10^{-6}+10^{-5}|UB|.
\]

`optimal` 表示在声明的浮点精度下具有数值最优性证书。时间限制返回 incumbent 和实际界限；所需 oracle/master 精度未达到时返回明确的 accuracy limit。旧版逐主问题搜索由 `search='master'` 保留，其中重复支持而间隙未闭合会直接报告精度限制。新版通过切线改善筛选和实际原始—对偶精度排除重复评估；精度重试属于同一组合的一次完整评估。

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

## 10. 可选的完整分式割：怎样检查新增主问题

默认算法仍是前述 affine tangent 主问题。`solve(..., cut_mode="ratio")` 是单独测试的扩展，只支持 `gate="risk"`。它没有修改 oracle、整数风险目标或部署规则。

对归一化证书保留 `tau`、非负 `mu` 和经数值余量降低的 `v`，完整下界为

\[
f(z)=\frac{v}{\tau+\mu^Tz}.
\]

分母在生成点约为 1，在别的支持上会改变。只存切线会丢掉这种曲率；保留分式能约束所有整数候选。`solver._ratio_master` 为每个坐标增加一个共享变量 `y_j=eta*z_j`，而不是为每条割复制一组变量。用当前可行检验的风险 U 写出

\[
0\le y_j\le Uz_j,\quad y_j\le\eta,\quad
y_j\ge\eta-U(1-z_j),\quad 0\le\eta\le U.
\]

当 z_j=0 时强制 y_j=0；当 z_j=1 时强制 y_j=eta。于是分式 epigraph 变成线性约束 `v <= tau*eta + mu@y`。满预算还给出 `sum(y)=d*eta`。检查代码时尤其要看：U 只能来自原始整数可行 test；每次 U 改变，所有 product bounds 必须一起更新。

连续 McCormick 松弛并不保证比原切线更强，所以代码同时保留 affine cuts。完整分式在整数支持上保留更多信息，也不自动保证运行更快；这由 `configs/ratio_cuts.json` 的配对补充实验衡量。独立文档 `docs/ratio_cut_derivation.md` 给出推导、反例和零值边界；当前理论笔记的附录 E 专门解释预测器。

`code/tests/test_ratio_master.py` 将共享变量主问题与直接枚举每个二元支持上的分式下界比较，再检查完整搜索与原始风险枚举一致。这比仅检查代码生成了若干约束更有意义。

## 11. 求解器数值失败怎样处理

正式实验发现少数 HiGHS MILP presolve 返回 status 4。相同模型关闭 presolve 可以继续求解。统一政策是：仅在 status 4 时重试一次，模型不变，使用同一个剩余时间预算；记录首次失败与重试。首次失败的 `x`、`fun`、`mip_dual_bound` 均不能贡献有效证书。

`_milp_with_retry` 实现这条政策，`test_master_retry.py` 模拟两种主问题的失败、成功和超时。status 1 仍然只是达到时间限制，status 4 不会包装成 optimal。详情见 `docs/numerical_remediation.md`。

对已经产生的旧失败记录，原文件与 trace 会先归档，再按同一政策重算全部同类失败项。重算只获得 `30秒 - 原失败选择耗时` 的剩余预算；原尝试与重算共享同一个选择预算。最终累计原失败耗时，保留首次状态和原因，并保留两次尝试中最好的可行支持和有效下界。因此时间包含失败代价，重算也可能留下 time limit，不能只挑第二次成功运行计时。

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
