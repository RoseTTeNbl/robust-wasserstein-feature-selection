# 阅读指南：公式、算法直觉、核心代码与 20 维结果

这份指南带你读懂三样东西：

1. 正式笔记 [robust_coordinate_selection.pdf](../note/robust_coordinate_selection.pdf) 里的公式和算法；
2. 核心代码 `src/robusthw/algorithm/`；
3. 20 维检查的结果 [results/study20-v1/summary.md](../results/study20-v1/summary.md)。

研究方向的想法和可能的定理在 [研究方向笔记](../note/directions/research_directions.pdf)（中文）里。

本指南里的公式编号都按当前正式笔记：Eq. (1)–(9)，附录 A 是命题 2.2 的证明。
标"方向笔记"的编号指研究方向笔记。

**建议的读法**

- 只想抓住主线（约 30 分钟）：读第 1 节、第 3.4 节、第 3.7 节、第 4 节、第 7.6 节。
- 要改代码：再读第 2 节和第 5 节，然后照第 5.10 节跑一个小例子。
- 要分析实验：读第 6 节和第 7 节。

---

## 目录

1. [一张图看懂整个问题](#1-一张图看懂整个问题)
2. [符号速查表](#2-符号速查表)
3. [正式笔记的公式逐个拆解](#3-正式笔记的公式逐个拆解)
4. [算法直觉：OA 在做什么，为什么在这里慢](#4-算法直觉oa-在做什么为什么在这里慢)
5. [核心代码导读](#5-核心代码导读)
6. [研究方向笔记的几个公式和实验代码](#6-研究方向笔记的几个公式和实验代码)
7. [怎么看 20 维结果](#7-怎么看-20-维结果)
8. [复现和修改实验](#8-复现和修改实验)

---

## 1. 一张图看懂整个问题

**数据。** 有两类样本：类 0 有 $n_0$ 个点 $x^{(0)}_1,\dots,x^{(0)}_{n_0}$，类 1 有 $n_1$ 个点，每个点是 $D$ 维向量。

**检测器。** 检测器是一个打分函数 $f:\mathbb R^D\to\mathbb R$。约定：$f(x)<0$ 判为类 1，$f(x)\ge0$ 判为类 0。

> 为什么是这个方向？风险里类 0 用 $\ell(-f)$、类 1 用 $\ell(f)$，而 $\ell$ 不减。
> 所以类 0 希望 $f$ 大，类 1 希望 $f$ 小。

**损失。** 理论对任何非负、不减、凸、$\ell(0)=1$、$t\to-\infty$ 时趋于 0 的 $\ell$ 都成立（方向笔记第 1.1 节）；代码和实验用 hinge 损失 $\ell(t)=(1+t)_+=\max(0,1+t)$。

- 若 $f\in[-1,1]$，类 0 的损失 $\ell(-f)=1-f$，类 1 的损失 $\ell(f)=1+f$，两者都是线性的。
- $f=0$（完全不判断）时，两类损失之和是 2。这就是"什么都不知道"的风险，代码里叫 `ZERO_SCORE_RISK = 2`。

**对手。** 真实分布未知，我们只有经验分布 $\widehat{\mathbb P}_k$。对手可以把每一类的经验分布在 Wasserstein-1 距离 $\rho_k$ 以内任意移动，距离用 $\ell_1$ 范数度量。

- 直观上，对手有一笔"搬运预算"：把质量 $q$ 从点 $x$ 搬到 $y$ 要花 $q\lVert x-y\rVert_1$，类 $k$ 总花费不超过 $\rho_k$。

**稳健风险。** 检测器的稳健风险是对手最坏情况下两类期望损失之和：

$$
R(f)=\sup_{\mathbb P_0\in\mathcal P_0,\ \mathbb P_1\in\mathcal P_1}\Bigl\{\mathbb E_{\mathbb P_0}\ell(-f(X))+\mathbb E_{\mathbb P_1}\ell(f(X))\Bigr\}.
$$

**变量选择。** 检测器只能看 $d$ 个坐标 $S$：$f(x)=g(x[S])$。

- 令 $V(S)$ 为只看 $S$ 的最好检测器的稳健风险。
- 目标是找 $\min_{|S|\le d}V(S)$。
- $V(S)$ 越小越好。它的范围是 $[0,2]$：0 表示完全可分，2 表示完全分不开。

**一句话的直觉（hinge 损失）。** 对手的最优策略是当"媒人"：把类 0 的点和类 1 的点成对地搬到同一个位置，制造重叠。重叠部分谁也分不清。

- 一对点 $(x^{(0)}_i,x^{(1)}_j)$ 只在坐标 $S$ 上被检测器看到，所以撮合它们的代价是 $\lVert x^{(0)}_i[S]-x^{(1)}_j[S]\rVert_1$。
- 对手用预算 $\rho_0+\rho_1$ 尽量多撮合，最终 $V(S)=2\times$ 撮合成功的质量。
- **好的 $S$ 让跨类点对在 $S$ 上都离得很远，对手撮合不起来。** 这是方向笔记命题 2，正式笔记没有直接写出来，但它是理解一切的钥匙。

一个两点例子：类 0 只有一点 $(0,0)$，类 1 只有一点 $(1,1)$，$\rho_0+\rho_1=1/2$，$d=1$。

- 只看坐标 1 时，两点距离是 1。预算 1/2 只够撮合一半质量，所以 $V(\{1\})=2\times 1/2=1$。
- 看两个坐标时距离是 2，只能撮合 1/4，所以 $V(\{1,2\})=1/2$。
- 坐标越多，点越"远"，对手越难撮合。这也是第 7 节"高维虚假可分"的来源。

---

## 2. 符号速查表

| 符号 | 读作 | 含义 | 代码里 |
|---|---|---|---|
| $D$ | 维数 | 坐标总数 | `problem.dimension` |
| $d$ | 预算 | 最多选几个坐标 | `budget` |
| $S$, $z=\mathbf 1_S$ | 选集、选择向量 | $z_m=1$ 表示选坐标 $m$；松弛时 $z\in[0,1]^D$ | `support`（0 起编号的元组），`problem.mask(S)` |
| $n_k$ | 样本量 | 类 $k$ 的样本数 | `len(problem.x0)` |
| $N=n_0+n_1$ | 总样本 | 两类合并后的样本数 | `problem.size` |
| $x[S]$ | 投影 | 只保留 $S$ 中坐标 | `x[:, list(support)]` |
| $\rho_k$ | 半径 | 类 $k$ 的 Wasserstein 预算 | `problem.radii[k]` |
| $\ell$ | 损失 | 理论：一般损失；代码：hinge $(1+t)_+$ | 代码中写死为 hinge |
| $\mathcal X$ | 目标点集 | 所有样本点（两类合并），对手只需把质量搬到这些点 | 合并样本 `problem.x` |
| $g_y$ | 分数 | 检测器在目标点 $y$ 上的分数 | `scores`（长度 $N$，限制在 $[-1,1]$） |
| $a^{(k)}_i$ | 源点上界 | 第 $i$ 个类 $k$ 源点能承受的"损失额度" | `source_bounds` |
| $\lambda_k$ | 价格 | 类 $k$ 每单位运输距离的价格 | `prices` |
| $\gamma^{(k)}_{i,y}$ | 运输计划 | 从源点 $x^{(k)}_i$ 搬到 $y$ 的质量 | `fit_fixed` 的乘子，修成可行后为 `plan`（$N\times N$） |
| $p_k(y)$ | 目标质量 | 搬完后类 $k$ 在 $y$ 的质量 | `hinge_value` 里的 `mass` |
| $H(\gamma)$ | 最小加权损失 | $\sum_y\psi(p_0(y),p_1(y))$；hinge 下 $=2\sum_y\min(p_0(y),p_1(y))$，即 2 倍重叠 | `hinge_value(problem, plan)` |
| $M(\gamma)$ | 平均质量 | 两类总质量的平均 | 不单独存储；`plan` 的行和为 $q_i$ 时 $M=1$ |
| $c_k(\gamma)[m]$ | 坐标代价 | 类 $k$ 的计划在坐标 $m$ 上的运输代价 | `ratio_cut` 里的 `cost` |
| $\mu_m$ | 豁免价格 | 见第 3.7 节；二元点上是 $\pi_m=\max_kc_k[m]/\rho_k$ | `ratio_cut` 里的 `mu`，割的斜率 `slope = mu / H` |
| $F(z)$ | 选择函数 | 凸函数，$F(\mathbf 1_S)=V(S)$；$1/F$ 凹 | 不直接计算；割给出 $1/F$ 的上界 |
| $c_t(z)$ | 割 | $1/F$ 在 $z^t$ 处的仿射上界 | `Cut(intercept, slope)` |
| $U_t$, $L_t$ | 上界、下界 | 外逼近的全局上下界 | `run.fit.upper`, `run.lower` |
| $q_i$ | 权重 | 合并样本 $i$ 的质量 $1/n_k$；$\sum_i q_i=2$ | `problem.q` |
| $\Delta_{ir}[m]$ | 坐标距离 | $\lvert x_i[m]-x_r[m]\rvert$ | `problem.delta`（$N\times N\times D$） |

---

## 3. 正式笔记的公式逐个拆解

### 3.1 Eq. (1)：最坏风险

$$
R(f)=\sup_{\mathbb P_0\in\mathcal P_0,\mathbb P_1\in\mathcal P_1}\bigl\{\mathbb E_{\mathbb P_0}\ell(-f(X))+\mathbb E_{\mathbb P_1}\ell(f(X))\bigr\}
$$

逐项读：

- $\mathbb E_{\mathbb P_0}\ell(-f(X))$：类 0 的期望损失。$f$ 越大，类 0 越安全。
- $\mathbb E_{\mathbb P_1}\ell(f(X))$：类 1 的期望损失。$f$ 越小，类 1 越安全。
- $\sup$：对手在两个球里各挑一个分布，让总损失最大。
- $\mathcal P_k=\{\mathbb P:\mathcal W_1(\widehat{\mathbb P}_k,\mathbb P)\le\rho_k\}$：以经验分布为中心、半径 $\rho_k$ 的 $\ell_1$-Wasserstein 球。

和错误率的关系：$\ell\ge1$ 在 $[0,\infty)$ 上，所以 $\mathbb P_0(f<0)+\mathbb P_1(f\ge0)\le$ 两类 hinge 损失之和。
**稳健 hinge 风险是两类错误率之和的上界。** 这就是后面说的"证书"。

### 3.2 Eq. (2)：选择问题

$$
\min_{S\subseteq[D],\,|S|\le d}\ \min_{f\in\mathcal F_S}R(f)=\min_{|S|\le d}V(S).
$$

外层选坐标（组合问题，$\binom Dd$ 种），内层给定坐标后选最好的检测器（凸问题）。
难点全在外层；内层的 $V(S)$ 可以精确算。

### 3.3 Lemma 2.1 / Eq. (3)：最不利分布

$$
V(S)=\max_{p,\gamma\ge0}\sum_{y\in\mathcal X}\inf_{g_y}\{p_0(y)\ell(-g_y)+p_1(y)\ell(g_y)\}
$$

约束有三条：

- 每个源点的质量全部搬走：$\sum_y\gamma^{(k)}_{i,y}=1/n_k$；
- 目标质量是搬来的总和：$p_k(y)=\sum_i\gamma^{(k)}_{i,y}$；
- 类 $k$ 的运输总代价不超过 $\rho_k$，代价只在 $S$ 的坐标上计：$\sum_{i,y}\gamma^{(k)}_{i,y}\lVert x^{(k)}_i[S]-y[S]\rVert_1\le\rho_k$。

拆开理解：

1. **目标点只需要是样本点**（$y\in\mathcal X$）。这是 Gao 等 (2018) 的结果：对手不需要把质量搬到别的地方。
2. **每个目标点上，检测器各自给一个最好的分数 $g_y$。** 因为检测器可以是任意可测函数，不同目标点互不影响，所以 $\inf$ 可以逐点取。
3. **hinge 下的逐点最小值：**
   $$\inf_t\{a(1-t)_++b(1+t)_+\}=2\min(a,b).$$
   在 $t\in[-1,1]$ 上它是 $a(1-t)+b(1+t)$，关于 $t$ 线性。$a>b$ 时取 $t=1$ 得 $2b$，反之取 $t=-1$ 得 $2a$。
4. 所以 hinge 下 **$V(S)=2\max\sum_y\min(p_0(y),p_1(y))$ = 2 × 对手能制造的最大重叠质量**。
5. "投影收缩代价"那句话的意思：对手在 $\mathbb R^D$ 里移动点，检测器只看 $S$；对手在不被看到的坐标上可以随便移动而不花钱（保留原坐标即可），所以只按 $S$ 上的距离计费。

代码不单独解这个 LP：`fit_fixed` 解 Eq. (4) 时，约束的乘子就是它的解（第 5.5 节）。

### 3.4 Eq. (4)：检测器 LP（最核心的一条）

$$
V(S)=\min_{g,a,\lambda}\ \frac1{n_0}\sum_ia^{(0)}_i+\frac1{n_1}\sum_ja^{(1)}_j+\rho_0\lambda_0+\rho_1\lambda_1
$$

约束对每个源点 $i$（或 $j$）和每个目标点 $y\in\mathcal X$ 各一条：

$$
\ell(-g_y)\le a^{(0)}_i+\lambda_0\lVert x^{(0)}_i[S]-y[S]\rVert_1,\qquad
\ell(g_y)\le a^{(1)}_j+\lambda_1\lVert x^{(1)}_j[S]-y[S]\rVert_1 .
$$

它是 Eq. (3) 的 Lagrange 对偶：$a$ 是"每个源点质量守恒"约束的乘子，$\lambda_k$ 是"运输预算"约束的乘子。

**约束怎么读。** 设想对手把类 0 的源点 $x^{(0)}_i$ 搬到 $y$。

- 收益：检测器在 $y$ 上给类 0 的损失 $\ell(-g_y)$；
- 成本：距离 $\lVert x^{(0)}_i[S]-y[S]\rVert_1$ 乘以价格 $\lambda_0$。
- 约束说：无论搬到哪个 $y$，"收益 − 成本"都不超过 $a^{(0)}_i$。所以 $a^{(0)}_i$ 是源点 $i$ 的最坏净损失。

**目标怎么读。** 稳健风险 = 所有源点的平均最坏净损失 + 预算 × 价格。

- 价格 $\lambda_k$ 高，对手搬得越远越亏，所以 $a$ 可以小；但要付 $\rho_k\lambda_k$。
- 价格低则反之。LP 在两者之间找平衡。

**目标点取 $y=x^{(0)}_i$ 自己时**（"对角约束"，距离为 0），约束给出 $a^{(0)}_i\ge\ell(-g_{x_i})\ge0$：
不搬的时候，损失就是检测器在原地的损失。

**hinge 下的具体形式（代码 `oracle.fit_fixed`）。** 分数可以限制在 $[-1,1]$（hinge 在 $|t|>1$ 外的部分只会更差），于是 $\ell(\sigma g)=1+\sigma g$。
这里 $\sigma=-1$ 对应类 0，$\sigma=+1$ 对应类 1。每条约束变成

$$
-a_i-\lambda_k\,\mathrm{dist}_{ir}+\sigma_k g_r\le-1 .
$$

这正是代码注释 `row (i, r): -a_i - lambda_k d_ir + sigma_k g_r <= -1`。
代码把源点 $i$ 和目标点 $r$ 都取遍合并样本 $1..N$，所以有 $N^2$ 行；目标里 $1/n_k$ 换成了权重 $q_i$。

### 3.5 Eq. (5)：把分数延拓到整个空间

LP 只给了样本点上的分数 $g_y$。新的点 $\xi=x[S]$ 怎么打分？Eq. (5) 的区间就是 Xie 等 (2021) 定理 1(ii) 的区间（hinge 下令 $\pi=(1+g)/2$ 逐项相同，记号对照见方向笔记第 1.3 节）；两点插值是在这个区间里取的一个具体的点。

1. 对每一类，找使"包络" $a^{(k)}_r+\lambda_k\lVert x^{(k)}_r[S]-\xi\rVert_1$ 最小的源点（并列时取最小下标），记为锚点 $i$（类 0）和 $j$（类 1）。
2. 令 $r_0=\lVert x^{(0)}_i[S]-\xi\rVert_1$（到类 0 锚点的距离），$b=\lVert x^{(0)}_i[S]-x^{(1)}_j[S]\rVert_1$（两个锚点之间的距离）。
3. $\tau=\min\{1,r_0/b\}$（$b=0$ 时 $\tau=0$），$g(\xi)=(1-\tau)g_{x^{(0)}_i}+\tau g_{x^{(1)}_j}$。

几何直觉：在两个锚点之间按距离做线性插值。离类 0 锚点越近（$r_0$ 小），越接近它的分数。

为什么这样就是最优：

- $\ell$ 是凸的，插值后的损失不超过两端损失的插值；
- 再用三角不等式，损失被包络 $E_k(\xi)=\min_r\{a^{(k)}_r+\lambda_k\lVert x_r-\xi\rVert_1\}$ 控制住；
- 包络对 $\xi$ 是 $\lambda_k$-Lipschitz 的，对任何搬运都成立，所以 $R(f)\le$ Eq. (4) 的目标 $=V(S)$。

代码：`FitResult.predict_score`，一一对应：`envelope`、`anchor`、`r0`、`b`、`tau`。
最后 `np.clip(result, -1, 1)`；查询点恰好是样本投影时直接返回拟合分数。

### 3.6 $H(\gamma)$、$M(\gamma)$、$c_k(\gamma)[m]$

为了把"选坐标"变成关于 $z$ 的问题，第 2 节后半段定义了三个量。这里允许运输计划 $\gamma$ 的总质量不是 1。

- $H(\gamma)=\sum_y\psi(p_0(y),p_1(y))$，$\psi(a,b)=\inf_t\{a\ell(-t)+b\ell(t)\}$：给定搬运结果，最好的检测器能达到的最小加权损失。
  hinge 下 $H(\gamma)=2\sum_y\min(p_0(y),p_1(y))$ = **2 × 重叠质量**。
  它是凹的、一次齐次的：$H(s\gamma)=sH(\gamma)$。
- $M(\gamma)=\frac12\sum_{k,i,y}\gamma^{(k)}_{i,y}$：两类质量的平均。正常的计划 $M=1$。
- $c_k(\gamma)[m]=\sum_{i,y}\gamma^{(k)}_{i,y}\lvert x^{(k)}_i[m]-y[m]\rvert$：类 $k$ 的计划在坐标 $m$ 上花的运输代价。
  因为 $\ell_1$ 距离按坐标相加，总代价 $=\sum_{m\in S}c_k(\gamma)[m]$。**这是能把选择写成线性的根本原因。**

### 3.7 命题 2.2，Eq. (6)–(7)：凸选择函数 $F(z)$

$$
\frac1{F(z)}=\min\bigl\{M(\gamma)+\mu^\top z:\ (\gamma,\mu)\in\mathcal D,\ H(\gamma)\ge1\bigr\}
$$

$\mathcal D$ 有两条约束：

- 行和：$\sum_y\gamma^{(k)}_{i,y}=M(\gamma)/n_k$（两类的总质量都等于 $M(\gamma)$）；
- 坐标预算：$\sum_m\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+\le\rho_kM(\gamma)$。

按下面四步拆开。

**第一步：它其实是一个比值。** $H$、$M$ 和 $\mathcal D$ 都随 $(\gamma,\mu)\mapsto(s\gamma,s\mu)$ 一起缩放，所以

$$
F(z)=\max_{(\gamma,\mu)\in\mathcal D}\ \frac{H(\gamma)}{M(\gamma)+\mu^\top z}
=\max_{(\gamma,\mu)\in\mathcal D,\ M(\gamma)=1}\ \frac{H(\gamma)}{1+\mu^\top z}.
$$

分子是对手制造的重叠，分母是"质量加豁免费"。Eq. (6) 只是把这个比值取了倒数：倒数是仿射函数的最小值，$z$ 线性出现。

**第二步：$\mu_m$ 是"豁免价格"。** $M=1$ 时坐标预算约束说的是：

- 坐标 $m$ 上的运输代价 $c_k[m]$，有 $\rho_k\mu_m$ 这么多可以被"豁免"，超出部分才计入预算 $\rho_k$；
- 对手为豁免付出代价：分母 $1+\mu^\top z$ 变大。但只为**被选中的**坐标付（$z_m=1$）。

**第三步：在二元点 $z=\mathbf 1_S$ 上看。**

- 没选的坐标（$z_m=0$）豁免是免费的。对手把 $\mu_m$ 取得够大，这些坐标上随便搬。这正对应"检测器看不到的坐标不计费"。
- 选中的坐标豁免要付钱，最优是 $\mu_m=0$。于是约束变成 $\sum_{m\in S}c_k[m]\le\rho_k$：只按 $S$ 上的代价计费。
- 目标变成 $\max H$。所以 $F(\mathbf 1_S)=V(S)$，这就是 Eq. (7)，证明在附录 A.1。

二元点上的值本来就由引理 2.1 给出，所以附录 A.1 "什么都没变"是对的：它检查这个函数在二元点上确实是 $V$。收获在二元点之间。

**第四步：为什么 $F$ 凸。** 可行集不依赖 $z$，每个可行的 $(\gamma,\mu)$ 给出 $z$ 的一个仿射函数 $M+\mu^\top z$。
$1/F$ 是一族仿射函数的最小值，所以是凹的；$t\mapsto1/t$ 在 $(0,\infty)$ 上凸且递减，所以 $F$ 凸。
这也给出了**割**：任何可行 $(\gamma,\mu)$ 都给出 $1/F$ 的一个仿射上界。

**旧版的平方根。** 旧版把同一个比值写成 $F(z)=\max\{2\sqrt{H}-M-\mu^\top z\}$。
原因是 $\max_{s\ge0}\{2\sqrt{sH}-sb\}=H/b$（$s=H/b^2$ 时取到，算术–几何平均不等式）：对 $(\gamma,\mu)$ 整体缩放，比值就变成"凹函数减去 $z$ 的线性项"。
新版直接取倒数，信息相同，不需要平方根（方向笔记第 1.4 节）。

**一个重要后果。** 在分数点 $z$（比如 $z_m=d/D$）上，对手对每个坐标都要付一点钱，又不能完全免费。
这相当于检测器"每个坐标都看一点"。第 4 节会说明，这让 $F$ 在分数点上很小，所以松弛弱。

### 3.8 Eq. (8)–(9) 和算法 1

**割，Eq. (8)：** 设 $(\gamma^t,\mu^t)$ 在 $z^t$ 处取到 Eq. (6) 的最小值，

$$
c_t(z)=M(\gamma^t)+(\mu^t)^\top z\ \ge\ \frac1{F(z)},\qquad\text{在 }z^t\text{ 处取等}.
$$

可行集不依赖 $z$，所以这个不等式对所有 $z$ 成立。在二元点 $z^t=\mathbf 1_S$ 上不需要再解任何问题：
取 $S$ 的最不利运输 $\gamma^\star$，令 $\pi_m=\max_kc_k(\gamma^\star)[m]/\rho_k$（$m\notin S$），$\pi_m=0$（$m\in S$），则 $\gamma^t=\gamma^\star/V(S)$、$\mu^t=\pi/V(S)$ 是最优的。代回去就是**比值割**：

$$
V(T)\ \ge\ \frac{V(S)}{1+\sum_{m\in T\setminus S}\pi_m}\qquad\text{（旧版切线割是 }V(S)\bigl(1-\textstyle\sum_{m\in T\setminus S}\pi_m\bigr)\text{）}.
$$

- 在 $S$ 自己处取等，在别的子集处是下界。
- 因为 $t\mapsto1/t$ 凸，比值割处处不低于旧版切线。
- 没选的坐标上 $\pi_m$ 往往大于 1（对手在看不见的坐标上搬得很远），所以对"换坐标"之后的子集给出的下界很弱。

**主问题，Eq. (9)：** $\max_{z\in\mathcal Z_d,\theta}\ \theta$，约束 $\theta\le c_s(z)$（$s=1,\dots,t$）。

- 它在所有二元选择上最大化"已有割的最小值"，得到下一个候选 $z^{t+1}$；$L_t=1/\theta_t$ 是全局下界。

**算法 1 逐行：**

| 行 | 做什么 | 代码 |
|---|---|---|
| 1 | $U=+\infty$，$t=1$ | `_Search.__init__`（初始为常数检测器，风险 2） |
| 3 | 当前选集 $S^t$ | `support` |
| 4 | 解 Eq. (4)：检测器、$V(S^t)$ 和最不利运输 | `oracle.fit_fixed` |
| 5 | 由最不利运输得到割 $c_t$ | `oracle.ratio_cut` |
| 6–8 | 更新最好的上界和检测器 | `_Search.evaluate` |
| 9 | 解主问题 Eq. (9)，得 $z^{t+1}$ 和 $L_t=1/\theta_t$ | `selection.master` |
| 10 | $U-L_t\le\varepsilon$ 则停 | `run.fit.upper - run.lower <= absolute_tolerance` |
| 11–12 | 用 Eq. (5) 延拓，返回 | `FitResult.predict_score` |

**为什么一定会停**：如果主问题又给出一个已评估过的 $z^s$，那么 $\theta_t\le c_s(z^s)=1/F(z^s)\le1/U_t$，即 $L_t\ge U_t$，间隙已关。
所以每次迭代要么停，要么评估一个新子集；最多 $\lvert\mathcal Z_d\rvert$ 次。

### 3.9 附录 A 的两步（只讲直觉）

正式笔记附录 A 直接从引理 2.1 证明命题 2.2，不用对偶。

1. **A.1：二元点上相等。** 两个方向各构造一个可行解。
   - $1/F(\mathbf 1_S)\le1/V(S)$：取 $S$ 的最不利运输 $\gamma^\star$，配上 $\mu=\pi$（$S$ 外是 $\max_kc_k/\rho_k$，$S$ 内是 0）。$S$ 外的项被 $\mu$ 抵消，剩下的正是式 (3) 的预算，所以 $(\gamma^\star,\mu)\in\mathcal D$；再除以 $V(S)$ 让 $H=1$。
   - $1/F(\mathbf 1_S)\ge1/V(S)$：任取可行 $(\gamma,\mu)$，令 $s=M(\gamma)+\mu^\top\mathbf 1_S$。把 $\gamma$ 除以 $s$，不足的质量留在原地（单位运输 $\iota$，不花钱）。坐标预算约束保证新的运输在 $S$ 上不超预算，所以 $V(S)\ge H(\gamma)/s\ge1/s$。
2. **A.2：最小值取得到，$F$ 凸。** 最小化序列有界（$\mu_m$ 可以降到 $\max_kc_k[m]/\rho_k$ 而不变差），极限可行；$\mathcal D\cap\{H\ge1\}$ 是凸集且不依赖 $z$，所以 $1/F$ 凹、$F$ 凸。

代码里没有单独的"Eq. (6) 求解器"：`fit_fixed` 解 Eq. (4)，它的乘子就是 $\gamma^\star$，`ratio_cut` 按 A.1 的第一个构造写出割。

---

## 4. 算法直觉：OA 在做什么，为什么在这里慢

**OA 的逻辑。** $1/F$ 是凹的，每评估一个子集就得到 $1/F$ 的一个仿射上界（割）。所有割的最小值是 $1/F$ 的上近似。

- 主问题在二元点上最大化这个上近似：它指向"按目前的信息，最可能更好的子集"，并给出下界 $L_t$。
- 如果割在别的子集上也比较紧，几条割就能把下界抬到最优值附近，OA 很快就停。

**为什么在这里慢。**

1. **每条割基本只认得自己那个子集。** 从 $S$ 得到的割要排除 $T$，需要 $V(S)/(1+\pi(T\setminus S))\ge U$。
   没选的坐标上 $\pi_m$ 常在 1.4–2.7（第 5.10 节的例子），好的 $S$ 又有 $V(S)\approx U$，所以几乎排除不了任何子集。
2. **根本原因：割把对手的计划原样缩小后搬到 $T$ 上用，而真正的对手看到 $T$ 之后会重新做计划。**
   哪怕允许对手事先挑一个对所有子集都不超预算的最好计划（方向笔记的"承诺界"），得到的下界也只有最优值的 43%–70%。
3. **混合坐标帮防守方。** 同一个承诺界也等于"防守方每个坐标看一部分"（加权 $\ell_1$ 度量）时的最好稳健值。
   加权把跨类点对的距离向均值集中，对手找不到在稀疏投影里总能找到的那几个便宜点对。连续松弛的间隙就是这个好处的大小。

**数据。** 方向笔记表 3：小例子上旧版切线割和比值割都要评估**全部**子集。
20 维检查（实验 3，旧版切线割，见第 7.4 节）：$d=2$ 时评估了全部 190 个子集；$d=3$ 时 600 秒内评估了 920–1090 个（共 1140 个），下界只有 0.17–0.28，最优值是 0.88–1.02。

**这不是实现问题，而是问题本身难。** 方向笔记命题 6 证明了对**每个**损失，$\min_SV(S)$ 都是 NP 难的（等价于 $k$-feature-set）；
异或例子说明它也没有边际递减，贪心不可靠。所以精确算法最坏情况下必然是搜索。方向笔记第 3 节给出的路线：

- 把选择问题精确改写成"稀疏的坐标价格"（方向笔记命题 4）：$G(w)$ 像损失、$\rho\lVert w\rVert_\infty$ 像罚项、$\lvert\mathrm{supp}\,w\rvert\le d$ 是稀疏约束；
- 罚项的凸包络给出承诺界的闭式（命题 5）；
- 用承诺界做分支定界：总是精确；$d=4$ 时只解了 3% 的子集，$D=50$ 时比 HiGHS 解紧凑程序快近 3 倍，但 $d=3$、$D=100$ 时线性规划个数和子集数相当。按统计上有意义的容差停止能再省一半以上。

---

## 5. 核心代码导读

### 5.1 目录和阅读顺序

```
src/robusthw/algorithm/
  model.py      数据结构：Problem, SolveOptions, Cut, FitResult（含 predict_score）
  oracle.py     fit_fixed：一个 LP 给出检测器、上下界和比值割（算法 1 第 4–5 行）
                feasible_plan、ratio_cut、hinge_value；highs 是唯一的求解器包装
  selection.py  select()：global（外逼近）/enumeration/forward/forward_swap；master = Eq. (9)
```

**建议顺序：** `model.Problem` → `oracle.fit_fixed` → `model.FitResult.predict_score` → `oracle.feasible_plan` → `oracle.ratio_cut` → `selection.master` → `selection._outer_approximation`。

[docs/algorithm.md](algorithm.md) 有一张"公式 ↔ 函数"对照表，可以和这里对着看。

### 5.2 数据结构：`Problem`

```python
Problem(x0, x1, radii)          # radii = [rho_0, rho_1]；可选权重 q0、q1
problem.x      # 合并样本 (N, D)：先 n0 行类 0，再 n1 行类 1
problem.y      # 每行的类别 0/1
problem.q      # 每行的质量 1/n_k；两类各 1，sum(q) = 2
problem.delta  # delta[i, r, m] = |x_i[m] - x_r[m]|，(N, N, D)
```

要点：

- **合并样本同时是源点和目标点。** 第 $r$ 行既是一个源点（属于类 `y[r]`），又是一个目标点 $x_r\in\mathcal X$。所以 LP 里有 $N\times N$ 的块。
- **`delta` 是 $O(N^2D)$ 的存储。** 它让任意选集的距离都是一次求和：`problem.distance(S)`；坐标代价 $c_k(\gamma)[m]=\sum\gamma_{ir}\Delta_{ir}[m]$。$N$ 大时内存是瓶颈。
- 坐标编号从 0 开始（笔记里从 1 开始）。

### 5.3 `oracle.fit_fixed`：Eq. (4)

变量分三块：`[a (N), lambda (2), g (N)]`。

- 目标：`np.r_[problem.q, problem.radii, np.zeros(n)]`，即 $\sum_iq_ia_i+\sum_k\rho_k\lambda_k$。
- 约束：每对 $(i,r)$ 一行，$-a_i-\lambda_{y_i}\,\mathrm{dist}_{ir}+\sigma_{y_i}g_r\le-1$，$\sigma=-1$（类 0）或 $+1$（类 1）。
  稀疏矩阵的三组列分别是 $a_i$、$\lambda_{y_i}$、$g_r$ 的系数。
- 变量范围：$a,\lambda\ge0$，$g\in[-1,1]$。

### 5.4 上界：为什么重新算 $a$

求解器的解只在容差（约 $10^{-9}$）内可行。代码拿求解器给出的 $(g,\lambda)$，重新取最小可行的
$a_i=\max\bigl(0,\max_r[\ell(\sigma g_r)-\lambda\,\mathrm{dist}_{ir}]\bigr)$，于是每条约束都精确成立，`fit.upper` 是 $V(S)$ 的上界。
投影重合的目标点先统一成同一个分数，Eq. (5) 才有定义。任何人拿 `(scores, source_bounds, prices)` 都能检查这个上界。

### 5.5 乘子就是最不利运输：`feasible_plan`

Eq. (4) 第 $(i,r)$ 行的乘子就是 Eq. (3) 的运输量 $\gamma_{ir}$（LP 对偶）。求解器的乘子也只是近似可行，`feasible_plan` 用两步把它修成严格可行：

1. 行质量不足 $q_i$ 的部分留在原地（单位运输 $\iota$，代价为 0），超出的部分按比例缩小；
2. 某类的代价超出 $\rho_k$，就和 $\iota$ 混合到刚好用完预算。

两步都只用到附录 A.1 里的"留在原地"。于是 `fit.lower = H(plan) <= V(S)`，`hinge_value` 算的就是 $H=2\sum_y\min(p_0(y),p_1(y))$。
**一次 LP 同时给出检测器、上界、下界和最不利运输。**

### 5.6 `ratio_cut`：从最不利运输到割

```python
cost = [c_0(plan), c_1(plan)]                    # 每类在每个坐标上的运输代价
mu = max_k cost[k] / rho_k；mu[S] = 0            # 附录 A.1 的 pi
Cut(intercept=1 / H, slope=mu / H)              # 1/F(z) <= (1 + mu @ z) / H
```

`Cut.evaluate(z)` 返回 $1/(\text{intercept}+\text{slope}\cdot z)$，即 $F(z)$ 的下界。
**割不需要是紧的才有效**：任何可行的运输都给出有效割，只是可能不紧。所以求解器超时时，外逼近仍然可以用这些割。

### 5.7 `fit_fixed` 返回什么

```python
fit = fit_fixed(problem, support, options, with_cut=True)
fit.upper    # Eq. (4) 的值：V(S) 的上界，也是检测器的稳健风险证书
fit.lower    # H(plan)：V(S) 的下界
fit.status   # optimal / time_limit / accuracy_limit / solver_error
fit.cut      # 比值割（with_cut=False 时为 None）
fit.predict_score(x)  # Eq. (5) 的检测器分数，<0 判类 1
```

- 若 LP 失败或时间为 0，退回常数检测器 $g=0$，上界为 2。
- `status == "optimal"` 的意思是内层间隙 `upper - lower <= inner_tolerance`（默认 $10^{-4}$）。

### 5.8 `selection.master` 和 `_outer_approximation`：算法 1

- `master`：变量 `[z (D), theta]`，约束 $\sum z=d$ 和 $\theta\le\text{intercept}_s+\text{slope}_s\cdot z$，`integrality` 让 $z$ 为二元，最大化 $\theta$。
  - $\theta$ 取 HiGHS 的全局上界 `mip_dual_bound`，**不是**主问题当前解的值，所以 $L=1/\theta$ 在超时时也有效。
- `highs` 包装处理 HiGHS 的两个已知问题：进程里已有 HiGHS 调用用不同线程数初始化了调度器时返回 status 4 "Not Set"（去掉线程数重试）；小主问题上预处理失败（关掉预处理重试）。
- `_outer_approximation`：主循环。每轮解主问题 → 取二元 $z$ → 若已评估过则以 `accuracy_limit` 结束（精确求解时这不会发生）→ 否则评估、加割。
- `select(problem, d, method="global")` 是入口；`"enumeration"` 逐个评估所有子集。

### 5.9 `predict_score`：Eq. (5)

见第 3.5 节。代码按 2048 个查询一批，避免 $N\times\text{查询数}$ 的距离矩阵太大。

### 5.10 一个最小例子

```python
import numpy as np
from robusthw.algorithm import Problem, SolveOptions, fit_fixed, select

rng = np.random.default_rng(0)
x0 = rng.normal(size=(20, 5))
x1 = rng.normal(size=(20, 5)); x1[:, :2] += 1.0        # 信号在前两个坐标
problem = Problem(x0, x1, [0.3, 0.3])                  # rho_0 + rho_1 = 0.6

fit = fit_fixed(problem, (0, 1))
print(fit.upper, fit.lower, fit.status)                # 上下界几乎相等, optimal
print(fit.predict_score(np.zeros((1, 5))))             # 原点的分数

result = select(problem, 2, method="global", options=SolveOptions(time_limit=120))
print(result.support, result.upper, result.lower, result.status, result.fit_count)
for t, cut in enumerate(result.cuts, 1):
    print(t, round(1 / cut.intercept, 3), np.round(cut.slope / cut.intercept, 2))   # V(S) 和 pi
```

运行：`PYTHONPATH=src python3 your_script.py`。实际输出（节选）：

```
1.5349242612025313 1.5349242612025293 optimal
[0.73793998]
(0, 4) 1.4656143489174238 1.4656143489174234 optimal 10
1 1.535 [0.   0.   2.02 1.87 2.15]
2 1.62 [2.1  2.18 0.   1.44 0.  ]
...
8 1.466 [0.   1.05 1.36 1.37 0.  ]
```

从这个输出可以读出四件事：

1. **内层是精确的。** `fit.upper` 和 `fit.lower` 相差约 $10^{-15}$：一个 LP 的乘子就给出了紧的下界。
   `fit.upper` 也等于实验代码的 `2 * Matching(cost).value(0.6)`，即方向笔记命题 2；`tests/test_study20.py::test_matching_value_equals_core_fit` 检查的就是这个。
2. **$\pi$ 的样子。** 每条割的 $\pi$ 在已选的两个坐标上为 0，在没选的坐标上是 1.0–2.7。
   这就是第 3.7 节说的"豁免价格"：没选的坐标对手可以免费用。
   它也是第 4 节说的"每条割只认得自己"：换一个坐标，下界就除以 2 到 3.7，远低于真实的 $V$。
3. **OA 评估了全部 $\binom52=10$ 个子集**（`fit_count = 10`）才证明最优。
4. **选出来的是 $(0,4)$，不是真正的信号对 $(0,1)$。**
   - 枚举可以验证：半径 0.6 时 $V(\{0,4\})=1.466<V(\{0,1\})=1.535$；
   - 把半径加到 1.0，两者几乎相等（1.772 对 1.773）；
   - 加到 1.3（约等于实验 1 标定的 $\rho_h(2,25)=1.28$），$(0,1)$ 以 1.900 对 1.921 胜出。
   这是第 7.3 节"半径太小会被噪声坐标骗"的一个缩影：$n=20$ 时，0.6 的半径对 $d=2$ 来说不够诚实。

### 5.11 读代码的几个陷阱

- **目标点是合并样本**，不是"另一类的样本"。$N^2$ 个约束里包括同类之间的搬运。
- **$q_i=1/n_k$**，不是 $1/N$；`sum(q) = 2`，因为风险是两类之和。
- **分数在 $[-1,1]$。** hinge 下这不损失一般性，并且让约束线性。
- **半径是分开的** `radii = [rho_0, rho_1]`。hinge 下 $V$ 只依赖 $\rho_0+\rho_1$（方向笔记命题 2），实验代码统一用 `[rho/2, rho/2]`。
- **状态的含义**：
  - `optimal`：间隙在容差内；
  - `time_limit`：时间到了，上下界仍然有效；
  - `accuracy_limit`：间隙没关上（内层数值精度，或主问题重复给出已评估的点）；
  - `solver_error`：求解器失败；
  - `constant_fallback`：搜索还没评估任何子集时的初始检测器（常数 0，风险 2）。

---

## 6. 研究方向笔记的几个公式和实验代码

实验代码 `src/robusthw/experiments/study20.py` 不调用核心的选择算法来选子集，而是用方向笔记里更快的等价形式。

**匹配形式（方向笔记式 (2)、命题 2）。** 记 $C^m_{ij}=\lvert x^{(0)}_i[m]-x^{(1)}_j[m]\rvert$，$c_{ij}(S)=\sum_{m\in S}C^m_{ij}$：

$$
m_S(\rho)=\max_{\pi\ge0}\Bigl\{\sum_{i,j}\pi_{ij}:\ \sum_j\pi_{ij}\le\tfrac1{n_0},\ \sum_i\pi_{ij}\le\tfrac1{n_1},\ \sum_{i,j}\pi_{ij}c_{ij}(S)\le\rho\Bigr\},\qquad V(S)=2m_S(\rho_0+\rho_1).
$$

逐项读：$\pi_{ij}$ 是撮合 0 类点 $i$ 和 1 类点 $j$ 的质量；每个点最多被撮合掉它自己的质量；撮合一单位要花它们在 $S$ 上的距离；总花费不超过 $\rho$。

| 函数 | 做什么 |
|---|---|
| `pair_costs(x0, x1)` | $C[i,j,m]$，形状 $(n_0,n_1,D)$ |
| `Matching(cost)` | 等样本量时的精确 $m_S$：`cost(k)` 是 $k$ 对不相交配对的最小代价 $/n$（补零后的匈牙利算法），关于 $k$ 凸 |
| `Matching.value(rho)` | 二分找最大的 $k$ 使 `cost(k) <= rho`，再线性插值，得 $m_S(\rho)$ |
| `Matching.budget(mass)` | 反过来：撮合到给定质量最少要多少预算（"所需半径"） |
| `matching_lp(cost, rho)` | 同一个量的 LP 版本，并返回对偶 $(u,v,\lambda)$ |
| `witness(x1, S, v, lam)` | 方向笔记命题 3 的检测器 $2h$：$h(\xi)=\mathrm{clip}(\min_j\{h'_j+\lambda\lVert x^{(1)}_j[S]-\xi\rVert_1\})$ |
| `compact_program(costs, rho, d)` | 方向笔记式 (4)：一个 MILP 直接求 $\min_SV(S)$；`relax=True` 给松弛（即承诺界，方向笔记命题 5） |

**对偶怎么读（方向笔记第 3.2 节）**：

$$
m_S(\rho)=\min_{u,v,\lambda\ge0}\{\rho\lambda+\bar u+\bar v:\ u_i+v_j+\lambda c_{ij}(S)\ge1\}.
$$

每个跨类点对要么被它的两个端点"付钱"（$u_i+v_j$），要么被价格乘距离 $\lambda c_{ij}(S)$ 隔开。
选 $S$ 就是决定对手要为哪些坐标付钱。令 $w_m=\lambda z_m$，就得到紧凑程序（方向笔记式 (4)），它是一般损失下"稀疏坐标价格"改写（方向笔记命题 4）在 hinge 下的样子。

`tests/test_study20.py` 核对了五件事：

- 匹配值 = LP 值；
- `budget` 是 `value` 的反函数；
- 方向笔记式 (4) = 枚举，且松弛 $\le$ 最优；
- 匹配值 = 核心 `fit_fixed` 的上界；
- 见证检测器的经验风险 $\le$ 切线截距。

---

## 7. 怎么看 20 维结果

### 7.1 文件在哪里

```
configs/study20.json                 所有参数（事先声明）
results/study20-v1/
  development.json                   开发阶段：标定的 rho_h 和选出的信号强度
  null.json signal.json              确认阶段：实验 1、2 的逐种子记录
  computation.json randomized.json   实验 3、4
  summary.md                         自动生成的表（英文）
  figures/null_radius.png            实验 1 的图
  figures/signal_levels.png          实验 2 的图
  provenance.md                      这次运行的代码版本说明
```

**两阶段设计**：先在开发种子（100–139）上确定参数，冻结；再在全新的确认种子（200–239）上跑。
确认阶段的数字才用来下结论，这样避免"调参调到结果好看"。

**数据**：$D=20$，类 0 是 $N(0,I_{20})$，类 1 只在前两个坐标上平移 1.0，每类 $n\in\{25,50,100\}$。

### 7.2 实验 1：两类同分布（检查虚假可分）

**问的问题**：两类完全一样时，正确答案是 $V=2$。半径要多大，稳健检验才不会"证明"它们可分？

**"所需半径"**：对子集 $S$，使 $V(S)\ge1.92$（对手能撮合至少 96% 质量）的最小 $\rho_0+\rho_1$。
用 `Matching.budget(0.96)` 算。半径超过它，$S$ 就不会被误判为可分。

**$\rho_h(d,n)$**：在 40 个开发种子上，取"所有 $\binom{20}d$ 个子集中最大的所需半径"的 95% 分位数。
意思是：用 $\rho_h$ 时，大约 95% 的数据集上**没有任何一个**大小为 $d$ 的子集被误判可分。

表的每一列：

| 列 | 含义 | 怎么看 |
|---|---|---|
| rho_h (dev) | 开发阶段标定的半径 | 随 $d$ 增大、随 $n$ 减小：维数诅咒 |
| max_S needed | 确认种子上最坏子集的所需半径（均值） | 应略低于 $\rho_h$ |
| median_S needed | 中位子集的所需半径 | 和最坏的差距 = "挑子集"带来的乐观 |
| Prop. 8 bound | 方向笔记命题 10 的理论下界（结果文件沿用旧编号命题 8） | 所需半径应该总是 $\ge$ 它 |
| subsets below bound | 有多少子集低于下界 | 必须是 0，否则命题 10 错了 |
| false sep. at rho_h | 确认种子中，最坏子集所需半径 $>\rho_h$ 的比例 | 应约等于 5% |
| false sep. at 0.6 | 用固定半径 0.6 时的同一比例 | 越大说明固定半径越不诚实 |
| best V at 0.6 | 固定半径 0.6 时最小的 $V(S)$（均值） | 正确值是 2；越小越"虚假可分" |

**读出来的结论**：

1. **维数效应非常清楚。** $n=100$ 时 $\rho_h=0.27,0.60,1.08,15.1$（$d=1,2,3,20$）。
   - 用全部 20 个坐标需要半径 15 才诚实，这时什么信号都检测不到。
   - 选 $d$ 个坐标把所需半径降到 $O(1)$。这是"选择让稳健检验诚实"的直接证据。
2. **固定半径会虚假可分。** 半径 0.6 下，$d=3$ 和 $d=20$ 在**每个**确认数据集上都"证明"两个相同的分布可分。
   $d=20$ 时最小的 $V$ 只有 0.10，相当于声称检验几乎不会犯错。
3. **$\rho_h$ 的确认结果。** 虚假可分率 3%–12%，目标 5%。40 次抽样时比例的标准误约 $\sqrt{0.05\times0.95/40}\approx3.4\%$。
   12% 比 5% 高约 2 个标准误，说明 40 个开发种子估 95% 分位数本身有误差。正式实验应该用更多种子标定。
4. **斜率。** 所需半径对 $n$ 的双对数斜率是 $-0.67,-0.53,-0.40,-0.07$。
   - 随 $d$ 变平，和 $n^{-1/d}$ 的方向一致（理论 $-1/2,-1/2,-1/3,-1/20$）。
   - $d=1$ 比 $-1/2$ 陡，可能有两个原因：只需撮合 96% 质量，可以丢掉最难撮合的尾部点；$n$ 只到 100，还没到渐近区。
5. **命题 10 的下界成立但很松**：比最坏子集的所需半径小约 5 倍（$d=20$）到 230–370 倍（$d=1$）。这是常数没优化的问题，不影响阶。

**看图** `figures/null_radius.png`：

- 横轴 $n$、纵轴所需半径，都是对数坐标；实线是最坏子集，虚线是中位子集，灰色点线是固定半径 0.6。
- 灰线以下的实线 = 固定半径会出现虚假可分。$d=3$ 和 $d=20$ 的实线全在灰线之上。

### 7.3 实验 2：两个坐标上有信号（有效性、恢复、选 $d$）

**问的问题**：

- 用不同半径，选出来的子集对不对（恢复）？
- 证书 $V(\widehat S)$ 是否真的盖住了检测器在新数据上的风险（有效性）？
- 用证书挑 $d$ 能不能挑对？

**半径的五个水平**：固定 0.6，以及 $0.25,0.5,1,2$ 倍的 $\rho_h(d,n)$。

**流程**：

1. 对每个 $d\in\{1,2,3\}$，枚举所有子集，用匹配形式算 $V(S)$，选最小的；
2. 用核心 `fit_fixed` 在选出的子集上拟合检测器；
3. 在每类 20000 个新样本上算真实的 hinge 风险和错误率。

**"有效"**：测试 hinge 风险 $\le$ 证书 + 2 倍 Monte Carlo 标准误。标准误的中位数约 0.005，很小。

**参照值**：只看信号坐标的 Bayes hinge 风险 $B=0.959$，Bayes 总错误（两类错误率之和）0.480。这是任何检测器的下限。

第一张表（$d=2$）的列：

| 列 | 含义 |
|---|---|
| radius used | 实际用的 $\rho_0+\rho_1$ |
| exact recovery | 选出的恰好是 $\{0,1\}$ 的比例 |
| certificate V(S_hat) | 证书：选出子集的稳健风险 |
| test hinge | 检测器在测试集上的真实 hinge 风险 |
| valid | 证书有效的比例 |
| test total error | 测试集上两类错误率之和（Bayes 是 0.480） |

**怎么读一行**：先看 valid（证书能不能信），再看 certificate 和 test hinge 的差（证书有多松），最后看 recovery 和 test total error（检测器好不好）。

**结论**：

1. **半径太小，证书无效。** $0.25\rho_h$ 时证书约 1.05，但真实 hinge 风险 1.30–1.59，有效率 0%。证书在"吹牛"。
2. **半径太小，选得也更差。** $0.25\rho_h$ 时恢复率 0.28/0.68/0.93，低于 $\rho_h$ 的 0.68/0.97/1.0；测试错误也更高（$n=100$：0.547 对 0.485）。
   原因：半径小时，噪声子集里偶然孤立的点让它们"看起来可分"。**稳健半径同时是正则化。**
3. **$\rho_h$ 时证书有效但偏松。** 有效率 97%–100%；证书比真实风险高 0.27–0.31。
   但 $n=100$ 时检测器的测试错误 0.485 已经接近 Bayes 的 0.480。**松的是证书，不是检测器。**
4. **$2\rho_h$ 过度保守。** 恢复率 100%，但 $n=25$ 时证书约 2，检测器几乎是常数（测试错误 0.975，等于乱猜）。
5. **固定 0.6** 在 $n=100$ 时几乎等于 $\rho_h(2,100)=0.602$，所以和 $\rho_h$ 那一行的数几乎相同；$n=25$ 时只有 33% 有效。

第二张表（按证书在 $d\in\{1,2,3\}$ 中选 $d$，方向笔记推论 13）：

- **半径不够大**（固定 0.6、$0.25\rho_h$、$0.5\rho_h$）时总是选 $d=3$，证书几乎全部无效。
  原因是惩罚不够：加坐标总让 $V$ 变小（坐标越多点越"远"），而半径没有相应增长到足以抵消。
- **$\rho_h$ 时**，$n\ge50$ 有 90%–95% 选对 $d=2$，证书 100% 有效；$n=25$ 时一半选 $d=1$（信号相对于 $d=2$ 的半径太弱）。
- **$2\rho_h$** 惩罚过度，62%–97% 选 $d=1$。
- 结论：半径既要**随 $d$ 增长**（形状对），**倍数**也要对（水平对）。

**看图** `figures/signal_levels.png`：三个面板，横轴都是"半径 / $\rho_h$"。

- 左：恢复率随半径上升；
- 中：有效率在 $0.5\rho_h$ 到 $\rho_h$ 之间从低升到 1；
- 右：证书（实线）随半径单调上升，真实风险（虚线）先降后升，灰线是 Bayes 风险。
  实线在虚线之上 = 有效。最佳点大约在两线刚交叉之后，也就是 $\rho_h$ 附近。

**最后一行** `Largest |core fit_fixed value - 2 m_S|` 是 $3.3\times10^{-11}$：核心求解器和匹配公式在所有拟合上一致。这是实现正确性的交叉检查。

### 7.4 实验 3：计算

| 列 | 含义 |
|---|---|
| enumeration | 用匹配形式枚举所有子集的最优值和时间 |
| program (2) | 紧凑程序（方向笔记式 (4)，结果文件沿用旧编号式 (2)）用 HiGHS 求解：值、时间、分支定界节点数 |
| relaxation of (2) | 紧凑程序的连续松弛值，即承诺界 |
| OA upper / lower | 核心算法 1 的上下界（当时的旧版切线割） |
| OA status | 结束状态、评估了多少子集、用时 |

**结论**：

- 紧凑程序每次都和枚举完全一致（正确）。但 $D=20$ 时枚举最快（0.1–1.2 秒），紧凑程序要 2–40 秒。
- 松弛（承诺界）只有最优值的 52%–59%，很弱。
- OA 在 $d=2$ 时评估了**全部** 190 个子集才证明最优；$d=3$ 时 600 秒超时，下界 0.17–0.28，远低于最优值。
- **实践建议**：$D\le20$ 左右直接枚举；更大时用方向笔记第 3.5 节的分支定界（方向笔记表 6：$D=50$ 时 85 秒，HiGHS 解紧凑程序要 235 秒）。

### 7.5 实验 4：随机化选择

两类数值，回答两个不同的问题：

- **左边（对抗意义）**：防守方随机挑子集，对手知道分布但不知道抽签结果。
  - fixed：固定最优值；
  - randomized $W_T$：随机化可达值（方向笔记定理 17）；
  - water-filling：$d=1$ 时的精确值（定理 18）；
  - relaxation：紧凑程序的松弛，即承诺界。
- **右边（面对抽样误差）**：直接从真分布抽测试数据，算固定最优检测器和最优混合的真实 hinge 风险。

**结论**：

- **对抗意义下随机化收益巨大**：$d=2$ 时 $W_T=0.446$，固定 1.394，连松弛 0.737 都比它差。
  $d=1$ 时 $W_T=1.200$ 略高于精确值 1.189（只用了有限条切线），符合定理 18。
- **面对抽样误差，混合更差**：测试 hinge 1.915/1.981，固定的是 1.475/1.527。
  原因：最优混合要把"价格暴露"摊到很多坐标上，它只把 11%–36%（$d=1$）和 18%–26%（$d=2$）的权重放在含信号坐标的子集上。
- 这正是方向笔记命题 19：随机化只对"先承诺、共用一笔预算的战略对手"有用，对抽样误差无用。

### 7.6 读结果的一般方法

1. **先问有效性，再问功效。** 证书无效时，它再小也没意义（实验 2 的 $0.25\rho_h$）。
2. **区分"证书"和"检测器"。** 证书是检测器风险的上界；证书松不代表检测器差（实验 2 的 $\rho_h$ 行）。
3. **看标准误。** 40 次抽样的比例，标准误约 3%–8%。0.07 和 0.12 的差别不显著。
4. **区分开发和确认。** 只有确认阶段的数字算数；参数在开发阶段冻结。
5. **对照理论值。** 斜率对 $-1/d$、Bayes 风险 0.959/0.480、命题 10 的下界、核心求解器和公式的一致性：每个都是对实现或理论的检查。
6. **这是开发检查，不是正式结论。** 一个分布、一种信号、$D=20$、40 个种子。正式实验要换分布、加大 $D$、做数据驱动标定。

### 7.7 这轮结果暴露的问题

1. **$\rho_h$ 用到了已知的零分布。** 实际中零分布未知。需要数据驱动的标定，例如：
   - 把两类合并后随机重分标签（置换）；
   - 或在每类内部随机对半分；
   然后算所需半径的分位数。这是下一步最重要的工作。
2. **40 个种子标定 95% 分位数不够稳**：确认集有一格虚假可分率到 12%。
3. **证书松约 0.3。** 来源是方向笔记定理 12 右端的 $V_{\mathrm{pop}}(S;2\rho)$。可能的改进：
   - 用估计的 $\mathcal W_1$ 误差扣回方向笔记命题 9 里的 $\lambda_k(\mathcal W_1-\rho_k)$；
   - 改用受限检测器类（线性/核），它的半径可以是 $\sqrt{d\log(eD/d)/n}$ 而不是 $n^{-1/d}$。
4. **小样本（$n=25$）时信号 1.0 太弱**：$\rho_h$ 下恢复率只有 68%，自适应 $d$ 有一半选成 1。
5. **OA 在这个问题上基本是枚举。** $D$ 大时要换成方向笔记第 3.5 节的分支定界（承诺界做节点的界）。
6. **方向笔记命题 10 的常数很松。**

---

## 8. 复现和修改实验

```bash
# 开发：标定 rho_h、选信号强度（约 1 分钟，32 进程）
PYTHONPATH=src python3 scripts/run_study20.py develop   --output results/study20-v2 --workers 32
# 确认：在新种子上跑四个实验（约 15 分钟，主要是 OA 的 600 秒时限）
PYTHONPATH=src python3 scripts/run_study20.py confirm   --output results/study20-v2 --workers 32
# 生成 summary.md 和图
PYTHONPATH=src python3 scripts/run_study20.py summarize --output results/study20-v2
```

- **新实验用新的输出目录**（如 `study20-v2`），不要覆盖 v1。
- 改参数改 `configs/study20.json`。主要字段：
  - `dimension`, `signal`, `budgets`, `train_sizes`；
  - `null_mass`：多少质量算"没被分开"，默认 0.96；
  - `honest_quantile`：默认 0.95；
  - `radius_factors`：半径的倍数水平；
  - `development_seeds`, `confirmation_seeds`；
  - `computation_*`、`randomized_*`：实验 3、4 的规模。
- **多进程注意**：设置 `OMP_NUM_THREADS=1`。HiGHS 有进程级的调度器，同一进程里先后用不同线程数调用会报 "Not Set"。核心代码已经处理了这种情况。
- 测试：`PYTHONPATH=src python3 -m pytest -q tests/test_study20.py`。
