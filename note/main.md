# Variable Selection for Wasserstein Robust Hypothesis Testing

18 October 2026

## Abstract

Variable selection for Wasserstein robust testing admits an exact finite formulation and a maximum-affine dual representation. Dual couplings generate global cuts.

## 1. Problem formulation

Throughout this paper, we consider the variable selection problem for robust hypothesis testing. Let $[D]=\{1,\ldots,D\}$ and $d$ be a positive integer denoting the sparsity budget.
Given a detector $f:~\mathbb{R}^D\to\mathbb{R}$, we follow the convention in [2] to define the worst-case risk as

$$
R(f)=\sup_{\mathbb{P}_0\in\mathcal{P}_0, \mathbb{P}_1\in\mathcal{P}_1}~
\left\{\mathbb E_{P_0}\ell(-f(X))+
       \mathbb E_{P_1}\ell(f(X))\right\},
\tag{1}
$$

where $\mathcal{P}_k, k=0,1$ denote the Wasserstein distance-based ambiguity sets, $\ell(\cdot)$ is the generating function that is nonnegative valued, non-decreasing, convex, and satisfies $\ell(0)=1, \lim_{t\to-\infty}\ell(t)=0$.
For $k=0,1$, assume we have $n$ available observations in the group $k$, which are denoted as $\{x_i^{(k)}\}_{i=1}^n$.
We denote the corresponding empirical distribution as $\widehat{\mathbb{P}}_k$, and construct the Wasserstein ambiguity set as

$$
\mathcal{P}_k=\Big\{
\mathbb{P}_k:~\mathcal{W}_1(
\widehat{\mathbb{P}}_k, \mathbb{P}_k
)\le \rho_k
\Big\}.
$$

Here, $\mathcal{W}_1(\cdot,\cdot)$ refers to the $1$-Wasserstein distance equipped with $\ell_1$-norm.

For $S\subseteq[D]$, define the collection of detectors whose input only depends on coordinates belonging to $S$:

$$
\mathcal{F}_S = \Big\{
f:~\mathbb{R}^D\to\mathbb{R}, f(x) = g(x[S])\text{ for some measurable function $g:~\mathbb{R}^{|S|}\to\mathbb{R}$}
\Big\}.
$$

Note that for a given vector $x\in\mathbb{R}^D$, we use $x[i]$ to denote its $i$-th coordinate, and $x[S]$ to denote the contamination of its coordinates whose index belongs to $S$.
Our proposed variable selection problem can be formulated as the following combinatorial optimization

$$
\min_{S\subseteq[D], |S|\le d}\min_{f\in\mathcal{F}_S}~R(f).
\tag{2}
$$

## 2. Finite reformulation and duality

### 2.1. The risk for a fixed set of coordinates

Fix a selection $S$. We first compute the best risk achievable using these
coordinates; the outer minimization will then compare different selections.
In the remainder, assume $\rho_0,\rho_1>0$ and that $\ell$ is finite on
$\mathbb R$. All probability distributions have finite first moments.
Pool the observations into a single list:

$$
N=2n,\qquad I_0=[n],\qquad I_1=\{n+1,\ldots,2n\},\qquad
x_i=x_i^{(0)},\quad x_{n+i}=x_i^{(1)}\quad(i\in[n]).
$$

Thus every source observation has probability mass $1/n$ within its class.
Write $\ell_0(s)=\ell(-s)$, $\ell_1(s)=\ell(s)$, and define

$$
\begin{gathered}
V(S)=\min_{f\in\mathcal F_S}R(f),\qquad
\mathrm{OPT}=\min_{|S|\le d}V(S),\\
y_i=x_i[S],\qquad \Delta_{ir,j}=|x_i[j]-x_r[j]|,\qquad
c_{ir}^S=\|y_i-y_r\|_1=\sum_{j\in S}\Delta_{ir,j}.
\end{gathered}
$$

Here $i,r\in[N]$ index observations and $j\in[D]$ indexes coordinates.

Restricting a detector to $S$ is equivalent to testing on the projected
observations $y_i$. This also preserves the ambiguity sets exactly. With
$\widehat Q_k^S=n^{-1}\sum_{i\in I_k}\delta_{y_i}$, where $\delta_y$ is a
unit mass at $y$, let

$$
\mathcal Q_k^S=\{Q:\mathcal W_1(Q,\widehat Q_k^S)\le\rho_k\}.
$$

The distributions of $X[S]$ as $\mathbb P_k$ ranges over $\mathcal P_k$
are precisely $\mathcal Q_k^S$. Projection cannot increase transport cost;
conversely, any transport in the selected coordinates can be lifted to the
full space by keeping the other coordinates of each source fixed.
Consequently,

$$
V(S)=\inf_{g\text{ measurable}}\sum_{k=0}^1
\sup_{Q\in\mathcal Q_k^S}\mathbb E_Q\ell_k(g(Y)).
$$

Gao et al. [2, Proposition 1 and Theorem 3] establish
minimax equality and an exact finite-support formulation for this convex-loss
testing problem. Applied to the projected observations, their result reduces
the optimized risk to transport between the pooled sample locations.
For class $k$, let $\pi_{ir}$ be the mass moved from $y_i$ to $y_r$, and set

$$
\mathcal T_k(S)=\left\{\pi\ge0:
\sum_{r=1}^N\pi_{ir}=\frac1n\ (i\in I_k),\quad
\sum_{i\in I_k,r\in[N]}c_{ir}^S\pi_{ir}\le\rho_k\right\}.
$$

If $u_r$ is the detector score assigned to target $y_r$, then

$$
V(S)=\inf_{u\in\mathbb R^N}\sum_{k=0}^1
\max_{\pi\in\mathcal T_k(S)}
\sum_{i\in I_k,r\in[N]}\pi_{ir}\ell_k(u_r).
\tag{3}
$$

The last sum is an expected loss: target $r$ receives class-$k$ mass
$p_{kr}=\sum_{i\in I_k}\pi_{ir}$, so its contribution is
$p_{kr}\ell_k(u_r)$. Exactness concerns the minimum over detectors; it does
not assert that a worst-case distribution for every fixed detector is
supported on the observations.

For fixed scores $u$, each maximization in (3) is a
transportation linear program. Its dual uses a variable $a_i\in\mathbb R$
for the source-mass equality and a price $\lambda_k\ge0$ for the transport
budget. Define

$$
J(a,\lambda)=\frac1n\sum_{i=1}^N a_i+\rho_0\lambda_0+\rho_1\lambda_1.
$$

Linear programming duality gives the finite convex problem

$$
\begin{aligned}
V(S)=\min_{a,u\in\mathbb R^N,\ \lambda\in\mathbb R_+^2}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S,
\quad i\in I_k,\ r\in[N],\ k=0,1.
\end{aligned}
\tag{4}
$$

Each constraint bounds the loss at a possible target by a source term plus
its transport cost. Thus (4) computes $\min_{f\in\mathcal F_S}R(f)$,
and minimizing its value over $|S|\le d$ solves the original problem.
Appendix A gives the projection argument, the precise
finite minimax and LP-duality steps, and an optimal detector defined beyond
the sample locations.

The later work of Xie et al. [3, Lemma 2 and Theorem 1]
develops finite-support and extension results for randomized tests whose
risk is the sum of the two error probabilities. Its transport objective is
$\sum_r\min\{p_{0r},p_{1r}\}$. For the general generating loss used here,
the corresponding objective is
$\sum_r\inf_s\{p_{0r}\ell(-s)+p_{1r}\ell(s)\}$, as in
Gao et al. [2, Theorems 2--3]. This is the convex-loss
result used to derive (4).

### 2.2. Representing the coordinate selection

Let $z_j=1$ when coordinate $j$ is selected and $z_j=0$ otherwise.
Following the logical-variable viewpoint of
Bertsimas et al. [1], an unselected coordinate will
force its associated continuous variables to zero. In our problem these
variables are the two coordinate prices $w_{0j},w_{1j}$, replacing the
products $\lambda_0z_j,\lambda_1z_j$ in the loss constraints.

For $z\in[0,1]^D$, define

$$
\begin{aligned}
F(z)=\min_{a,u,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\sum_{j=1}^D\Delta_{ir,j}w_{kj},
\quad i\in I_k,\ r\in[N],\ k=0,1,\\
&0\le w_{kj}\le\lambda_k,\quad k=0,1,\ j\in[D],\\
&\rho_0w_{0j}+\rho_1w_{1j}\le z_jJ(a,\lambda),\quad j\in[D],\\
&a,u\in\mathbb R^N,\qquad\lambda\in\mathbb R_+^2.
\end{aligned}
\tag{5}
$$

For fixed $z$ this is a convex program. The last family of inequalities
implements the selection: if $z_j=0$, positivity of the radii forces
$w_{0j}=w_{1j}=0$. If $z_j=1$, it adds no restriction, because the
constraints with $r=i$ give $a_i\ge0$ and hence

$$
\rho_0w_{0j}+\rho_1w_{1j}
\le\rho_0\lambda_0+\rho_1\lambda_1\le J(a,\lambda).
$$

At $z=\mathbf1_S$, every feasible solution of (4) lifts to
(5) by $w_{kj}=\lambda_kz_j$. Conversely, a feasible solution
of (5) satisfies

$$
\ell_k(u_r)\le a_i+\sum_{j\in S}\Delta_{ir,j}w_{kj}
\le a_i+\lambda_kc_{ir}^S.
$$

The objective is unchanged in both directions, so
$F(\mathbf1_S)=V(S)$. This particular reformulation is derived for the
testing problem here; the cited logical-constraint paper supplies the
modelling and outer-approximation viewpoint.

### 2.3. A dual that is affine in the selection

The purpose of the next duality step is to obtain lower bounds that can be
reused across selections. Introduce nonnegative multipliers $\Pi_{ir}$ for
the loss constraints and $\mu_j$ for the selection constraints in
(5). The matrix $\Pi$ is a scaled transport plan. To describe
its objective and feasible set, write

$$
\begin{gathered}
m_{kr}(\Pi)=\sum_{i\in I_k}\Pi_{ir},\qquad
C_{kj}(\Pi)=\sum_{i\in I_k,r\in[N]}\Pi_{ir}\Delta_{ir,j},\\
H(\Pi)=\sum_{r=1}^N\inf_{s\in\mathbb R}
\{m_{0r}(\Pi)\ell(-s)+m_{1r}(\Pi)\ell(s)\}.
\end{gathered}
$$

Here $m_{kr}$ is the class-$k$ mass at target $r$, $C_{kj}$ is the
transport expenditure in coordinate $j$, and each term of $H$ is the
infimum of the weighted loss at one target. Unlike $\pi\in\mathcal T_k(S)$,
$\Pi$ need not have total class mass one.

The Lagrangian calculation in Appendix B gives

$$
\begin{aligned}
F(z)=\max_{\Pi,\alpha,\mu}\quad&H(\Pi)\\
\text{subject to}\quad&(\Pi,\alpha,\mu)\in\mathcal D,\qquad
\alpha+\mu^\top z=1,
\end{aligned}
\tag{6}
$$

where $\alpha$ scales the source masses and, with $(v)_+=\max\{v,0\}$,

$$
\mathcal D=\left\{(\Pi,\alpha,\mu)\ge0:
\begin{array}{ll}
\displaystyle\sum_{r=1}^N\Pi_{ir}=\alpha/n,&i\in[N],\\[3pt]
\displaystyle\sum_{j=1}^D(C_{kj}(\Pi)-\rho_k\mu_j)_+
\le\alpha\rho_k,&k=0,1
\end{array}\right\}.
\tag{7}
$$

The row equalities come from minimizing over $a$, while the positive-part
inequalities come from minimizing over $\lambda$ and $w$. These formulas
are derived from (5), rather than imported from the references.

The normalization in (6) still depends on $z$.
Since $\mathcal D$ is a cone, scaling a normalized tuple by $\tau\ge0$
keeps it feasible and gives

$$
H(\tau\Pi)=\tau H(\Pi),\qquad
\max_{\tau\ge0}\{2\sqrt{\tau H(\Pi)}-\tau\}=H(\Pi).
$$

Optimizing the scale on each ray therefore yields

$$
\boxed{F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.}
\tag{8}
$$

Each feasible tuple now defines an affine lower bound for every selection.
At a given $z$, a normalized optimizer
$(\bar\Pi,\bar\alpha,\bar\mu)$ must be multiplied by $F(z)$ to give an
optimizer of (8). In particular, its affine slope is
$-F(z)\bar\mu$, not $-\bar\mu$. Proposition B.1 proves the
equivalence, attainment, and contact with $F$ at the evaluation point.

## 3. Cutting-plane algorithm

The affine representation allows the outer-approximation strategy of
Bertsimas et al. [1, Section 3.1]: solve the
continuous problem for one selection, retain its affine lower bound, and
use a binary linear program to choose the next selection.
Let $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
At $z^t$, take an optimizer $(\Pi^t,\alpha^t,\mu^t)$ of
(8). It defines

$$
\begin{aligned}
c_t(z)&=2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z,\\
F(z)&\ge c_t(z)=F(z^t)-(\mu^t)^\top(z-z^t),\qquad
c_t(z^t)=F(z^t).
\end{aligned}
\tag{9}
$$

Dual feasibility gives the lower bound, and dual optimality gives equality
at $z^t$. Thus this is a supporting affine function, not merely an
arbitrary lower bound; no differentiability of $F$ is required.
The next selection and the master lower bound solve

$$
\begin{aligned}
(z^{t+1},\eta^{t+1})\in
\operatorname*{arg\,min}_{z\in\mathcal Z_d,\ \eta\ge0}\quad&\eta\\
\text{subject to}\quad&
\eta\ge F(z^s)-(\mu^s)^\top(z-z^s),\qquad s=1,\ldots,t.
\end{aligned}
\tag{10}
$$

With $U_t=\min_{s\le t}F(z^s)$, all retained cuts lie below $F$, so

$$
\eta^{t+1}\le\mathrm{OPT}\le U_t.
$$

Stop when $U_t-\eta^{t+1}\le\varepsilon$ and return the best evaluated
selection. Appendix A recovers its detector from a solution
of the finite convex problem.

**Algorithm 1. Exact cutting-plane method for variable selection.**

**Input:** Problem data, budget $d$, tolerance $\varepsilon>0$, and $z^1\in\mathcal Z_d$.

**Output:** A subset $S^*$, its optimal detector $f^*$, and a risk interval of width at most $\varepsilon$.

1. $t\gets1$, $U\gets+\infty$.

2. **while** true **do**

    3. Solve (8) at $z^t$ and take an optimizer $(\Pi^t,\alpha^t,\mu^t)$.

    4. **if** $F(z^t)<U$ **then**

        5. $(U,z^*)\gets(F(z^t),z^t)$.

    6. **end if**

    7. Retain the cut (9) with index $t$.

    8. Solve (10) to obtain $(z^{t+1},\eta^{t+1})$.

    9. **if** $U-\eta^{t+1}\le\varepsilon$ **then**

        10. $S^*\gets\{j:z_j^*=1\}$.

        11. Solve (4) for $S^*$ and construct $g^*$ by (14).

        12. Set $f^*(x)=g^*(x[S^*])$ and

        13. **return** $(S^*,f^*,[\eta^{t+1},U])$.

    14. **end if**

    15. $t\gets t+1$.

16. **end while**

## Appendix A. Equivalence with the finite testing problem

**Proposition A.1 (Finite representation).**

For every $S\subseteq[D]$, the finite convex problem (4)
attains the value $V(S)=\min_{f\in\mathcal F_S}R(f)$.
Every optimal finite solution yields an optimal measurable detector on
the full observation space.

**Proof.**

Let $\operatorname{Proj}_Sx=x[S]$ and let $(\operatorname{Proj}_S)_\#\mathbb P$ denote the distribution of
$X[S]$ under $\mathbb P$. Nonexpansiveness of $\operatorname{Proj}_S$ gives

$$
\mathcal W_1((\operatorname{Proj}_S)_\#\mathbb P,\widehat Q_k^S)
\le\mathcal W_1(\mathbb P,\widehat{\mathbb P}_k).
$$

Conversely, disintegrate a coupling from $\widehat Q_k^S$ to
$Q\in\mathcal Q_k^S$ into source-indexed measures $\nu_i$ of mass $1/n$.
For $i\in I_k$, define $L_i(y)[S]=y$ and
$L_i(y)[S^c]=x_i[S^c]$. Then

$$
\mathbb P=\sum_{i\in I_k}(L_i)_\#\nu_i,\qquad
(\operatorname{Proj}_S)_\#\mathbb P=Q,\qquad
\mathcal W_1(\mathbb P,\widehat{\mathbb P}_k)
\le\sum_{i\in I_k}\int\|y-y_i\|_1\,\nu_i(dy)\le\rho_k.
$$

Repeated projected sources can be split into their original labels.
Thus $(\operatorname{Proj}_S)_\#\mathcal P_k=\mathcal Q_k^S$, and the projected robust
testing problem has value $V(S)$.

For $S\ne\varnothing$, apply Proposition 1 and Theorem 3 of
Gao et al. [2] on the projected Euclidean space.
With $p_{kr}=\sum_{i\in I_k}\pi^k_{ir}$, their finite-support objective is
the sum of the pointwise optimal weighted losses. It gives

$$
\begin{aligned}
V(S)
&=\max_{\pi^0\in\mathcal T_0(S),\,\pi^1\in\mathcal T_1(S)}
\sum_{r=1}^N\inf_{s\in\mathbb R}
\{p_{0r}\ell(-s)+p_{1r}\ell(s)\}\\
&=\max_{\pi^0,\pi^1}\inf_{u\in\mathbb R^N}
\sum_{k=0}^1\sum_{i\in I_k,r\in[N]}\pi^k_{ir}\ell_k(u_r)\\
&=\inf_{u\in\mathbb R^N}\sum_{k=0}^1
\max_{\pi^k\in\mathcal T_k(S)}
\sum_{i\in I_k,r\in[N]}\pi^k_{ir}\ell_k(u_r).
\end{aligned}
\tag{11}
$$

In the second line the plans range over the same product set as in the
first. The last equality is finite-dimensional minimax duality: the
transport set is compact and convex, and the objective is continuous,
affine in the plans, and convex in $u$. Coincident target locations can
be merged; keeping their labels gives the same value. If
$S=\varnothing$, the projected space is a singleton and both sides are
$\min_s\{\ell(-s)+\ell(s)\}=2$, so (3) still holds.

For fixed $u$ and class $k$, the transport LP has Lagrangian

$$
\begin{aligned}
L_k={}&\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r)
+\sum_{i\in I_k}a_i\left(\frac1n-\sum_r\pi_{ir}\right)
+\lambda_k\left(\rho_k-\sum_{i\in I_k,r}c_{ir}^S\pi_{ir}\right)\\
={}&\frac1n\sum_{i\in I_k}a_i+\rho_k\lambda_k
+\sum_{i\in I_k,r}\pi_{ir}[\ell_k(u_r)-a_i-\lambda_kc_{ir}^S].
\end{aligned}
\tag{12}
$$

Its supremum over $\pi\ge0$ is finite exactly when all bracketed
coefficients are nonpositive. LP strong duality therefore gives

$$
\begin{aligned}
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r)
=\min_{(a_i)_{i\in I_k},\,\lambda_k\ge0}\quad&
\frac1n\sum_{i\in I_k}a_i+\rho_k\lambda_k\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S,\quad i\in I_k,\ r\in[N].
\end{aligned}
\tag{13}
$$

Summing over the classes and minimizing over $u$ proves
(4). Its sublevel $J\le2$ has
$0\le a_i\le2n$ and $0\le\lambda_k\le2/\rho_k$.
The constraints then bound both $\ell(u_r)$ and $\ell(-u_r)$.
Since $\ell(t)\to\infty$ as $t\to\infty$, this sublevel is bounded
and closed, and a minimum is attained.

To recover a detector, take a feasible $(a,u,\lambda)$ in
(4). At $y\in\mathbb R^{|S|}$, choose the smallest minimizing
index in each class:

$$
\begin{gathered}
E_k(y)=\min_{i\in I_k}\{a_i+\lambda_k\|y_i-y\|_1\},\qquad
i_k=\min\operatorname*{arg\,min}_{i\in I_k}
\{a_i+\lambda_k\|y_i-y\|_1\},\\
r_k=\|y_{i_k}-y\|_1,\qquad
b=\|y_{i_0}-y_{i_1}\|_1\le r_0+r_1.
\end{gathered}
$$

Define

$$
g(y)=\begin{cases}
(1-\gamma)u_{i_0}+\gamma u_{i_1},\quad
\gamma=\min\{1,r_0/b\},&b>0,\\
u_{i_0},&b=0.
\end{cases}
\tag{14}
$$

If $b>0$, then $\gamma b\le r_0$ and $(1-\gamma)b\le r_1$.
Convexity and the four source--target constraints give

$$
\begin{aligned}
\ell_0(g(y))
&\le(1-\gamma)\ell_0(u_{i_0})+\gamma\ell_0(u_{i_1})
\le a_{i_0}+\lambda_0\gamma b\le E_0(y),\\
\ell_1(g(y))
&\le(1-\gamma)\ell_1(u_{i_0})+\gamma\ell_1(u_{i_1})
\le a_{i_1}+\lambda_1(1-\gamma)b\le E_1(y).
\end{aligned}
$$

For $b=0$, the same bounds follow from the constraints at target $i_0$.
The finite tie rule makes $g$ Borel measurable and
$|g(y)|\le\max_r|u_r|$. Integrating its loss bounds against a transport
coupling from $\widehat Q_k^S$ to any $Q\in\mathcal Q_k^S$ yields

$$
\mathbb E_Q\ell_k(g(Y))\le\frac1n\sum_{i\in I_k}a_i+\rho_k\lambda_k,
\qquad R(f)\le J(a,\lambda),\quad f(x)=g(x[S]).
$$

For an optimal finite solution, the right side is $V(S)$; the reverse
inequality follows from $f\in\mathcal F_S$. Hence the recovered detector
is optimal.

$\square$

## Appendix B. Dual representation and supporting affine functions

**Proposition B.1 (Duality and subgradients).**

(i) For every $S\subseteq[D]$,
$F(\mathbf1_S)=V(S)$. For every $z\in[0,1]^D$, both the primal minimum
and the normalized dual maximum are attained, with

$$
0<F(z)=\max_{\substack{(\Pi,\alpha,\mu)\in\mathcal D\\
\alpha+\mu^\top z=1}}H(\Pi)\le2.
\tag{15}
$$

(ii) The value also has the attained representation

$$
F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.
\tag{16}
$$

For a normalized optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$, an optimizer
of this affine representation is

$$
(\Pi,\alpha,\mu)=F(z)(\bar\Pi,\bar\alpha,\bar\mu),\qquad
H(\Pi)=F(z)^2.
\tag{17}
$$

(iii) The function $F$ is convex on $[0,1]^D$. An affine
optimizer $(\Pi^t,\alpha^t,\mu^t)$ at $z^t$ gives

$$
\begin{aligned}
c_t(z)&=2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z\le F(z),\\
c_t(z^t)&=F(z^t),\qquad
F(z)\ge F(z^t)-(\mu^t)^\top(z-z^t).
\end{aligned}
\tag{18}
$$

Thus $-\mu^t\in\partial F(z^t)$, where the subgradient inequality is
understood on $[0,1]^D$.

**Proof.**

(i) In either primal formulation, the constraints with $r=i$
imply $a_i\ge\ell_k(u_i)\ge0$. If $(a,u,\lambda)$ is feasible in
(4), set $z=\mathbf1_S$ and $w_{kj}=\lambda_kz_j$.
Then

$$
\ell_k(u_r)\le a_i+\sum_j\Delta_{ir,j}w_{kj},\qquad
\sum_k\rho_kw_{kj}=z_j\sum_k\rho_k\lambda_k\le z_jJ.
$$

Conversely, a feasible solution of (5) with $z=\mathbf1_S$
satisfies

$$
w_{kj}=0\ (j\notin S),\qquad
\ell_k(u_r)\le a_i+\sum_{j\in S}\Delta_{ir,j}w_{kj}
\le a_i+\lambda_kc_{ir}^S.
$$

Both directions preserve $J$, proving $F(\mathbf1_S)=V(S)$.

To derive the dual, assign multipliers $\Pi_{ir},\beta_{kj},\mu_j\ge0$
to the loss inequalities, the caps $w_{kj}\le\lambda_k$, and the
selection inequalities, respectively. Retain $a,u$ free and
$\lambda,w\ge0$, and put $\alpha=1-\mu^\top z$. The Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J+
\sum_{k,i\in I_k,r}\Pi_{ir}
\left[\ell_k(u_r)-a_i-\sum_j\Delta_{ir,j}w_{kj}\right]
+\sum_{k,j}\beta_{kj}(w_{kj}-\lambda_k)\\
&+\sum_j\mu_j\left[\sum_k\rho_kw_{kj}-z_jJ\right]\\
={}&\sum_i a_i\left(\frac\alpha n-\sum_r\Pi_{ir}\right)
+\sum_{r=1}^N\{m_{0r}(\Pi)\ell(-u_r)+m_{1r}(\Pi)\ell(u_r)\}\\
&+\sum_k\lambda_k\left(\alpha\rho_k-\sum_j\beta_{kj}\right)
+\sum_{k,j}w_{kj}[\beta_{kj}+\rho_k\mu_j-C_{kj}(\Pi)].
\end{aligned}
\tag{19}
$$

Its infimum over $a,\lambda,w$ is finite precisely when

$$
\sum_r\Pi_{ir}=\alpha/n,\qquad
\sum_j\beta_{kj}\le\alpha\rho_k,\qquad
\beta_{kj}\ge(C_{kj}(\Pi)-\rho_k\mu_j)_+.
\tag{20}
$$

The row sums force $\alpha\ge0$. Minimizing the remaining terms over $u$
separates across targets and gives $H(\Pi)$. Choosing the smallest
admissible $\beta_{kj}$ in (20) gives exactly
$\mathcal D$ and $\alpha+\mu^\top z=1$, which proves the dual formula
subject to strong duality.

For strong duality, first remove every variable $w_{kj}$ with $z_j=0$
and its selection row, since those variables are forced to zero.
The reduced convex problem is strictly feasible at

$$
u=0,\qquad a_i=2,\qquad\lambda_k=1,\qquad
w_{kj}=z_j/2\quad(z_j>0).
$$

Indeed, the loss constraints and caps are strict and

$$
\sum_k\rho_kw_{kj}=\frac{z_j}{2}(\rho_0+\rho_1)
<z_j(4+\rho_0+\rho_1)=z_jJ\quad(z_j>0).
$$

Slater's theorem gives strong duality and dual attainment for the reduced
problem. Restore a deleted coordinate using
$\mu_j=\max_k C_{kj}(\Pi)/\rho_k$ and $\beta_{kj}=0$;
neither the objective nor normalization changes. Thus the full dual also
attains $F(z)$, including at the boundary of the cube.

The constant feasible point $(a,u,\lambda,w)=(\mathbf1,0,0,0)$ has value
$2$. The sublevel $J\le2$ bounds $a,\lambda,w$, and the loss constraints
bound $u$ on both sides, as in Appendix A. Hence a primal
minimum exists. A zero value would force $a=\lambda=w=0$ and therefore

$$
2=2\ell(0)\le\ell(-u_r)+\ell(u_r)\le0,
$$

a contradiction. This proves (15).

(ii) The set $\mathcal D$ is a cone, and
$H(\tau\Pi)=\tau H(\Pi)\ge0$ for $\tau\ge0$.
For any feasible tuple let $b=\alpha+\mu^\top z$.
If $b>0$, normalization and part (i) imply
$H(\Pi)/b\le F(z)$, so

$$
2\sqrt{H(\Pi)}-b
=\frac{H(\Pi)}b-
\left(\sqrt{\frac{H(\Pi)}b}-\sqrt b\right)^2\le F(z).
\tag{21}
$$

If $b=0$, then $\alpha=0$ and $\Pi=0$, giving affine value zero.
For a normalized optimizer, $H(\bar\Pi)=F(z)$ and
$\bar\alpha+\bar\mu^\top z=1$. Scaling it by $F(z)$ yields

$$
H(F(z)\bar\Pi)=F(z)^2,\qquad
2\sqrt{H(F(z)\bar\Pi)}-F(z)(\bar\alpha+\bar\mu^\top z)=F(z).
$$

Thus the bound is attained, proving (16) and
(17).

(iii) Since $\mathcal D$ is independent of $z$,
(16) expresses $F$ as a maximum of affine
functions. It is therefore convex. A feasible tuple defines a lower
bound for all $z$, and an optimal tuple at $z^t$ attains equality there.
Hence

$$
c_t(z)=c_t(z^t)-(\mu^t)^\top(z-z^t)
=F(z^t)-(\mu^t)^\top(z-z^t)\le F(z).
$$

$\square$

## References

[1] Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A Unified Approach to Mixed-Integer Optimization Problems With Logical Constraints. *SIAM Journal on Optimization*, 31(3): 2340–2367, 2021. [DOI](https://doi.org/10.1137/20M1346778).

[2] Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust Hypothesis Testing Using Wasserstein Uncertainty Sets. *Advances in Neural Information Processing Systems*, 31. Curran Associates, Inc., 2018. [Source](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a08e32d2f9a8b78894d964ec7fd4172e-Abstract.html).

[3] Liyan Xie, Rui Gao, and Yao Xie. Robust Hypothesis Testing with Wasserstein Uncertainty Sets. *arXiv:2105.14348*, 2021. [Source](https://arxiv.org/abs/2105.14348).
