# Variable Selection for Wasserstein Robust Hypothesis Testing

## Abstract

Starting from the finite least-favorable-distribution formulation of Wasserstein robust testing, we derive a convex model with coordinate selection. A uniform risk bound introduces the selection variables without changing binary optimal values. The resulting convex value function admits affine dual cuts for optimization over coordinate subsets.

## 1. Problem formulation

Throughout this paper, we consider the variable selection problem for robust hypothesis testing. Let $[D]=\{1,\ldots,D\}$ and $d$ be a positive integer denoting the sparsity budget.
Given a detector $f:~\mathbb{R}^D\to\mathbb{R}$, we follow the convention in [2] to define the worst-case risk as

$$
R(f)=\sup_{\mathbb{P}_0\in\mathcal{P}_0, \mathbb{P}_1\in\mathcal{P}_1}~
\left\{\mathbb E_{\mathbb P_0}\ell(-f(X))+
       \mathbb E_{\mathbb P_1}\ell(f(X))\right\},
\tag{1}
$$

where $\mathcal{P}_k, k=0,1$ denote the Wasserstein distance-based ambiguity sets, and $\ell(\cdot)$ is the generating function that is nonnegative-valued, non-decreasing, convex, and satisfies $\ell(0)=1, \lim_{t\to-\infty}\ell(t)=0$.
For $k=0,1$, assume we have $n_k$ available observations in the group $k$, which are denoted as $\{x_i^{(k)}\}_{i=1}^{n_k}$.
We denote the corresponding empirical distribution as $\widehat{\mathbb{P}}_k=\frac1{n_k}\sum_{i=1}^{n_k}\delta_{x_i^{(k)}}$, and construct the Wasserstein ambiguity set as

$$
\mathcal{P}_k=\Big\{
\mathbb{P}_k:~\mathcal{W}_1(
\widehat{\mathbb{P}}_k, \mathbb{P}_k
)\le \rho_k
\Big\}.
$$

Here, $\mathcal{W}_1(\cdot,\cdot)$ refers to the $1$-Wasserstein distance equipped with the $\ell_1$-norm.

For $S\subseteq[D]$, define the collection of detectors whose input only depends on coordinates belonging to $S$:

$$
\mathcal{F}_S = \Big\{
f:~\mathbb{R}^D\to\mathbb{R}, f(x) = g(x[S])\text{ for some measurable function $g:~\mathbb{R}^{|S|}\to\mathbb{R}$}
\Big\}.
$$

Note that for a given vector $x\in\mathbb{R}^D$, we use $x[i]$ to denote its $i$-th coordinate, and $x[S]$ to denote the concatenation of its coordinates whose indices belong to $S$.
Our proposed variable selection problem can be formulated as the following combinatorial optimization problem

$$
\min_{S\subseteq[D], |S|\le d}\min_{f\in\mathcal{F}_S}~R(f).
\tag{2}
$$

## 2. Finite reformulation

Assume $n_0,n_1\ge1$, $\rho_0,\rho_1>0$, and that $\ell$ is finite
on $\mathbb R$. All distributions have finite first moments. Fix
$S\subseteq[D]$ and write

$$
V(S)=\min_{f\in\mathcal F_S}R(f),\qquad
\mathcal X=\{x_i^{(0)}:i\in[n_0]\}\cup
\{x_j^{(1)}:j\in[n_1]\}.
$$

Sources retain their class indices; a target is denoted by $y\in\mathcal X$.
Projection onto $S$ does not increase the transport cost. Conversely,
any transport of projected observations lifts to the original space by
keeping each source's unselected coordinates unchanged, with the same
cost. Thus the projected Wasserstein balls describe exactly the
uncertainty relevant to a detector $g(x[S])$.

Our starting point is the full-support transport formulation in
[3, Lemma 2, equation (7)]. With our class indices, its randomized-testing
objective uses $\min\{p_0(y),p_1(y)\}$ at each target. For the generating
loss of Section 1, this term becomes the infimum of the weighted
loss over a real score. The following lemma retains the source paper's
transport constraints and our class indices. Its extension to the present
loss is proved in Appendix A.

**Lemma 2.1 (Finite LFD reformulation).**

The fixed-selection risk is

$$
\begin{aligned}
V(S)=\max_{p,\gamma\ge0}\quad&
\sum_{y\in\mathcal X}\inf_{v\in\mathbb R}
\{p_0(y)\ell(-v)+p_1(y)\ell(v)\}\\
\text{subject to}\quad&
\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}=\frac1{n_k},
&&i\in[n_k],\ k=0,1,\\
&p_k(y)=\sum_{i=1}^{n_k}\gamma_{i,y}^{(k)},
&&y\in\mathcal X,\ k=0,1,\\
&\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}\|x_i^{(k)}[S]-y[S]\|_1\le\rho_k,
&&k=0,1.
\end{aligned}
\tag{3}
$$

Here $\gamma_{i,y}^{(k)}$ is the mass moved from $x_i^{(k)}[S]$ to
$y[S]$, and $p_k(y)$ is its resulting target mass. All targets are
retained, including the source itself. If projected targets coincide,
their masses are aggregated. Concavity and positive homogeneity of the
objective imply that aggregation cannot decrease it; conversely, an
aggregate can be split in the same proportions for both classes without
changing the objective.
Hence the fixed target labels $y\in\mathcal X$ do not affect the value.

Write $g_y$ for the score assigned to $y[S]$. Substituting the target
masses gives

$$
\sum_{y\in\mathcal X}\inf_{v\in\mathbb R}
\{p_0(y)\ell(-v)+p_1(y)\ell(v)\}
=\inf_g\left\{
\sum_{i,y}\gamma_{i,y}^{(0)}\ell(-g_y)
+\sum_{j,y}\gamma_{j,y}^{(1)}\ell(g_y)\right\}.
$$

The transport set is compact and convex, and the expression inside braces is convex
in the finite score vector $g$ and linear in $\gamma$. Finite-dimensional
minimax expresses the finite LFD value as

$$
V(S)=\inf_g\left\{
\max_{\gamma^{(0)}}\sum_{i,y}\gamma_{i,y}^{(0)}\ell(-g_y)
+\max_{\gamma^{(1)}}\sum_{j,y}\gamma_{j,y}^{(1)}\ell(g_y)
\right\},
$$

where each maximum satisfies its source-mass and transport-budget
constraints in (3).

For fixed $g$, the class-$0$ maximization is a transport LP.
Assign a free multiplier $a_i^{(0)}$ to source-mass conservation and
a nonnegative multiplier $\lambda_0$ to the transport budget. Its
Lagrangian is

$$
\begin{aligned}
\mathcal L_0={}&\sum_{i,y}\gamma_{i,y}^{(0)}\ell(-g_y)
+\sum_i a_i^{(0)}\left(\frac1{n_0}-\sum_y\gamma_{i,y}^{(0)}\right)\\
&+\lambda_0\left(\rho_0-\sum_{i,y}\gamma_{i,y}^{(0)}
\|x_i^{(0)}[S]-y[S]\|_1\right)\\
={}&\frac1{n_0}\sum_i a_i^{(0)}+\rho_0\lambda_0\\
&+\sum_{i,y}\gamma_{i,y}^{(0)}
\left[\ell(-g_y)-a_i^{(0)}
-\lambda_0\|x_i^{(0)}[S]-y[S]\|_1\right].
\end{aligned}
\tag{4}
$$

The supremum over $\gamma^{(0)}\ge0$ is finite exactly when each
bracketed coefficient is nonpositive. Applying LP duality to both
classes, then minimizing over $g$, yields

$$
\begin{aligned}
V(S)=\min_{g,a,\lambda}\quad&
\frac1{n_0}\sum_i a_i^{(0)}+\frac1{n_1}\sum_j a_j^{(1)}
+\rho_0\lambda_0+\rho_1\lambda_1\\
\text{subject to}\quad&
\ell(-g_y)\le a_i^{(0)}+\lambda_0\|x_i^{(0)}[S]-y[S]\|_1,
&&i\in[n_0],\ y\in\mathcal X,\\
&\ell(g_y)\le a_j^{(1)}+\lambda_1\|x_j^{(1)}[S]-y[S]\|_1,
&&j\in[n_1],\ y\in\mathcal X,\\
&g\in\mathbb R^{\mathcal X},\quad a^{(k)}\in\mathbb R^{n_k},
\quad\lambda_k\ge0,&&k=0,1.
\end{aligned}
\tag{5}
$$

Denote this objective by $J(a,\lambda)$. Scores at coincident
projections can be chosen equal because their constraint columns are
identical. Appendix A extends these finite scores to
the measurable detector $g$ in Section 1 and verifies
attainment of the minimum.

The zero detector has risk $2$. Taking $y=x_i^{(k)}$ in the constraints
gives $a_i^{(k)}\ge0$, so an optimum satisfies

$$
\rho_0\lambda_0+\rho_1\lambda_1\le J(a,\lambda)=V(S)\le2.
\tag{6}
$$

This bound will introduce coordinate selection without changing the
binary optimal values.

## 3. Selection variables and duality

Let $z=\mathbf1_S$ and
$\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
The transport cost decomposes directly into coordinates:

$$
\|x_i^{(k)}[S]-y[S]\|_1
=\sum_{m=1}^D z_m|x_i^{(k)}[m]-y[m]|.
$$

Introduce $w_{km}$ in place of $\lambda_kz_m$. The conditions
$0\le w_{km}\le\lambda_k$ and
$\rho_0w_{0m}+\rho_1w_{1m}\le2z_m$ force both weights to zero
when a coordinate is unselected. When it is selected, setting
$w_{km}=\lambda_k$ is allowed at an optimum by
(6). For $z\in[0,1]^D$, define

$$
\begin{aligned}
F(z)=\min_{g,a,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell(-g_y)\le a_i^{(0)}+\sum_m w_{0m}|x_i^{(0)}[m]-y[m]|,
&&i\in[n_0],\ y\in\mathcal X,\\
&\ell(g_y)\le a_j^{(1)}+\sum_m w_{1m}|x_j^{(1)}[m]-y[m]|,
&&j\in[n_1],\ y\in\mathcal X,\\
&0\le w_{km}\le\lambda_k,&&k=0,1,\ m\in[D],\\
&\rho_0w_{0m}+\rho_1w_{1m}\le2z_m,&&m\in[D],\\
&g\in\mathbb R^{\mathcal X},\quad a^{(k)}\in\mathbb R^{n_k},
\quad\lambda_k\ge0,&&k=0,1.
\end{aligned}
\tag{7}
$$

This has the logical structure of problems (2.1)--(2.2) in
[1]: the outer problem selects coordinates,
each activating a pair $(w_{0m},w_{1m})$ in a convex inner problem.

**Proposition 3.1 (Selection formulation).**

The minimum in (7) is attained. The function $F$ is convex
on $[0,1]^D$, and

$$
F(\mathbf1_S)=V(S),\qquad
\min_{|S|\le d}V(S)=\min_{z\in\mathcal Z_d}F(z).
\tag{8}
$$

**Proof.**

An optimum of (5) extends to (7) by
$w_{km}=\lambda_kz_m$ at $z=\mathbf1_S$, using
(6). Conversely, a feasible point at that binary
selection has $w_{km}=0$ outside $S$, hence

$$
\sum_m w_{km}|x_i^{(k)}[m]-y[m]|
\le\lambda_k\sum_{m\in S}|x_i^{(k)}[m]-y[m]|
=\lambda_k\|x_i^{(k)}[S]-y[S]\|_1.
$$

It is therefore feasible in (5) with the same objective.
This proves equality of the binary optimal values. For any $z$, the
feasible choice $g=0$, $a=\mathbf1$, $\lambda=w=0$ has value $2$.
The sublevel $J\le2$ is compact by the bounds in
Appendix A and $0\le w_{km}\le\lambda_k$.
All constraints are jointly convex, including in $z$. Taking convex
combinations of feasible solutions gives

$$
F(tz+(1-t)z')\le tF(z)+(1-t)F(z'),\qquad 0\le t\le1.
$$

$\square$

### 3.1. Dual representation

Assign multipliers $\gamma_{i,y}^{(0)}\ge0$ and
$\gamma_{j,y}^{(1)}\ge0$ to the two families of loss constraints in
(7), and $\mu_m\ge0$ to each coordinate-capacity constraint.
Minimizing over the free source variables $a$ recovers the source-mass
equalities $\sum_y\gamma_{i,y}^{(k)}=1/n_k$. Thus the same transport
notation reappears through duality. With
$p_k(y)=\sum_{i=1}^{n_k}\gamma_{i,y}^{(k)}$ as in (3), put

$$
H(\gamma)=\sum_{y\in\mathcal X}\inf_{v\in\mathbb R}
\{p_0(y)\ell(-v)+p_1(y)\ell(v)\}.
\tag{9}
$$

This is the LFD objective, obtained by minimizing over the target scores.
The remaining dual constraints define the set

$$
\mathcal D=\left\{(\gamma,\mu)\ge0:
\begin{aligned}
&\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}=1/n_k,
&&i\in[n_k],\ k=0,1,\\
&\sum_{m=1}^D\left(
\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|-\rho_k\mu_m
\right)_+\le\rho_k,
&&k=0,1
\end{aligned}\right\},
\tag{10}
$$

where $(v)_+=\max\{v,0\}$. Appendix B derives the second
line by minimizing over $\lambda,w$. In particular, $\mathcal D$ does
not depend on $z$.

**Proposition 3.2 (Dual representation).**

For every $z\in[0,1]^D$, the following maximum is attained:

$$
F(z)=\max_{(\gamma,\mu)\in\mathcal D}
\{H(\gamma)-2\mu^\top z\}.
\tag{11}
$$

An optimizer $(\gamma^t,\mu^t)$ at $z^t$ gives the supporting function

$$
c_t(z)=H(\gamma^t)-2(\mu^t)^\top z
=F(z^t)-2(\mu^t)^\top(z-z^t)\le F(z),
\tag{12}
$$

with $c_t(z^t)=F(z^t)$.

The function $H$ is concave as an infimum of linear functions, and
$\mathcal D$ is convex. The dual at fixed $z$ is therefore a convex
optimization problem. Its affine dependence on $z$ gives the
cutting-plane construction underlying Theorem 2.5 and Algorithm 3.1
of [1].

## 4. Cutting-plane algorithm

Write $\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z)$.
After evaluating $z^1,\ldots,z^t$ and obtaining their dual optimizers,
let $U_t=\min_{s\le t}F(z^s)$ and solve

$$
\begin{aligned}
(z^{t+1},L_t)\in\operatorname*{arg\,min}_{z\in\mathcal Z_d,\,\eta\ge0}
\quad&\eta\\
\text{subject to}\quad&
\eta\ge F(z^s)-2(\mu^s)^\top(z-z^s),\qquad s=1,\ldots,t.
\end{aligned}
\tag{13}
$$

Every retained cut is a global lower bound and is exact at its evaluated
selection. Consequently,

$$
L_t\le\mathrm{OPT}\le U_t.
\tag{14}
$$

Starting from any $z^1\in\mathcal Z_d$, alternate the convex subproblem
(7)--(11) and the binary linear master
(13). Stop when $U_t-L_t\le\varepsilon$ and return the
best evaluated selection. Appendix A recovers its
detector, with risk $U_t$ and optimality gap at most $\varepsilon$.
For an inexact inner solve, use the actual feasible dual expression
$H(\gamma^t)-2(\mu^t)^\top z$ as the cut and a primal feasible objective
as the upper bound; the master must provide a global lower bound.
The binary formulation is exact, although its continuous relaxation
can be weak.

## Appendix A. Proof of the finite reformulation

For any feasible $(g,a,\lambda)$ in (5), choose equal
scores at coincident projected targets. This preserves feasibility
and the objective. As in the off-support extension in
[3, Theorem 1], we bound both losses at every point.
For the present convex generating loss, we construct an extension satisfying

$$
\begin{aligned}
\ell(-g(\xi))&\le E_0(\xi),&
\ell(g(\xi))&\le E_1(\xi),\\
E_k(\xi)&=\min_{i\in[n_k]}
\{a_i^{(k)}+\lambda_k\|x_i^{(k)}[S]-\xi\|_1\},
&&k=0,1.
\end{aligned}
\tag{15}
$$

**Proof of Lemma 2.1.**

At an observed projection, set $g(y[S])=g_y$; the constraints of
(5) give (15) there. Elsewhere, choose
the smallest minimizing indices $i,j$ in $E_0(\xi),E_1(\xi)$, and set

$$
r_0=\|x_i^{(0)}[S]-\xi\|_1,\qquad
r_1=\|x_j^{(1)}[S]-\xi\|_1,\qquad
b=\|x_i^{(0)}[S]-x_j^{(1)}[S]\|_1.
$$

For $b>0$, let $\tau=\min\{1,r_0/b\}$ and define

$$
g(\xi)=(1-\tau)g_{x_i^{(0)}}+\tau g_{x_j^{(1)}}.
\tag{16}
$$

The triangle inequality gives $\tau b\le r_0$ and
$(1-\tau)b\le r_1$. Convexity and (5) imply

$$
\begin{aligned}
\ell(-g(\xi))
&\le(1-\tau)\ell(-g_{x_i^{(0)}})+\tau\ell(-g_{x_j^{(1)}})
\le a_i^{(0)}+\lambda_0\tau b\le E_0(\xi),\\
\ell(g(\xi))
&\le(1-\tau)\ell(g_{x_i^{(0)}})+\tau\ell(g_{x_j^{(1)}})
\le a_j^{(1)}+\lambda_1(1-\tau)b\le E_1(\xi).
\end{aligned}
$$

For $b=0$, take $g(\xi)=g_{x_i^{(0)}}$; the same bounds follow from
(5). The finite tie rule makes $g$ Borel measurable,
and $|g|\le\max_{y\in\mathcal X}|g_y|$. For any
$\mathbb P_k\in\mathcal P_k$, integrate (15) against
an optimal empirical-to-$\mathbb P_k$ coupling, using that projection cannot
increase its cost. This gives

$$
\begin{aligned}
\mathbb E_{\mathbb P_0}\ell(-g(X[S]))
&\le\frac1{n_0}\sum_i a_i^{(0)}+\rho_0\lambda_0,\\
\mathbb E_{\mathbb P_1}\ell(g(X[S]))
&\le\frac1{n_1}\sum_j a_j^{(1)}+\rho_1\lambda_1.
\end{aligned}
$$

Thus $f(x)=g(x[S])$ satisfies $R(f)\le J(a,\lambda)$.
On the nonempty sublevel $J\le2$,
$0\le a_i^{(k)}\le2n_k$ and $0\le\lambda_k\le2/\rho_k$.
Both losses of every $g_y$ are bounded by the two classes' constraints.
Since $\ell(t)\to\infty$ as $t\to\infty$, all scores are bounded.
This sublevel is compact, so (5) attains its minimum.
Weak minimax and restriction to the finite transport supports give
a lower bound on $V(S)$ equal to the finite LFD value in
(3). Finite minimax and transport LP duality identify
that value with the minimum of (5). The constructed
extension gives the reverse bound $V(S)\le\min J(a,\lambda)$.
Thus both finite formulations equal $V(S)$, and an optimal extension
attains this risk.

$\square$

## Appendix B. Proof of the dual representation

**Proof of Proposition 3.2.**

In addition to $\gamma,\mu$, assign multipliers $\beta_{km}\ge0$ to
$w_{km}\le\lambda_k$. Keeping $g,a$ free and $\lambda,w\ge0$, the
minimization Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J(a,\lambda)
+\sum_{i,y}\gamma_{i,y}^{(0)}
\left[\ell(-g_y)-a_i^{(0)}
-\sum_m w_{0m}|x_i^{(0)}[m]-y[m]|\right]\\
&+\sum_{j,y}\gamma_{j,y}^{(1)}
\left[\ell(g_y)-a_j^{(1)}
-\sum_m w_{1m}|x_j^{(1)}[m]-y[m]|\right]\\
&+\sum_{k,m}\beta_{km}(w_{km}-\lambda_k)
+\sum_m\mu_m(\rho_0w_{0m}+\rho_1w_{1m}-2z_m).
\end{aligned}
\tag{17}
$$

The infimum over $a$ is finite exactly when
$\sum_y\gamma_{i,y}^{(k)}=1/n_k$.
The terms depending on $g$ are
$\sum_y[p_0(y)\ell(-g_y)+p_1(y)\ell(g_y)]$; their infimum is
$H(\gamma)$. The coefficients of $\lambda,w\ge0$ must be nonnegative:

$$
\rho_k-\sum_m\beta_{km}\ge0,\qquad
\beta_{km}+\rho_k\mu_m
-\sum_{i,y}\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|\ge0.
$$

Together with $\beta\ge0$, these are equivalent to

$$
\sum_m\left(
\sum_{i,y}\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|
-\rho_k\mu_m\right)_+\le\rho_k.
$$

Eliminating $\beta$ therefore leaves the domain $\mathcal D$ and
objective $H(\gamma)-2\mu^\top z$.

For every fixed $z\in[0,1]^D$, the point
$g=0$, $a=2\mathbf1$, $\lambda=\mathbf1$, $w=0$
strictly satisfies all loss inequalities and satisfies all affine
constraints. Treating nonnegativity as affine constraints, the refined
Slater condition gives strong duality and dual attainment, including
at boundary selections. Finally, any dual feasible point gives a
global lower bound. Optimality of $(\gamma^t,\mu^t)$ at $z^t$ gives

$$
F(z)\ge H(\gamma^t)-2(\mu^t)^\top z
=F(z^t)-2(\mu^t)^\top(z-z^t),
$$

which proves (12).

$\square$

## References

[1] Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

[2] Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.

[3] Liyan Xie, Rui Gao, and Yao Xie. Robust hypothesis testing with Wasserstein uncertainty sets, 2021. arXiv:2105.14348.
