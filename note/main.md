# Certified Feature Selection for Wasserstein Robust Hypothesis Testing

9 October 2026.

## Abstract

We study how to select a limited number of features while designing a hypothesis test that remains reliable under class-conditional Wasserstein uncertainty. The objective is the sum of the two worst-case classwise losses. For nonnegative, nondecreasing convex losses that are finite on the real line, we obtain an exact finite optimization problem by constructing a whole-space test from its feasible solutions. The construction uses only convexity of the loss and the triangle inequality of the transport distance. We then use the risk itself to limit the coordinatewise transport prices. Eliminating the continuous variables gives a family of lower bounds: each is a residual loss divided by a transport correction. Their maximum is a convex function that agrees with the original risk at every binary feature selection. Tangents to these bounds generate a global search over feature subsets, with feasible tests providing upper bounds. Exact continuous and discrete solves terminate after finitely many subsets. The resulting certificates concern optimization within the specified uncertainty sets; population error control requires their separate calibration.

## 1. The selection problem

We can observe at most $d$ of $D$ features. The choice of features and the test using them must be optimized together: a good subset is one on which a test can keep both classwise losses small when the distributions move within their specified uncertainty sets.

Pool the observations from the two classes into $x_1,\ldots,x_N\in\mathbb R^D$. Observation $i$ has class $c_i\in\{0,1\}$, sign $y_i=2c_i-1$, and weight $q_i>0$. Write $I_k=\{i:c_i=k\}$. The empirical distributions and uncertainty sets are

$$
\widehat P_k=\sum_{i\in I_k}q_i\delta_{x_i},\qquad
\sum_{i\in I_k}q_i=1,\qquad
\mathcal U_k=\{P:W_1(P,\widehat P_k)\le\theta_k\},\quad k=0,1.
$$

Here $\delta_x$ denotes unit mass at $x$. The distance $W_1$ is the minimum expected $L^1$ distance needed to transport one distribution to the other; the distributions have finite first moments. The radii $\theta_k>0$ and feature scales are fixed before selection. Each class has total weight one, so the pooled weights sum to two.

For an integer budget $0\le d\le D$, a binary vector $z$ selects $S(z)=\{j:z_j=1\}$, with $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$. A measurable score $g$ depends only on the selected coordinates. A positive score favors class zero; the losses are therefore $\Phi(-g)$ for class zero and $\Phi(g)$ for class one. The problem is

$$
\begin{aligned}
V(z)&=\inf_{g=g(x_{S(z)})}
\left\{\sup_{P\in\mathcal U_0}\mathbb E_P\Phi(-g)
       +\sup_{P\in\mathcal U_1}\mathbb E_P\Phi(g)\right\},\\
\mathrm{OPT}&=\min_{z\in\mathcal Z_d}V(z).
\end{aligned}
\tag{1}
$$

Thus $V(z)$ already includes the best possible test for the chosen features; selection does not hold a previously fitted test fixed.

The function $\Phi:\mathbb R\to[0,\infty)$ is finite, convex and nondecreasing. Nondecreasing means that a larger score against the true class cannot reduce its loss. We consider nonconstant losses with $\Phi(0)>0$. If $\Phi(0)=0$ or the loss is constant, the constant test already solves the problem. Specific formulas for $\Phi$ are not needed for the construction below; examples are given in Section 6. The constant score $g=0$ provides the common benchmark

$$
0\le V(z)\le B:=2\Phi(0).
$$

## 2. A finite problem that still designs the whole test

The same pooled observations serve as candidate target locations. To compare source $i$ with target $\ell$, use one vector of coordinatewise distances:

$$
\mathbf d_{i\ell}=|x_i-x_\ell|\in\mathbb R_+^D,\qquad
d_z(x_i,x_\ell)=\mathbf d_{i\ell}^{\top}z.
$$

Absolute value is taken coordinatewise. The vector stores the geometry once; its inner product with $z$ is the distance using the selected features.

Let $u_\ell$ be the score assigned to target $\ell$. Let $a_i$ be a loss allowance at source $i$, and let $\lambda_k\ge0$ be the price per unit of transport for class $k$. Write $q=(q_i)$, $\theta=(\theta_0,\theta_1)^\top$, $\lambda=(\lambda_0,\lambda_1)^\top$, and $J(a,\lambda)=q^\top a+\theta^\top\lambda$.

**Theorem 1 (Exact finite formulation).**

For every binary $z$,

$$
\begin{aligned}
V(z)=\min_{a,u,\lambda}\quad &J(a,\lambda)\\
\text{subject to}\quad
&\Phi(y_i u_\ell)\le a_i+
  \lambda_{c_i}\mathbf d_{i\ell}^{\top}z,
  \qquad i,\ell=1,\ldots,N,\\
&\lambda\ge0,\qquad a,u\in\mathbb R^N.
\end{aligned}
\tag{2}
$$

The minimum is attained. Every feasible solution yields a measurable test on the entire observation space, using only the selected coordinates, with robust risk at most $J(a,\lambda)$.

The inequality in (2) has a direct meaning: after moving from source $i$ to target $\ell$, the loss is at most its source allowance plus the transport price. The class sign is $y_i$, not the original label of target $\ell$; either class can send mass to any target location.

The whole-space conclusion is essential. It allows a finite feasible objective to be used as an upper bound for the original test problem. Its proof blends two source scores, one from each class. Convexity controls the blended losses, and the triangle inequality controls the required transport allowances. The construction and both directions of the equality are given in Appendix A; no particular inverse or closed form of $\Phi$ is used.

## 3. Use the risk to control which coordinates are available

Taking $\ell=i$ in (2) gives $a_i\ge\Phi(y_i u_i)\ge0$. Consequently,

$$
\theta^\top\lambda\le J(a,\lambda).
\tag{3}
$$

The transport-price contribution is part of the total risk, so it cannot exceed that risk. This inequality supplies the bound needed to connect feature selection with continuous test design.

Introduce two vectors $w_0,w_1\in\mathbb R_+^D$ of coordinate prices. On a binary selection we can take $w_{kj}=\lambda_k z_j$: an unobserved coordinate has price zero. To obtain a useful continuous extension, use the risk bound $\eta$ to constrain these prices:

$$
\begin{aligned}
v(z)=\min_{\eta,a,u,\lambda,w_0,w_1}\quad &\eta\\
\text{subject to}\quad
&\Phi(y_i u_\ell)\le a_i+\mathbf d_{i\ell}^{\top}w_{c_i},
  \qquad \forall i,\ell,\\
&J(a,\lambda)\le\eta,\qquad \lambda\ge0,\\
&0\le w_k\le\lambda_k\mathbf1,\qquad k=0,1,\\
&\theta_0w_0+\theta_1w_1\le\eta z.
\end{aligned}
\tag{4}
$$

Vector inequalities hold coordinatewise; $a,u,\eta$ are real variables. The first line designs the test using the coordinate prices. The next lines bound its risk and keep each coordinate price below the class price. The last line makes an unselected coordinate unavailable and ties the allowed prices to the risk being minimized.

For binary $z$, an optimizer of (2) enters (4) with $\eta=J$ and $w_k=\lambda_k z$, by (3). Conversely, a feasible point of (4) has zero prices on unselected coordinates and $w_k\le\lambda_k z$. The same $a,u,\lambda$ therefore satisfy (2), with objective at most $\eta$. Hence $v(z)=V(z)$ at every binary selection. For fractional $z\in[0,1]^D$, (4) defines the extension used to construct lower bounds.

Fixing $z$ makes (4) a convex problem. This alone does not prove that its value $v(z)$ is convex in $z$, since the last line contains the product $\eta z$. The next step establishes that property and supplies information that remains useful when the feature subset changes.

## 4. One solved subset gives lower bounds for all subsets

Consider a nonnegative matrix $\Pi$ with $\Pi_{i\ell}$ units of mass sent from source $i$ to target $\ell$. Its rows retain the source weights, $\Pi\mathbf1=q$. The masses received from the two classes and their coordinatewise movements are

$$
p_{k\ell}=\sum_{i\in I_k}\Pi_{i\ell},\qquad
A_k(\Pi)=\sum_{i\in I_k,\ell}\Pi_{i\ell}\mathbf d_{i\ell}
\in\mathbb R_+^D.
$$

At a target receiving both classes, one score must serve both. The least weighted loss that this common score can achieve is

$$
\Psi(p_0,p_1)=\inf_s\{p_0\Phi(-s)+p_1\Phi(s)\},
\qquad H(\Pi)=\sum_\ell\Psi(p_{0\ell},p_{1\ell}).
\tag{5}
$$

The infimum permits limiting scores when one class has zero mass. $H$ measures the loss left after the best score is chosen at every target. It is nonnegative. Scaling all masses by $t\ge0$ scales $H$ by $t$; this property concerns the masses and does not require $\Phi$ itself to be homogeneous.

To use a matrix that may move more mass than the original budget allows, introduce a nonnegative correction vector $\mu\in\mathbb R_+^D$. The pair $(\Pi,\mu)$ must satisfy

$$
\Pi\ge0,\qquad \Pi\mathbf1=q,\qquad \mu\ge0,\qquad
\mathbf1^\top\bigl(A_k(\Pi)-\theta_k\mu\bigr)_+\le\theta_k,
\quad k=0,1.
\tag{6}
$$

Here $t_+=\max(t,0)$, applied coordinatewise to a vector. After $\theta_k\mu_j$ is deducted from movement in coordinate $j$, the remaining positive movements must fit inside the class budget $\theta_k$. These conditions contain no $z$: the same pair can be used at any selection. The matrix need not be a transport plan inside the original uncertainty balls. The correction is what makes it useful as a risk lower bound.

The reason for the correction in the denominator can be seen directly. For $0\le z\le1$, (6) implies $z^\top A_k(\Pi)\le\theta_k(1+\mu^\top z)$. For any feasible point of (2) on a binary selection,

$$
\begin{aligned}
H(\Pi)
&\le\sum_{i,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)\\
&\le q^\top a+\sum_k\lambda_k z^\top A_k(\Pi)\\
&\le J(a,\lambda)+(\mu^\top z)\theta^\top\lambda\\
&\le(1+\mu^\top z)J(a,\lambda).
\end{aligned}
$$

The four steps use, respectively, best-score loss, finite loss constraints, the corrected transport budgets, and (3). Thus $H(\Pi)/(1+\mu^\top z)$ is a lower bound on $V(z)$.

**Theorem 2 (Exact representation and convexity).**

For every $z\in[0,1]^D$,

$$
v(z)=\max_{\Pi,\mu\ \text{satisfying }(6)}
\frac{H(\Pi)}{1+\mu^\top z}.
\tag{7}
$$

The maximum is attained, $v$ is convex, and $v(z)=V(z)$ on binary selections.

The preceding inequalities explain why each fraction is a valid lower bound. Equality with $v$ requires more: minimizing the Lagrangian of (4) gives the row masses, transport inequalities and one normalization equation. Rescaling those multipliers gives exactly (7). Appendix B derives the objective and every constraint by eliminating one group of variables at a time. For fixed $(\Pi,\mu)$, the fraction is convex because its numerator is nonnegative and its denominator is a positive affine function. The maximum of these convex lower functions is therefore convex.

## 5. From lower bounds to a global algorithm

At a selection $z^t$, solving (4) and its dual gives a pair $(\Pi^t,\mu^t)$. Write $h_t=H(\Pi^t)$ for its remaining target loss and $r_t=1+(\mu^t)^\top z^t$ for the correction at the current selection. The function $f_t(z)=h_t/[1+(\mu^t)^\top z]$ is a lower bound everywhere. Its gradient at $z^t$ is $-h_t\mu^t/r_t^2$, so its tangent is

$$
L_t(z)=\frac{h_t}{r_t}
-\frac{h_t}{r_t^2}(\mu^t)^\top(z-z^t)\le v(z).
\tag{8}
$$

This affine inequality is the global cut. Convexity puts the tangent below $f_t$, and (7) puts $f_t$ below $v$. An optimal pair makes $L_t(z^t)=v(z^t)$. A merely feasible pair also gives a valid cut, using its own $h_t$. This argument does not assume that $v$ is differentiable.

Given the accumulated cuts, solve the discrete master problem

$$
\min_{z\in\mathcal Z_d,\eta}\ \eta
\qquad\text{subject to}\qquad
\eta\ge0,\quad \eta\ge L_t(z)\ \text{for every stored cut}.
\tag{9}
$$

It searches the entire set of allowed feature subsets. Each new cut changes the lower model for all subsets, while a continuous solve refits the entire test at the next candidate. The search is not confined to a neighborhood of the current subset or to a fixed combination of earlier tests.

**Algorithm 1: Global feature selection**

**Input:** Initial $z^0\in\mathcal Z_d$; tolerance $\varepsilon\ge0$.

1. Set $t=0$, $\mathrm{LB}=0$, $\mathrm{UB}=B$; retain $z^0$ and $g=0$ as the best feasible test.
2. **While** $\mathrm{UB}-\mathrm{LB}>\varepsilon$:
   1. At the current binary selection $z^t$, solve the full problem (4).
   2. Construct an original feasible test. If its risk bound is smaller than $\mathrm{UB}$, update $\mathrm{UB}$ and retain this selection and test.
   3. Obtain a feasible dual pair and add the cut (8) using its own numerator $h_t$.
   4. Solve (9); update $\mathrm{LB}$ with its globally valid solver bound, take its binary candidate as $z^{t+1}$ and set $t\gets t+1$.
   5. If the time or accuracy budget is exhausted, return the best test and the achieved bounds.
3. **Return:** Best feature subset and test, with interval $[\mathrm{LB},\mathrm{UB}]$.

**Theorem 3 (Bounds and exact finite termination).**

Every collection of valid cuts and every feasible original integer test give

$$
\mathrm{LB}\le\mathrm{OPT}\le\mathrm{UB}.
\tag{10}
$$

With exact continuous solutions, optimal dual pairs and globally solved masters, the algorithm terminates after finitely many different subsets. Termination with $\mathrm{UB}-\mathrm{LB}\le\varepsilon$ guarantees $\mathrm{UB}-\mathrm{OPT}\le\varepsilon$.

To see why, a previously solved subset has both a touching cut and an equally good feasible test. If an exact master selects it again, the global gap closes. Otherwise a new subset is evaluated, and there are only finitely many subsets. Appendix C proves the bound and termination statements in detail. Valid but nonoptimal certificates retain the bound guarantee, not the exact finite-termination argument. An interrupted master's global solver bound is usable; its current feasible objective is not a lower bound.

Adding available coordinates cannot worsen $V$, because a test can ignore them. The search may therefore use $\mathbf1^\top z=d$ when features have no additional charge. Cuts from fractional selections may strengthen the master, but only binary feasible tests update the original upper bound.

## 6. Loss choices and the scope of the result

The loss enters through $\Phi$ in the continuous problem and through $\Psi$ in the dual objective. Examples covered by the same derivation are shown below; boundary values use continuous limits.

| Loss | $\Phi(s)$ | $\Psi(p_0,p_1)$ |
| --- | --- | --- |
| Half hinge | $\tfrac12(1+s)_+$ | $\min(p_0,p_1)$ |
| Logistic | $\log(1+e^s)$ | $(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1$ |
| Exponential | $e^s$ | $2\sqrt{p_0p_1}$ |
| Squared hinge | $(1+s)_+^2$ | $4p_0p_1/(p_0+p_1)$ |

For example, half-hinge masses $p_0=0.2$ and $p_1=0.5$ leave loss $0.2$: the best common score favors the larger mass but cannot distinguish the smaller mass at the same location. Appendix D derives all four formulas. For half hinge, a score clipped to $[-1,1]$ gives randomized class-zero probability $(1+g)/2$, so $V/2$ is the equal-prior worst-case error. Other losses define surrogate objectives. The current implementation uses half hinge and logistic; the other rows illustrate the theoretical scope.

The loss formulas need not be fixed before deriving the algorithm. The transport geometry still matters: the whole-space construction uses the triangle inequality, and the coordinate-price formulation uses the $L^1$ decomposition. The certified interval concerns the stated robust optimization problem; whether its uncertainty sets cover an unknown population is a separate statistical question.

## Appendix A. Exact finite formulation

### A.1 From distributions to finite transports

Fix binary $z$ and $S=\{j:z_j=1\}$. Write $P_S$ for the distribution of the selected coordinates under $P$. Projection and lifting give

$$
\begin{aligned}
W_1(P_S,\widehat P_{k,S})&\le W_1(P,\widehat P_k),\\
\|x_i-(x_{\ell,S},x_{i,S^c})\|_1&=d_z(x_i,x_\ell),\\
g(x_{\ell,S},x_{i,S^c})&=g(x_\ell).
\end{aligned}
\tag{A.1}
$$

The lifted target retains the unselected source coordinates. Coincident projected sources are split according to their weights; the same construction applies to arbitrary target distributions and preserves finite first moments.

For finite scores $u$, linear programming duality yields

$$
\begin{aligned}
Q_k(u)
&=\max_{\substack{\Pi\ge0,\ \sum_\ell\Pi_{i\ell}=q_i\ (i\in I_k)\\
\sum_{i\in I_k,\ell}\Pi_{i\ell}d_z(x_i,x_\ell)\le\theta_k}}
\sum_{i\in I_k,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)\\
&=\min_{\substack{a_i\in\mathbb R,\ \lambda_k\ge0\\
\Phi(y_i u_\ell)\le a_i+\lambda_kd_z(x_i,x_\ell)\ (i\in I_k,\ \forall\ell)}}
\left[\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k\right].
\end{aligned}
\tag{A.2}
$$

The matrix in this calculation contains only rows $i\in I_k$, so the first feasible set is nonempty and compact. Let $T(z)$ be the infimum of (2). Applying (A.1) to the scores induced by any original test gives

$$
\begin{aligned}
T(z)&=\inf_u[Q_0(u)+Q_1(u)]\\
&\le\inf_g\sum_{k=0}^1\sup_{P\in\mathcal U_k}
\mathbb E_P\Phi((2k-1)g)=V(z).
\end{aligned}
\tag{A.3}
$$

Independent $u_\ell$ at coincident projected targets only relax this inequality.

### A.2 Constructing a test on the whole space

Take any feasible $(a,u,\lambda)$ in (2). Define

$$
E_k(x)=\min_{i\in I_k}\{a_i+\lambda_kd_z(x_i,x)\},\qquad
i_k(x)=\min\operatorname*{arg\,min}_{i\in I_k}\{a_i+\lambda_kd_z(x_i,x)\}.
\tag{A.4}
$$

At a fixed $x$, abbreviate $i_k=i_k(x)$ and set

$$
\begin{gathered}
c=d_z(x_{i_0},x_{i_1}),\qquad r_k=d_z(x_{i_k},x),\qquad c\le r_0+r_1,\\
\alpha=\begin{cases}\min(c,r_0)/c,&c>0,\\0,&c=0,\end{cases}
\qquad g(x)=(1-\alpha)u_{i_0}+\alpha u_{i_1},\\
0\le\alpha\le1,\qquad \alpha c\le r_0,\qquad (1-\alpha)c\le r_1.
\end{gathered}
\tag{A.5}
$$

The finite constraints at the two endpoints give

$$
\begin{aligned}
\Phi(-u_{i_0})&\le a_{i_0},&\Phi(-u_{i_1})&\le a_{i_0}+\lambda_0c,\\
\Phi(u_{i_0})&\le a_{i_1}+\lambda_1c,&\Phi(u_{i_1})&\le a_{i_1}.
\end{aligned}
\tag{A.6}
$$

Convexity and (A.5) imply, including $c=0$,

$$
\begin{aligned}
\Phi(-g(x))
&\le(1-\alpha)\Phi(-u_{i_0})+\alpha\Phi(-u_{i_1})
\le a_{i_0}+\lambda_0\alpha c\le E_0(x),\\
\Phi(g(x))
&\le(1-\alpha)\Phi(u_{i_0})+\alpha\Phi(u_{i_1})
\le a_{i_1}+\lambda_1(1-\alpha)c\le E_1(x).
\end{aligned}
\tag{A.7}
$$

The smallest-index choice uses finite comparisons of continuous functions, so $g$ is Borel measurable. It depends only on $x_S$ and satisfies $\min_\ell u_\ell\le g(x)\le\max_\ell u_\ell$; interpolation of every prescribed $u_\ell$ is not required.

For a coupling $\gamma_k$ from $\widehat P_k$ to $P\in\mathcal U_k$ with cost at most $\theta_k$,

$$
\begin{aligned}
\mathbb E_P\Phi((2k-1)g)
&\le\sum_{i\in I_k}q_i a_i+\lambda_k\int d_z(x_i,x)\,d\gamma_k
\le\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k,\\
\sum_{k=0}^1\sup_{P\in\mathcal U_k}\mathbb E_P\Phi((2k-1)g)
&\le J(a,\lambda)
\quad\Longrightarrow\quad V(z)\le T(z).
\end{aligned}
\tag{A.8}
$$

Together with (A.3), this proves $T(z)=V(z)$. Attainment follows below.

## Appendix B. Deriving the dual

### B.1 Existence and strong duality

Convexity, monotonicity and nonconstancy give $s_1<s_2$ with $b=[\Phi(s_2)-\Phi(s_1)]/(s_2-s_1)>0$, and hence

$$
\Phi(t)\ge\Phi(s_2)+b(t-s_2)\quad(t\ge s_2),
\qquad\lim_{t\to\infty}\Phi(t)=\infty.
\tag{B.1}
$$

The constant test is feasible with value $B$. At feasible points with $J\le B$ in (2), or $J\le\eta\le B$ in (4),

$$
\begin{gathered}
0\le a_i\le B/q_i,\qquad0\le\lambda_k\le B/\theta_k,
\qquad0\le w_{kj}\le B/\theta_k,\\
\Phi(y_i u_\ell)\le B/q_i+(B/\theta_{c_i})\|\mathbf d_{i\ell}\|_1.
\end{gathered}
\tag{B.2}
$$

Using one source from each class bounds both $\Phi(u_\ell)$ and $\Phi(-u_\ell)$; (B.1) bounds $u_\ell$. These feasible sets are nonempty, closed and bounded, so both minima are attained.

For fixed $z\in[0,1]^D$, delete $w_{kj}=0$ where $z_j=0$. A strictly feasible point for the remaining constraints is

$$
\begin{gathered}
u=0,\quad a_i=\Phi(0)+1,\quad\lambda_k=1,\quad w_k=\delta z,\quad0<\delta<1,\\
\eta>J+\delta(\theta_0+\theta_1),\qquad
\theta_0w_{0j}+\theta_1w_{1j}<\eta z_j\quad(z_j>0).
\end{gathered}
\tag{B.3}
$$

Finite-dimensional convex strong duality therefore gives zero duality gap and attained multipliers. The deleted coordinates can be restored by sufficiently large nonnegative coordinate multipliers; their coefficients in $\bar\mu^\top z$ are zero.

### B.2 Eliminating the primal variables

For (4), assign nonnegative multipliers $\bar\Pi_{i\ell}$ to the loss constraints, $\tau$ to $J-\eta$, $\bar\beta_k$ to $w_k-\lambda_k\mathbf1$, and $\bar\mu$ to $\theta_0w_0+\theta_1w_1-\eta z$. With $\bar p_{k\ell}=\sum_{i\in I_k}\bar\Pi_{i\ell}$,

$$
\begin{aligned}
\mathcal L={}&\eta+\tau(J-\eta)
+\sum_{i,\ell}\bar\Pi_{i\ell}[\Phi(y_i u_\ell)-a_i-\mathbf d_{i\ell}^{\top}w_{c_i}]\\
&+\sum_k\bar\beta_k^{\top}(w_k-\lambda_k\mathbf1)
+\bar\mu^{\top}(\theta_0w_0+\theta_1w_1-\eta z)\\
={}&\eta(1-\tau-\bar\mu^{\top}z)
+\sum_i a_i(\tau q_i-\sum_\ell\bar\Pi_{i\ell})\\
&+\sum_\ell[\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)]\\
&+\sum_k\lambda_k(\tau\theta_k-\mathbf1^{\top}\bar\beta_k)
+\sum_k w_k^{\top}[\bar\beta_k+\theta_k\bar\mu-A_k(\bar\Pi)].
\end{aligned}
\tag{B.4}
$$

The elementary rules are

$$
\inf_{t\in\mathbb R}bt=\begin{cases}0,&b=0,\\-\infty,&b\ne0,\end{cases}
\qquad
\inf_{t\ge0}bt=\begin{cases}0,&b\ge0,\\-\infty,&b<0.\end{cases}
$$

Applying them to each variable group gives

$$
\begin{aligned}
\inf_{\eta\in\mathbb R}\mathcal L> -\infty
&\ \Longleftrightarrow\ \tau+\bar\mu^{\top}z=1,\\
\inf_{a\in\mathbb R^N}\mathcal L> -\infty
&\ \Longleftrightarrow\ \bar\Pi\mathbf1=\tau q,\\
\inf_{\lambda\ge0}\mathcal L> -\infty
&\ \Longleftrightarrow\ \mathbf1^{\top}\bar\beta_k\le\tau\theta_k,\\
\inf_{w_k\ge0}\mathcal L> -\infty
&\ \Longleftrightarrow\ A_k(\bar\Pi)\le\bar\beta_k+\theta_k\bar\mu.
\end{aligned}
\tag{B.5}
$$

The scores contribute precisely the dual objective:

$$
\inf_{u\in\mathbb R^N}\sum_\ell
[\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)]
=\sum_\ell\Psi(\bar p_{0\ell},\bar p_{1\ell})=H(\bar\Pi).
\tag{B.6}
$$

For any $\varepsilon>0$, choose each score with scalar loss within $\varepsilon/N$ of its infimum; then the sum lies in $[H(\bar\Pi),H(\bar\Pi)+\varepsilon]$. Thus (B.6) also holds when a scalar infimum is not attained. The domains of $a$ and $\eta$ remain free; their implied nonnegativity is not added as a separate domain constraint.

The allocation multipliers disappear by an exact equivalence:

$$
\begin{aligned}
\exists\bar\beta_k\ge0:\quad
A_k(\bar\Pi)&\le\bar\beta_k+\theta_k\bar\mu,\quad
\mathbf1^{\top}\bar\beta_k\le\tau\theta_k\\
&\Longleftrightarrow\quad
\mathbf1^{\top}(A_k(\bar\Pi)-\theta_k\bar\mu)_+\le\tau\theta_k.
\end{aligned}
\tag{B.7}
$$

For the reverse direction, take $\bar\beta_k=(A_k(\bar\Pi)-\theta_k\bar\mu)_+$. The minimized Lagrangian is $H(\bar\Pi)$ under (B.5), and $-\infty$ otherwise. Therefore

$$
\begin{aligned}
v(z)=\max_{\tau,\bar\Pi,\bar\mu\ge0}\quad&H(\bar\Pi)\\
\text{subject to}\quad&\bar\Pi\mathbf1=\tau q,\qquad\tau+\bar\mu^{\top}z=1,\\
&\mathbf1^{\top}(A_k(\bar\Pi)-\theta_k\bar\mu)_+\le\tau\theta_k,
\quad k=0,1.
\end{aligned}
\tag{B.8}
$$

### B.3 Removing the normalization

For $\tau>0$, set $\Pi=\bar\Pi/\tau$ and $\mu=\bar\mu/\tau$. The constraints become (6), while

$$
H(\tau\Pi)=\sum_\ell\inf_s\tau[p_{0\ell}\Phi(-s)+p_{1\ell}\Phi(s)]=\tau H(\Pi).
$$

Consequently,

$$
\tau+\bar\mu^{\top}z=1
\ \Longleftrightarrow\ \tau=\frac1{1+\mu^{\top}z},\qquad
H(\bar\Pi)=\frac{H(\Pi)}{1+\mu^{\top}z}.
\tag{B.9}
$$

Conversely, every pair in (6) scales to a feasible point of (B.8).

To exclude $\tau=0$ at an optimum, fix a target $\ell_*$ and set

$$
\begin{gathered}
\Pi^0=q e_{\ell_*}^{\top},\qquad
H(\Pi^0)=\Psi(1,1)=2\Phi(0)=B>0,\\
\mu^0_j=\max_{k=0,1}\frac{A_{kj}(\Pi^0)}{\theta_k},\qquad
\tau^0=\frac1{1+(\mu^0)^{\top}z},\\
\bar\Pi^0=\tau^0\Pi^0,\qquad\bar\mu^0=\tau^0\mu^0,
\qquad H(\bar\Pi^0)=\tau^0B>0.
\end{gathered}
\tag{B.10}
$$

Here $e_{\ell_*}$ is the unit vector at the chosen target, and $\Psi(1,1)=B$ follows from $\Phi(-s)+\Phi(s)\ge2\Phi(0)$, with equality at $s=0$. In contrast,

$$
\tau=0\ \Longrightarrow\ \bar\Pi\mathbf1=0\ \Longrightarrow\ \bar\Pi=0\ \Longrightarrow\ H(\bar\Pi)=0.
$$

An attained normalized optimum therefore has $\tau>0$, proving the attained maximum in (7).

For binary $z$, the two primal mappings are

$$
\begin{aligned}
(2)\text{ feasible}
&\ \Longrightarrow\ \eta=J,\ w_k=\lambda_kz\text{ satisfy }(4),\\
(4)\text{ feasible}
&\ \Longrightarrow\ w_k\le\lambda_kz,\ (a,u,\lambda)\text{ satisfy }(2),\ J\le\eta.
\end{aligned}
\tag{B.11}
$$

Together with (3), this gives $v(z)=V(z)$.

## Appendix C. Global cuts and termination

For any feasible pair in (6), let $h=H(\Pi)\ge0$ and $f(z)=h/(1+\mu^{\top}z)$. Then

$$
\nabla f(z)=-\frac{h\mu}{(1+\mu^{\top}z)^2},\qquad
\nabla^2f(z)=\frac{2h\mu\mu^{\top}}{(1+\mu^{\top}z)^3}\succeq0.
\tag{C.1}
$$

The maximum in (7) is convex. At $z^t$, writing $r_t=1+\mu^{\top}z^t$ gives

$$
v(z)\ge f(z)\ge
f(z^t)+\nabla f(z^t)^{\top}(z-z^t)
=\frac h{r_t}-\frac{h\mu^{\top}(z-z^t)}{r_t^2}=L_t(z).
\tag{C.2}
$$

For an optimal pair, $L_t(z^t)=v(z^t)$. A nonoptimal feasible pair still gives a valid cut using its own $h$; replacing $h$ by any verified value in $[0,H(\Pi)]$ also preserves validity.

Let $m_t(z)=\max(0,L_0(z),\ldots,L_t(z))$. The master and stored original feasible tests give

$$
\mathrm{LB}=\min_{z\in\mathcal Z_d}m_t(z)
\le\min_{z\in\mathcal Z_d}V(z)=\mathrm{OPT}
\le\mathrm{UB}.
\tag{C.3}
$$

Under exact continuous and discrete solves, if the master selects a previously evaluated $z^s$,

$$
\mathrm{LB}=m_t(z^s)\ge L_s(z^s)=V(z^s)\ge\mathrm{UB}.
\tag{C.4}
$$

Equations (C.3)--(C.4) imply $\mathrm{LB}=\mathrm{OPT}=\mathrm{UB}$. Otherwise the next subset is new; $|\mathcal Z_d|<\infty$ proves finite termination. Finally,

$$
\mathrm{UB}-\mathrm{LB}\le\varepsilon
\ \Longrightarrow\ 0\le\mathrm{UB}-\mathrm{OPT}\le\varepsilon.
$$

With inexact or interrupted solves, use a valid global solver bound for $\mathrm{LB}$ and an original feasible test for $\mathrm{UB}$; the bounds remain valid, while exact finite termination requires the stated exact solves.

## Appendix D. The loss formulas

For $p_0,p_1>0$, the four rows in Section 6 follow from the following scalar calculations.

For half hinge, clipping to $[-1,1]$ cannot increase either loss:

$$
\Psi(p_0,p_1)=\min_{|s|\le1}
\left[\frac{p_0+p_1}{2}+\frac{p_1-p_0}{2}s\right]=\min(p_0,p_1).
\tag{D.1}
$$

For logistic,

$$
\begin{aligned}
\frac d{ds}[p_0\log(1+e^{-s})+p_1\log(1+e^s)]
&=\frac{-p_0+p_1e^s}{1+e^s}=0
\quad\Longrightarrow\quad s^*=\log(p_0/p_1),\\
\Psi(p_0,p_1)&=(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1.
\end{aligned}
\tag{D.2}
$$

For exponential,

$$
p_0e^{-s}+p_1e^s\ge2\sqrt{p_0p_1},\qquad
s^*=\tfrac12\log(p_0/p_1),\qquad\Psi(p_0,p_1)=2\sqrt{p_0p_1}.
\tag{D.3}
$$

For squared hinge, clipping again permits $|s|\le1$, and

$$
\begin{aligned}
p_0(1-s)^2+p_1(1+s)^2
&=(p_0+p_1)\left(s-\frac{p_0-p_1}{p_0+p_1}\right)^2
+\frac{4p_0p_1}{p_0+p_1},\\
s^*&=\frac{p_0-p_1}{p_0+p_1}\in[-1,1],\qquad
\Psi(p_0,p_1)=\frac{4p_0p_1}{p_0+p_1}.
\end{aligned}
\tag{D.4}
$$

Each of these losses has $\inf_s\Phi(s)=0$, so $\Psi(p,0)=\Psi(0,p)=0$ for $p\ge0$ and $\Psi(0,0)=0$, with $0\log0=0$. A boundary value may be an infimum without a finite minimizing score.
