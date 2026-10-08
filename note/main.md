# Certified Feature Selection for Wasserstein Robust Hypothesis Testing

8 October 2026.
## Abstract

We study feature selection for binary hypothesis testing under class-conditional Wasserstein uncertainty. The objective jointly chooses a budgeted feature subset and a measurable test to minimize the sum of worst-case classwise losses. For half-hinge and logistic losses with positive uncertainty radii and an $L^1$ transport cost, we derive an exact finite convex formulation and construct a valid whole-space test from every feasible solution. Our key step uses the optimized risk to scale the admissible coordinatewise transport prices. A normalized homogeneous dual then expresses the relaxed value as a maximum of convex certificate fractions. This representation is exact at binary selections, strengthens fixed-bound relaxations, and yields affine lower bounds valid across all feature subsets. Combining these bounds with feasible tests gives a global-cut algorithm that maintains certified lower and upper bounds on the optimal selection risk. Under exact oracle and master solves, it terminates after finitely many distinct subsets. These are optimization guarantees within the specified uncertainty sets; population error control requires separate calibration.

## 1. Problem and assumptions

For classes $k\in\{0,1\}$, let
\[
\widehat P_k=\sum_{i=1}^{n_k}q_{ki}\delta_{x_{ki}},\qquad
x_{ki}\in\mathbb R^D,\quad q_{ki}>0,\quad\sum_iq_{ki}=1.
\tag{1.1}
\]
The uncertainty set is
\[
\mathcal U_k=\{P\in\mathcal P_1(\mathbb R^D):W_1(P,\widehat P_k)\le\theta_k\},
\qquad \theta_k>0,
\tag{1.2}
\]
where $\mathcal P_1$ denotes distributions with finite first moment and $W_1$ uses the full-space $L^1$ cost. Write $\theta=(\theta_0,\theta_1)^\top$. Coordinate scales and radii are fixed before selection. For an integer budget $0\le d\le D$, define
\[
\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\},\qquad
S(z)=\{j:z_j=1\}.
\tag{1.3}
\]
A measurable score $g$ depends only on $x_{S(z)}$. Positive scores favor class $0$; write $\sigma_0=-1$ and $\sigma_1=1$. Throughout, $\Phi$ is one of the two losses
\[
\Phi_{\rm hh}(v)=\tfrac12(1+v)_+,
\qquad \Phi_{\log}(v)=\log(1+e^v).
\tag{1.4}
\]
The fixed-support risk and selection objective are
\[
V(z)=\inf_{g=g(x_{S(z)})}\sum_{k=0}^1\sup_{P\in\mathcal U_k}
\mathbb E_P\Phi(\sigma_kg),\qquad
\mathrm{OPT}=\min_{z\in\mathcal Z_d}V(z).
\tag{1.5}
\]
This is a sum of classwise risks, rather than a maximum-error criterion. For half hinge, clipping $g$ to $[-1,1]$ cannot increase either loss. Then $r(x)=(1+g(x))/2$ is the probability of declaring class $0$, and $V(z)/2$ is the corresponding optimal worst-case equal-prior randomized error. Logistic risk is a surrogate objective.

The constant score $g=0$ gives the uniform benchmark
\[
0\le V(z)\le B:=2\Phi(0),\qquad
B=1\ \text{for half hinge},\quad B=2\log2\ \text{for logistic}.
\tag{1.6}
\]
Our construction specializes the logical-constraint framework of Bertsimas, Cory-Wright, and Pauphilet [1] to this testing objective and then uses its risk bound to strengthen the block domain. The testing setting is related to the Wasserstein formulations in [2, 3]; the risk normalization and assumptions are specified by (1.1)--(1.5).

## 2. An exact finite testing formulation

Pool the source observations into $t_1,\ldots,t_N$, where $N=n_0+n_1$; repeated points are allowed. Define
\[
\Delta_{kimj}=|x_{ki,j}-t_{m,j}|,\qquad
d_z(x,y)=\sum_jz_j|x_j-y_j|.
\tag{2.1}
\]
The distance $d_z$ is the projected transport cost for a binary support. Projection of a full-space coupling cannot increase cost, and a projected coupling can be lifted by retaining each source atom's unselected coordinates. Thus a selected-coordinate test sees exactly the projected Wasserstein balls, with the original radii.

Let $u_m$ be a target score, $a_{ki}$ a source intercept, and $\lambda_k$ a nonnegative transport price. Put
\[
J(a,\lambda)=\sum_{ki}q_{ki}a_{ki}+\sum_k\theta_k\lambda_k.
\tag{2.2}
\]
**Theorem 1 (Exact finite representation).**

Under the assumptions of Section 1, for every binary $z$,
\[
\begin{aligned}
V(z)=\min_{a,u,\lambda}\quad &J(a,\lambda)\\
\text{subject to}\quad &\Phi(\sigma_ku_m)
\le a_{ki}+\lambda_k\sum_jz_j\Delta_{kimj},\quad\forall k,i,m,\\
&\lambda_k\ge0,\quad a_{ki},u_m\in\mathbb R.
\end{aligned}
\tag{2.3}
\]
The minimum is attained. Every feasible solution extends to a measurable selected-coordinate test whose original robust risk is at most $J(a,\lambda)$.

Appendix A proves equality by finite adversarial LP duality and a constructive whole-space extension; it does not exchange an infinite-dimensional infimum and supremum. The extension uses the envelopes
\[
E_k(x)=\min_i\{a_{ki}+\lambda_kd_z(x_{ki},x)\}
\tag{2.4}
\]
to bound both class losses. This connects a finite feasible solution to a deployable test, which is essential when using its objective as an upper bound.

Every source occurs among the targets. Its zero-distance constraint gives $a_{ki}\ge\Phi(\sigma_ku_m)\ge0$. Consequently, on the constant-test sublevel,
\[
J(a,\lambda)\le B
\quad\Longrightarrow\quad
\theta^\top\lambda\le J(a,\lambda)\le B,
\qquad \lambda_k\le U_k:=B/\theta_k.
\tag{2.5}
\]
This is an analytically certified bound on relevant optimal prices, not an assumption about every feasible $\lambda$. It is valid even when $B$ is coarse. Discarding candidates with $J>B$ cannot remove the optimum, because the constant test remains available.

## 3. Logical gates and objective-scaled duality

### 3.1 Fixed block domains

Introduce a two-component price block $w_j=(w_{0j},w_{1j})$. The common convex constraints are
\[
\mathcal K=\left\{(a,u,\lambda,w):
\begin{array}{l}
a,u\text{ free},\quad \lambda,w\ge0,\quad w_{kj}\le\lambda_k,\\
\Phi(\sigma_ku_m)\le a_{ki}+\sum_j\Delta_{kimj}w_{kj},\quad\forall k,i,m
\end{array}\right\}.
\tag{3.1}
\]
The logical condition is $z_j=0\Rightarrow w_j=0$. Equation (2.5) supplies two bounded versions:
\[
\begin{aligned}
G_{\rm box}(z)&=\min\{J:(a,u,\lambda,w)\in\mathcal K,\\
&\qquad w_{kj}\le U_kz_j\}.
\end{aligned}
\tag{3.2}
\]

\[
\begin{aligned}
R_B(z)&=\min\{J:(a,u,\lambda,w)\in\mathcal K,\\
&\qquad \theta^\top w_j\le Bz_j\}.
\end{aligned}
\tag{3.3}
\]
No additional constraint $\lambda_k\le U_k$ is imposed in (3.2). For a binary support, an original optimizer enters either model with $w_{kj}=\lambda_kz_j$. Conversely, any gated feasible point satisfies $w_{kj}\le\lambda_kz_j$; increasing $w$ to $\lambda z$ relaxes the loss inequalities. Hence both gates preserve the integer optimal value. Appendix C gives the two directional proofs, including a near-optimal-sequence argument independent of attainment.

For fractional $z$, (3.2) and (3.3) define continuous extensions. They are not asserted to equal the original product model with a fractional $z$ substituted into $\lambda_kz_j$. A larger valid $U_k$ retains integer exactness while weakening the box extension.

To express the certificates, let $\Gamma_{kim}\ge0$ denote source-to-target masses and define
\[
p_{km}(\Gamma)=\sum_i\Gamma_{kim},\qquad
A_{kj}(\Gamma)=\sum_{im}\Gamma_{kim}\Delta_{kimj}.
\tag{3.4}
\]
The residual loss at a target and its aggregate are
\[
\psi(p_0,p_1)=\inf_v\{p_0\Phi(-v)+p_1\Phi(v)\},\qquad
\mathcal H(\Gamma)=\sum_m\psi(p_{0m},p_{1m}).
\tag{3.5}
\]
For half hinge, $\psi(p_0,p_1)=\min(p_0,p_1)$; for logistic,
$\psi=(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1$, with $0\log0=0$.
Both are nonnegative, concave, and homogeneous of degree one. Thus $\mathcal H$ measures unresolved class overlap, whereas $A_{kj}$ measures coordinatewise movement.

The fixed-triangle certificate domain is
\[
\mathcal D=\left\{(\Gamma,\beta,\mu)\ge0:
\begin{array}{l}
\sum_m\Gamma_{kim}=q_{ki},\quad \sum_j\beta_{kj}\le\theta_k,\\
A_{kj}(\Gamma)\le\beta_{kj}+\theta_k\mu_j
\end{array}\right\}.
\tag{3.6}
\]
Here $\beta$ allocates each class's transport allowance, and $\mu$ covers the normalized residual in each coordinate block. These are expanded dual certificates: their row masses are normalized, but $\Gamma$ need not satisfy the original adversarial transport budgets at a given support.

**Proposition 2 (Fixed logical extensions).**

For $z\in[0,1]^D$, the fixed-triangle value admits the attained representation
\[
R_B(z)=\max_{(\Gamma,\beta,\mu)\in\mathcal D}
\{\mathcal H(\Gamma)-B\mu^\top z\}.
\tag{3.7}
\]
Both $G_{\rm box}$ and $R_B$ are convex continuous extensions satisfying
$G_{\rm box}(z)=R_B(z)=V(z)$ at binary supports, and $R_B\ge G_{\rm box}$ on the continuous box.

The underlying block domain is $C_B=\{w\ge0:\theta^\top w\le B\}$, whose support function is
$B\max(0,\alpha_0/\theta_0,\alpha_1/\theta_1)$. Applying the conjugate mechanism of [1, Theorem 2.5] produces (3.7); Appendix C derives this specialization and the independent box dual explicitly. Because $\mathcal D$ does not depend on $z$, each certificate in (3.7) gives a global affine lower bound. An optimal certificate makes that bound touch the risk at its generating binary support.

### 3.2 Replacing the constant allowance by the optimized risk

The stronger inequality in (2.5) is $\theta^\top\lambda\le J$, not merely $\theta^\top\lambda\le B$. It motivates an objective-scaled gate:
\[
R(z)=\min_{\eta,a,u,\lambda,w}
\{\eta:(a,u,\lambda,w)\in\mathcal K,\ J\le\eta,
\ \theta^\top w_j\le\eta z_j\ \forall j\}.
\tag{3.8}
\]
The variable $\eta$ is free; feasibility implies $\eta\ge0$. At a binary support, an original optimizer satisfies this gate with $\eta=J$ and $w=\lambda z$. The reverse saturation argument recovers an original test with objective at most $\eta$. The model is convex for fixed $z$, but $\eta z_j$ prevents joint convexity in the selection and oracle variables. Convexity of its value function therefore requires a separate argument.

**Theorem 3 (Normalized dual and stronger convex extension).**

For every $z\in[0,1]^D$, the primal (3.8) and its dual attain equal values. The dual is
\[
\begin{aligned}
R(z)=\max_{\tau,\widehat\Gamma,\widehat\beta,\widehat\mu\ge0}\quad&
\mathcal H(\widehat\Gamma)\\
\text{subject to}\quad&\sum_m\widehat\Gamma_{kim}=\tau q_{ki},\qquad
\sum_j\widehat\beta_{kj}\le\tau\theta_k,\\
&A_{kj}(\widehat\Gamma)\le\widehat\beta_{kj}+\theta_k\widehat\mu_j,\\
&\tau+\widehat\mu^\top z=1.
\end{aligned}
\tag{3.9}
\]
Equivalently,
\[
R(z)=\max_{(\Gamma,\beta,\mu)\in\mathcal D}
\frac{\mathcal H(\Gamma)}{1+\mu^\top z}.
\tag{3.10}
\]
The function $R$ is convex, $R(z)=V(z)$ at binary supports, and
\[
R(z)\ge R_B(z)\ge G_{\rm box}(z),\qquad z\in[0,1]^D.
\tag{3.11}
\]
The multiplier $\tau$ prices $J\le\eta$; elimination of $\eta$ yields the normalization in (3.9). Homogeneity then turns the fixed penalty in (3.7) into the fraction in (3.10). For a fixed certificate, $h=\mathcal H(\Gamma)\ge0$ and $\mu\ge0$ make $h/(1+\mu^\top z)$ convex. Taking a maximum over the common domain proves value-function convexity. Moreover, $h\le B$ implies
$h/(1+s)\ge h-Bs$ for $s\ge0$, proving the first comparison in (3.11). Appendix D supplies the full Lagrangian derivation and an independent inequality tying each fraction directly to the original risk.

Half-hinge oracles are linear programs; logistic oracles are finite convex programs with exponential-cone representations. The strengthening is specific: low-risk solutions receive a smaller transport-price allowance without adding a penalty to the testing objective. Pointwise dominance of complete extensions does not imply dominance of arbitrary finite cut collections or a universal runtime improvement.

## 4. Certified global search

At an anchor $z^t$, take a feasible normalized certificate with $\tau_t>0$ and let
$v_t=\mathcal H(\widehat\Gamma^t)$. Rescaling that certificate at any new $z$ gives the global lower function
\[
f_t(z)=\frac{v_t}{\tau_t+(\widehat\mu^t)^\top z}\le R(z).
\tag{4.1}
\]
Its denominator equals one at the anchor. Its tangent therefore gives
\[
L_t(z)=v_t-v_t(\widehat\mu^t)^\top(z-z^t)\le R(z).
\tag{4.2}
\]
A zero-scale certificate has zero objective and contributes the zero cut. If the certificate is optimal, $v_t=R(z^t)$ and the cut touches $R$ there. A merely feasible certificate gives a valid, possibly non-tight cut using its own dual value; a primal oracle objective cannot be substituted for $v_t$. No differentiability of $R$ is required.

Given cuts indexed by $\mathcal T$, the master is
\[
\min_{z,\eta}\{\eta:z\in\mathcal Z_d,\ \eta\ge0,
\ \eta\ge L_t(z)\ \forall t\in\mathcal T\}.
\tag{4.3}
\]
At integer candidates, a feasible finite test provides an upper bound after the extension in Theorem 1. Fractional anchors can strengthen the master but do not supply deployable integer-support upper bounds.

**Algorithm 1. Global selection with full convex oracles.**

Input: $z^0\in\mathcal Z_d$ and $\varepsilon\ge0$. Initialize $(z^{\rm inc},g^{\rm inc})$ as $z^0$ and the constant test $g\equiv0$, with $\underline v=0$, $\bar v=B$, $\mathcal T=\varnothing$, and $t=0$.

While $\bar v-\underline v>\varepsilon$:

1. Reoptimize the continuous oracle at $z^t$. Obtain a feasible original test $g_t$ with certified risk $b_t$ and a feasible normalized certificate.
2. If $b_t<\bar v$, store $(z^{\rm inc},g^{\rm inc})=(z^t,g_t)$ before updating $\bar v=\min(\bar v,b_t)$. Add the certificate's cut (4.2).
3. Solve the integer master (4.3), obtaining a global lower bound $\ell_t$ and an integer candidate $z^{t+1}$.
4. Update $\underline v=\max(\underline v,\ell_t)$ and increment $t$.
5. If the time or accuracy budget is exhausted, return the achieved bounds and status.

Return $(z^{\rm inc},g^{\rm inc})$ and $[\underline v,\bar v]$.

**Theorem 4 (Global bounds and ideal finite termination).**

Every collection of cuts (4.2) gives a valid master lower bound, and every feasible original integer-support test gives a valid upper bound:
\[
\underline v\le\mathrm{OPT}\le\bar v.
\tag{4.4}
\]
With optimal oracle certificates, exact primal oracle solutions, and globally solved masters, Algorithm 1 terminates after finitely many distinct supports. Any tolerance-based termination certifies $\bar v-\mathrm{OPT}\le\varepsilon$.
The proof is in Appendix E. A previously evaluated support has a touching cut and an equally good feasible test; if an exact master chooses it again, the global gap closes. Otherwise a new support is evaluated. Adding coordinates cannot worsen $V$, since a test may ignore them, so the search may use $\mathbf1^\top z=d$ when there is no feature charge. Full reoptimization at each candidate is essential; the method does not restrict later tests to combinations of earlier fitted tests.

For an interrupted master, its globally valid solver bound is usable; its incumbent objective is not a lower bound. Numerical certificates must satisfy the stated constraints, and non-tight certificates do not inherit the exact finite-termination claim. Optional fractional root cuts and selection among dual-optimal cuts leave the target unchanged. Appendix F describes a supplementary master that retains entire certificate fractions rather than only tangents.

## 5. Scope

The results concern the two losses in (1.4), positive radii, fixed feature scales, and full-space $L^1$ transport. The finite reduction uses a loss-specific whole-space extension; it is not claimed for arbitrary convex losses or restricted ambient supports. The price bounds and normalization are not obtained by substituting zero for a radius. Worst-case integer search remains exponential.

A Wasserstein certificate controls distributions inside the specified balls. It does not establish that an unknown population lies there, guarantee a prescribed Type-I level, or identify a unique scientific signal. These statistical questions and empirical runtime comparisons require separate evaluation. The objective-scaled block construction and its normalized dual provide the mathematical basis for the selection algorithm; empirical comparisons are documented separately.

## Appendix A. Projection, finite adversaries, and whole-space tests

### A.1 Projection of the uncertainty sets

Let $\Pi_Sx=x_S$. Projection of any coupling gives
\[
W_1((\Pi_S)_\#P,(\Pi_S)_\#\widehat P_k)
\le W_1(P,\widehat P_k).
\tag{A.1}
\]
Conversely, disaggregate a projected empirical marginal by source identity if projected atoms coincide. For mass originating at $x_{ki}$, retain $x_{ki,S^c}$ and move only its selected coordinates. This lifts any projected coupling to $\mathbb R^D$ at exactly the projected cost. Thus the projected uncertainty set is precisely the Wasserstein ball centered at $(\Pi_S)_\#\widehat P_k$ with radius $\theta_k$. The construction uses the full ambient space.

### A.2 Finite transport duality

For fixed finite scores $u$, restrict class $k$'s adversary to the target pool. Its value is
\[
\begin{aligned}
Q_k(u)=\max_{\Gamma_k\ge0}\quad&\sum_{im}\Gamma_{kim}\Phi(\sigma_ku_m)\\
\text{subject to}\quad&\sum_m\Gamma_{kim}=q_{ki},\qquad
\sum_{im}\Gamma_{kim}d_z(x_{ki},t_m)\le\theta_k.
\end{aligned}
\tag{A.2}
\]
Mass can stay at its own source, so the LP is feasible. Its mass set is bounded, and its objective is finite. Free row multipliers $a_{ki}$ and a nonnegative budget multiplier $\lambda_k$ give
\[
\begin{aligned}
\mathcal L_k={}&\sum_iq_{ki}a_{ki}+\theta_k\lambda_k\\
&+\sum_{im}\Gamma_{kim}
[\Phi(\sigma_ku_m)-a_{ki}-\lambda_kd_z(x_{ki},t_m)].
\end{aligned}
\tag{A.3}
\]
The supremum over $\Gamma_k\ge0$ is finite exactly when all bracketed coefficients are nonpositive. LP strong duality therefore identifies $Q_k(u)$ with the classwise part of (2.3). Minimizing $Q_0+Q_1$ jointly over $u,a,\lambda$ gives a finite value $T(z)$ equal to that program's infimum. Every original test induces pool scores, and a pool-supported adversary is admissible, so
\[
T(z)\le V(z).
\tag{A.4}
\]
Allowing different scores at targets with identical projections can only relax this inequality. The following extension establishes the reverse inequality and produces a consistent selected-coordinate score.

### A.3 Extension of a feasible finite test

Take a feasible $(a,u,\lambda)$ and the envelopes (2.4). We construct $g$ satisfying
\[
\Phi(-g(x))\le E_0(x),\qquad \Phi(g(x))\le E_1(x),\qquad x\in\mathbb R^D.
\tag{A.5}
\]
For a class-0 source $x_{0i}$ and a class-1 source $x_{1h}$, put
$c=d_z(x_{0i},x_{1h})$, $r_0=d_z(x_{0i},x)$, and $r_1=d_z(x_{1h},x)$. Then $r_0+r_1\ge c$.

For logistic, the identity $e^{-\Phi(-u)}+e^{-\Phi(u)}=1$ and the finite constraints at the two sources imply
\[
e^{-a_{0i}}+e^{-a_{1h}-\lambda_1c}\le1,
\qquad e^{-a_{0i}-\lambda_0c}+e^{-a_{1h}}\le1.
\tag{A.6}
\]
Choose $r\in[0,c]$ with $r\le r_0$ and $c-r\le r_1$, possible because $r_0+r_1\ge c$. The function
\[
F(r)=e^{-a_{0i}-\lambda_0r}+e^{-a_{1h}-\lambda_1(c-r)}
\tag{A.7}
\]
is convex and hence at most the larger of its endpoint values. By (A.6) and monotonicity in distance,
$e^{-a_{0i}-\lambda_0r_0}+e^{-a_{1h}-\lambda_1r_1}\le1$.
Taking the maximum over source pairs gives
\[
e^{-E_0(x)}\le r(x)\le1-e^{-E_1(x)}.
\tag{A.8}
\]
This interval is nonempty. Its midpoint and $g(x)=\log(r(x)/(1-r(x)))$ satisfy (A.5). For finite logistic scores the self-target constraints give $a_{ki}>0$, so the midpoint lies strictly between zero and one. The envelopes are finite minima of continuous selected-coordinate functions; $g$ is finite and measurable.

For half hinge, the loss sum is at least one. The two source targets give
\[
a_{0i}+a_{1h}+\lambda_1c\ge1,
\qquad a_{0i}+a_{1h}+\lambda_0c\ge1.
\tag{A.9}
\]
Therefore
$a_{0i}+a_{1h}+\lambda_0r_0+\lambda_1r_1
\ge a_{0i}+a_{1h}+\min(\lambda_0,\lambda_1)c\ge1$.
Minimizing over source pairs yields $E_0+E_1\ge1$. Since $E_k\ge0$, the interval
\[
\max(0,1-E_0(x))\le r(x)\le\min(1,E_1(x))
\tag{A.10}
\]
is nonempty. Its midpoint and $g(x)=2r(x)-1$ satisfy (A.5) and are measurable.

For either loss, integrate the envelope bound against a coupling from the empirical class to an admissible $P$:
\[
\mathbb E_P\Phi(\sigma_kg)
\le\sum_iq_{ki}a_{ki}+\lambda_k\int d_z(x_{ki},x)\,d\gamma_k
\le\sum_iq_{ki}a_{ki}+\theta_k\lambda_k.
\tag{A.11}
\]
An optimal coupling exists for these finite-first-moment measures; near-optimal couplings also give the same bound by a limit. Summing and minimizing yields $V(z)\le T(z)$, proving equality in Theorem 1. Primal attainment follows from Appendix B.

### A.4 Confusion functions and the original dual

Clipping half-hinge scores permits $r=(1+u)/2\in[0,1]$, and
\[
\psi_{\rm hh}(p_0,p_1)=\min_{0\le r\le1}[p_0(1-r)+p_1r]=\min(p_0,p_1).
\tag{A.12}
\]
For logistic with positive masses, stationarity is
$(-p_0+p_1e^u)/(1+e^u)=0$. Substitution of $e^u=p_0/p_1$ gives
\[
\psi_{\log}(p_0,p_1)
=(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1.
\tag{A.13}
\]
Boundary values follow by continuity, even when the inner score infimum is not attained. An infimum of functions linear in $(p_0,p_1)$ is concave, and scaling the masses proves
\[
\mathcal H(c\Gamma)=c\mathcal H(\Gamma),\quad c\ge0;
\qquad 0\le\mathcal H(\Gamma)\le B\quad\text{if }\sum_m\Gamma_{kim}=q_{ki}.
\tag{A.14}
\]
The upper bound uses the score $u=0$.

Direct dualization of (2.3) gives
\[
V(z)=\max_{\Gamma\ge0}
\left\{\mathcal H(\Gamma):\sum_m\Gamma_{kim}=q_{ki},\quad
\sum_jz_jA_{kj}(\Gamma)\le\theta_k\right\}.
\tag{A.15}
\]
Free intercepts impose the row equalities, score minimization gives $\mathcal H$, and nonnegative transport prices impose the budget inequalities. These feasible masses are admissible original finite adversaries. The expanded domain (3.6) is larger and must not be given that interpretation without checking these budgets.

## Appendix B. Existence and finite convex strong duality

**Lemma 5.**

For fixed $z\in[0,1]^D$, the finite product program, box gate, fixed-triangle gate, and objective-scaled gate are feasible, attain their finite primal minima, and have attained dual optima with no gap.
**Proof.**
The constant test is feasible in all four models, with $w=0$ and, where needed, $\eta=B$. Self-target constraints imply $a\ge0$. On the objective sublevel at most $B$,
\[
0\le a_{ki}\le B/q_{ki},\qquad
0\le\lambda_k\le B/\theta_k,\qquad
0\le w_{kj}\le B/\theta_k.
\tag{B.1}
\]
For the objective-scaled model, $0\le J\le\eta\le B$. At each target, a constraint from each class bounds $\Phi(-u_m)$ and $\Phi(u_m)$ above by uniformly bounded quantities. Each stated loss grows in its positive argument, so $u$ is bounded. These sublevels are nonempty, closed, and bounded; minima are attained. Half-hinge scores may alternatively be clipped to $[-1,1]$.

For the finite product program, choose positive $\lambda$, finite $u$, and sufficiently large intercepts to satisfy every loss inequality strictly. For the gates, remove the variables $w_j$ fixed to zero by $z_j=0$, or equivalently work in that face of the linear domain. For every active coordinate, choose small positive $w_{kj}$, positive $\lambda_k$ strictly exceeding them, and sufficiently large $a$. The fixed box and triangle allowances are then strict. In the objective-scaled model, take $\eta$ sufficiently large that both $J<\eta$ and $\theta^\top w_j<\eta z_j$ are strict. This gives relative Slater points. Finite-dimensional convex strong duality yields zero gap and attained multiplier optima. Strict feasibility in the unreduced zero-coordinate inequalities is neither possible nor required.
$\square$

## Appendix C. Logical domains, box bounds, and fixed-gate duals

### C.1 Optimal-value equivalence of the bounds

For a binary $z$, any box feasible point has $w_{kj}=0$ when $z_j=0$ and $w_{kj}\le\lambda_k$ when $z_j=1$. Thus $w\le\lambda z$ and nonnegative distances imply original feasibility with the same $J$; consequently $V(z)\le G_{\rm box}(z)$. The same argument applies to the triangle, since positive radii force an inactive block to zero.

Conversely, any original feasible point with $J\le B$ satisfies (2.5). Taking $w=\lambda z$ gives
\[
w_{kj}\le U_kz_j,\qquad
\theta^\top w_j=z_j\theta^\top\lambda\le Bz_j.
\tag{C.1}
\]
It enters both gated models with unchanged objective. This proves equality using an attained original optimizer. More generally, if $V(z)<B$, original solutions with $J<V(z)+\varepsilon<B$ enter both gates; letting $\varepsilon\downarrow0$ proves the reverse value inequality. If $V(z)=B$, the constant test supplies it directly. Thus the argument does not require preservation of every original feasible solution, nor even primal attainment.

Any fixed price bounds retaining at least one optimizer (or an appropriate near-optimal sequence) at each considered binary support suffice. Increasing such bounds weakens the fractional box model but preserves binary values. A bound on the global minimum alone is not automatically a bound on every fixed-support risk. For fractional $z$, $w\le\lambda$ and $w\le Uz$ do not imply $w\le\lambda z$; for example $\lambda=1,U=2,z=1/2,w=1$. This is precisely why the gated model is a distinct continuous extension.

### C.2 The box Lagrangian

Assign nonnegative multipliers $\Gamma,\beta,\xi$ to the loss, $w-\lambda$, and $w-Uz$ inequalities. With $a,u$ free and $\lambda,w\ge0$,
\[
\begin{aligned}
\mathcal L={}&J+\sum_{kim}\Gamma_{kim}
[\Phi(\sigma_ku_m)-a_{ki}-\sum_jw_{kj}\Delta_{kimj}]\\
&+\sum_{kj}\beta_{kj}(w_{kj}-\lambda_k)
+\sum_{kj}\xi_{kj}(w_{kj}-U_kz_j)\\
={}&\sum_{ki}a_{ki}(q_{ki}-\sum_m\Gamma_{kim})
+\sum_m[p_{0m}\Phi(-u_m)+p_{1m}\Phi(u_m)]\\
&+\sum_k\lambda_k(\theta_k-\sum_j\beta_{kj})
+\sum_{kj}w_{kj}[\beta_{kj}+\xi_{kj}-A_{kj}(\Gamma)]
-\sum_{kj}U_kz_j\xi_{kj}.
\end{aligned}
\tag{C.2}
\]
The finite infimum conditions are respectively row conservation, the contribution $\mathcal H$, classwise allocation budgets, and coverage of coordinate residuals. Lemma 5 gives
\[
\begin{aligned}
G_{\rm box}(z)=\max_{\Gamma,\beta,\xi\ge0}\quad&
\mathcal H(\Gamma)-\sum_{kj}U_kz_j\xi_{kj}\\
\text{subject to}\quad&\sum_m\Gamma_{kim}=q_{ki},\quad
\sum_j\beta_{kj}\le\theta_k,\quad
A_{kj}(\Gamma)\le\beta_{kj}+\xi_{kj}.
\end{aligned}
\tag{C.3}
\]
The independent box multiplier is denoted $\xi_{kj}$ to distinguish it from the shared triangle multiplier $\mu_j$. There is no additional $\theta_k$ factor in its coverage inequality. Taking $\xi_{kj}=(A_{kj}-\beta_{kj})_+$ is sufficient; larger residuals at zero-weight coordinates can give weaker cuts without changing the anchor value. Every feasible certificate in (C.3) defines a global affine lower bound, and an optimal certificate touches $G_{\rm box}$ at its anchor. With residuals chosen minimally, the mass and allocation budgets bound all certificate variables; continuity of $\mathcal H$ then gives continuity of the fixed-gate value functions.

### C.3 Conjugate construction and the shared triangle

For arbitrary real $w$, define
\[
\mathcal F(w)=\inf_{a,u,\lambda}\{J:
\Phi(\sigma_ku_m)\le a_{ki}+\sum_jw_{kj}\Delta_{kimj},\quad
w_{kj}\le\lambda_k,\quad\lambda\ge0\}.
\tag{C.4}
\]
Nonnegativity of $w$ is imposed by its block domain, rather than in this definition. Partial minimization proves convexity. Choosing $u=0$, sufficiently large intercepts, and $\lambda_k\ge\max(0,\max_jw_{kj})$ proves finiteness everywhere; self-target constraints bound the value below by zero. Hence $\mathcal F$ is continuous.

For $C_B=\{w\ge0:\theta^\top w\le B\}$ and $\Omega=I_{C_B}$, the grouped logical problem is
$\min\mathcal F(w)+\sum_j\Omega(w_j)$ subject to $w_j=0$ if $z_j=0$.
For binary $z$, introduce $v_j=z_jx_j$ with $x_j\in C_B$ and use multipliers $\alpha_j$ on $v_j-z_jx_j=0$. Minimization yields
\[
V(z)=\max_\alpha\left\{h(\alpha)-\sum_jz_j\Omega^*(\alpha_j)\right\},\qquad
h(\alpha)=\inf_v[\mathcal F(v)+\langle\alpha,v\rangle].
\tag{C.5}
\]
Continuity of $\mathcal F$ and the nonempty compact block domain give Fenchel strong duality. The plus sign in $h$ follows from the stated equality multiplier; the block is not symmetric. Its conjugate is
\[
\Omega^*(\alpha)=\max_{w\in C_B}\alpha^\top w
=B\max(0,\alpha_0/\theta_0,\alpha_1/\theta_1).
\tag{C.6}
\]
This follows by setting $r_k=\theta_kw_k$ and allocating the budget $r_0+r_1\le B$ to the largest positive coefficient. An independent box gives instead $\sum_kU_k(\alpha_k)_+$.

Dualizing $h$ gives row conservation, $\sum_j\beta_{kj}\le\theta_k$, and
$\alpha_{kj}=A_{kj}(\Gamma)-\beta_{kj}$ because $w$ is free in (C.4). Substitution into (C.6) and introduction of its epigraph multiplier $\mu_j$ yield (3.6)--(3.7). For fractional $z$, positive homogeneity also gives the support function of $z_jC_B$, proving the same formula for (3.3). Equivalently, direct dualization assigns $\mu_j$ to $\theta^\top w_j-Bz_j\le0$ and gives the identical conditions. This establishes the dual and convexity claims of Proposition 2.

Finally, the triangle gate implies each box gate. Its feasible set is smaller, so $R_B\ge G_{\rm box}$. This compares full value functions, not arbitrary choices of individual cuts.

## Appendix D. Objective-scaled duality and convex certificates

### D.1 Binary exactness and the risk epigraph

An original optimizer enters (3.8) with $\eta=J$ and $w=\lambda z$, since $\theta^\top\lambda\le J$. Conversely, a binary gated point has inactive blocks zero and $w\le\lambda z$, so saturation gives an original feasible test with $J\le\eta$. Thus $R=V$ on binary supports.

Eliminating $\eta$ by imposing $\theta^\top w_j\le z_jJ$ and minimizing $J$ preserves the value even at fractional $z$. Indeed, given a feasible point with $J<\eta$, add $(\eta-J)/2$ to every intercept. Each class's weights sum to one, so the new objective is exactly $\eta$, all loss inequalities are relaxed, and the block allowance is unchanged. This proves the elimination without incorrectly assuming every epigraph optimizer already has $J=\eta$.

### D.2 Lagrangian elimination

Use multipliers $\tau$, $\widehat\Gamma$, $\widehat\beta$, and $\widehat\mu$ for $J-\eta$, the losses, $w-\lambda$, and $\theta^\top w_j-\eta z_j$. The Lagrangian groups as
\[
\begin{aligned}
\mathcal L={}&\eta(1-\tau-\widehat\mu^\top z)
+\sum_{ki}a_{ki}(\tau q_{ki}-\sum_m\widehat\Gamma_{kim})\\
&+\sum_m[\widehat p_{0m}\Phi(-u_m)+\widehat p_{1m}\Phi(u_m)]\\
&+\sum_k\lambda_k(\tau\theta_k-\sum_j\widehat\beta_{kj})
+\sum_{kj}w_{kj}[\widehat\beta_{kj}+\theta_k\widehat\mu_j-A_{kj}(\widehat\Gamma)].
\end{aligned}
\tag{D.1}
\]
The coefficients of free $\eta,a$ must vanish. Score minimization gives $\mathcal H(\widehat\Gamma)$, and nonnegative $\lambda,w$ require nonnegative coefficients. These are exactly (3.9); Lemma 5 supplies strong duality and attainment.

If $\tau>0$, set $(\Gamma,\beta,\mu)=(\widehat\Gamma,\widehat\beta,\widehat\mu)/\tau$. The first constraints put this triple in $\mathcal D$, and normalization gives
\[
\tau(1+\mu^\top z)=1,
\qquad \mathcal H(\widehat\Gamma)=\tau\mathcal H(\Gamma).
\tag{D.2}
\]
Conversely, every element of $\mathcal D$ can be scaled by $\tau=(1+\mu^\top z)^{-1}$ to satisfy (3.9). A zero-scale point has $\widehat\Gamma=0$ and objective zero, which cannot increase the nonnegative maximum. This proves (3.10). The maximum is attained: $\mu_j$ can be taken as the smallest coverage residual, while $\Gamma$ and $\beta$ range over compact sets.

For $h=\mathcal H(\Gamma)$ and $s=\mu^\top z$,
\[
\nabla\frac h{1+s}=-\frac{h\mu}{(1+s)^2},\qquad
\nabla^2\frac h{1+s}=\frac{2h\mu\mu^\top}{(1+s)^3}\succeq0.
\tag{D.3}
\]
The maximum over a fixed domain is therefore convex. Moreover, $0\le h\le B$ implies
\[
\frac h{1+s}-(h-Bs)=s\left(B-\frac h{1+s}\right)\ge0.
\tag{D.4}
\]
Taking maxima and applying Proposition 2 proves (3.11), completing Theorem 3.

### D.3 A certificate bound directly on the original objective

Let $(a,u,\lambda)$ be any original feasible solution at a binary $z$, and let $(\Gamma,\beta,\mu)\in\mathcal D$, with $s=\mu^\top z$. Then
\[
\begin{aligned}
\mathcal H(\Gamma)
&\le\sum_{kim}\Gamma_{kim}\Phi(\sigma_ku_m)\\
&\le\sum_{ki}q_{ki}a_{ki}+\sum_k\lambda_k\sum_jz_jA_{kj}(\Gamma)\\
&\le\sum_{ki}q_{ki}a_{ki}
+\sum_k\lambda_k(\sum_jz_j\beta_{kj}+\theta_ks)\\
&\le J+s\theta^\top\lambda\le J(1+s).
\end{aligned}
\tag{D.5}
\]
The last steps use $0\le z\le1$, the allocation budgets, and nonnegative intercepts. Minimizing over the original feasible set proves the fractional certificate lower bound on $V$. This independent check does not depend on interpreting expanded $\Gamma$ as an admissible adversary.

### D.4 Touching and non-touching tangents

For a positive-scale normalized certificate at $z^t$, the function in (4.1) is convex and globally below $R$. Normalization makes its denominator one at $z^t$, so its gradient is $-v_t\widehat\mu^t$. Its tangent is consequently (4.2). An optimal certificate gives equality at the anchor; a feasible nonoptimal one need not. This remains a valid support inequality at a nondifferentiable point of $R$, since it is obtained from one smooth certificate function, not by assuming $R$ differentiable.

In numerical use, enforce or conservatively repair certificate feasibility and recompute its own $\mathcal H$. A downward shift of a valid tangent remains valid. If the numerator of the fraction is reduced instead, its matched tangent has the reduced numerator too; the two constructions should not be conflated. The theoretical touching property requires an optimal certificate, not a solver status alone.

## Appendix E. Master bounds, termination, and an analytic example

### E.1 Proof of Theorem 4

Each cut is below $R=V$ on binary supports. Therefore every original optimal support with $\eta=V(z)$ is feasible for (4.3), making its optimum at most $\mathrm{OPT}$. A globally valid lower bound on an interrupted master is also at most $\mathrm{OPT}$. A feasible original test gives an upper bound by Theorem 1, proving (4.4) and the stated tolerance guarantee.

With exact solves, consider a master optimizer $z^s$ already evaluated. Its touching cut equals $V(z^s)$, and no other cut exceeds $V(z^s)$. Thus the master objective is $V(z^s)$, while the stored feasible test gives $\bar v\le V(z^s)$. Hence
\[
\underline v=V(z^s)\ge\bar v\ge\mathrm{OPT}\ge\underline v,
\tag{E.1}
\]
and the gap closes. Otherwise the next support is new. The finite set $\mathcal Z_d$ rules out an infinite exact loop. A repeated support with a nonzero numerical oracle gap does not satisfy this argument and must be refined or reported as an accuracy limit.

If $S\subseteq S'$, every test based on $S$ is admissible on $S'$ by ignoring the additional observations. Therefore $V(S')\le V(S)$, and any optimal support can be padded to cardinality $d$ without a feature charge. Budgets $0$ and $D$ then require only one full-budget support.

### E.2 Fractional anchors and dual-optimal cut selection

Relaxing the master to $0\le z\le1$ gives fractional candidates at which (3.9) produces valid cuts. Their gate-primal objective is not an original integer-test upper bound. For an exact positive oracle value $v_t$ and a fixed interior reference $\bar z$, a strongest reference-point tangent on the dual-optimal face can be selected by
\[
\min\{(\bar z-z^t)^\top\widehat\mu:
\text{constraints of (3.9) at }z^t,
\ \mathcal H(\widehat\Gamma)\ge v_t\}.
\tag{E.2}
\]
Weak duality makes the last inequality an equality. For half hinge this is an LP; for logistic its superlevel constraint is convex. An approximate optimal-face condition requires the returned certificate's own value in its cut. Such selection can improve a cut without guaranteeing a universal speedup.

### E.3 A one-coordinate comparison

Take $x_0=0$, $x_1=2$, one unit-mass atom per class, $\theta_0=\theta_1=1/2$, and half hinge, so $B=1$. Reflection and class interchange permit a symmetric optimizer with
$w_0=w_1=\lambda_0=\lambda_1=r$. Combining endpoint constraints gives
$a_0+a_1+2r\ge1$, and self-targets give $a_0+a_1\ge0$. For $0\le r\le1/2$, the bounds are attained by
$a_0=a_1=1/2-r$, $u(0)=2r$, $u(2)=-2r$; for larger $r$, zero intercepts and scores $1,-1$ are feasible. Thus
\[
J(r)=(1-2r)_++r.
\tag{E.3}
\]
The box has $r\le2z$, and the triangle has $r\le z$, giving
\[
G_{\rm box}(z)=\max(1/2,1-2z),\qquad R_B(z)=\max(1/2,1-z).
\tag{E.4}
\]
For the objective-scaled gate, $r\le\eta z$ and $\eta\ge J(r)$ imply
$\eta\ge1-r\ge1-\eta z$. Equality is feasible with
$r=z/(1+z)$, $\eta=1/(1+z)$, including the bound $\eta\ge r$. Therefore
\[
R(z)=1/(1+z),\qquad 0\le z\le1.
\tag{E.5}
\]
All three values are $1$ at $z=0$ and $1/2$ at $z=1$, but at $z=1/4$ they are respectively $1/2$, $3/4$, and $4/5$. The strengthened extension leaves integer risk unchanged. Its tangent at $z=1$ is $3/4-z/4$.

## Appendix F. An optional master retaining entire fractions

Let a positive-scale certificate satisfying the first three constraints of (3.9) have $h=\mathcal H(\widehat\Gamma)$, and retain any numerator $0\le v\le h$. Set
\[
f(z)=\frac{v}{\tau+\widehat\mu^\top z},\qquad
\nu=\tau+\widehat\mu^\top z^t>0.
\tag{F.1}
\]
Homogeneous rescaling gives $f\le R$, and its matched tangent $T$ satisfies
\[
T(z)=v/\nu-v\widehat\mu^\top(z-z^t)/\nu^2,
\qquad f(z)-T(z)=\frac{v[\tau+\widehat\mu^\top z-\nu]^2}
{(\tau+\widehat\mu^\top z)\nu^2}\ge0.
\tag{F.2}
\]
Keeping the full fraction retains information discarded by a tangent.

Let $\bar v>0$ be an independently feasible original integer-test risk. Since $\mathrm{OPT}\le\bar v$, the restriction $0\le\eta\le\bar v$ preserves an original optimum. Introduce shared products $y_j=\eta z_j$ through
\[
\begin{aligned}
&0\le\eta\le\bar v,\qquad 0\le y_j\le\bar v z_j,\qquad y_j\le\eta,\qquad
y_j\ge\eta-\bar v(1-z_j),\\
&\mathbf1^\top z=d,\qquad\mathbf1^\top y=d\eta.
\end{aligned}
\tag{F.3}
\]
For binary $z$, these conditions force $y=\eta z$. With an unpadded budget inequality, only $\mathbf1^\top y\le d\eta$ is justified. For a family of certificates, the master adds
\[
v_s\le\tau_s\eta+(\widehat\mu^s)^\top y,\qquad \eta\ge L_s(z),\qquad\forall s.
\tag{F.4}
\]
There are only $D$ shared additional variables. Every original optimal support with $\eta=V(z)$ and $y=\eta z$ remains feasible, so its global master bound is a valid selection lower bound. Optimal certificates touch the true risk at evaluated supports, and the exact finite-termination argument is unchanged. Update product bounds consistently when $\bar v$ improves; a fractional gate objective cannot supply this incumbent cap.

Any support capable of improving the incumbent satisfies
\[
(\widehat\mu^s)^\top z\ge v_s/\bar v-\tau_s.
\tag{F.5}
\]
This is implied by (F.3)--(F.4), since
$v_s\le\bar v[\tau_s+(\widehat\mu^s)^\top z]$; explicitly adding it does not strengthen that full LP relaxation. An exact feasible test of zero risk proves optimality directly, without division by $\bar v$.

The integer strengthening does not imply dominance of the lifted fractional relaxation over retained affine cuts. For example, let
$f(z)=(1+4z_1+5z_2+z_3)^{-1}$, $d=1$, $z^t=(1,0,0)$, and $\bar v=1/5$.
The point $z=(0,5/6,1/6)$, $\eta=1/6$, $y=(0,1/6,0)$ satisfies the product inequalities and ratio row, but its anchor tangent is $T(z)=14/75>1/6$. Keeping the affine floors avoids this loss. Neither the fraction's dominance over its matched tangent nor the product lift proves a runtime improvement; this master is an optional algorithmic supplement.

## References

[1] D. Bertsimas, R. Cory-Wright, and J. Pauphilet.
*A Unified Approach to Mixed-Integer Optimization Problems with Logical Constraints*.
arXiv:1907.02109v3.

[2] L. Xie, R. Gao, and Y. Xie.
*Robust Hypothesis Testing with Wasserstein Uncertainty Sets*.
arXiv:2105.14348v1, 2021.

[3] R. Gao, L. Xie, Y. Xie, and H. Xu.
*Robust Hypothesis Testing Using Wasserstein Uncertainty Sets*.
arXiv:1805.10611v1, 2018.
