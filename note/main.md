# Variable Selection for Wasserstein Robust Hypothesis Testing

## 1. Problem formulation

Throughout this paper, we consider the variable selection problem for robust hypothesis testing. Let $[D]=\{1,\ldots,D\}$ and $d$ be a positive integer denoting the sparsity budget.
Given a detector $f:~\mathbb{R}^D\to\mathbb{R}$, we define the worst-case risk as

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

<a id="sec-finite"></a>

## 2. Convex formulation

For $S\subseteq[D]$, denote $V(S)=\min_{f\in\mathcal F_S}R(f)$ and define

$$
\mathcal X=\{x_i^{(0)}:i\in[n_0]\}\cup
\{x_j^{(1)}:j\in[n_1]\}.
$$

Assume $n_0,n_1\ge1$, $\rho_0,\rho_1>0$, and that $\ell$ is finite
on $\mathbb R$. Each target $y\in\mathcal X$ receives a detector score
$g_y$, and $g=(g_y)_{y\in\mathcal X}$.
For masses $a,b\ge0$, let
$\psi(a,b)=\inf_{t\in\mathbb R}\{a\ell(-t)+b\ell(t)\}$ be the smallest
loss at a point that carries mass $a$ of class 0 and $b$ of class 1;
it is the auxiliary function of [[2]](#ref-gao2018robust) in homogeneous form.
The next lemma is [[2, Proposition 1 and Theorem 3]](#ref-gao2018robust)
applied to the projected observations: projection contracts the
$\ell_1$ transport cost, and every projected transport lifts at the
same cost by keeping each source's unselected coordinates.

<a id="prop-lfd"></a>

**Lemma 2.1 (Least favorable distributions).**

<a id="eq-lfd"></a>

$$
\begin{aligned}
V(S)=\max_{p,\gamma\ge0}\quad&
\sum_{y\in\mathcal X}\psi\bigl(p_0(y),p_1(y)\bigr)\\
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

Here $\gamma_{i,y}^{(k)}$ is the mass transported from
$x_i^{(k)}[S]$ to $y[S]$, and $p_k(y)$ is the mass of class $k$
at that target. Let $\Gamma_S$ denote the feasible transports in
[Eq. (3)](#eq-lfd) after eliminating $p$. Since $\psi(a,b)$ is an
infimum over scores, the objective equals
$\inf_g\sum_{k}\sum_{i,y}\gamma_{i,y}^{(k)}\ell((-1)^{k+1}g_y)$.
This sum is convex in $g$ and linear in $\gamma$, and $\Gamma_S$ is
compact and convex, so Sion's minimax theorem [[3]](#ref-sion1958minimax)
lets us minimize over $g$ outside:

$$
V(S)=\inf_g\max_{\gamma\in\Gamma_S}
\sum_{k=0}^1\sum_{i,y}\gamma_{i,y}^{(k)}\ell((-1)^{k+1}g_y).
$$

For fixed $g$, the inner maximum splits into one linear program per
class: the worst expectation of $y\mapsto\ell((-1)^{k+1}g_y)$ over
transports of $\widehat{\mathbb P}_k$ with budget $\rho_k$. Leaving
every observation in place is feasible, so linear programming duality
gives the familiar Wasserstein dual

$$
\min_{\lambda_k\ge0}\ \rho_k\lambda_k+\frac1{n_k}\sum_{i=1}^{n_k}
\max_{y\in\mathcal X}\bigl\{\ell((-1)^{k+1}g_y)
-\lambda_k\|x_i^{(k)}[S]-y[S]\|_1\bigr\}.
$$

Naming the inner maximum $a_i^{(k)}$ and minimizing over $g$ gives

<a id="eq-finite"></a>

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
\tag{4}
$$

[Eq. (4)](#eq-finite) is a finite convex program, and it attains its
minimum: the constraints with $y=x_i^{(k)}$ give $a_i^{(k)}\ge0$, so
an objective value at most $B$ forces $a_i^{(k)}\le n_kB$ and
$\lambda_k\le B/\rho_k$; the constraints then bound $\ell(\pm g_y)$,
hence $g$, since $\ell(t)\to\infty$ as $t\to\infty$ ($\ell$ is convex with
$\ell(0)=1>\lim_{t\to-\infty}\ell(t)$).

A detector is recovered from an optimal $(g,a,\lambda)$ by extending
the scores off the samples. For $\xi\in\mathbb R^{|S|}$, let
$E_k(\xi)=\min_{i\in[n_k]}\{a_i^{(k)}+\lambda_k\|x_i^{(k)}[S]-\xi\|_1\}$.
The constraints of [Eq. (4)](#eq-finite) say exactly that
$\ell(-g_y)\le E_0(y[S])$ and $\ell(g_y)\le E_1(y[S])$, so we require
the extension to satisfy

<a id="eq-extension"></a>

$$
\ell(-g(\xi))\le E_0(\xi),\qquad \ell(g(\xi))\le E_1(\xi),
\qquad\xi\in\mathbb R^{|S|}.
\tag{5}
$$

This is the analogue, for a general $\ell$, of the interval that
[[4, Theorem 1]](#ref-xie2021robust) gives for randomized tests.
Coincident projected targets have identical constraints, so
$g(y[S])=g_y$ is well defined and satisfies [Eq. (5)](#eq-extension).
For any other $\xi$, let $i$ and $j$ be the smallest indices attaining
$E_0(\xi)$ and $E_1(\xi)$, let $r_0=\|x_i^{(0)}[S]-\xi\|_1$,
$r_1=\|x_j^{(1)}[S]-\xi\|_1$ and $b=\|x_i^{(0)}[S]-x_j^{(1)}[S]\|_1$,
and weigh the two anchors by distance:
$g(\xi)=(1-\tau)g_{x_i^{(0)}}+\tau g_{x_j^{(1)}}$ with
$\tau=r_0/(r_0+r_1)$, where $r_0,r_1>0$ because $\xi$ is not a
projected sample.
The constraints of [Eq. (4)](#eq-finite) at source $i$ give
$\ell(-g_{x_i^{(0)}})\le a_i^{(0)}$ and
$\ell(-g_{x_j^{(1)}})\le a_i^{(0)}+\lambda_0b$; by convexity,
$\ell(-g(\xi))\le a_i^{(0)}+\lambda_0\tau b$. Likewise
$\ell(g(\xi))\le a_j^{(1)}+\lambda_1(1-\tau)b$. The triangle inequality
$b\le r_0+r_1$ gives $\tau b\le r_0$ and $(1-\tau)b\le r_1$, so the two
bounds are at most $E_0(\xi)$ and $E_1(\xi)$.

The detector $f(x)=g(x[S])$ is Borel and attains $R(f)=V(S)$.
Indeed, $E_k$ is $\lambda_k$-Lipschitz for $\|\cdot\|_1$ and
$E_k(x_i^{(k)}[S])\le a_i^{(k)}$. Projection does not increase
$\mathcal W_1$, so by Kantorovich–Rubinstein duality every
$\mathbb P_k\in\mathcal P_k$ satisfies
$\mathbb E_{\mathbb P_k}E_k(X[S])\le
\frac1{n_k}\sum_ia_i^{(k)}+\lambda_k\rho_k$. With
[Eq. (5)](#eq-extension), summing over $k$ bounds $R(f)$ by the value of
[Eq. (4)](#eq-finite), which is $V(S)$; the reverse inequality holds
because $f\in\mathcal F_S$.

To optimize the coordinate selection, define $z=\mathbf1_S$ and
$\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
For $\gamma=(\gamma^{(0)},\gamma^{(1)})$ with
$\gamma^{(k)}\in\mathbb R_+^{n_k\times|\mathcal X|}$, let
$p_k(y)=\sum_i\gamma_{i,y}^{(k)}$ as in [Eq. (3)](#eq-lfd) and define the
objective of [Eq. (3)](#eq-lfd)

$$
H(\gamma)=\sum_{y\in\mathcal X}\psi\bigl(p_0(y),p_1(y)\bigr).
$$

Since $\psi$ is an infimum of linear functions with nonnegative
coefficients, $H$ is concave, nondecreasing, and homogeneous:
$H(s\gamma)=sH(\gamma)$ for $s\ge0$.
Define the average of the two class masses and the transport cost
in coordinate $m$, respectively, by

$$
\begin{aligned}
M(\gamma)&=\frac12\sum_{k=0}^1\sum_{i=1}^{n_k}
\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)},\\
c_k(\gamma)[m]&=\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|,
\qquad k=0,1,\ m\in[D].
\end{aligned}
$$

For a scalar $t\in\mathbb R$, write $t_+=\max\{t,0\}$.
The next result writes the reciprocal of the selection objective as
a minimum of functions affine in $z$.

<a id="prop-dual"></a>

**Proposition 2.2 (Convex selection function).**
*Let $\mathcal D$ be the set of pairs $(\gamma,\mu)$ with $\gamma\ge0$
and $\mu\in\mathbb R_+^D$ such that*

$$
\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}=\frac{M(\gamma)}{n_k},
\quad i\in[n_k],\qquad
\sum_{m=1}^D\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+\le\rho_kM(\gamma),
\qquad k=0,1.
$$

*For $z\in[0,1]^D$, define $F(z)>0$ by*

<a id="eq-dual"></a>

$$
\frac1{F(z)}=\min\bigl\{M(\gamma)+\mu^\top z:\
(\gamma,\mu)\in\mathcal D,\ H(\gamma)\ge1\bigr\}.
\tag{6}
$$

*Then the minimum is attained, $F$ is convex and nonincreasing with
$F\le2$, and*

<a id="eq-binary-exact"></a>

$$
F(\mathbf1_S)=V(S),\qquad
\min_{|S|\le d}V(S)=\min_{z\in\mathcal Z_d}F(z).
\tag{7}
$$

The row constraints make each class's total mass equal to
$M(\gamma)$; this common mass may vary, whereas each class has unit
mass in [Eq. (3)](#eq-lfd). Because $H$, $M$ and $\mathcal D$ scale
together, $F(z)$ is the largest ratio
$H(\gamma)/(M(\gamma)+\mu^\top z)$ over $\mathcal D$.
The variable $\mu_m$ is a price: paying $\mu_mz_m$ relieves each class
of $\rho_k\mu_m$ of transport cost in coordinate $m$. At $z=\mathbf1_S$
the unselected coordinates are relieved for free, which leaves the
budget of [Eq. (3)](#eq-lfd) on $S$ alone.
[Appendix A](#app-dual) proves the proposition directly from
[Lemma 2.1](#prop-lfd). Minimizing $F$ over $\mathcal Z_d$ therefore solves
the original problem; replacing $\mathcal Z_d$ by its continuous
relaxation gives a lower bound.

<a id="sec-algorithm"></a>

## 3. An outer-approximation approach

We use outer approximation [[1]](#ref-bertsimas2021unified) on the concave
function $1/F$. For fixed $z$, [Eq. (6)](#eq-dual) minimizes a linear
function over a convex set, since $\mathcal D$ is convex and $H$ is
concave. For an optimizer $(\gamma^t,\mu^t)$ at $z^t$, define

<a id="eq-cut"></a>

$$
c_t(z)=M(\gamma^t)+(\mu^t)^\top z.
\tag{8}
$$

The feasible set in [Eq. (6)](#eq-dual) does not depend on $z$, so
$1/F(z)\le c_t(z)$ for every $z\in[0,1]^D$, with equality at $z^t$.
Equivalently, $F(z)\ge1/c_t(z)$; as $t\mapsto1/t$ is convex, this
bound lies above the tangent of $F$ at $z^t$.
At a binary $z^t=\mathbf1_{S^t}$ no extra solve is needed:
if $\gamma^\star$ is a least favorable transport of [Eq. (3)](#eq-lfd) at
$S^t$, [Appendix A](#app-dual) shows that
$\gamma^t=\gamma^\star/V(S^t)$ with
$\mu^t_m=\max_kc_k(\gamma^\star)[m]/(\rho_kV(S^t))$ for $m\notin S^t$
and $\mu^t_m=0$ for $m\in S^t$ is optimal.
After $t$ evaluations, we maximize the accumulated upper bounds on
$1/F$ to obtain the next selection:

<a id="eq-master"></a>

$$
\begin{aligned}
(z^{t+1},\theta_t)\in\operatorname*{arg\,max}_{z\in\mathcal Z_d,\,\theta}
\quad&\theta\\
\text{subject to}\quad&\theta\le c_s(z),\qquad s=1,\ldots,t.
\end{aligned}
\tag{9}
$$

The value $L_t=1/\theta_t$ is a lower bound on
$\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z)$, and the best evaluated
value $U_t=\min_{s\le t}F(z^s)$ is an upper bound.

[Algorithm 1](#alg-oa) alternates between evaluating a selection
and solving [Eq. (9)](#eq-master). It retains the best evaluated
detector and stops when $U_t-L_t\le\varepsilon$.
With exact solves, if $z^{t+1}=z^s$ for some $s\le t$, then

$$
\theta_t\le c_s(z^{t+1})=1/F(z^s)\le1/U_t,
$$

so $L_t\ge U_t$ and the stopping condition holds. Each nonterminal
iteration therefore produces a new selection, and at most
$|\mathcal Z_d|$ selections are evaluated.
Since $F$ is nonincreasing, the budget can
also be imposed as $\mathbf1^\top z=d$ when $d\le D$.

<a id="alg-oa"></a>

**Algorithm 1. Outer approximation for variable selection**

**Require:** Samples $x_i^{(k)}$, radii $\rho_k>0$, budget $d$,
tolerance $\varepsilon>0$, initial $z^1\in\mathcal Z_d$

1. $U\gets+\infty$, $t\gets1$
2. **loop**
3. &emsp;$S^t\gets\{m:z_m^t=1\}$
4. &emsp;Solve [Eq. (4)](#eq-finite) at $S^t$ for $(g^t,a^t,\lambda^t)$, value $v_t=V(S^t)$, and a least favorable transport of [Eq. (3)](#eq-lfd)
5. &emsp;Form $(\gamma^t,\mu^t)$ from it and $c_t$ by [Eq. (8)](#eq-cut)
6. &emsp;**if** $v_t<U$ **then**
7. &emsp;&emsp;$U\gets v_t$; store $S^\star\gets S^t$ and $(g^\star,a^\star,\lambda^\star)\gets(g^t,a^t,\lambda^t)$
8. &emsp;**end if**
9. &emsp;Solve [Eq. (9)](#eq-master) with $c_1,\ldots,c_t$ for $(z^{t+1},\theta_t)$; $L_t\gets1/\theta_t$
10. &emsp;**if** $U-L_t\le\varepsilon$ **then**
11. &emsp;&emsp;Extend $g^\star$ to $\mathbb R^{|S^\star|}$ by [Eq. (5)](#eq-extension)
12. &emsp;&emsp;**return** $S^\star$ and $f^\star(x)=g^\star(x[S^\star])$
13. &emsp;**end if**
14. &emsp;$t\gets t+1$
15. **end loop**

When a solver stops early, feasible solutions of [Eq. (4)](#eq-finite)
give upper bounds, and every $(\gamma,\mu)\in\mathcal D$ with
$H(\gamma)>0$ gives the valid cut
$(M(\gamma)+\mu^\top z)/H(\gamma)\ge1/F(z)$.
Take $\theta_t$ as the solver's global upper bound for
[Eq. (9)](#eq-master).

<a id="app-dual"></a>

## Appendix A. Proof of Proposition [2.2](#prop-dual)

<a id="app-dual-binary"></a>

### A.1 Equality at binary selections

Fix $S$. We first note $V(S)>0$. Convexity of $\ell$ gives
$\ell(t)+\ell(-t)\ge2\ell(0)=2$, hence
$\psi(a,b)\ge2\min\{a,b\}$. Moving a small mass $\varepsilon>0$ of
$x_1^{(0)}$ onto the target $x_1^{(1)}$ is feasible in [Eq. (3)](#eq-lfd)
and puts masses at least $\varepsilon$ and $1/n_1$ on that target, so
$V(S)\ge2\min\{\varepsilon,1/n_1\}>0$.

*Upper bound on $1/F(\mathbf1_S)$.*
The maximum in [Eq. (3)](#eq-lfd) is attained, because $\Gamma_S$ is
compact and $H$, an infimum of linear functions, is upper
semicontinuous. Let $\gamma^\star$ attain it, so $M(\gamma^\star)=1$,
$H(\gamma^\star)=V(S)$ and $\sum_{m\in S}c_k(\gamma^\star)[m]\le\rho_k$.
Set $\mu_m=0$ for $m\in S$ and
$\mu_m=\max_kc_k(\gamma^\star)[m]/\rho_k$ for $m\notin S$.
The terms with $m\notin S$ then vanish, and

$$
\sum_m\bigl(c_k(\gamma^\star)[m]-\rho_k\mu_m\bigr)_+
=\sum_{m\in S}c_k(\gamma^\star)[m]\le\rho_k M(\gamma^\star).
$$

Thus $(\gamma^\star,\mu)\in\mathcal D$. The constraints of $\mathcal D$
are invariant under scaling $(\gamma,\mu)\mapsto(s\gamma,s\mu)$ with
$s>0$, and $H$ and $M$ scale by $s$. The pair
$(\gamma^\star,\mu)/V(S)$ is therefore feasible in [Eq. (6)](#eq-dual)
with $H=1$, and its objective is
$(M(\gamma^\star)+\mu^\top\mathbf1_S)/V(S)=1/V(S)$.
Hence $1/F(\mathbf1_S)\le1/V(S)$.

*Lower bound on $1/F(\mathbf1_S)$.*
Let $(\gamma,\mu)\in\mathcal D$ with $H(\gamma)\ge1$, and let
$s=M(\gamma)+\mu^\top\mathbf1_S$. Taking $t=0$ in the definition of
$\psi$ gives $\psi(a,b)\le a+b$, hence $H(\gamma)\le2M(\gamma)$, so
$M(\gamma)\ge1/2$ and $s>0$.
Let $\iota_{i,y}^{(k)}=n_k^{-1}\mathbf1_{\{y=x_i^{(k)}\}}$ be the
transport that leaves each observation in place; then $M(\iota)=1$ and
$c_k(\iota)=0$. Rescale $\gamma$ and complete its mass with $\iota$:

$$
\gamma'=\frac{\gamma}{s}+\Bigl(1-\frac{M(\gamma)}{s}\Bigr)\iota .
$$

Since $M(\gamma)\le s$, $\gamma'\ge0$, and its rows sum to $1/n_k$.
For each $m\in S$,
$c_k(\gamma)[m]\le(c_k(\gamma)[m]-\rho_k\mu_m)_++\rho_k\mu_m$.
Summing over $m\in S$, bounding the positive parts by their sum over
all $m$, and using the cost constraint of $\mathcal D$,

$$
\sum_{m\in S}c_k(\gamma')[m]=\frac1s\sum_{m\in S}c_k(\gamma)[m]
\le\frac{\rho_kM(\gamma)+\rho_k\mu^\top\mathbf1_S}{s}=\rho_k .
$$

Hence $\gamma'\in\Gamma_S$. As $H$ is nondecreasing and homogeneous,
$V(S)\ge H(\gamma')\ge H(\gamma/s)=H(\gamma)/s\ge1/s$.
Every feasible point of [Eq. (6)](#eq-dual) at $\mathbf1_S$ therefore has
objective $s\ge1/V(S)$, so $1/F(\mathbf1_S)\ge1/V(S)$.
Together, $F(\mathbf1_S)=V(S)$, and $(\gamma^\star,\mu)/V(S)$ attains
the minimum; this is the optimizer used in [Section 3](#sec-algorithm).

<a id="app-dual-convex"></a>

### A.2 Attainment and convexity

Fix $z\in[0,1]^D$. The feasible set of [Eq. (6)](#eq-dual) is nonempty by
[Appendix A.1](#app-dual-binary), and every feasible point has
$M(\gamma)+\mu^\top z\ge M(\gamma)\ge1/2$, so $1/F(z)\ge1/2$ and
$F(z)\le2$. Take a minimizing sequence $(\gamma^j,\mu^j)$. Its
objective values are bounded, so $M(\gamma^j)$ and hence $\gamma^j$
are bounded. If $\mu_m^j$ exceeds $\max_kc_k(\gamma^j)[m]/\rho_k$,
lowering it to this value keeps the terms
$(c_k(\gamma^j)[m]-\rho_k\mu_m^j)_+$ at zero and does not increase the
objective, so we may take $\mu^j$ bounded as well. A subsequence
converges; its limit is feasible, because $\mathcal D$ is closed and
$H$ is upper semicontinuous, and it attains the minimum.

The set $\mathcal D\cap\{H\ge1\}$ is convex, because the constraints
of $\mathcal D$ are linear equations and a sum of positive parts of
affine functions bounded by a linear function, and $H$ is concave.
This set does not depend on $z$, and the objective is affine in $z$
with slope $\mu\ge0$. Hence $1/F$, a pointwise minimum of such
functions, is concave, nondecreasing and positive. Since $t\mapsto1/t$
is convex and decreasing on $(0,\infty)$, $F$ is convex and
nonincreasing. With [Appendix A.1](#app-dual-binary), the correspondence
$S\leftrightarrow\mathbf1_S$ gives [Eq. (7)](#eq-binary-exact). $\square$

## References

<a id="ref-bertsimas2021unified"></a>

1. Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

<a id="ref-gao2018robust"></a>

2. Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.

<a id="ref-sion1958minimax"></a>

3. Maurice Sion. On general minimax theorems. *Pacific Journal of Mathematics*, 8(1):171–176, March 1958.

<a id="ref-xie2021robust"></a>

4. Liyan Xie, Rui Gao, and Yao Xie. Robust hypothesis testing with Wasserstein uncertainty sets, 2021. arXiv:2105.14348.
