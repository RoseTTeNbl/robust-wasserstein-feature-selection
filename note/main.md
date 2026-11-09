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
on $\mathbb R$. The points of $\mathcal X$ serve as transport targets.
For $y\in\mathcal X$, let $g_y\in\mathbb R$ denote the detector score
at $y$, and write $g=(g_y)_{y\in\mathcal X}$. For masses $a,b\ge0$,
define

$$
\psi(a,b)=\inf_{t\in\mathbb R}\bigl\{a\ell(-t)+b\ell(t)\bigr\}.
$$

This is the smallest loss at a point that carries mass $a$ of class 0
and mass $b$ of class 1. The function $p\mapsto\psi(1-p,p)$ on $[0,1]$
is the auxiliary function of [[2]](#ref-gao2018robust), and $\psi$ is
positively homogeneous.

The next lemma is [[2, Proposition 1 and Theorem 3]](#ref-gao2018robust)
applied to the projected observations $x_i^{(k)}[S]$. The reduction to
projections rests on two facts: projection does not increase $\ell_1$
transport costs, and every transport between projected points lifts to
a transport in $\mathbb R^D$ of the same cost that keeps the unselected
coordinates of each observation.

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
$x_i^{(k)}[S]$ to $y[S]$, and $p_k(y)$ is the mass of class $k$ at the
target $y$. Let $\Gamma_S$ denote the set of transports
$\gamma=(\gamma^{(0)},\gamma^{(1)})$ that are feasible in
[Eq. (3)](#eq-lfd) after $p$ is eliminated.

<a id="prop-finite"></a>

**Proposition 2.2 (Finite convex program).**
*The value $V(S)$ equals the optimal value of the convex program*

<a id="eq-finite"></a>

$$
\begin{aligned}
\min_{g,a,\lambda}\quad&
\frac1{n_0}\sum_i a_i^{(0)}+\frac1{n_1}\sum_j a_j^{(1)}
+\rho_0\lambda_0+\rho_1\lambda_1\\
\text{subject to}\quad&
\ell(-g_y)\le a_i^{(0)}+\lambda_0\|x_i^{(0)}[S]-y[S]\|_1,
&&i\in[n_0],\ y\in\mathcal X,\\
&\ell(g_y)\le a_j^{(1)}+\lambda_1\|x_j^{(1)}[S]-y[S]\|_1,
&&j\in[n_1],\ y\in\mathcal X,\\
&g\in\mathbb R^{\mathcal X},\quad a^{(k)}\in\mathbb R^{n_k},
\quad\lambda_k\ge0,&&k=0,1,
\end{aligned}
\tag{4}
$$

*and the minimum is attained.*

In [Eq. (4)](#eq-finite), $\lambda_k$ is the price of the transport budget
$\rho_k$. At an optimal solution, each $a_i^{(k)}$ takes its smallest
feasible value

$$
a_i^{(k)}=\max_{y\in\mathcal X}\bigl\{\ell\bigl((-1)^{k+1}g_y\bigr)
-\lambda_k\|x_i^{(k)}[S]-y[S]\|_1\bigr\},
$$

the largest loss that the observation $x_i^{(k)}$ can reach at a target,
net of the transport price. [Appendix A](#app-finite) derives the program
from [Lemma 2.1](#prop-lfd) by Sion's minimax theorem
[[3]](#ref-sion1958minimax) and linear programming duality.

A solution of [Eq. (4)](#eq-finite) assigns scores only to the targets. To
obtain a detector on $\mathbb R^{|S|}$, let $(g,a,\lambda)$ be optimal
and define, for $\xi\in\mathbb R^{|S|}$,

$$
E_k(\xi)=\min_{i\in[n_k]}
\bigl\{a_i^{(k)}+\lambda_k\|x_i^{(k)}[S]-\xi\|_1\bigr\},
\qquad k=0,1.
$$

The constraints of [Eq. (4)](#eq-finite) state that
$\ell(-g_y)\le E_0(y[S])$ and $\ell(g_y)\le E_1(y[S])$ for every
$y\in\mathcal X$. We extend the scores so that these bounds hold on
all of $\mathbb R^{|S|}$:

<a id="eq-extension"></a>

$$
\ell(-g(\xi))\le E_0(\xi),\qquad \ell(g(\xi))\le E_1(\xi),
\qquad\xi\in\mathbb R^{|S|}.
\tag{5}
$$

For a general $\ell$, [Eq. (5)](#eq-extension) plays the role of the
interval of optimal randomized tests in
[[4, Theorem 1]](#ref-xie2021robust).

We construct one such extension. If $y,y'\in\mathcal X$ and
$y[S]=y'[S]$, the constraints of [Eq. (4)](#eq-finite) on $g_y$ and
$g_{y'}$ coincide, so replacing $g_{y'}$ by $g_y$ preserves feasibility
and the objective. We therefore take $g_y=g_{y'}$ and set
$g(y[S])=g_y$. For $\xi\notin\{y[S]:y\in\mathcal X\}$, let $i$ and $j$
be the smallest indices attaining $E_0(\xi)$ and $E_1(\xi)$,
respectively, and set

$$
r_0=\|x_i^{(0)}[S]-\xi\|_1,\qquad
r_1=\|x_j^{(1)}[S]-\xi\|_1,
$$

$$
\tau=\frac{r_0}{r_0+r_1},\qquad
g(\xi)=(1-\tau)g_{x_i^{(0)}}+\tau g_{x_j^{(1)}}.
$$

Both distances are positive because $\xi$ is not a projected
observation. Thus $g(\xi)$ is a convex combination of the scores at
$x_i^{(0)}$ and $x_j^{(1)}$, with the larger weight on the point closer
to $\xi$.

<a id="prop-detector"></a>

**Proposition 2.3 (Optimal detector).**
*The extended score $g$ is Borel measurable and satisfies
[Eq. (5)](#eq-extension). The detector $f(x)=g(x[S])$ belongs to
$\mathcal F_S$ and attains $R(f)=V(S)$.*

The proof in [Appendix B](#app-detector) uses the interpolation rule only
to verify [Eq. (5)](#eq-extension); every Borel extension that satisfies
[Eq. (5)](#eq-extension) is optimal.

To optimize the coordinate selection, define $z=\mathbf1_S$ and
$\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
For $\gamma=(\gamma^{(0)},\gamma^{(1)})$ with
$\gamma^{(k)}\in\mathbb R_+^{n_k\times|\mathcal X|}$, let
$p_k(y)=\sum_i\gamma_{i,y}^{(k)}$ as in [Eq. (3)](#eq-lfd), and denote the
objective of [Eq. (3)](#eq-lfd) by

$$
H(\gamma)=\sum_{y\in\mathcal X}\psi\bigl(p_0(y),p_1(y)\bigr).
$$

Since $\psi$ is an infimum of linear functions with nonnegative
coefficients, $H$ is concave, nondecreasing and positively homogeneous:
$H(s\gamma)=sH(\gamma)$ for $s\ge0$. Define the average class mass and
the class-$k$ transport cost in coordinate $m$ by

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
The next result expresses the reciprocal of the selection objective as
a minimum of functions affine in $z$.

<a id="prop-dual"></a>

**Proposition 2.4 (Convex selection function).**
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

The row constraints of $\mathcal D$ require both classes to carry the
same total mass $M(\gamma)$, which may differ from one; in
[Eq. (3)](#eq-lfd), each class carries unit mass. Since $\mathcal D$ is a
cone and $H$ and $M$ are positively homogeneous, [Eq. (6)](#eq-dual) is
equivalent to

$$
F(z)=\max\Bigl\{\frac{H(\gamma)}{M(\gamma)+\mu^\top z}:\
(\gamma,\mu)\in\mathcal D,\ H(\gamma)>0\Bigr\}.
$$

The variable $\mu_m$ is a price on coordinate $m$. Paying $\mu_mz_m$
in the objective exempts the first $\rho_k\mu_m$ units of the class-$k$
cost in coordinate $m$ from the budget constraint. At $z=\mathbf1_S$,
the coordinates outside $S$ are exempt at no charge, and only the cost
on $S$ remains constrained, as in [Eq. (3)](#eq-lfd).

[Appendix C](#app-dual) proves the proposition from
[Lemma 2.1](#prop-lfd). Minimizing $F$ over $\mathcal Z_d$ therefore
solves the original problem, and minimizing it over the continuous
relaxation $\{z\in[0,1]^D:\mathbf1^\top z\le d\}$ gives a lower bound.

<a id="sec-algorithm"></a>

## 3. An outer-approximation approach

We apply outer approximation [[1]](#ref-bertsimas2021unified) to the
concave function $1/F$. For fixed $z$, [Eq. (6)](#eq-dual) minimizes a
linear function over the convex set $\mathcal D\cap\{H\ge1\}$. Let
$(\gamma^t,\mu^t)$ be a minimizer at $z^t$, and define

<a id="eq-cut"></a>

$$
c_t(z)=M(\gamma^t)+(\mu^t)^\top z.
\tag{8}
$$

The feasible set of [Eq. (6)](#eq-dual) does not depend on $z$, so

$$
\frac1{F(z)}\le c_t(z),\quad z\in[0,1]^D,\qquad
\frac1{F(z^t)}=c_t(z^t).
$$

Equivalently, $F(z)\ge1/c_t(z)$. Because $t\mapsto1/t$ is convex,
$1/c_t$ dominates its linearization at $z^t$, which is the standard
outer-approximation cut for $F$.

At a binary point $z^t=\mathbf1_{S^t}$, the cut requires no additional
solve. Let $\gamma^\star$ be a least favorable transport of
[Eq. (3)](#eq-lfd) at $S^t$. [Appendix C.1](#app-dual-binary) shows that
the pair

$$
\gamma^t=\frac{\gamma^\star}{V(S^t)},\qquad
\mu^t_m=\begin{cases}
0,&m\in S^t,\\[2pt]
\displaystyle\max_{k=0,1}\frac{c_k(\gamma^\star)[m]}{\rho_kV(S^t)},
&m\notin S^t,
\end{cases}
$$

is a minimizer of [Eq. (6)](#eq-dual) at $z^t$.

After $t$ evaluations, the next selection maximizes the smallest of the
accumulated upper bounds on $1/F$:

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

[Algorithm 1](#alg-oa) alternates between evaluating a selection and
solving [Eq. (9)](#eq-master). It retains the best evaluated detector and
stops when $U_t-L_t\le\varepsilon$. With exact solves, the algorithm
terminates after finitely many iterations. Indeed, if $z^{t+1}=z^s$ for
some $s\le t$, then

$$
\theta_t\le c_s(z^{t+1})=\frac1{F(z^s)}\le\frac1{U_t},
$$

so $L_t\ge U_t$ and the stopping condition holds. Each nonterminal
iteration therefore produces a new selection, and at most
$|\mathcal Z_d|$ selections are evaluated. Since $F$ is nonincreasing,
the budget can also be imposed as $\mathbf1^\top z=d$ when $d\le D$.

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
11. &emsp;&emsp;Extend $g^\star$ to $\mathbb R^{|S^\star|}$ as in [Proposition 2.3](#prop-detector)
12. &emsp;&emsp;**return** $S^\star$ and $f^\star(x)=g^\star(x[S^\star])$
13. &emsp;**end if**
14. &emsp;$t\gets t+1$
15. **end loop**

When a solver stops early, the bounds remain valid. Every feasible
point of [Eq. (4)](#eq-finite) gives an upper bound on $V(S^t)$, and every
$(\gamma,\mu)\in\mathcal D$ with $H(\gamma)>0$ gives the cut

$$
\frac1{F(z)}\le\frac{M(\gamma)+\mu^\top z}{H(\gamma)},
\qquad z\in[0,1]^D.
$$

For [Eq. (9)](#eq-master), take $\theta_t$ to be the solver's global
upper bound.

<a id="app-finite"></a>

## Appendix A. Proof of Proposition [2.2](#prop-finite)

*Proof.* For $g\in\mathbb R^{\mathcal X}$ and $\gamma\in\Gamma_S$, let

$$
\Phi(g,\gamma)=\sum_{k=0}^1\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}\,\ell\bigl((-1)^{k+1}g_y\bigr)
=\sum_{y\in\mathcal X}\bigl\{p_0(y)\ell(-g_y)+p_1(y)\ell(g_y)\bigr\}.
$$

*Exchange of maximum and infimum.*
The infimum of $\Phi(\cdot,\gamma)$ separates over the targets, and the
infimum of the term at $y$ over $g_y$ is $\psi(p_0(y),p_1(y))$. Hence
$\inf_g\Phi(g,\gamma)$ is the objective of [Eq. (3)](#eq-lfd) at $\gamma$,
and [Lemma 2.1](#prop-lfd) gives

$$
V(S)=\max_{\gamma\in\Gamma_S}\ \inf_{g}\ \Phi(g,\gamma).
$$

The set $\Gamma_S$ is nonempty, convex and compact: it contains the
transport that leaves every observation in place, it is defined by
linear constraints, and its entries lie in $[0,\max_k1/n_k]$. The
function $\Phi(\cdot,\gamma)$ is convex and continuous, because $\ell$
is convex and finite and $\gamma\ge0$; the function $\Phi(g,\cdot)$ is
linear, hence concave and continuous. Sion's minimax theorem
[[3]](#ref-sion1958minimax) therefore gives

<a id="eq-sion"></a>

$$
V(S)=\inf_{g}\ \max_{\gamma\in\Gamma_S}\ \Phi(g,\gamma).
\tag{10}
$$

*Duality for fixed scores.*
Fix $g$. The constraints of $\Gamma_S$ do not couple the two classes,
so the inner maximum in [Eq. (10)](#eq-sion) is the sum over $k=0,1$ of
the linear programs

$$
\begin{aligned}
\max_{\gamma^{(k)}\ge0}\quad&\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}\ell\bigl((-1)^{k+1}g_y\bigr)\\
\text{subject to}\quad&
\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}=\frac1{n_k},
\qquad i\in[n_k],\\
&\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}
\|x_i^{(k)}[S]-y[S]\|_1\le\rho_k.
\end{aligned}
$$

With a free multiplier $a_i^{(k)}$ for the $i$th row constraint and a
multiplier $\lambda_k\ge0$ for the budget constraint, the dual linear
program is

$$
\begin{aligned}
\min_{a^{(k)},\,\lambda_k\ge0}\quad&
\frac1{n_k}\sum_{i=1}^{n_k}a_i^{(k)}+\rho_k\lambda_k\\
\text{subject to}\quad&
\ell\bigl((-1)^{k+1}g_y\bigr)\le a_i^{(k)}
+\lambda_k\|x_i^{(k)}[S]-y[S]\|_1,
\qquad i\in[n_k],\ y\in\mathcal X.
\end{aligned}
$$

The primal program is feasible, since leaving every observation in
place has cost $0\le\rho_k$, and its feasible set is bounded. By linear
programming duality, the primal and dual optimal values are equal.
Substituting the two dual programs into [Eq. (10)](#eq-sion) and
combining the infimum over $g$ with the minima over $(a,\lambda)$ shows
that $V(S)$ is the infimum of [Eq. (4)](#eq-finite).

*Attainment.*
The point $g=0$, $a_i^{(k)}=1$, $\lambda=0$ is feasible because
$\ell(0)=1$, and its objective value is $2$. Let $\mathcal K$ be the
set of feasible points with objective at most $2$. Since the objective
is continuous, it suffices to show that $\mathcal K$ is compact. The
set $\mathcal K$ is closed because $\ell$ is continuous. Let
$(g,a,\lambda)\in\mathcal K$. The constraint with source $x_i^{(k)}$
and target $y=x_i^{(k)}$ reads
$\ell((-1)^{k+1}g_{x_i^{(k)}})\le a_i^{(k)}$, so $a^{(k)}\ge0$. Every
term of the objective is then nonnegative, and therefore

$$
0\le a_i^{(k)}\le2n_k,\qquad 0\le\lambda_k\le\frac2{\rho_k}.
$$

With $\Delta=\max\{\|x[S]-y[S]\|_1:x,y\in\mathcal X\}$, the
constraints give, for every $y\in\mathcal X$,

$$
\ell(-g_y)\le2n_0+\frac{2\Delta}{\rho_0},\qquad
\ell(g_y)\le2n_1+\frac{2\Delta}{\rho_1}.
$$

Moreover, $\ell(t)\to\infty$ as $t\to\infty$. Indeed, since
$\lim_{t\to-\infty}\ell(t)=0<1$, there is $t_0<0$ with $\ell(t_0)<1$,
and convexity of $\ell$ applied to $t_0<0<t$ gives

$$
\ell(t)\ge\ell(0)+\frac{1-\ell(t_0)}{-t_0}\,t,\qquad t>0.
$$

Hence $g_y$ and $-g_y$ are bounded above on $\mathcal K$. Thus
$\mathcal K$ is bounded, and therefore compact. $\square$

<a id="app-detector"></a>

## Appendix B. Proof of Proposition [2.3](#prop-detector)

*Proof.* Let $P=\{y[S]:y\in\mathcal X\}$ be the set of projected targets.

*The bounds in [Eq. (5)](#eq-extension) on $P$.*
Let $\xi=y[S]$ with $y\in\mathcal X$. The constraints of
[Eq. (4)](#eq-finite) with target $y$ give, for every $i\in[n_0]$,

$$
\ell(-g(\xi))=\ell(-g_y)\le a_i^{(0)}+\lambda_0\|x_i^{(0)}[S]-\xi\|_1.
$$

Minimizing over $i$ gives $\ell(-g(\xi))\le E_0(\xi)$. The same
argument for class 1 gives $\ell(g(\xi))\le E_1(\xi)$.

*The bounds in [Eq. (5)](#eq-extension) off $P$.*
Let $\xi\notin P$, let $i$, $j$, $r_0$, $r_1$ and $\tau$ be as in
[Section 2](#sec-finite), and let

$$
b=\|x_i^{(0)}[S]-x_j^{(1)}[S]\|_1.
$$

The constraints of [Eq. (4)](#eq-finite) with source $x_i^{(0)}$ and
targets $x_i^{(0)}$, $x_j^{(1)}$, and with source $x_j^{(1)}$ and targets
$x_j^{(1)}$, $x_i^{(0)}$, give

<a id="eq-anchors"></a>

$$
\begin{aligned}
\ell(-g_{x_i^{(0)}})&\le a_i^{(0)},&\qquad
\ell(-g_{x_j^{(1)}})&\le a_i^{(0)}+\lambda_0b,\\
\ell(g_{x_j^{(1)}})&\le a_j^{(1)},&\qquad
\ell(g_{x_i^{(0)}})&\le a_j^{(1)}+\lambda_1b.
\end{aligned}
\tag{11}
$$

The triangle inequality gives $b\le r_0+r_1$, hence

$$
\tau b=\frac{r_0b}{r_0+r_1}\le r_0,\qquad
(1-\tau)b=\frac{r_1b}{r_0+r_1}\le r_1.
$$

By convexity of $\ell$, [Eq. (11)](#eq-anchors), $\lambda_k\ge0$, and the
choice of $i$ and $j$,

$$
\begin{aligned}
\ell(-g(\xi))&\le(1-\tau)\ell(-g_{x_i^{(0)}})+\tau\ell(-g_{x_j^{(1)}})
\le a_i^{(0)}+\lambda_0\tau b\le a_i^{(0)}+\lambda_0r_0=E_0(\xi),\\
\ell(g(\xi))&\le(1-\tau)\ell(g_{x_i^{(0)}})+\tau\ell(g_{x_j^{(1)}})
\le a_j^{(1)}+\lambda_1(1-\tau)b\le a_j^{(1)}+\lambda_1r_1=E_1(\xi).
\end{aligned}
$$

*Measurability.*
For $i\in[n_0]$ and $j\in[n_1]$, let $A_{ij}$ be the set of
$\xi\notin P$ for which $i$ and $j$ are the smallest indices attaining
$E_0(\xi)$ and $E_1(\xi)$. Each $A_{ij}$ is Borel, being defined by
finitely many equalities and inequalities between continuous functions
of $\xi$, and $g$ is continuous on $A_{ij}$. The sets $A_{ij}$ and the
finitely many points of $P$ partition $\mathbb R^{|S|}$, so $g$ is
Borel. Hence $f\in\mathcal F_S$.

*The risk of $f$.*
Fix $k$ and $\mathbb P_k\in\mathcal P_k$. By [Eq. (5)](#eq-extension),

$$
\ell\bigl((-1)^{k+1}f(x)\bigr)\le E_k(x[S]),\qquad x\in\mathbb R^D.
$$

Since $\|u[S]-v[S]\|_1\le\|u-v\|_1$ for $u,v\in\mathbb R^D$, each
function

$$
x\mapsto a_i^{(k)}+\lambda_k\|x_i^{(k)}[S]-x[S]\|_1,\qquad i\in[n_k],
$$

is $\lambda_k$-Lipschitz for $\|\cdot\|_1$ on $\mathbb R^D$, and so is
their minimum $x\mapsto E_k(x[S])$. This function is integrable under
$\mathbb P_k$, which has a finite first moment because
$\mathcal W_1(\widehat{\mathbb P}_k,\mathbb P_k)<\infty$. For every
coupling $\pi$ of $\widehat{\mathbb P}_k$ and $\mathbb P_k$,

$$
\mathbb E_{\mathbb P_k}E_k(X[S])
=\int E_k(x'[S])\,d\pi(x,x')
\le\frac1{n_k}\sum_{i=1}^{n_k}E_k\bigl(x_i^{(k)}[S]\bigr)
+\lambda_k\int\|x-x'\|_1\,d\pi(x,x').
$$

Taking the infimum over couplings and using
$E_k(x_i^{(k)}[S])\le a_i^{(k)}$, the $i$th term of the minimum that
defines $E_k$, gives

$$
\mathbb E_{\mathbb P_k}\ell\bigl((-1)^{k+1}f(X)\bigr)
\le\frac1{n_k}\sum_{i=1}^{n_k}a_i^{(k)}+\lambda_k\rho_k.
$$

Summing over $k$ and taking the supremum over
$\mathcal P_0\times\mathcal P_1$ bounds $R(f)$ by the objective of
[Eq. (4)](#eq-finite) at $(g,a,\lambda)$, which equals $V(S)$ by
[Proposition 2.2](#prop-finite). Conversely, $R(f)\ge V(S)$ because
$f\in\mathcal F_S$. $\square$

<a id="app-dual"></a>

## Appendix C. Proof of Proposition [2.4](#prop-dual)

Three facts are used throughout. First, for $a,b\ge0$,

<a id="eq-psi-bounds"></a>

$$
2\min\{a,b\}\le\psi(a,b)\le a+b.
\tag{12}
$$

The upper bound takes $t=0$ in the definition of $\psi$. For the lower
bound, convexity gives $\ell(t)+\ell(-t)\ge2\ell(0)=2$, and since
$\ell\ge0$,

$$
a\ell(-t)+b\ell(t)\ge\min\{a,b\}\bigl(\ell(-t)+\ell(t)\bigr)
\ge2\min\{a,b\},\qquad t\in\mathbb R.
$$

Summing the upper bound over the targets gives $H(\gamma)\le2M(\gamma)$.
Second, $H$ is upper semicontinuous, because $\psi$ is an infimum of
continuous functions. Third, the transport that leaves every
observation in place,

$$
\iota_{i,y}^{(k)}=\frac1{n_k}\mathbf1\{y=x_i^{(k)}\},
$$

satisfies $M(\iota)=1$ and $c_k(\iota)=0$.

<a id="app-dual-binary"></a>

### C.1 Equality at binary selections

*Proof of $F(\mathbf1_S)=V(S)$.* Fix $S$.

*Positivity of $V(S)$.*
Let $b_0=\|x_1^{(0)}[S]-x_1^{(1)}[S]\|_1$. Set $\varepsilon=1/n_0$ if
$b_0=0$, and $\varepsilon=\min\{1/n_0,\rho_0/b_0\}$ otherwise. Move mass
$\varepsilon$ of $x_1^{(0)}$ to the target $x_1^{(1)}$ and leave all
other mass in place. This transport is feasible in [Eq. (3)](#eq-lfd),
because its cost is $\varepsilon b_0\le\rho_0$, and it places class-0
mass at least $\varepsilon$ and class-1 mass at least $1/n_1$ at
$x_1^{(1)}$. Since $\psi$ is nonnegative and nondecreasing,
[Eq. (12)](#eq-psi-bounds) gives

$$
V(S)\ge\psi(\varepsilon,1/n_1)\ge2\min\{\varepsilon,1/n_1\}>0.
$$

*The bound $1/F(\mathbf1_S)\le1/V(S)$.*
The maximum in [Eq. (3)](#eq-lfd) is attained, because $\Gamma_S$ is
compact and $H$ is upper semicontinuous. Let $\gamma^\star$ attain it.
Since $\|x[S]-y[S]\|_1=\sum_{m\in S}|x[m]-y[m]|$, the constraints of
[Eq. (3)](#eq-lfd) give

$$
M(\gamma^\star)=1,\qquad H(\gamma^\star)=V(S),\qquad
\sum_{m\in S}c_k(\gamma^\star)[m]\le\rho_k,\quad k=0,1.
$$

Define $\mu\in\mathbb R_+^D$ by

<a id="eq-mu-star"></a>

$$
\mu_m=0\quad(m\in S),\qquad
\mu_m=\max_{k=0,1}\frac{c_k(\gamma^\star)[m]}{\rho_k}\quad(m\notin S).
\tag{13}
$$

For $m\notin S$, this choice gives
$(c_k(\gamma^\star)[m]-\rho_k\mu_m)_+=0$. Hence

$$
\sum_{m=1}^D\bigl(c_k(\gamma^\star)[m]-\rho_k\mu_m\bigr)_+
=\sum_{m\in S}c_k(\gamma^\star)[m]\le\rho_k=\rho_kM(\gamma^\star).
$$

The row constraints of $\mathcal D$ hold because $M(\gamma^\star)=1$,
so $(\gamma^\star,\mu)\in\mathcal D$. Since $\mathcal D$ is a cone and
$H$ and $M$ are positively homogeneous, $(\gamma^\star,\mu)/V(S)$ is
feasible in [Eq. (6)](#eq-dual) with $H=1$, and its objective at
$z=\mathbf1_S$ is

$$
\frac{M(\gamma^\star)+\mu^\top\mathbf1_S}{V(S)}=\frac1{V(S)}.
$$

*The bound $1/F(\mathbf1_S)\ge1/V(S)$.*
Let $(\gamma,\mu)\in\mathcal D$ with $H(\gamma)\ge1$, and let
$s=M(\gamma)+\mu^\top\mathbf1_S$. Since $H(\gamma)\le2M(\gamma)$, we
have $M(\gamma)\ge1/2$, and $s\ge M(\gamma)>0$ because $\mu\ge0$.
Define

$$
\gamma'=\frac{\gamma}{s}+\Bigl(1-\frac{M(\gamma)}{s}\Bigr)\iota.
$$

The coefficient of $\iota$ is nonnegative, so $\gamma'\ge0$, and the
rows of $\gamma'$ sum to

$$
\frac{M(\gamma)}{sn_k}+\Bigl(1-\frac{M(\gamma)}{s}\Bigr)\frac1{n_k}
=\frac1{n_k}.
$$

The cost $c_k$ is linear and $c_k(\iota)=0$, so
$c_k(\gamma')=c_k(\gamma)/s$. For every $m$,

$$
c_k(\gamma)[m]\le\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_++\rho_k\mu_m.
$$

Summing over $m\in S$ and using the cost constraint of $\mathcal D$,

$$
\sum_{m\in S}c_k(\gamma)[m]
\le\sum_{m=1}^D\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+
+\rho_k\mu^\top\mathbf1_S
\le\rho_k\bigl(M(\gamma)+\mu^\top\mathbf1_S\bigr)=\rho_ks.
$$

Hence $\sum_{m\in S}c_k(\gamma')[m]\le\rho_k$, and $\gamma'\in\Gamma_S$.
Since $\gamma'\ge\gamma/s$ entrywise and $H$ is nondecreasing and
positively homogeneous,

$$
V(S)\ge H(\gamma')\ge H(\gamma/s)=\frac{H(\gamma)}{s}\ge\frac1s.
$$

Thus every feasible point of [Eq. (6)](#eq-dual) at $z=\mathbf1_S$ has
objective $s\ge1/V(S)$.

The two bounds give $F(\mathbf1_S)=V(S)$. Moreover,
$(\gamma^\star,\mu)/V(S)$, with $\mu$ given by [Eq. (13)](#eq-mu-star),
attains the minimum in [Eq. (6)](#eq-dual); this is the minimizer used in
[Section 3](#sec-algorithm). $\square$

<a id="app-dual-convex"></a>

### C.2 Attainment and convexity

*Proof of the remaining claims.* Fix $z\in[0,1]^D$.

*Attainment and the bound $F\le2$.*
The feasible set of [Eq. (6)](#eq-dual) is nonempty by
[Appendix C.1](#app-dual-binary). Since $\mu,z\ge0$ and
$H(\gamma)\le2M(\gamma)$, every feasible point satisfies

$$
M(\gamma)+\mu^\top z\ge M(\gamma)\ge\frac12.
$$

Let $(\gamma^j,\mu^j)$ be a minimizing sequence with objective values
at most $B<\infty$. Then $M(\gamma^j)\le B$, and each entry of
$\gamma^j$ is at most the total mass $M(\gamma^j)$ of its class, so the
sequence $\gamma^j$ is bounded. Replacing $\mu_m^j$ by

$$
\min\Bigl\{\mu_m^j,\ \max_{k=0,1}\frac{c_k(\gamma^j)[m]}{\rho_k}\Bigr\}
$$

changes $\mu_m^j$ only if $\rho_k\mu_m^j\ge c_k(\gamma^j)[m]$ for both
$k$, and then the terms $(c_k(\gamma^j)[m]-\rho_k\mu_m^j)_+$ are zero
before and after the replacement. The replacement therefore keeps the
point in $\mathcal D$, and it does not increase the objective because
$z\ge0$. We may thus assume that $\mu^j$ is bounded. A subsequence
converges to some $(\gamma,\mu)$. The limit lies in $\mathcal D$, which
is closed because its constraints involve only continuous functions,
and it satisfies $H(\gamma)\ge1$ because $H$ is upper semicontinuous.
The objective is continuous, so the limit attains the minimum. Hence
$1/F(z)\ge1/2$, that is, $F(z)\le2$.

*Convexity and monotonicity.*
The set $\mathcal D\cap\{H\ge1\}$ is convex. Indeed, the row
constraints are linear equations; the cost constraint bounds a convex
function of $(\gamma,\mu)$, a sum of positive parts of affine
functions, by the linear function $\rho_kM(\gamma)$; and $\{H\ge1\}$ is
a superlevel set of a concave function. This set does not depend on
$z$, and at each of its points the objective $M(\gamma)+\mu^\top z$ is
affine in $z$ with slope $\mu\ge0$. Therefore $1/F$, a pointwise
minimum of such functions, is concave, nondecreasing and positive on
$[0,1]^D$. Since $t\mapsto1/t$ is convex and decreasing on
$(0,\infty)$, $F$ is convex and nonincreasing.

*Proof of [Eq. (7)](#eq-binary-exact).*
The first identity is proved in [Appendix C.1](#app-dual-binary). The
second follows because $S\mapsto\mathbf1_S$ is a bijection from
$\{S\subseteq[D]:|S|\le d\}$ onto $\mathcal Z_d$. $\square$

## References

<a id="ref-bertsimas2021unified"></a>

1. Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

<a id="ref-gao2018robust"></a>

2. Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.

<a id="ref-sion1958minimax"></a>

3. Maurice Sion. On general minimax theorems. *Pacific Journal of Mathematics*, 8(1):171–176, March 1958.

<a id="ref-xie2021robust"></a>

4. Liyan Xie, Rui Gao, and Yao Xie. Robust hypothesis testing with Wasserstein uncertainty sets, 2021. arXiv:2105.14348.
