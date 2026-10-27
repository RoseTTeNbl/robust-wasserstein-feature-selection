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

## 2. Finite reformulation

Assume $n_0,n_1\ge1$ and $\rho_0,\rho_1>0$. Being real-valued and
convex, $\ell$ is continuous. For $S\subseteq[D]$ let
$V(S)=\min_{f\in\mathcal F_S}R(f)$; the minimum is attained by
[Lemma 2.1](#lem-finite). Put $\sigma_0=-1$ and $\sigma_1=1$, so that the
class-$k$ loss in Eq. (1) is $\ell(\sigma_kf(X))$.

**Notation.**
Pool the samples as $x_1,\ldots,x_N$, where $N=n_0+n_1$,
$x_r=x_r^{(0)}$ for $r\le n_0$ and $x_{n_0+j}=x_j^{(1)}$; they serve as
transport targets. For a source $x_i^{(k)}$ and a target $x_r$, let
$\Delta_{ir}^{(k)}\in\mathbb R_+^D$ have entries
$\Delta_{ir}^{(k)}[m]=|x_i^{(k)}[m]-x_r[m]|$, so that
$\|x_i^{(k)}[S]-x_r[S]\|_1=\mathbf 1_S^\top\Delta_{ir}^{(k)}$.
A plan is a pair $\gamma=(\gamma^{(0)},\gamma^{(1)})$ with
$\gamma^{(k)}\in\mathbb R_+^{n_k\times N}$; the entry
$\gamma_{ir}^{(k)}$ is the mass moved from $x_i^{(k)}$ to the point
that agrees with $x_r$ on $S$ and with $x_i^{(k)}$ off $S$, at cost
$\mathbf 1_S^\top\Delta_{ir}^{(k)}$. A plan has target masses and
coordinate costs

$$
p_k^\gamma(r)=\sum_{i=1}^{n_k}\gamma_{ir}^{(k)},\qquad
c_k(\gamma)=\sum_{i=1}^{n_k}\sum_{r=1}^N\gamma_{ir}^{(k)}
\Delta_{ir}^{(k)}\in\mathbb R_+^D,\qquad k=0,1.
$$

Let $\Pi$ be the set of plans with $\sum_r\gamma_{ir}^{(k)}=1/n_k$ for
all $k$ and $i$, and let

$$
\Gamma_S=\bigl\{\gamma\in\Pi:\ \mathbf 1_S^\top c_k(\gamma)\le\rho_k,
\ k=0,1\bigr\}.
$$

For $a,b\ge0$ let $\psi(a,b)=\inf_{t\in\mathbb R}\{a\ell(-t)+b\ell(t)\}$,
and for every plan $\gamma$ define

<a id="eq-H"></a>

$$
H(\gamma)=\sum_{r=1}^N\psi\bigl(p_0^\gamma(r),p_1^\gamma(r)\bigr)
=\inf_{g\in\mathbb R^N}\sum_{k=0}^1\sum_{i=1}^{n_k}\sum_{r=1}^N
\gamma_{ir}^{(k)}\ell(\sigma_kg_r).
\tag{3}
$$

As an infimum of linear functions with nonnegative coefficients,
$\psi$ is concave, positively homogeneous, upper semicontinuous and
nonnegative. Since $p_k^\gamma$ is linear in $\gamma$, $H$ has the same
properties. For the hinge loss $\ell(t)=(1+t)_+$, where
$t_+=\max\{t,0\}$, one has $\psi(a,b)=2\min\{a,b\}$.

<a id="lem-finite"></a>

**Lemma 2.1 (Finite reformulation).** *For every $S\subseteq[D]$,*

<a id="eq-lfd"></a>

$$
V(S)=\max_{\gamma\in\Gamma_S}H(\gamma),
\tag{4}
$$

*and*

<a id="eq-finite"></a>

$$
\begin{aligned}
V(S)=\min_{g,a,\lambda}\quad&
\sum_{k=0}^1\Bigl(\frac1{n_k}\sum_{i=1}^{n_k}a_i^{(k)}
+\rho_k\lambda_k\Bigr)\\
\text{subject to}\quad&
\ell(\sigma_kg_r)\le a_i^{(k)}+\lambda_k\mathbf 1_S^\top\Delta_{ir}^{(k)},
\qquad k=0,1,\ i\in[n_k],\ r\in[N],\\
&g\in\mathbb R^N,\quad a^{(k)}\in\mathbb R^{n_k},\quad
\lambda\in\mathbb R_+^2.
\end{aligned}
\tag{5}
$$

*Every minimizer of [Eq. (5)](#eq-finite) extends to a detector
$f\in\mathcal F_S$ with $R(f)=V(S)$.*

[Eq. (4)](#eq-lfd) is the least favorable distribution program
of [[8, Lemma 2]](#ref-xie2021robust) (see also [[5]](#ref-gao2018robust)) with
the transport cost restricted to $S$: least favorable distributions
move mass only along selected coordinates, onto the $S$-projections of
the pooled samples. [Eq. (5)](#eq-finite) is its dual. There, $g_r$ is the
detector score at $x_r[S]$, $a_i^{(k)}$ bounds the loss that the
adversary extracts from the source $x_i^{(k)}$, and $\lambda_k$ prices
the budget $\rho_k$. [Appendix A](#app-lemma) proves the lemma and constructs
the detector. We use two consequences below. First, $V$ is
nonincreasing, since $\mathcal F_S\subseteq\mathcal F_{S'}$ for
$S\subseteq S'$. Second, $0\le V\le V(\emptyset)=2$: the bound
$\psi(a,b)\le(a+b)\ell(0)$ gives $H\le2$ on $\Pi$, and for
$S=\emptyset$ the plan that moves all mass onto one target is feasible
with $H=\psi(1,1)=2$, because $\ell(-t)+\ell(t)\ge2\ell(0)$.

<a id="sec-selection"></a>

## 3. A convex selection function

[Lemma 2.1](#lem-finite) gives one program for each subset. The subset
enters only through the budgets $\mathbf 1_S^\top c_k(\gamma)\le\rho_k$,
which are bilinear in $(\mathbf 1_S,\gamma)$. We replace them by a
priced form. A price $\mu_m\ge0$ removes the transport cost on
coordinate $m$ from the budget, and the objective charges for it only on
selected coordinates. Let
$\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf 1^\top z\le d\}$.

<a id="thm-selection"></a>

**Theorem 3.1 (Convex selection function).** *For $z\in\mathbb R_+^D$ define*

<a id="eq-F"></a>

$$
F(z)=\max_{\gamma\in\Pi,\ \mu\in\mathbb R_+^D}\
\frac{H(\gamma)}{1+\mu^\top z}
\quad\text{subject to}\quad
\sum_{m=1}^D\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+\le\rho_k,
\quad k=0,1.
\tag{6}
$$

- *(a) $F$ is convex and nonincreasing, and $0\le F\le2$.*
- *(b) $F(\mathbf 1_S)=V(S)$ for every $S\subseteq[D]$. Hence
  $\min_{|S|\le d}V(S)=\min_{z\in\mathcal Z_d}F(z)$.*
- *(c) Let $(\gamma,\mu)$ be feasible in [Eq. (6)](#eq-F) and
  $\bar z\in\mathbb R_+^D$. With $v=H(\gamma)/(1+\mu^\top\bar z)$ and
  $\bar\mu=\mu/(1+\mu^\top\bar z)$,*

<a id="eq-tangent"></a>

$$
F(z)\ge v\bigl(1-\bar\mu^\top(z-\bar z)\bigr)
\qquad\text{for all }z\in\mathbb R_+^D,
\tag{7}
$$

  *with equality at $z=\bar z$ if $(\gamma,\mu)$ is optimal at $\bar z$.*

*Proof.* The maximum in [Eq. (6)](#eq-F) exists: the feasible set is closed, the
objective is upper semicontinuous, and replacing $\mu_m$ by
$\min\{\mu_m,\kappa_m\}$ with
$\kappa_m=\max_{k,i,r}\Delta_{ir}^{(k)}[m]/\rho_k$ keeps feasibility
(as $c_k(\gamma)[m]\le\rho_k\kappa_m$ on $\Pi$) and does not decrease
the objective. So the maximization may be restricted to a compact set.

(a) For fixed feasible $(\gamma,\mu)$, the map
$z\mapsto H(\gamma)/(1+\mu^\top z)$ composes the convex decreasing
function $u\mapsto H(\gamma)/(1+u)$ on $[0,\infty)$ with the
nondecreasing linear map $z\mapsto\mu^\top z$, so it is convex and
nonincreasing on $\mathbb R_+^D$. Pointwise maxima keep both
properties, and $0\le H\le2$ on $\Pi$.

(b) Fix $S$. Let $\gamma^\star$ attain [Eq. (4)](#eq-lfd), and set
$\mu^\star_m=0$ for $m\in S$ and
$\mu^\star_m=\max_kc_k(\gamma^\star)[m]/\rho_k$ for $m\notin S$. Then
$\sum_m(c_k(\gamma^\star)[m]-\rho_k\mu^\star_m)_+
=\mathbf 1_S^\top c_k(\gamma^\star)\le\rho_k$ and
$(\mu^\star)^\top\mathbf 1_S=0$, so
$F(\mathbf 1_S)\ge H(\gamma^\star)=V(S)$.
Conversely, let $(\gamma,\mu)$ be feasible and
$\theta=1/(1+\mu^\top\mathbf 1_S)$. Let $\iota\in\Pi$ leave every
sample in place: $\iota_{ir}^{(k)}=1/n_k$ if $r$ is the pooled index of
$x_i^{(k)}$, and $\iota_{ir}^{(k)}=0$ otherwise. Then
$\gamma'=\theta\gamma+(1-\theta)\iota\in\Pi$, $c_k(\iota)=0$, and

$$
\mathbf 1_S^\top c_k(\gamma')
=\theta\sum_{m\in S}c_k(\gamma)[m]
\le\theta\sum_{m\in S}\bigl[(c_k(\gamma)[m]-\rho_k\mu_m)_+
+\rho_k\mu_m\bigr]
\le\theta\rho_k(1+\mu^\top\mathbf 1_S)=\rho_k.
$$

Hence $\gamma'\in\Gamma_S$, and since $H$ is concave and nonnegative,

$$
V(S)\ge H(\gamma')\ge\theta H(\gamma)+(1-\theta)H(\iota)
\ge\frac{H(\gamma)}{1+\mu^\top\mathbf 1_S}.
$$

Maximizing over $(\gamma,\mu)$ gives $V(S)\ge F(\mathbf 1_S)$. The
last claim follows from the correspondence $S\leftrightarrow\mathbf 1_S$.

(c) Convexity of $u\mapsto1/(1+u)$ gives
$1/(1+u)\ge1/(1+\bar u)-(u-\bar u)/(1+\bar u)^2$ for $u,\bar u\ge0$.
Take $u=\mu^\top z$ and $\bar u=\mu^\top\bar z$ and multiply by
$H(\gamma)\ge0$. The left side becomes $H(\gamma)/(1+\mu^\top z)\le F(z)$
and the right side becomes that of [Eq. (7)](#eq-tangent). If
$(\gamma,\mu)$ is optimal at $\bar z$, then $v=F(\bar z)$. □

**What is relaxed.**
[Eq. (6)](#eq-F) keeps the plan set $\Pi$ and the objective $H$ of
[Lemma 2.1](#lem-finite); only the budgets are priced. At
$z=\mathbf 1_S$ the proof shows that pricing is free off $S$, which
removes those coordinates from the budgets, and that pricing on $S$
gains nothing over mixing the plan with $\iota$. At fractional $z$,
coordinate $m$ is charged in proportion to $z_m$. Thus $F$ is exact on
$\{0,1\}^D$, and its minimum over the convex hull of $\mathcal Z_d$ is a
lower bound on the optimal value.

<a id="rem-saddle"></a>

**Remark 3.2 (Saddle-point form).** For $h\ge0$ and $u>0$, $h/u=\max_{s\ge0}\{2\sqrt{sh}-su\}$, attained at
$s=h/u^2$. Hence

<a id="eq-saddle"></a>

$$
F(z)=\max\Bigl\{2\sqrt{sH(\gamma)}-s-s\mu^\top z:\
(\gamma,\mu)\text{ feasible in Eq. (6)},\ s\ge0\Bigr\},
\tag{8}
$$

a maximum of functions affine in $z$ over a set that does not depend on
$z$. This is the structure of [[1, Theorem 2.5]](#ref-bertsimas2021unified)
on which outer approximation rests. The tangent [Eq. (7)](#eq-tangent) is
the affine piece with $s=v/(1+\mu^\top\bar z)$.

<a id="sec-algorithm"></a>

## 4. Outer approximation

**Evaluating $F$.**
For a fixed $z$, [Eq. (6)](#eq-F) is a concave–convex fractional program.
The next proposition gives its Charnes–Cooper form [[3]](#ref-charnes1962programming),
which the algorithm solves, and its dual, which explains the coupling of
$z$ with the detector variables.

<a id="prop-evaluation"></a>

**Proposition 4.1 (Evaluation of $F$).** *Let $z\in[0,1]^D$.*

- *(a) $F(z)$ is the optimal value of*

<a id="eq-cc"></a>

$$
\begin{aligned}
\max_{\hat\gamma,\hat\mu,\beta,\alpha\ge0}\quad&H(\hat\gamma)\\
\text{subject to}\quad&
\textstyle\sum_r\hat\gamma_{ir}^{(k)}=\alpha/n_k,
&&k=0,1,\ i\in[n_k],\\
&c_k(\hat\gamma)[m]\le\beta_{km}+\rho_k\hat\mu_m,
\quad\textstyle\sum_m\beta_{km}\le\rho_k\alpha,
&&k=0,1,\ m\in[D],\\
&\alpha+\hat\mu^\top z=1.
\end{aligned}
\tag{9}
$$

  *If $(\hat\gamma,\hat\mu,\beta,\alpha)$ is optimal and $\alpha>0$, then
  $(\hat\gamma,\hat\mu)/\alpha$ is optimal in [Eq. (6)](#eq-F), with
  $v=H(\hat\gamma)$ and $\bar\mu=\hat\mu$ in [Eq. (7)](#eq-tangent).*
- *(b) $F(z)$ is also the optimal value of*

<a id="eq-primal"></a>

$$
\begin{aligned}
\min_{g,a,\lambda,w,\eta}\quad&\eta\\
\text{subject to}\quad&
\ell(\sigma_kg_r)\le a_i^{(k)}+\textstyle\sum_mw_{km}\Delta_{ir}^{(k)}[m],
&&k=0,1,\ i\in[n_k],\ r\in[N],\\
&0\le w_{km}\le\lambda_k,\quad\rho_0w_{0m}+\rho_1w_{1m}\le\eta z_m,
&&k=0,1,\ m\in[D],\\
&\textstyle\sum_k\bigl(\frac1{n_k}\sum_ia_i^{(k)}+\rho_k\lambda_k\bigr)
\le\eta,\quad\lambda\in\mathbb R_+^2.
\end{aligned}
\tag{10}
$$

- *(c) If the coupling $\rho_0w_{0m}+\rho_1w_{1m}\le\eta z_m$ in
  [Eq. (10)](#eq-primal) is replaced by the big-$M$ constraints
  $w_{km}\le(2/\rho_k)z_m$, the optimal value $F_M(z)$ satisfies
  $F_M\le F$ on $[0,1]^D$ and $F_M(\mathbf 1_S)=V(S)$.*

[Appendix B](#app-evaluation) gives the proof. In [Eq. (9)](#eq-cc),
$(\hat\gamma,\hat\mu,\alpha)=(\gamma,\mu,1)/(1+\mu^\top z)$, and
$\beta$ linearizes the positive parts. For the hinge loss,
$H(\hat\gamma)=\max\{2\sum_rh_r:\ h_r\le p_0^{\hat\gamma}(r),\
h_r\le p_1^{\hat\gamma}(r)\}$, so [Eq. (9)](#eq-cc) is a linear program.
[Eq. (10)](#eq-primal) is the epigraph of [Eq. (5)](#eq-finite) with
$\lambda_kz_m$ replaced by $w_{km}$. Its coupling bounds $w_{km}$ by
$\eta z_m/\rho_k$, a bound that scales with the objective. No big-$M$
constant is needed, and by (c) the resulting relaxation is never weaker
than the big-$M$ one.

**Cuts and master problem.**
Let $z^t=\mathbf 1_{S^t}$, let $(\hat\gamma^t,\hat\mu^t,\beta^t,\alpha_t)$
solve [Eq. (9)](#eq-cc) at $z^t$, and let $h_t=H(\hat\gamma^t)$. By
[Theorem 3.1](#thm-selection)(c) and [Proposition 4.1](#prop-evaluation)(a),

<a id="eq-cut"></a>

$$
c_t(z)=h_t\bigl(1-(\hat\mu^t)^\top(z-z^t)\bigr)
\tag{11}
$$

satisfies $c_t\le F$ on $[0,1]^D$ and $c_t(z^t)=F(z^t)=V(S^t)$. If
$\alpha_t=0$, then $\hat\gamma^t=0$, $h_t=0$ and $c_t\equiv0$, which is
still valid. Since $F$ is nonincreasing and $d\le D$, some minimizer
of $F$ over $\mathcal Z_d$ satisfies $\mathbf 1^\top z=d$. The master
problem is therefore

<a id="eq-master"></a>

$$
L_t=\min_{z\in\{0,1\}^D,\ \eta\ge0}\ \eta
\quad\text{subject to}\quad
\mathbf 1^\top z=d,\quad\eta\ge c_s(z),\ s=1,\ldots,t,
\tag{12}
$$

and $z^{t+1}$ denotes a minimizer. Let
$\mathrm{OPT}=\min_{|S|\le d}V(S)$. Then
$L_t\le\mathrm{OPT}\le U_t=\min_{s\le t}V(S^s)$, because every cut lies
below $F$. [Algorithm 1](#alg-oa) alternates evaluations and master problems,
as in [[4]](#ref-duran1986outer) and [[1, Algorithm 3.1]](#ref-bertsimas2021unified).

<a id="alg-oa"></a>

**Algorithm 1: Outer approximation for variable selection**

**Require:** Samples $x_i^{(k)}$, radii $\rho_k>0$, budget $d\le D$,
tolerance $\varepsilon>0$, initial $S^1$ with $|S^1|=d$.

1. $U\gets+\infty$, $t\gets1$.
2. **loop**
3. &nbsp;&nbsp;Solve [Eq. (5)](#eq-finite) at $S^t$ for $(g^t,a^t,\lambda^t)$
   and its value $V(S^t)$.
4. &nbsp;&nbsp;Solve [Eq. (9)](#eq-cc) at $z^t=\mathbf 1_{S^t}$ and form $c_t$ by
   [Eq. (11)](#eq-cut).
5. &nbsp;&nbsp;**if** $V(S^t)<U$ **then**
6. &nbsp;&nbsp;&nbsp;&nbsp;$U\gets V(S^t)$, $S^\star\gets S^t$,
   $(g^\star,a^\star,\lambda^\star)\gets(g^t,a^t,\lambda^t)$.
7. &nbsp;&nbsp;**end if**
8. &nbsp;&nbsp;Solve [Eq. (12)](#eq-master) with $c_1,\ldots,c_t$ for $z^{t+1}$
   and $L_t$.
9. &nbsp;&nbsp;**if** $U-L_t\le\varepsilon$ **then**
10. &nbsp;&nbsp;&nbsp;&nbsp;**return** $S^\star$ and the detector $f^\star$ built from
    $(g^\star,a^\star,\lambda^\star)$ by [Eq. (13)](#eq-extension).
11. &nbsp;&nbsp;**end if**
12. &nbsp;&nbsp;$S^{t+1}\gets\{m:z^{t+1}_m=1\}$, $t\gets t+1$.
13. **end loop**

<a id="prop-termination"></a>

**Proposition 4.2 (Finite termination).** *If [Eq. (5)](#eq-finite), [Eq. (9)](#eq-cc) and [Eq. (12)](#eq-master) are solved
exactly, [Algorithm 1](#alg-oa) stops after at most $\binom Dd$ iterations
and returns $S^\star$ with $V(S^\star)\le\mathrm{OPT}+\varepsilon$.*

*Proof.* If $z^{t+1}=z^s$ for some $s\le t$, then
$L_t\ge c_s(z^{t+1})=F(z^s)\ge U_t$, and the algorithm stops. Hence
the iterates before termination are distinct points of
$\{z\in\{0,1\}^D:\mathbf 1^\top z=d\}$. On return,
$V(S^\star)=U\le L_t+\varepsilon\le\mathrm{OPT}+\varepsilon$. □

**Implementation.**
The code implements the hinge loss. Its scores may then be restricted
to $[-1,1]$ without loss, and [Eq. (5)](#eq-finite), [Eq. (9)](#eq-cc) and
[Eq. (10)](#eq-primal) are linear programs. The code also accepts general empirical
weights $q_i^{(k)}$ in place of $1/n_k$; nothing above changes. Because solvers
return only approximately feasible points, every bound is certified. A
solution of [Eq. (5)](#eq-finite) is made feasible by resetting
$a_i^{(k)}=\max_r[\ell(\sigma_kg_r)-\lambda_k\mathbf 1_S^\top
\Delta_{ir}^{(k)}]_+$, which gives an upper bound on $V(S^t)$ and a
detector. A solution of [Eq. (9)](#eq-cc) is rescaled to $\Pi$, $\mu$ is
raised until the constraint of [Eq. (6)](#eq-F) holds, and $H$ is lowered
by an explicit error bound. [Theorem 3.1](#thm-selection)(c) then keeps
the cut valid, tight or not; it is stored in the form of
[Remark 3.2](#rem-saddle). $L_t$ is the master solver's global dual
bound, not its incumbent value. The pair [Eq. (10)](#eq-primal) and
[Eq. (9)](#eq-cc) evaluates $F$ at fractional $z$ with matching bounds.

<a id="rem-cuts"></a>

**Remark 4.3 (Cost and choice of cuts).** The $2D$ coupling rows of [Eq. (9)](#eq-cc) contain about $DN^2$ nonzeros,
against about $N^2$ for [Eq. (4)](#eq-lfd), so the cut problems dominate
the running time. A tight cut does not require [Eq. (9)](#eq-cc). By the
proof of [Theorem 3.1](#thm-selection)(b), a solution $\gamma^\star$ of
[Eq. (4)](#eq-lfd) at $S^t$ with the prices $\mu^\star$ defined there is
optimal in [Eq. (6)](#eq-F) at $\mathbf 1_{S^t}$. Then [Eq. (7)](#eq-tangent)
gives the tight cut $V(S^t)\bigl(1-(\mu^\star)^\top z\bigr)$, and
[Proposition 4.2](#prop-termination) still applies. Optimal pairs at
binary points are generally not unique, and different choices give
different cuts. Selecting Pareto-optimal ones [[6]](#ref-magnanti1981accelerating)
is a natural refinement.

<a id="app-lemma"></a>

## Appendix A. Proof of Lemma [2.1](#lem-finite)

*Proof.* Fix $S$ and let $v_S$ be the infimum in [Eq. (5)](#eq-finite). We show
$\max_{\Gamma_S}H\le V(S)\le v_S=\max_{\Gamma_S}H$, and then
attainment.

*Duality.* Let
$\Phi(\gamma,g)=\sum_{k,i,r}\gamma_{ir}^{(k)}\ell(\sigma_kg_r)$, so that
$H(\gamma)=\inf_g\Phi(\gamma,g)$ by [Eq. (3)](#eq-H). The set $\Gamma_S$ is
a nonempty polytope, since it contains $\iota$. The function $\Phi$ is
continuous, linear in $\gamma$ and convex in $g$. Sion's
theorem [[7]](#ref-sion1958minimax) therefore gives
$\max_{\gamma\in\Gamma_S}H(\gamma)
=\inf_g\max_{\gamma\in\Gamma_S}\Phi(\gamma,g)$. The outer maximum is
attained because $H$ is upper semicontinuous. For fixed $g$, the inner
maximum splits into one feasible, bounded linear program per class,

$$
\max\Bigl\{\textstyle\sum_{i,r}\gamma_{ir}^{(k)}\ell(\sigma_kg_r):\
\gamma^{(k)}\ge0,\ \sum_r\gamma_{ir}^{(k)}=\frac1{n_k},\
\sum_{i,r}\gamma_{ir}^{(k)}\mathbf 1_S^\top\Delta_{ir}^{(k)}\le\rho_k\Bigr\},
$$

whose dual is
$\min\{\frac1{n_k}\sum_ia_i^{(k)}+\rho_k\lambda_k:\ \lambda_k\ge0,\
a_i^{(k)}+\lambda_k\mathbf 1_S^\top\Delta_{ir}^{(k)}\ge\ell(\sigma_kg_r)
\ \forall i,r\}$. Adding the two classes and minimizing over $g$ gives
$\max_{\Gamma_S}H=v_S$.

*Lower bound.* Let $\gamma\in\Gamma_S$. Let $y_{ir}^{(k)}$ agree
with $x_r$ on $S$ and with $x_i^{(k)}$ off $S$, and set
$\mathbb P_k^\gamma=\sum_{i,r}\gamma_{ir}^{(k)}\delta_{y_{ir}^{(k)}}$.
Moving mass $\gamma_{ir}^{(k)}$ from $x_i^{(k)}$ to $y_{ir}^{(k)}$
costs $\mathbf 1_S^\top c_k(\gamma)\le\rho_k$ in total, so
$\mathbb P_k^\gamma\in\mathcal P_k$. Every $f\in\mathcal F_S$ satisfies
$f(y_{ir}^{(k)})=f(x_r)$, hence

$$
R(f)\ge\sum_{k=0}^1\mathbb E_{\mathbb P_k^\gamma}\ell(\sigma_kf(X))
=\Phi\bigl(\gamma,(f(x_r))_{r=1}^N\bigr)\ge H(\gamma).
$$

Minimizing over $f$ and maximizing over $\gamma$ gives
$\max_{\Gamma_S}H\le V(S)$.

*Upper bound.* Let $(g,a,\lambda)$ be feasible in
[Eq. (5)](#eq-finite). If $x_r[S]=x_{r'}[S]$, the constraints on $g_r$ and
$g_{r'}$ coincide, so we may take $g_r=g_{r'}$. Then
$\hat g(x_r[S])=g_r$ is well defined on the projected samples. For
$\xi\in\mathbb R^{|S|}$ and $k=0,1$ let

$$
E_k(\xi)=\min_{i\in[n_k]}\bigl\{a_i^{(k)}
+\lambda_k\|x_i^{(k)}[S]-\xi\|_1\bigr\},
$$

let $i_0,i_1$ be the smallest minimizing indices, and write
$r_k=\|x_{i_k}^{(k)}[S]-\xi\|_1$ and
$b=\|x_{i_0}^{(0)}[S]-x_{i_1}^{(1)}[S]\|_1$. At every $\xi$ that is not
a projected sample, set

<a id="eq-extension"></a>

$$
\hat g(\xi)=(1-\tau)\,\hat g(x_{i_0}^{(0)}[S])
+\tau\,\hat g(x_{i_1}^{(1)}[S]),\qquad
\tau=\begin{cases}\min\{1,r_0/b\},&b>0,\\0,&b=0.\end{cases}
\tag{13}
$$

We claim that

<a id="eq-envelopes"></a>

$$
\ell(-\hat g(\xi))\le E_0(\xi),\qquad
\ell(\hat g(\xi))\le E_1(\xi),\qquad\xi\in\mathbb R^{|S|}.
\tag{14}
$$

At a projected sample $\xi=x_r[S]$ this is the constraint of
[Eq. (5)](#eq-finite) at target $r$, minimized over sources. Otherwise the
definition of $\tau$ and the triangle inequality $b\le r_0+r_1$ give
$\tau b\le r_0$ and $(1-\tau)b\le r_1$. By
convexity of $\ell$ and the constraints for the sources
$x_{i_0}^{(0)}$ and $x_{i_1}^{(1)}$ at these two targets,

$$
\begin{aligned}
\ell(-\hat g(\xi))&\le(1-\tau)a_{i_0}^{(0)}
+\tau\bigl(a_{i_0}^{(0)}+\lambda_0b\bigr)
\le a_{i_0}^{(0)}+\lambda_0r_0=E_0(\xi),\\
\ell(\hat g(\xi))&\le(1-\tau)\bigl(a_{i_1}^{(1)}+\lambda_1b\bigr)
+\tau a_{i_1}^{(1)}\le a_{i_1}^{(1)}+\lambda_1r_1=E_1(\xi).
\end{aligned}
$$

The anchors are chosen by finitely many comparisons of continuous
functions, so $\hat g$ is Borel measurable, and
$|\hat g|\le\max_r|g_r|$. Let $f(x)=\hat g(x[S])$, let
$\mathbb P_k\in\mathcal P_k$ and $\epsilon>0$, and take a coupling of
$\widehat{\mathbb P}_k$ and $\mathbb P_k$ with cost at most
$\rho_k+\epsilon$, written as
$\frac1{n_k}\sum_i\delta_{x_i^{(k)}}\otimes\pi_i$. By
[Eq. (14)](#eq-envelopes), $\ell(\sigma_kf(x))\le a_i^{(k)}
+\lambda_k\|x_i^{(k)}-x\|_1$ for every $i$, so

$$
\mathbb E_{\mathbb P_k}\ell(\sigma_kf(X))
=\frac1{n_k}\sum_{i=1}^{n_k}\int\ell(\sigma_kf(x))\,d\pi_i(x)
\le\frac1{n_k}\sum_{i=1}^{n_k}a_i^{(k)}+\lambda_k(\rho_k+\epsilon).
$$

Letting $\epsilon\to0$ and adding the classes gives that $R(f)$ is at
most the objective of $(g,a,\lambda)$. Hence $V(S)\le v_S$.

*Attainment.* The point $g=0$, $a=\mathbf 1$, $\lambda=0$ is
feasible. The constraint whose target is the source itself gives
$a_i^{(k)}\ge0$. On the set where the objective is at most $B$,
therefore, $0\le a_i^{(k)}\le n_kB$ and $0\le\lambda_k\le B/\rho_k$, so
every $\ell(g_r)$ and $\ell(-g_r)$ is bounded. Being convex,
nondecreasing and nonconstant, $\ell$ satisfies $\ell(t)\to\infty$ as
$t\to\infty$, so $g$ is bounded as well. Thus [Eq. (5)](#eq-finite) attains
its minimum. For a minimizer, $R(f)\le v_S=V(S)\le R(f)$, so $f$
attains $V(S)$. □

<a id="app-evaluation"></a>

## Appendix B. Proof of Proposition [4.1](#prop-evaluation)

*Proof.* (a) Let $(\gamma,\mu)$ be feasible in [Eq. (6)](#eq-F) and put
$\alpha=1/(1+\mu^\top z)$, $(\hat\gamma,\hat\mu)=\alpha(\gamma,\mu)$
and $\beta_{km}=\alpha(c_k(\gamma)[m]-\rho_k\mu_m)_+$. As $c_k$ and $H$
are positively homogeneous, this point is feasible in [Eq. (9)](#eq-cc)
with value $H(\hat\gamma)=H(\gamma)/(1+\mu^\top z)$. Conversely, let
$(\hat\gamma,\hat\mu,\beta,\alpha)$ be feasible in [Eq. (9)](#eq-cc). If
$\alpha=0$, the row constraints force $\hat\gamma=0$, and the value is
$0\le F(z)$. If $\alpha>0$, then $\beta_{km}\ge(c_k(\hat\gamma)[m]
-\rho_k\hat\mu_m)_+$ shows that $(\gamma,\mu)=(\hat\gamma,\hat\mu)/\alpha$
is feasible in [Eq. (6)](#eq-F). Moreover $1+\mu^\top z=1/\alpha$, so its
ratio is $H(\hat\gamma)$ and $\bar\mu=\hat\mu$. Both programs therefore
have the same value, and optimal solutions correspond.

(b) Attach multipliers $\gamma\ge0$ to the loss constraints, $\beta\ge0$
to $w\le\lambda$, $\mu\ge0$ to the coupling and $\alpha\ge0$ to the
epigraph constraint. The Lagrangian is

$$
\begin{aligned}
\mathcal L={}&\eta(1-\alpha-\mu^\top z)
+\sum_{k,i,r}\gamma_{ir}^{(k)}\ell(\sigma_kg_r)
+\sum_{k,i}a_i^{(k)}\Bigl(\frac\alpha{n_k}-\sum_r\gamma_{ir}^{(k)}\Bigr)\\
&+\sum_k\lambda_k\Bigl(\rho_k\alpha-\sum_m\beta_{km}\Bigr)
+\sum_{k,m}w_{km}\bigl(\beta_{km}+\rho_k\mu_m-c_k(\gamma)[m]\bigr).
\end{aligned}
$$

Its infimum over $g$ is $H(\gamma)$ by [Eq. (3)](#eq-H). Its infimum over
the free variables $\eta,a$ and the nonnegative variables $\lambda,w$
is finite exactly when the constraints of [Eq. (9)](#eq-cc) hold. Hence
the Lagrangian dual of [Eq. (10)](#eq-primal) is [Eq. (9)](#eq-cc). The program
[Eq. (10)](#eq-primal) is convex. Its only nonaffine constraints are the
loss constraints, and they hold strictly at $g=0$, $a=2\cdot\mathbf 1$,
$\lambda=0$, $w=0$, $\eta=4$, which satisfies the affine constraints.
Its value is finite, because the constraints whose target is the
source give $a\ge0$ and hence $\eta\ge0$. Slater's
condition [[2, §5.2.3]](#ref-boyd2004convex) gives equal values, which
equal $F(z)$ by (a).

(c) As in [Appendix A](#app-lemma), the sublevel sets of [Eq. (10)](#eq-primal)
are compact, now with $0\le w_{km}\le\lambda_k\le\eta/\rho_k$. Hence a
minimizer exists, and $\eta=F(z)\le2$ there. It satisfies
$\rho_kw_{km}\le\eta z_m\le2z_m$, so it is feasible in the big-$M$
program with objective at most $F(z)$, and $F_M(z)\le F(z)$. At
$z=\mathbf 1_S$ the big-$M$ constraints force $w_{km}=0$ for
$m\notin S$, and $w_{km}\le\lambda_k$ gives
$\sum_mw_{km}\Delta_{ir}^{(k)}[m]\le\lambda_k\mathbf 1_S^\top
\Delta_{ir}^{(k)}$. Every feasible point therefore yields a feasible
point of [Eq. (5)](#eq-finite) with the same objective, and
$F_M(\mathbf 1_S)\ge V(S)=F(\mathbf 1_S)$. □

## References

<a id="ref-bertsimas2021unified"></a>

1. Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

<a id="ref-boyd2004convex"></a>

2. Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*. Cambridge University Press, March 2004.

<a id="ref-charnes1962programming"></a>

3. Abraham Charnes and William W. Cooper. Programming with linear fractional functionals. *Naval Research Logistics Quarterly*, 9(3–4):181–186, 1962.

<a id="ref-duran1986outer"></a>

4. Marco A. Duran and Ignacio E. Grossmann. An outer-approximation algorithm for a class of mixed-integer nonlinear programs. *Mathematical Programming*, 36(3):307–339, 1986.

<a id="ref-gao2018robust"></a>

5. Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.

<a id="ref-magnanti1981accelerating"></a>

6. Thomas L. Magnanti and Richard T. Wong. Accelerating Benders decomposition: Algorithmic enhancement and model selection criteria. *Operations Research*, 29(3):464–484, 1981.

<a id="ref-sion1958minimax"></a>

7. Maurice Sion. On general minimax theorems. *Pacific Journal of Mathematics*, 8(1):171–176, March 1958.

<a id="ref-xie2021robust"></a>

8. Liyan Xie, Rui Gao, and Yao Xie. Robust hypothesis testing with Wasserstein uncertainty sets, 2021. arXiv:2105.14348.
