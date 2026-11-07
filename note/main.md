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
The next lemma is [[2, Proposition 1 and Theorem 3]](#ref-gao2018robust)
applied to the projected observations, in the transport notation
of [[4]](#ref-xie2021robust): projection contracts the $\ell_1$ transport
cost, and every projected transport lifts at the same cost by keeping
each source's unselected coordinates.

<a id="prop-lfd"></a>

**Lemma 2.1 (Least favorable distributions).**

<a id="eq-lfd"></a>

$$
\begin{aligned}
V(S)=\max_{p,\gamma\ge0}\quad&
\sum_{y\in\mathcal X}\inf_{g_y\in\mathbb R}
\{p_0(y)\ell(-g_y)+p_1(y)\ell(g_y)\}\\
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
[Eq. (3)](#eq-lfd) after eliminating $p$. Since $\Gamma_S$ is compact and
convex, and the sum below is convex in $g$ and linear in $\gamma$,
Sion's minimax theorem [[3]](#ref-sion1958minimax) gives

$$
\begin{aligned}
V(S)
&=\max_{\gamma\in\Gamma_S}\inf_g
\sum_{k=0}^1\sum_{i,y}\gamma_{i,y}^{(k)}\ell((-1)^{k+1}g_y)\\
&=\inf_g\max_{\gamma\in\Gamma_S}
\sum_{k=0}^1\sum_{i,y}\gamma_{i,y}^{(k)}\ell((-1)^{k+1}g_y).
\end{aligned}
$$

For fixed $g$, the inner maximum is a linear program in each class,
feasible because leaving every observation in place costs nothing.
Its Lagrangian is

$$
\begin{aligned}
\mathcal L_k={}&\sum_{i,y}\gamma_{i,y}^{(k)}\ell((-1)^{k+1}g_y)
+\sum_i a_i^{(k)}\left(\frac1{n_k}-\sum_y\gamma_{i,y}^{(k)}\right)\\
&+\lambda_k\left(\rho_k-\sum_{i,y}\gamma_{i,y}^{(k)}
\|x_i^{(k)}[S]-y[S]\|_1\right),
\end{aligned}
$$

with multipliers $a_i^{(k)}\in\mathbb R$ for source-mass conservation
and $\lambda_k\ge0$ for the transport budget. Strong duality, followed
by minimization over $g$, gives

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

Following [[2]](#ref-gao2018robust), [[4]](#ref-xie2021robust), a detector is recovered
from an optimal $(g,a,\lambda)$ as follows. Coincident projected
targets have identical constraints, so set $g(y[S])=g_y$.
For any other $\xi\in\mathbb R^{|S|}$, let $i$ and $j$ be the smallest
indices minimizing $a_r^{(0)}+\lambda_0\|x_r^{(0)}[S]-\xi\|_1$ over
$r\in[n_0]$ and $a_r^{(1)}+\lambda_1\|x_r^{(1)}[S]-\xi\|_1$ over
$r\in[n_1]$, respectively.
With $r_0=\|x_i^{(0)}[S]-\xi\|_1$ and
$b=\|x_i^{(0)}[S]-x_j^{(1)}[S]\|_1$, define

<a id="eq-extension"></a>

$$
\tau=\begin{cases}\min\{1,r_0/b\},&b>0,\\0,&b=0,\end{cases}
\qquad g(\xi)=(1-\tau)g_{x_i^{(0)}}+\tau g_{x_j^{(1)}}.
\tag{5}
$$

Convexity of $\ell$ and the triangle inequality bound
$\ell(-g(\xi))$ by $a_i^{(0)}+\lambda_0r_0$, and $\ell(g(\xi))$ by the
analogous quantity at $j$; by the choice of $i$ and $j$, the
constraints of [Eq. (4)](#eq-finite) therefore hold at every
$\xi\in\mathbb R^{|S|}$. Integrating them against couplings within
the budgets shows that the Borel detector $f(x)=g(x[S])$ attains
$R(f)=V(S)$.

To optimize the coordinate selection, define $z=\mathbf1_S$ and
$\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
For $\gamma=(\gamma^{(0)},\gamma^{(1)})$ with
$\gamma^{(k)}\in\mathbb R_+^{n_k\times|\mathcal X|}$, define the
minimum weighted loss over the finite scores in [Eq. (4)](#eq-finite):

$$
H(\gamma)=\inf_{g\in\mathbb R^{\mathcal X}}
\sum_{y\in\mathcal X}\left[
\sum_{i=1}^{n_0}\gamma_{i,y}^{(0)}\ell(-g_y)
+\sum_{j=1}^{n_1}\gamma_{j,y}^{(1)}\ell(g_y)\right].
$$

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
Following [[1]](#ref-bertsimas2021unified), the next result expresses the
selection objective as a maximum of functions affine in $z$.

<a id="prop-dual"></a>

**Proposition 2.2 (Convex selection function).**
*For $z\in[0,1]^D$, define*

<a id="eq-dual"></a>

$$
\begin{aligned}
F(z)=\max_{\gamma\ge0,\,\mu\in\mathbb R_+^D}\quad&
2\sqrt{H(\gamma)}-M(\gamma)-\mu^\top z\\
\text{subject to}\quad&
\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}
=\frac{M(\gamma)}{n_k},&&i\in[n_k],\ k=0,1,\\
&\sum_{m=1}^D\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+
\le\rho_k M(\gamma),&&k=0,1.
\end{aligned}
\tag{6}
$$

*Then $F$ is convex and nonincreasing, the maximum in [Eq. (6)](#eq-dual)
is attained, and*

<a id="eq-binary-exact"></a>

$$
F(\mathbf1_S)=V(S),\qquad
\min_{|S|\le d}V(S)=\min_{z\in\mathcal Z_d}F(z).
\tag{7}
$$

The row constraints make each class's total mass equal to
$M(\gamma)$; this common mass is allowed to vary, whereas each
class has unit mass in [Eq. (3)](#eq-lfd).
[Appendix A](#app-dual) derives [Eq. (6)](#eq-dual) from the
epigraph of [Eq. (4)](#eq-finite) by duality and a change of scale.
Minimizing $F$ over $\mathcal Z_d$ therefore solves the original
problem; replacing $\mathcal Z_d$ by its continuous relaxation
gives a lower bound.

<a id="sec-algorithm"></a>

## 3. An outer-approximation approach

We use an outer-approximation approach following
[[1]](#ref-bertsimas2021unified), which iteratively refines affine lower
bounds on $F$ over $\mathcal Z_d$.
Since $H$ is nonnegative and concave, [Eq. (6)](#eq-dual) is a convex
optimization problem for fixed $z$.
For an optimizer $(\gamma^t,\mu^t)$ at $z^t$, define

<a id="eq-cut"></a>

$$
\begin{aligned}
c_t(z)&=2\sqrt{H(\gamma^t)}-M(\gamma^t)-(\mu^t)^\top z\\
&=F(z^t)-(\mu^t)^\top(z-z^t).
\end{aligned}
\tag{8}
$$

The feasible set in [Eq. (6)](#eq-dual) is independent of $z$, so
$c_t(z)\le F(z)$ for every $z\in[0,1]^D$, with equality at $z^t$.
After $t$ evaluations, we minimize the accumulated lower bounds
to obtain the next selection:

<a id="eq-master"></a>

$$
\begin{aligned}
(z^{t+1},L_t)\in\operatorname*{arg\,min}_{z\in\mathcal Z_d,\,\eta\ge0}
\quad&\eta\\
\text{subject to}\quad&\eta\ge c_s(z),\qquad s=1,\ldots,t.
\end{aligned}
\tag{9}
$$

The value $L_t$ is a lower bound on
$\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z)$, and the best evaluated
value $U_t=\min_{s\le t}F(z^s)$ is an upper bound.

[Algorithm 1](#alg-oa) alternates between evaluating a selection
and solving [Eq. (9)](#eq-master). It retains the best evaluated
detector and stops when $U_t-L_t\le\varepsilon$.
With exact solves, if $z^{t+1}=z^s$ for some $s\le t$, then

$$
L_t\ge c_s(z^{t+1})=F(z^s)\ge U_t,
$$

so the stopping condition holds. Each nonterminal iteration
therefore produces a new selection, and at most $|\mathcal Z_d|$
selections are evaluated.
Since $F$ is nonincreasing, the budget can
also be imposed as $\mathbf1^\top z=d$ when $d\le D$.

<a id="alg-oa"></a>

**Algorithm 1. Outer approximation for variable selection**

**Require:** Samples $x_i^{(k)}$, radii $\rho_k>0$, budget $d$,
tolerance $\varepsilon>0$, initial $z^1\in\mathcal Z_d$

1. $U\gets+\infty$, $t\gets1$
2. **loop**
3. &emsp;$S^t\gets\{m:z_m^t=1\}$
4. &emsp;Solve [Eq. (4)](#eq-finite) at $S^t$ for $(g^t,a^t,\lambda^t)$ and value $v_t=V(S^t)$
5. &emsp;Solve [Eq. (6)](#eq-dual) at $z^t$ for $(\gamma^t,\mu^t)$; form $c_t$ by [Eq. (8)](#eq-cut)
6. &emsp;**if** $v_t<U$ **then**
7. &emsp;&emsp;$U\gets v_t$; store $S^\star\gets S^t$ and $(g^\star,a^\star,\lambda^\star)\gets(g^t,a^t,\lambda^t)$
8. &emsp;**end if**
9. &emsp;Solve [Eq. (9)](#eq-master) with $c_1,\ldots,c_t$ for $(z^{t+1},L_t)$
10. &emsp;**if** $U-L_t\le\varepsilon$ **then**
11. &emsp;&emsp;Recover $g^\star$ on $\mathbb R^{|S^\star|}$ by [Eq. (5)](#eq-extension)
12. &emsp;&emsp;**return** $S^\star$ and $f^\star(x)=g^\star(x[S^\star])$
13. &emsp;**end if**
14. &emsp;$t\gets t+1$
15. **end loop**

When a solver stops early, feasible solutions of [Eq. (4)](#eq-finite)
give upper bounds, and feasible solutions of [Eq. (6)](#eq-dual) give
cuts $2\sqrt{H(\gamma)}-M(\gamma)-\mu^\top z$.
Use the solver's global lower bound for [Eq. (9)](#eq-master) as $L_t$.

<a id="app-dual"></a>

## Appendix A. Proof of Proposition [2.2](#prop-dual)

<a id="app-dual-duality"></a>

### A.1 Duality

Let $\mathcal D$ denote the feasible set in [Eq. (6)](#eq-dual).
Introduce an epigraph variable $\eta$ in [Eq. (4)](#eq-finite) and write
$\|x_i^{(k)}[S]-y[S]\|_1
=\sum_m z_m|x_i^{(k)}[m]-y[m]|$ for $z=\mathbf1_S$.
The diagonal constraints give $a_i^{(k)}\ge0$, so the epigraph
inequality implies $\sum_k\rho_k\lambda_k\le\eta$.
Introducing $w_{km}$ for $\lambda_k z_m$, consider the following
problem for fixed $z\in[0,1]^D$:

<a id="eq-value"></a>

$$
\begin{aligned}
v(z)=\min_{g,a,\lambda,w,\eta}\quad&\eta\\
\text{subject to}\quad&
\ell((-1)^{k+1}g_y)\le a_i^{(k)}
+\sum_m w_{km}|x_i^{(k)}[m]-y[m]|,
&&k=0,1,\ i\in[n_k],\ y\in\mathcal X,\\
&0\le w_{km}\le\lambda_k,&&k=0,1,\ m\in[D],\\
&\sum_{k=0}^1\rho_k w_{km}\le\eta z_m,&&m\in[D],\\
&\sum_{k=0}^1\left(\frac1{n_k}\sum_i a_i^{(k)}
+\rho_k\lambda_k\right)\le\eta,\\
&g\in\mathbb R^{\mathcal X},\quad a^{(k)}\in\mathbb R^{n_k},
\quad\eta\in\mathbb R.
\end{aligned}
\tag{10}
$$

Using nonnegative Lagrange multipliers $\gamma,\beta,\mu,\alpha$
for the loss, weight, coordinate, and epigraph inequalities,
respectively, we obtain

$$
\begin{aligned}
\mathcal L={}&(1-\alpha-\mu^\top z)\eta
+\sum_{k,i,y}\gamma_{i,y}^{(k)}\ell((-1)^{k+1}g_y)\\
&+\sum_{k,i}a_i^{(k)}
\left(\frac\alpha{n_k}-\sum_y\gamma_{i,y}^{(k)}\right)
+\sum_k\lambda_k\left(\rho_k\alpha-\sum_m\beta_{km}\right)\\
&+\sum_{k,m}w_{km}
\bigl[\beta_{km}+\rho_k\mu_m-c_k(\gamma)[m]\bigr].
\end{aligned}
$$

Minimization over $g$ gives $H(\gamma)$. Minimization over the free
variables $a,\eta$ and the nonnegative variables $\lambda,w$ is
finite precisely when

$$
\begin{gathered}
\sum_y\gamma_{i,y}^{(k)}=\alpha/n_k,\qquad
\alpha+\mu^\top z=1,\\
\sum_m\beta_{km}\le\rho_k\alpha,\qquad
\beta_{km}\ge c_k(\gamma)[m]-\rho_k\mu_m,\qquad\beta_{km}\ge0.
\end{gathered}
$$

Summing the row equations over $i$ shows that each class has total
mass $\alpha$, so $\alpha=M(\gamma)$. The smallest admissible
multiplier is $\beta_{km}=(c_k(\gamma)[m]-\rho_k\mu_m)_+$, so a suitable
$\beta$ exists exactly when the last constraint of [Eq. (6)](#eq-dual)
holds. Eliminating $\alpha$ and $\beta$ thus leaves
$(\gamma,\mu)\in\mathcal D$ and $M(\gamma)+\mu^\top z=1$.

Problem [Eq. (10)](#eq-value) is feasible (take $w=0$ and a feasible point
of [Eq. (4)](#eq-finite) with $S=\emptyset$), and its minimum is attained by
the argument given for [Eq. (4)](#eq-finite), since
$0\le w_{km}\le\lambda_k$. Adding $\delta>0$ to every $a_i^{(k)}$ and
$3\delta$ to $\eta$ makes the loss constraints strict and keeps the
affine ones, so Slater's condition gives strong duality with an
attained dual optimum:

<a id="eq-normalized-dual"></a>

$$
v(z)=\max_{\substack{(\gamma,\mu)\in\mathcal D\\
M(\gamma)+\mu^\top z=1}}H(\gamma).
\tag{11}
$$

<a id="app-dual-binary"></a>

### A.2 Equality at binary selections

We prove $v(\mathbf1_S)=V(S)$. Let $\gamma^\star$ maximize
[Eq. (3)](#eq-lfd) and set

$$
\mu_m=\begin{cases}
0,&m\in S,\\
\displaystyle\max_{k=0,1}\frac{c_k(\gamma^\star)[m]}{\rho_k},&m\notin S.
\end{cases}
$$

Then $M(\gamma^\star)=1$, $\mu^\top\mathbf1_S=0$, and

$$
\sum_m\bigl(c_k(\gamma^\star)[m]-\rho_k\mu_m\bigr)_+
=\sum_{m\in S}c_k(\gamma^\star)[m]\le\rho_k.
$$

Thus $(\gamma^\star,\mu)$ is feasible in
[Eq. (11)](#eq-normalized-dual), giving $v(\mathbf1_S)\ge V(S)$.

For the reverse inequality, let $(\gamma,\mu)$ be feasible in
[Eq. (11)](#eq-normalized-dual) at $z=\mathbf1_S$.
The normalization gives $0\le M(\gamma)\le1$.
Let $\iota_{i,y}^{(k)}=n_k^{-1}\mathbf1_{\{y=x_i^{(k)}\}}$
be the transport that leaves each observation in place, so
$c_k(\iota)[m]=0$ for all $k,m$.
Completing the missing mass gives
$\gamma'=\gamma+(1-M(\gamma))\iota$, whose rows sum to $1/n_k$ and
whose cost satisfies

$$
\begin{aligned}
\sum_{m\in S}c_k(\gamma')[m]=\sum_{m\in S}c_k(\gamma)[m]
&\le\sum_m\bigl(c_k(\gamma)[m]-\rho_k\mu_m\bigr)_+
+\rho_k\sum_{m\in S}\mu_m\\
&\le\rho_k\left(M(\gamma)+\mu^\top\mathbf1_S\right)=\rho_k.
\end{aligned}
$$

Hence $\gamma'\in\Gamma_S$, and since $\gamma'\ge\gamma$ and
$\ell\ge0$, $V(S)\ge H(\gamma')\ge H(\gamma)$. Maximizing over [Eq. (11)](#eq-normalized-dual) proves
$V(S)\ge v(\mathbf1_S)$.

<a id="app-dual-scaling"></a>

### A.3 Scaling and convexity

It remains to remove the normalization.
The constraints defining $\mathcal D$ are homogeneous in
$(\gamma,\mu)$, and $H(s\gamma)=sH(\gamma)$ for $s\ge0$.
If $M(\gamma)+\mu^\top z=0$, the row constraints force
$\gamma=0$, and the objective in [Eq. (6)](#eq-dual) is zero.
Otherwise, division by $M(\gamma)+\mu^\top z$ produces a feasible
pair in [Eq. (11)](#eq-normalized-dual). Since

<a id="eq-scaling"></a>

$$
2\sqrt{sh}-s=h-(\sqrt s-\sqrt h)^2,
\qquad
\max_{s\ge0}\{2\sqrt{sh}-s\}=h
\quad(h\ge0),
\tag{12}
$$

we obtain

$$
\begin{aligned}
F(z)
&=\max_{\substack{(\gamma,\mu)\in\mathcal D\\
M(\gamma)+\mu^\top z=1}}
\max_{s\ge0}\{2\sqrt{sH(\gamma)}-s\}\\
&=\max_{\substack{(\gamma,\mu)\in\mathcal D\\
M(\gamma)+\mu^\top z=1}}H(\gamma)=v(z).
\end{aligned}
$$

The scalar maximum is attained at $s=h$, so attainment in
[Eq. (11)](#eq-normalized-dual) also gives attainment in [Eq. (6)](#eq-dual).
In particular, $F(\mathbf1_S)=V(S)$.

Finally, $\mathcal D$ is convex, $\sqrt H$ is concave, and $M$ is
linear. Hence [Eq. (6)](#eq-dual) is a convex optimization problem
for fixed $z$. Its objective is affine in $z$ with slope
$-\mu\le0$, and its feasible set is independent of $z$, so $F$ is
convex and nonincreasing. The correspondence
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
