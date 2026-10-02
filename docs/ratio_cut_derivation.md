# Keeping the whole dual certificate: a ratio-cut supplement

**Design date: 3 October 2026. Status: an experiment-motivated, exploratory extension.** The primary experiment protocol and the default tangent algorithm remain unchanged. This note derives a separate master formulation to be evaluated in a paired supplementary experiment. It does not establish a runtime improvement before that experiment is run.

The idea is simple: our risk-scaled dual already gives a global lower bound that is a fraction. The current algorithm keeps a tangent to that fraction. We can instead keep the entire fraction at integer supports, using a small shared set of product variables. The continuous oracle and the original robust testing objective stay the same.

## 1. Start from the optimization target

Let \(z_j\in\{0,1\}\) indicate whether coordinate \(j\) is selected. We use

\[
\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf 1^\top z=d\},
\qquad
\operatorname{OPT}=\min_{z\in\mathcal Z_d} V(z).
\tag{1}
\]

Here \(V(z)\) is the best robust testing risk when the test can use only those coordinates. The original problem allows at most \(d\) coordinates. Since a test can ignore an added coordinate, an optimum can be padded to exactly \(d\) coordinates; this justifies the equality in (1). It is also what permits the strengthening in Section 3.

For a fixed support, the finite test formulation minimizes

\[
J(a,\lambda)
=\sum_{k=0}^1\sum_i q_{ki}a_{ki}
 +\sum_{k=0}^1\theta_k\lambda_k
\tag{2}
\]

subject to

\[
a_{ki}+\lambda_k\sum_j z_j\Delta_{kimj}
\ge \Phi(\sigma_k u_m),
\qquad a_{ki}\ge0,\quad\lambda_k\ge0.
\tag{3}
\]

The source weights \(q_{ki}>0\) sum to one separately in each class. The Wasserstein radii \(\theta_k\) are strictly positive. The quantity \(\Delta_{kimj}\) is the absolute distance from source atom \(i\) in class \(k\) to target atom \(m\) along coordinate \(j\). The targets are the pooled source atoms. The main experiment uses half hinge; the same argument applies to the homogeneous Bayes envelope of logistic loss.

The existing note establishes a convex risk-scaled extension \(R(z)\) on \([0,1]^D\), with \(R(z)=V(z)\) at binary supports. We only need one consequence of its dual: a feasible dual certificate remains useful away from the support at which it was computed.

### A certificate that can be reused at every support

Write a certificate as

\[
c=(\tau,\widehat\Gamma,\widehat\beta,\widehat\mu),
\qquad \tau>0,
\tag{4}
\]

where all entries are nonnegative and

\[
\begin{aligned}
\sum_m\widehat\Gamma_{kim}&=\tau q_{ki},\\
\sum_j\widehat\beta_{kj}&\le\tau\theta_k,\\
A_{kj}(\widehat\Gamma)
 :=\sum_{i,m}\widehat\Gamma_{kim}\Delta_{kimj}
 &\le\widehat\beta_{kj}+\theta_k\widehat\mu_j.
\end{aligned}
\tag{5}
\]

These three conditions contain no \(z\). Thus they describe a reusable cone of certificates. The masses \(\widehat\Gamma\) produce received class masses \(p_{km}=\sum_i\widehat\Gamma_{kim}\). Their unavoidable testing loss is

\[
h=\mathcal H(\widehat\Gamma)
 =\sum_m\psi(p_{0m},p_{1m}),
\qquad
\psi(p_0,p_1)=\inf_u\{p_0\Phi(-u)+p_1\Phi(u)\}.
\tag{6}
\]

For half hinge, \(\psi(p_0,p_1)=\min(p_0,p_1)\). For logistic, it is
\((p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1\), using \(0\log0=0\). Both envelopes are nonnegative and homogeneous of degree one.

Choose a retained numerator \(v\) with \(0\le v\le h\). Exact arithmetic permits \(v=h\). A repaired numerical certificate can use \(v=\max(0,h-\delta)\) with a conservative margin \(\delta\ge0\).

At any support, define

\[
D_c(z)=\tau+\widehat\mu^\top z>0,
\qquad
f_c(z)=\frac{v}{D_c(z)}.
\tag{7}
\]

Dividing every part of (4) by \(D_c(z)\) preserves (5) and gives the dual normalization
\(\tau/D_c(z)+(\widehat\mu/D_c(z))^\top z=1\).
The objective divides by the same amount, by homogeneity. Weak duality therefore gives

\[
\boxed{R(z)\ge f_c(z)\quad\text{for all }z\in[0,1]^D,
\qquad V(z)\ge f_c(z)\quad\text{for binary }z.}
\tag{8}
\]

This is the essential result. A single oracle call provides a whole global lower model, rather than only a number at the queried support. Appendix A proves the same statement directly from the original testing objective (2).

## 2. What is lost when we keep only a tangent?

Suppose certificate \(c\) was generated at \(z^t\), and let
\(\nu_t=D_c(z^t)\). Its tangent is

\[
T_c(z)
=\frac{v}{\nu_t}
 -\frac{v}{\nu_t^2}\widehat\mu^\top(z-z^t).
\tag{9}
\]

The denominator \(\nu_t\) is the actual stored value. It need not be exactly one after floating-point repairs. The familiar expression \(v-v\widehat\mu^\top(z-z^t)\) is the special case \(\nu_t=1\).

The tangent is valid because the fraction is convex. More explicitly,

\[
\nabla^2 f_c(z)
=\frac{2v}{D_c(z)^3}\widehat\mu\widehat\mu^\top\succeq0,
\qquad
\boxed{f_c(z)-T_c(z)
=\frac{v\,[D_c(z)-\nu_t]^2}{D_c(z)\nu_t^2}\ge0.}
\tag{10}
\]

The right side describes precisely what the tangent discards: curvature when the new support has a different total certificate price. The full fraction can rule out a support that its tangent leaves attractive. At the anchor, they agree; they also agree anywhere with the same denominator.

This is a refinement of the existing global-cut method, rather than a new testing objective. It retains more information from each oracle certificate. It gives no general guarantee that the resulting optimization will be faster: a stronger constraint at integer points also costs additional master variables and may change the branch-and-bound behavior.

## 3. Keep the fraction using shared linear variables

Let \(U>0\) be the objective of an independently feasible original test on an admissible support. Then \(\operatorname{OPT}\le U\). We can search only for an objective \(\eta\) between zero and \(U\), because a globally optimal solution remains in that range.

For each coordinate, introduce one shared variable

\[
y_j=\eta z_j.
\tag{11}
\]

For binary \(z_j\), this product has the exact linear representation

\[
\boxed{
\begin{aligned}
0&\le\eta\le U,\\
0&\le y_j,\\
y_j&\le U z_j,\\
y_j&\le\eta,\\
y_j&\ge\eta-U(1-z_j).
\end{aligned}}
\tag{12}
\]

Indeed, \(z_j=0\) forces \(y_j=0\), while \(z_j=1\) forces \(y_j=\eta\). These are the standard McCormick inequalities for a binary variable multiplied by a bounded continuous variable.

The rational constraint \(\eta\ge f_c(z)\) is equivalent to

\[
v\le\eta(\tau+\widehat\mu^\top z)
  =\tau\eta+\widehat\mu^\top y.
\tag{13}
\]

The final expression is linear. Crucially, every certificate uses the **same** \(y\): these variables represent the product of the common risk bound \(\eta\) with the common support \(z\). We add \(D\) continuous variables once, and each new certificate adds one linear row. We do not add a separate product vector for every cut.

Because \(\mathbf1^\top z=d\), multiplication by \(\eta\) also gives the valid identity

\[
\boxed{\mathbf1^\top y=d\eta.}
\tag{14}
\]

It is redundant when \(z\) is binary, but can strengthen the fractional master relaxation. If a future formulation uses \(\mathbf1^\top z\le d\) without padding to full budget, the corresponding valid statement is only \(\mathbf1^\top y\le d\eta\); equality must not be imposed in that case.

With stored certificates \(c_s=(\tau_s,\widehat\mu^s,v_s)\), the supplementary master is

\[
\boxed{
\begin{aligned}
\min_{z,\eta,y}\quad&\eta\\
\text{s.t.}\quad&z\in\mathcal Z_d,\quad (12),\quad (14),\\
&v_s\le\tau_s\eta+(\widehat\mu^s)^\top y,
                         &&s=1,\ldots,t,\\
&\eta\ge T_s(z),         &&s=1,\ldots,t.
\end{aligned}}
\tag{15}
\]

The last line keeps valid affine cuts as well. It matters: linearizing a product with McCormick inequalities is exact at integer supports, but its fractional relaxation need not dominate a tangent. Keeping both families ensures that the supplemented LP relaxation retains the affine information already available. Appendix B gives a small counterexample, even with (14) and an upper bound equal to the exact anchor value.

For direct matrix assembly, the signs of the inequalities in (12)--(13) are

\[
\begin{aligned}
y_j-Uz_j&\le0,&
y_j-\eta&\le0,&
\eta+Uz_j-y_j&\le U,\\
-\tau_s\eta-(\widehat\mu^s)^\top y&\le-v_s.
\end{aligned}
\tag{16}
\]

Together with the variable bounds \(0\le z\le1\), \(0\le\eta\le U\), and \(y\ge0\), these rows are sufficient. The \(z\) variables are binary; \(\eta,y\) are continuous.

### An interpretable support exclusion rule

Any support with \(V(z)\le U\) must satisfy

\[
\frac{v_s}{\tau_s+(\widehat\mu^s)^\top z}\le U
\quad\Longrightarrow\quad
\boxed{(\widehat\mu^s)^\top z\ge\frac{v_s}{U}-\tau_s.}
\tag{17}
\]

This rule has an intuitive reading: a support must collect enough certificate price to make the unavoidable overlap small enough to compete with the incumbent. A support violating it cannot improve the current feasible test. This is a necessary condition, not a sufficient condition for low risk.

The rule is already implied by (12)--(13), since
\(v_s\le\tau_s\eta+(\widehat\mu^s)^\top y\le U(\tau_s+(\widehat\mu^s)^\top z)\).
Thus an explicit (17) may help presolve or diagnostics, but it does not strengthen the complete LP relaxation in (15). Also, it is an **incumbent-dependent restriction**: it is valid for the region \(V(z)\le U\), rather than for every feasible support in the original problem.

## 4. Why the algorithm still gives global bounds

Take an original optimal support \(z^\star\), with value \(V(z^\star)=\operatorname{OPT}\le U\). In the master, set

\[
\eta=\operatorname{OPT},\qquad y=\operatorname{OPT}\,z^\star.
\tag{18}
\]

The product rows and cardinality identity hold exactly. Every retained ratio row holds by (8), and every valid tangent holds by (9)--(10). Thus (18) is feasible for the master, so its optimal value \(L_t\) satisfies

\[
\boxed{L_t\le\operatorname{OPT}\le U.}
\tag{19}
\]

This argument also explains why reducing \(U\) is safe after finding a better feasible original test: an original optimum is still represented. The McCormick rows must be rebuilt or updated consistently with the new \(U\).

If the integer master stops early, use its **global solver bound** for \(L_t\), not the objective of its best master incumbent. A master incumbent is a candidate support with a lower model value; it is not an original feasible-test upper bound and not a global lower bound. If no valid solver bound is available, retain the preceding valid lower bound. Time limits remain time limits, with both bounds reported.

The practical loop is:

1. Start from an admissible support and a feasible original test. Store its upper bound \(U\); the constant test gives a fallback bound \(B\).
2. At a root point or integer candidate, reoptimize the complete risk-gate oracle, repair its certificate, and retain the numerator, scale, prices, and valid tangent.
3. At an integer candidate, also retain an independently feasible original test and reduce \(U\) if its objective improves.
4. Solve (15), update the global lower bound, and query its integer candidate.
5. Stop on the declared certified gap, or report a time/accuracy limit with the achieved bounds.

Fractional root calls supply cuts, but cannot by themselves supply a deployable original-test upper bound. This supplement does not change that separation.

### Exact finite termination, and its numerical limit

Under exact oracle and master solves, an optimal certificate at an evaluated binary support \(z^t\) has
\(f_{c_t}(z^t)=V(z^t)\). Once stored, its ratio row forces the master lower model to equal the true value at that support. If the master revisits an evaluated support, its exact optimum \(L_t\) equals that support's value. Meanwhile the best feasible upper bound is no greater than that value. Together with (19), this forces equality of the lower and upper bounds. Otherwise the master visits a new support. Since \(\mathcal Z_d\) is finite, termination is finite.

A repaired feasible but nonoptimal certificate need not touch \(V\). This preserves (19), but does not prove exact finite termination: a support can be revisited while its oracle gap remains unresolved. The implementation must stop with an accuracy-limit status or refine the oracle, rather than treating a repeat as optimality. Ordinary floating-point repair and margins are not a directed-rounding proof for every decimal on arbitrarily ill-conditioned inputs.

## 5. Scope of the supplementary experiment

The comparison will keep the same objective, risk gate, root stage, initial support, stopping rules, and oracle implementation. The only treatment is the master cut representation: the primary tangent master versus the supplemented ratio-and-tangent master. This prevents a change of selector or a changed uncertainty set from being mistaken for a cut improvement.

The paired exploratory design is \(D=8,d=2\), \(D=12,d=3\), and \(D=20,d=3\); eight selection observations per class; mean and mixture mechanisms; seeds 10--14; a 30-second limit; five root rounds; and no core-point selection. The configuration is to be frozen separately before the first supplementary pilot. Report global bounds, achieved gap, solved fraction, total runtime, oracle/master time, calls, and distinct integer supports. Retain timeouts and failures. Neither a favorable pilot nor stronger integer cuts establishes universal speedup.

The main contribution claimed by this derivation is specific: **the normalized dual can be retained as an exact rational lower model at binary supports, with a shared \(D\)-variable linear lift and an incumbent-based exclusion interpretation.** The experiment determines whether that additional information is computationally worthwhile.

---

## Appendix A. Check the fraction directly against the original risk

This proof keeps the optimization target visible throughout. Let \((a,u,\lambda)\) satisfy the original finite test constraints (3), and put \(s=\widehat\mu^\top z\). From the definition of the Bayes envelope,

\[
\begin{aligned}
\mathcal H(\widehat\Gamma)
&\le\sum_{k,i,m}\widehat\Gamma_{kim}\Phi(\sigma_k u_m)\\
&\le\tau\sum_{k,i}q_{ki}a_{ki}
    +\sum_{k,j}\lambda_k z_j A_{kj}(\widehat\Gamma)\\
&\le\tau\sum_{k,i}q_{ki}a_{ki}
    +\sum_k\lambda_k\left(\sum_jz_j\widehat\beta_{kj}+\theta_k s\right)\\
&\le\tau\sum_{k,i}q_{ki}a_{ki}
    +(\tau+s)\sum_k\theta_k\lambda_k\\
&\le(\tau+s)\,J(a,\lambda).
\end{aligned}
\tag{A1}
\]

The first line optimizes each target's score. The second uses (3) and the source-row equality in (5). The third uses the coordinate coverage inequality. The fourth uses \(0\le z_j\le1\) and the allocation budget. The last uses \(a\ge0\). Hence \(v\le\mathcal H(\widehat\Gamma)\le D_c(z)J\), and minimizing the original feasible-test objective gives \(V(z)\ge v/D_c(z)\) at binary supports. No original budget-feasible adversarial distribution is being assumed: the certificate satisfies the expanded coverage conditions (5).

## Appendix B. Integer dominance does not imply LP-relaxation dominance

The pointwise statement \(f_c(z)\ge T_c(z)\) is exact. However, a fractional McCormick point need not satisfy \(y=\eta z\), even after the budget identity (14). Therefore the projection of the lifted LP can allow \(\eta<f_c(z)\), and even \(\eta<T_c(z)\).

For an explicit example, take \(D=3,d=1\) and

\[
f(z)=\frac{1}{1+4z_1+5z_2+z_3},
\quad z^t=(1,0,0),\quad U=f(z^t)=\frac15.
\tag{B1}
\]

The tangent at the anchor is

\[
T(z)=\frac15-\frac1{25}(4z_1+5z_2+z_3-4).
\tag{B2}
\]

Consider the fractional lifted point

\[
z=(0,5/6,1/6),\quad\eta=1/6,\quad y=(0,1/6,0).
\tag{B3}
\]

It satisfies every McCormick inequality with \(U=1/5\), \(\sum_j z_j=1\), and \(\sum_j y_j=\eta\). The ratio row holds at equality:
\(1\le\eta+4y_1+5y_2+y_3=1\).
Nevertheless,

\[
\eta=\frac16
<T(z)=\frac{14}{75}
<f(z)=\frac3{16}.
\tag{B4}
\]

Thus the cardinality identity strengthens the lift, but does not make its LP projection dominate all tangents. The combined master (15) keeps those tangents explicitly. At binary points there is no such issue: (12) forces the exact product, so each whole-ratio row dominates its matched tangent.

## Appendix C. Conditions that must survive numerical implementation

**A feasible numerator is essential.** The numerator is a lower certificate derived from the repaired dual envelope. A feasible original-test upper objective must not be substituted for \(v\). Retain \(0\le v\le\mathcal H(\widehat\Gamma)\); recompute the envelope after repairing all masses and coverage prices. With both class masses equal to \(\tau\), it also obeys \(\mathcal H(\widehat\Gamma)\le\tau B\), where \(B\) is the constant-test risk.

**Use the actual scale.** Check \(\tau>0\), finite nonnegative prices, and the residuals in (5). Use \(\nu_t=\tau+\widehat\mu^\top z^t\) when constructing a matched tangent. If desired, divide the entire certificate, including \(v\), by \(\nu_t\); this leaves the fraction unchanged. Changing only \(\tau\) to force normalization would generally destroy feasibility.

**A zero-scale certificate carries no information.** If \(\tau=0\), the nonnegative source rows force \(\widehat\Gamma=0\), hence \(h=0\). A valid retained numerator is then zero. Discard that row and keep the nonnegative objective bound. A positive \(v\) with \(\tau=0\) is not a valid certificate. A certificate with \(\tau>0\) but \(v=0\) can likewise be omitted. Clamping a negative, margin-lowered numerator to zero is valid because the envelope itself is nonnegative.

**Zero upper bounds need no division.** If an independently feasible original test has objective exactly zero, nonnegativity proves \(\operatorname{OPT}=0\); terminate before forming \(v/U\). A small positive objective or a rounded display of zero is not that proof. Otherwise use a conservative positive upper bound and avoid rounding it downward in (12) or (17).

**Numerical floor planes and matched tangents are distinct.** The primary implementation constructs a tangent from \(h\) and then shifts it downward by a margin. The new ratio may instead use \(v=h-\delta\). Its exact matched tangent uses \(v\) in both its value and gradient. Do not claim that \(v/D_c(z)\) pointwise dominates a plane constructed with gradient \(-h\widehat\mu\) and a separately shifted intercept: those are different lower models. Keeping both valid rows is safe. The ideal same-certificate dominance statement (10) applies to a matched numerator and scale.

**The cap must come from the original problem.** A fractional gate-primal objective is not a feasible-test upper bound for the binary selection problem. The constant test or an independently repaired original fixed-support test supplies \(U\). If the master with \(\eta\le U\) becomes infeasible despite such an incumbent, the exact theory says that incumbent should still be represented; treat this as numerical/certificate inconsistency, not an optimality certificate. A deliberately strict improvement cutoff requires its own stated tolerance argument.

**No stronger scalability claim follows from this proof.** There are only \(D\) extra continuous variables and one ratio row per certificate, but support search can still be exponential. Poorly scaled prices can also make a lifted master harder. The supplement should be judged using paired computation records, including unsuccessful solves.
