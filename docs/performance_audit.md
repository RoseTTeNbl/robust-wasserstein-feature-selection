# Predeclared core-point diagnostic pilot

Declared on 2026-10-03 before these four diagnostic runs. This is a small performance audit, separate from the frozen primary computational comparison. It will not replace an unfavorable primary run or change the default algorithm after observing outcomes.

Run exactly two instances: mean and mixture, seed 3, D=12, n=8 selection observations per class, feature budget 3, theta0=theta1=0.25, overlap loss. Use the frozen synthetic generator, preprocessing, and univariate MMD warm support. For each instance compare risk gates with five root rounds and core-point selection disabled/enabled, under the same 30-second wall limit. Execute sequentially with one BLAS/HiGHS thread. The main computational and statistical runners may be active concurrently, so wall times are diagnostic and share host load.

Save bounds/status, visited supports, oracle/master time, root bound, convergence trace, exact settings and source hashes under `results/performance_audit_20261003/`. Do not execute extra instances or choose a heuristic based on this pilot. Any later full core-point ablation needs a separately frozen supplement before its runs.

## Observed pilot results

All four runs reached the same numerical optimum within the original stopping tolerance. Counts are distinct integer supports out of 220 possible supports. Root calls are included in oracle calls. Times are seconds.

| Instance | Core point | Final UB | Root LB | Total | Oracle | Master | Supports | Calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean, seed 3 | Off | 0.2596741098 | 0.1073940327 | 16.535 | 1.372 | 15.144 | 206 | 210 |
| Mean, seed 3 | On | 0.2596741098 | 0.1076425698 | 16.436 | 2.310 | 14.109 | 187 | 191 |
| Mixture, seed 3 | Off | 0.2933196020 | 0.1186163553 | 17.084 | 1.436 | 15.629 | 206 | 210 |
| Mixture, seed 3 | On | 0.2933196020 | 0.1205500336 | 15.818 | 2.312 | 13.488 | 189 | 193 |

Final absolute gaps are approximately 1.3e-9 without the core point and 2.3e-9 with it. Full JSON records retain traces, sources, parameters and checksums in `results/performance_audit_20261003/`. These four runs are a diagnostic pilot, not a statistical runtime comparison.

The main cost is the integer master: about 86--92% of wall time. Core-point selection visits fewer supports but requires a second oracle LP. In the mean case, lower master time is almost entirely offset by increased oracle time; in the mixture case there is a modest observed reduction. With only two instances and concurrent host load, neither observation supports a general speed claim. Both variants still visit approximately 85--94% of all supports.

## Why stronger convex values need not make this master faster

The theory compares complete convex extensions pointwise. The implemented master stores only a finite set of supporting planes chosen from independently optimized certificates. A supporting plane of a larger convex function can be below a supporting plane of a smaller function elsewhere.

The analytic one-coordinate instance already provides a concrete counterexample. Its triangle extension and risk extension are

\[
T(z)=\max\{0.4,1-1.5z\},\qquad
R(z)=\frac{1}{1+1.5z},\qquad R(z)\ge T(z).
\]

At the common fractional anchor z=0.2, their supporting planes are

\[
\ell_T(z)=1-1.5z,\qquad
\ell_R(z)=\frac{160}{169}-\frac{150}{169}z.
\]

At z=0 the risk plane is 0.946745 while the triangle plane is 1. Thus pointwise extension dominance does not order the planes generated at this anchor. Add an uninformative second coordinate and a budget of one; z=0 then corresponds to selecting that second coordinate and is a legitimate competing support in the same model.

The overlap envelope is piecewise linear in transported masses, and many optimal transport/cost-allocation certificates can exist. Maximizing the normalized dual value determines tightness at the current point, but does not uniquely determine how useful its tangent will be on other supports. The optional core-point solve selects among near-optimal certificates; its validity is preserved by the same independent repairs. It adds a solver call and has no universal speed guarantee.

This outer loop also starts a fresh SciPy MILP search on every iteration. Bertsimas et al., Section 3.1, explicitly identify embedding cuts in one branch-and-bound tree through lazy callbacks as an important implementation improvement. The SciPy `milp` interface used here does not expose that callback workflow. Rebuilding roughly 190--210 integer master searches is a plausible and directly timed explanation of the observed overhead. A callback-capable exact implementation would be a separately planned engineering comparison; it was not added after inspecting these pilot outcomes.

No new mathematical-validity bug or mask/matrix-sign error was identified in this audit. The observed weakness is practical cut/search efficiency, not a license bottleneck or a claim that a dual objective can replace a primal upper bound.

## Declared supplement, not executed

If a fuller core-point comparison is requested, freeze the following paired supplement before its first run. It leaves the primary protocol and default configuration unchanged:

- New seeds 10--19, chosen before seeing their outcomes, so the two diagnostic seed-3 observations are not used as confirmatory instances.
- Mean and mixture; D=12, n_select=8 per class, signal size 3, budget 3, overlap, theta0=theta1=0.25.
- Risk gate, five root rounds, the same MMD warm support; pair core-point off/on, with a 30-second cap per run.
- Use the unchanged frozen data generator and preprocessing. One solver/BLAS thread and sequential paired runs; alternate order by seed parity. Record host load or run on an otherwise idle host.
- Preserve each status, UB/LB/gap, visited supports, root bound, oracle/master/total time, and full trace. Use separate output paths for off/on variants to avoid runner resume collisions.
- Report all 40 runs, including timeouts and failures. Compare paired total time, supports visited, and master/oracle decomposition; do not call a lower mean over solved-only instances an improvement.

This supplement has not been launched. No additional cases were explored after the declared two-instance pilot.
