# How to read and regenerate the experimental report

Updated 11 October 2026. The English and Chinese reports contain the same experiments, formulas, numbers, figures and section order. Neither version has an appendix. The report is organized by scientific questions; complete historical comparisons remain in `docs/experiment_development_record.md` and the original result directories.

## The reading order

1. **Twelve variables, three observations slots.** The first section explains what selection is intended to preserve, how four synthetic population differences are constructed, and what information each method may use. Selection, prediction fitting and final testing use disjoint observations.
2. **Can a difference still be detected?** All selected supports first receive one common RBF two-sample test. Power measures detection when there is a real difference; false alarms measure rejection when there is none. The first figure shows power, including the known-signal and all-feature diagnostic references. A small table reports false alarms.
3. **What changes when learned weights are retained?** The implemented sparse additive selector also learns coordinate weights. A separate table compares testing the same support by the common RBF statistic and by the weighted additive statistic. Common support-only power is not the performance of the complete weighted method.
4. **What was selected and missed?** FDP is the fraction of selected variables that are irrelevant; NDP is the fraction of true signals omitted. Both should be low. They coincide only for the six selectors whose budget equals the true signal size, three. The recovery figure reports this common value. The all-feature reference has FDP 0.75 and NDP zero, so it is not part of that equality.
5. **Does selection help new predictions?** A separate figure fixes the prediction model to SVM and compares its balanced error. A table then distinguishes a refitted robust classifier's randomized error from its empirical-ball bound. Preset distribution shifts and radii check the full procedure; they are not adversarial optimization over an unknown population.
6. **What happens on real data?** The report gives the same six three-variable selectors on WDBC and Ionosphere. There is no known true support. The robust classifier's randomized error and the SVM's hard-decision error have different definitions. All 40 global real-data selections reach the time cap and are feasible incumbent supports, not certified optima.
7. **Does fewer inner solves mean less time?** The current confirmation separates evaluated subsets, integer-master time and total time. Counts fall from 598 to 453, but summed within-configuration median time increases by 4.3%. The unchanged Gaussian process is not credited as the cause of the reduction. A brief earlier scaling comparison remains separate.
8. **Parameters and full records.** Data/sample counts, bandwidth and fitting choices, numerical tolerances, current prediction settings and software appear together at the end. Detailed development and failure history does not interrupt the experimental questions.

## What is and is not reproduced

The source paper motivates separate testing-power, recovery and computational measurements. Our four mechanisms use independent fixed mean and variance shifts, a pure-dependence change, and a shared-sign mixture. These are not the source paper's four data models. The current statistical study fixes D = 12, d = 3 and 20 selection observations per class; it does not implement the source paper's statistical sample-size/dimension/budget sweeps.

Only the paper's linear additive sparse-kernel criterion is implemented. Quadratic and Gaussian learned kernels, L0-constrained logistic regression and projected-Wasserstein selection are not implemented. L1 coefficient ranking is an additional baseline, not an L0 reproduction. The additive variance coefficient is fixed at 0.1 rather than reproduced through the original cross-validation. The mathematical criterion and evaluation adaptations are described in the last report section and `docs/baseline_reproduction.md`; no complete author-code replay is claimed.

## Version and uncertainty boundaries

The earlier 1,640-record study supplies statistical, real-data and scaling results. Current small-case confirmation is separate: 16 configurations reuse four generated datasets, with three rotated-order timings for each method. Those timings are reduced to a median within each configuration and then summed. They are not 48 independent datasets. The confirmation does not establish new statistical utility or large-instance speed.

Synthetic error bars resample training seeds after averaging test batches within seed. Real-data tables show overlapping-split standard deviation. Failures and time limits remain in counts and charged costs. The unknown population has not been established to belong to an empirical uncertainty ball; an empirical robust-risk optimum is therefore not automatically a population guarantee.

## Generation and code map

`reports/report_template.tex` and `report_template_zh.tex` hold the complete readable text. `scripts/build_report.py` fills nine blocks from audited records. `scripts/report_evidence.py` checks the separately versioned search records and seals, including all four data arrays and exact raw/summary agreement. `scripts/plot_results.py` generates matched English/Chinese figures. No experiment is rerun by this report build.

| Insertion block | Meaning |
|---|---|
| OVERVIEW | Short scientific question, observed statistical limitation and search-cost tradeoff. |
| STATISTICAL_TABLE | Common power figure and result, false-alarm table, separate weighted-additive evaluation. |
| RECOVERY_TABLE | FDP/NDP figure and interpretation; replaces the obsolete RATIO_TABLE insertion name. |
| SENSITIVITY_TABLE | Common SVM prediction, robust fitted-bound versus fresh/stress error, preset radius comparison. |
| REAL_TABLE | Real-data balanced errors and time-limit interpretation. |
| SEARCH_TABLE | Current independent search confirmation and the evaluations/time tradeoff. |
| COMPUTATIONAL_TABLE | Brief earlier scaling comparison, separately labeled. |
| SETUP_TABLE | Data and historical optimization sample settings, placed after results. |
| REMEDIATION | Concentrated implementation parameters and paths to complete records. |

Use `reports/build.ps1 -PrepareOnly` to generate matched figures and both sources from immutable code/template snapshots in an exclusive system TEMP directory, then compile both PDFs in three passes. This does not replace the stable artifacts. Review every rendered page before publishing the pair. `-Draft` is explicitly marked and also stays in TEMP. The default final build validates complete evidence, rejects changes during the build and publishes both languages together with rollback protection.

Each manifest records actual input bytes. A known source snapshot receives a portable repository-relative label only if its SHA matches the canonical source; a different custom external source keeps its real path. Every figure actually included is fingerprinted. Canonical stable PDFs are `reports/experiment_report.pdf` and `reports/experiment_report_zh.pdf`.
