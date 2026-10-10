# Experimental report structure and generation contract

The 10 October 2026 report has complete English and Chinese templates, `reports/report_template.tex` and `reports/report_template_zh.tex`. `scripts/build_report.py` fills both from audited records. Stable products are `reports/experiment_report.pdf` and `reports/experiment_report_zh.pdf`, with corresponding editable TeX and input manifests. The report has no appendix.

## Main body

The design section defines data roles, fixed parameters, six budget-matched selectors, baseline fidelity, metrics and uncertainty. Separate sections give revised-search validation, frozen computation, frozen statistical utility and recorded outcomes. The robust formulation and proofs remain in the companion theory note. The old statistical study and new search study are explicitly different evidence cohorts.

Compact tables show settings, selectors, development/confirmation costs, D=12 computation, the earlier fraction supplement, native versus common Wang evaluation, empirical classifier bounds versus independent errors, radius sensitivity and real-data outcomes. Figures combine runtime/success, power/null rejection, recovery/prediction and paired confirmation costs. Complete secondary metrics and raw traces are linked rather than reproduced as an appendix.

## Nine insertion blocks

| Marker | Content |
|---|---|
| %%OVERVIEW%% | Data-derived abstract; explicit draft stamp if requested. |
| %%SETUP_TABLE%% | Fixed data and optimization settings from frozen configurations. |
| %%SEARCH_TABLE%% | Version-separated development and frozen confirmation results. |
| %%COMPUTATIONAL_TABLE%% | Main D=12 comparison and concise computation findings. |
| %%STATISTICAL_TABLE%% | Mechanism findings, native Wang distinction and empirical-head-bound comparison. |
| %%REAL_TABLE%% | Main twelve-row real-data classifier table and recorded selection status. |
| %%SENSITIVITY_TABLE%% | Joint support/classifier radius results and source-coupling checks. |
| %%RATIO_TABLE%% | Earlier, separately labeled paired fraction results. |
| %%REMEDIATION%% | Compact outcomes, original replay cohort, hardware/backend and charged cost. |

All markers are complete TeX insertion blocks. Tables and equations use stable labels; both languages preserve their order. The generator hashes every figure actually included, not unused legacy assets.

## Scientific and integrity constraints

Keep original-risk bounds distinct from independently fitted classifier bounds and held-out error. The shifted-source check has an explicit feasible coupling; population stress and population coverage require separate interpretation. Common support-only RBF power and Wang's native signed additive power are distinct outputs. Source variance-sign differences and approximate support search are disclosed, not silently called an exact author-code replay.

Synthetic intervals resample training seeds after conditional batches are averaged. Real entries show split SD. Confirmation conditions may reuse generated data across losses/radii and are descriptive paired cases, not independent population trials. Time limits and failures remain in solver denominators. The original failure inventory has twelve archived cases; later sensitivity failures are not added to that original cohort.

New-search timing repetitions are reduced to their within-case median before paired comparisons; their complete charged runtime and status inventory are also retained. The displayed visit count measures distinct evaluated subsets, including a possible inaccurate final point in a limited run. Accuracy checks and the theory's precision-controlled evaluation count are separate. Final confirmation requires a sealed case inventory, nonempty source snapshots and one unambiguous frozen alternative.

The generator rejects missing or duplicate planned trials, stale summaries/CSV, inconsistent raw identities or numerical fields, missing final replay provenance and erased retry cost. It records configuration, metadata, every raw-record and figure SHA-256 hash. Drafts live only in unique system TEMP directories. Original experiment verification is labeled as such; editorial verification is recorded separately by the parent task.

`reports/build.ps1` generates matched figures and both TeX sources in an exclusive system TEMP directory, then compiles both languages in three passes. Only when both logs pass does it publish the pair, editable sources, manifests and generated report figures. Copy failures restore replaced files. `-Draft` leaves all products in TEMP; a final build requires complete separately frozen confirmation evidence. The parent task renders and reviews every page before release.

## Implementation map

`report_evidence.py` reads the versioned search evidence, verifies every sealed input and the four-array data digest, and checks raw/summary equality. It aggregates timing repeats by configuration median, so three timings do not become three independent datasets. `build_report.py` retains the 1,640-record reference audit, builds nine readable insertion blocks and writes source/figure fingerprints. `plot_results.py` produces the same four figures in both languages from those audited readers. No optimization or statistical experiment is rerun by the report build.

The revised-search confirmation is limited to D = 7/10 with six observations per class: 16 configurations reuse four generated datasets. It establishes fewer full evaluations for the frozen stored-fraction rule (598 to 453), but median-summed time increases by 4.3%. The historical scaling and statistical results belong to the earlier implementation.
