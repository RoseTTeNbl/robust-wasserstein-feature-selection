# Experimental report structure and generation contract

The 7 October 2026 revision presents the fixed study as an empirical paper. It does not rerun experiments, alter numerical results or replace the original protocol. The source is reports/report_template.tex; scripts/build_report.py fills it from results/research_20261003 and produces reports/experiment_report.tex with an input manifest.

## Main body

The abstract reports the main findings. The method section states the original robust risk, gate comparisons and certified stopping rule, with a reference to the companion theory note. The design section specifies data roles, the six budget-matched selectors, baseline fidelity, metrics and uncertainty. Results cover computation, testing and recovery, prediction and radius sensitivity, then the two real datasets. Discussion states the scope of the findings once.

There are three main result tables: the D=12 gate comparison on both mechanisms, the four-mechanism empirical-head-bound comparison, and the twelve-row common-classifier real-data comparison. Four main figures use existing assets: combined runtime/success, power/calibration, recovery, and radius sensitivity. Complete tables, diagnostics, native Wang evaluation, numerical replay and the ratio supplement are appendices.

## Eight insertion blocks

| Marker | Content |
|---|---|
| %%OVERVIEW%% | Data-derived abstract; explicit draft stamp if requested. |
| %%COMPUTATIONAL_TABLE%% | Main D=12 comparison and concise computation findings. |
| %%STATISTICAL_TABLE%% | Mechanism findings, independent prediction and empirical-head-bound table. |
| %%REAL_TABLE%% | Main twelve-row real-data classifier table and recorded selection status. |
| %%SENSITIVITY_TABLE%% | Short data-derived joint support/classifier radius findings. |
| %%RATIO_TABLE%% | Full separately labeled paired ratio results in the appendix. |
| %%REMEDIATION%% | Original replay cohort, final outcomes and cumulative cost in the appendix. |
| %%APPENDIX_TABLES%% | Full settings-level results, original study metadata and compact suite/method inventories. |

All markers are complete TeX insertion blocks. Main table references use labels; equation references use \eqref. The generator also detects manual figure inclusions when hashing assets, so all ten figure PDFs remain covered.

## Scientific and integrity constraints

Keep original-risk bounds distinct from independently fitted classifier bounds and held-out error. The shifted-source check has an explicit feasible coupling; population stress and population coverage require separate interpretation. Common support-only RBF power and Wang's native signed additive power are distinct outputs. Source variance-sign differences and approximate support search are disclosed, not silently called an exact author-code replay.

Synthetic intervals resample training seeds after conditional batches are averaged. Real entries show split SD. Time limits and failures remain in solver denominators. The original failure inventory has twelve archived cases; two later sensitivity failures must not be added to that original cohort. Final outcomes remain derivable from published records.

The generator rejects missing or duplicate planned trials, stale summaries/CSV, inconsistent raw identities or numerical fields, missing final replay provenance and erased retry cost. It records configuration, metadata, every raw-record and figure SHA-256 hash. Drafts live only in unique system TEMP directories. Original experiment verification is labeled as such; editorial verification is recorded separately by the parent task.

The parent task compiles, renders and reviews the two revised PDF deliverables. Inspect all pages for table/equation overflow, legibility, float order and continuity; do not force a page count by shrinking text or deleting unfavorable findings.
