# Final review of version 0.5.0

The current algorithm is `full_inner_uncertainty_v3`, with predictor `fixed_hamming_gp_v1`. Every selected binary subset receives a complete inner solve and one tangent at that evaluated point. Only verified actual tests update the upper bound. Predictions rank eligible proposals; they do not supply tests, bounds or cuts.

Read the current bilingual theory note, especially Section 5, Appendix C (optimization guarantee) and Appendix E (risk predictor), together with `docs/adaptive_search.md`. The complete English reading copy has the same mathematical content.

The final test records are `tests_final_gp.xml`, `tests_final_gp.log` and `ruff_final_gp.log`: 150 passing tests, zero failures, one existing Clarabel accuracy warning, and a passing lint check. Earlier numbered logs are development checks, not additional final runs.

`markdown_kernel_conversion.json` checks complete preservation of the final English source. `visual_qa.json` binds the inspected PDFs, matching numbered references and final-page layout to their hashes. `final_gp_audit.json` supplies independent checks of the actual tests, gradients, prediction activity and final source snapshots. `revision_final.json` binds the current theory, implementation, tests and matched validation. The geometric `markdown_conversion.json` remains an earlier prototype record.

The current paired optimization validation is in `../full_inner_uncertainty/`. All 26 cases attained the requested precision against exhaustive reference intervals. The predictor changed proposal ordering, but total full inner calls were 475 versus 467 for master-only search, so these measurements do not show an aggregate reduction in expensive evaluations. All per-case outcomes, traces, generated data and numerical residuals remain available. Single timing measurements do not establish general runtime superiority.

The two geometric-score prototypes are retained separately in `../full_inner_uncertainty_pre_cache_guard/` and `../full_inner_uncertainty_geometric_guarded/`. Their constant historical risk estimates motivated replacing that ranking with the learned predictor. They are not pooled with the current results. The cache-failure before/after records document why stale solver values must not be reused after an unsuccessful solve. The initial-stop records document avoiding unnecessary work when the first real test already meets the target.

The 1,640 historical study records and frozen configurations were preserved. `frozen_records.json` verifies their record hashes. This revision's small computational checks do not support claims about population testing power or variable recovery.
