# Validation of the 10 October 2026 revision

The final matched validation is `adaptive_search/`. It binds the source snapshot to 26 paired cases and exhaustive reference solves. All headline comparisons use only that directory. Earlier runs are preserved separately for provenance and are not pooled into the final comparison.

- `adaptive_search_initial_fixed_options/`: initial partial run, before numerical solver tolerances were made responsive to requested refinement accuracy.
- `adaptive_search_pre_immediate_stop/`: before the additional early-stop checks.
- `adaptive_search_before_bounded_proposals/`: before bounding candidate construction work.
- `tests_final.json` and `pytest_final.xml`: the final complete test and lint results.
- `markdown_conversion.json`: source-to-reading-copy completeness check.
- `revision_final.json`: source/output fingerprints, bilingual review, final PDF visual checks and the scope of the quantitative theorem.

This revision does not rewrite the 1,640 historical study records, frozen configurations or the earlier empirical report. The new small matched validation assesses optimization accuracy and computational effort; it establishes no support-recovery or population-testing-power result. Wall-clock measurements have one repetition per case and no timing confidence intervals.
