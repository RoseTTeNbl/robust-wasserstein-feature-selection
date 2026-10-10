# Validation of the 10 October 2026 revisions

The current algorithm is version 0.5.0. Its matched validation is `full_inner_uncertainty/`, and its final note/test review is `full_inner_revision/`. Every selected binary support receives a full inner solve and one tangent at that point. A fixed Hamming-kernel Gaussian predictor uses completed actual risk observations to rank candidates by mean minus standard deviation. Its estimates never update optimization bounds or provide the incumbent test. The final search and predictor tags are `full_inner_uncertainty_v3` and `fixed_hamming_gp_v1`.

The earlier version 0.4.0 validation is `adaptive_search/`. Its 26 paired cases remain evidence for that version alone. Earlier attempts are preserved separately and are not pooled into either comparison:

- `adaptive_search_initial_fixed_options/`: initial partial run, before numerical tolerances were made responsive to refinement.
- `adaptive_search_pre_immediate_stop/`: before the additional early-stop checks.
- `adaptive_search_before_bounded_proposals/`: before bounding candidate construction work.
- `full_inner_uncertainty_pre_cache_guard/`: the first completed geometric-score run before cached solver-failure guards; a prototype, not the final predictor.
- `full_inner_uncertainty_geometric_guarded/`: the guarded geometric-score run, retained because its constant risk estimates motivated the new learned predictor.
- `tests_final.json`, `pytest_final.xml`, `markdown_conversion.json`, `revision_final.json`: final review records for version 0.4.0.

Neither revision rewrites the 1,640 historical study records, frozen configurations or the earlier empirical report. Small matched validation assesses optimization accuracy and computational effort; it establishes no support-recovery or population-testing-power result. Wall-clock measurements have one repetition per case and no timing confidence intervals.

The current 26-case comparison attained the requested accuracy against exhaustive reference intervals. In the 24 primary cases, adaptive search used fewer/equal/more full inner calls in 4/12/8 cases (416 versus 410 calls in total), so it does not establish an aggregate reduction in inner solves. Of 449 acquisition batches across all 26 cases, 287 had nonconstant model standard deviations and 197 selected a different subset from the minimum-cut proposal. These diagnostics establish active uncertainty-based ranking, not calibrated probabilities or general speed superiority. See the current validation README and raw records for all statuses and measurements.
