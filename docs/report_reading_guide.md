# How the experimental report is written and filled

The English report is designed to be read in order by a group-meeting audience. It begins with the concrete question, explains one complete oracle/master cycle, separates the data roles, and introduces metrics by the questions they answer. Figures appear after their measurement has been explained. Detailed settings, algebra, code responsibilities, and full result tables are in the appendices.

## Reading route

1. **Sections 1–2:** what is selected, what the risk means, why a fixed-support problem supplies a feasible-test upper bound, and why a reusable dual certificate supplies a global lower cut.
2. **Sections 3–5:** which data a method may see, the four mechanisms, why power/calibration/prediction are different, and precisely what each baseline reproduces.
3. **Sections 6–7:** correctness evidence first, then runtime, solved fraction, achieved bounds, search effort, and the presolve-error remedy. A stronger continuous value is not interpreted as an automatic speedup.
4. **Sections 8–10:** independent power and null calibration, support recovery, common-head prediction, finite population stress, radius sensitivity, and real-data split variability. The report retains cases where robust-risk selection is inferior to marginal selectors.
5. **Section 11:** the separately declared full-ratio master supplement. This is an experiment-motivated extension; it retains affine rows, shares one product vector, and does not replace the frozen primary study.
6. **Section 12 and appendices:** supported conclusions, numerical/statistical limits, complete settings, code map, outcome inventory, and exact source/code provenance.

The terms “upper bound,” “lower bound,” “randomized error,” and “power” always keep their distinct meanings. Repeated conditional test batches are aggregated within training seed. Real-data split SD is not labeled an independent-population confidence interval.

## Template contract for final generation

The authoring source is `reports/report_template.tex`. The root report generator should read it, replace the eight markers below, and write `reports/finalreport.tex`. It must not edit the final PDF's displayed numbers manually. Each marker represents a **complete TeX block**: it may contain paragraphs, captions, complete table environments, and interpretation text. No marker is nested inside a `tabular` or `table` in the template.

| Exact marker | Required final content |
|---|---|
| `%%OVERVIEW%%` | Completed planned/actual run inventory, independent verification result, repository/run version, principal positive and negative findings. Do not use provisional counts. |
| `%%COMPUTATIONAL_TABLE%%` | Compact primary comparisons: all-run solved fraction, total time including unsuccessful outcomes, upper/lower or gap, oracle/master decomposition, calls and supports. State whether original or remedied outcomes are used. |
| `%%STATISTICAL_TABLE%%` | Power/null calibration and common-head classification summaries with effective seed counts; recovery may be in a separate compact table. Keep randomized and deterministic risk distinct. |
| `%%REAL_TABLE%%` | Both public datasets, common-head metrics, finite stress, selection bounds/status/time, valid split counts, and mean ± split SD. |
| `%%SENSITIVITY_TABLE%%` | The three predeclared radii with effective seed counts, downstream results, and a short interpretation without choosing a preferred radius from final outcomes. |
| `%%RATIO_TABLE%%` | Paired exploratory tangent versus ratio-plus-tangent results, achieved bounds, solved fraction, calls/supports, total and component time, and all outcome statuses. |
| `%%REMEDIATION%%` | Original reproducible presolve errors, identical-model retry outcomes, charged time, version/attempt provenance, and remaining unresolved cases. Preserve original attempts. |
| `%%APPENDIX_TABLES%%` | Complete settings-level tables and failure/time-limit inventory. Include effective denominators and correct uncertainty units. A missing value is a dash, not a zero. |

All template numbers outside these blocks are design parameters or mathematical examples, not provisional observed results. The negative qualitative interpretation reflects the primary observations, but final numerical comparisons and scope must be checked against the completed inventory.

## Figure contract and captions

Figures are referenced relative to the `reports/` directory as vector PDFs in `figures/`. Required stems are:

- `gate_mechanism`, `evaluation_split`;
- `optimization_runtime`, `optimization_success`;
- `power_and_calibration`, `support_recovery`;
- `population_stress`, `radius_sensitivity`;
- `real_support_frequency`, `ratio_supplement`.

Captions in the template state the experimental unit, uncertainty construction, and what the panel can support. Final plots must match those captions. In particular, power uses nominal level 0.05 and 199 plus-one permutations; seed uncertainty resamples trained seeds; the population-stress caption never asserts empirical Wasserstein-ball inclusion. Real-data frequencies use the disclosed valid-outcome denominator.

The source uses Times New Roman 11pt body text, Arial restrained navy headings, 23 mm margins, light table rules, and bounded figure heights. Floats are allowed to fit naturally and are stopped at major result boundaries. Full tables belong in appendices; shrink a table only after considering shorter labels or splitting by question. Avoid microscopic all-metric tables and avoid inserting a page break before every figure.

## Final checks

After all replacements, verify that no `%%...%%` marker remains, every cited figure exists, generated counts match the inventory, and no table includes incomplete results as complete. Check that the original and remedied master outcomes are not mixed silently. The baseline must remain “Wang additive-MMD, paper objective,” with the inspected official positive-variance sign discrepancy and evaluation adaptations disclosed.

Compile and render `experiment_report.tex`, then inspect all pages for equation overflow, table readability, caption alignment, float drift, and mostly empty pages. Keep the main narrative readable and place comprehensive settings-level tables in appendices; the amount of evidence determines the final length. The delivered PDF is `reports/experiment_report.pdf`. Draft previews stay in a unique system TEMP directory.
