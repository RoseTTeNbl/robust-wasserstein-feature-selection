# Experimental-report storyboard

Revised 12 October 2026. English and Chinese remain complete corresponding versions. Begin with selecting three of twelve variables. Each result section gives the design, the question, and one observed implication. Definitions precede their first use; numerical settings are collected at the end.

| Question | Main display | What it can establish |
|---|---|---|
| Can selected variables preserve a population difference? | Four-panel common-RBF power; six equal-budget methods plus two diagnostic references. | Detection performance of the selected coordinates under a fixed evaluator. |
| Are false alarms controlled? | Eight-row false-alarm table across the four mechanisms. | Recorded rejection on same-population batches relative to nominal upper level 0.05. |
| Does the weighted additive test change the comparison? | Same-support common/native table. | Why dropping learned weights understates the implemented weighted method. |
| Are irrelevant variables selected or true signals omitted? | Four-panel FDP/NDP figure with ideal zero and random expectation 0.75. | Recovery for the six budget-three methods; equality of FDP/NDP is explicitly limited to those methods. |
| Do selected inputs help new predictions? | Separate four-panel SVM balanced-error figure. | Prediction on independently fitted and evaluated common models. |
| What do the robust bound and preset shifts measure? | Robust bound/fresh-error/stress table; preset-radius table. | Empirical-ball guarantees are distinct from population error and three measured shifts. |
| Does the result carry to real data? | Two datasets, six selectors, two clearly defined balanced errors. | Feasible-support predictive utility; real global selections are all time-limited. |
| Do fraction bounds reduce search cost? | Current tangent/fraction ablation table and paired evaluation/time ratios. | Recorded costs of global-master search at every step, with and without fractions. The two earlier prediction procedures are process-level references. |

The statistical study uses the earlier implementation. The revised search solves a global integer master at every step, then fully evaluates its chosen binary subset; it has no prediction model, candidate pool, or periodic refresh. Its 16-case ablation reuses four generated datasets and previous configurations. It is not independent confirmation. No new utility or scaling claim is transferred from the earlier study. State source-paper model and baseline differences once, without claiming complete reproduction.

After the result sections, collect numerical settings, fitting and bandwidth choices, uncertainty calculations, time allowances and exact record paths. Full four-stage development tables, integration replay, earlier ratio results and failure inventory are in `docs/experiment_development_record.md`. The original 1,640 records, configurations, sources and baselines remain unchanged.

Before each figure or table, state the question and observed implication. Each statistical figure contains four matched panels, one metric, common scales, short method ticks, and a shared legend. Use stable method colors and markers throughout, seed means with 95% intervals, and vector PDF/SVG output. Reference supports are separated from budget-matched methods. The recovery and prediction chance lines are defined in captions. Avoid repeated long labels, decorative frames, and reduced paper-size fonts.

The testing section defines a kernel as similarity and MMD as two within-class averages minus twice the cross-class average. The additive comparison states the sparse unit-norm objective `t^T w - lambda w^T Sigma w`, defines every symbol, and distinguishes its variance-penalized training from the raw weighted-MMD permutation test. Signed weights are explicit.

The new search display is generated only from sealed real records in `reports/verification/20261012/no_prediction`. Until those inputs exist, only an explicitly marked draft may be generated; no placeholder values enter a final report. October 10 GP comparisons remain historical in `docs/experiment_development_record.md` and their original sealed directories.

The recorded 192-run ablation passes all checks. Current tangent/fraction search uses 601/446 full evaluations and 15.672/18.509 summed median seconds: fractions reduce evaluations by 25.8% while increasing time by 18.1%. Earlier complete prediction procedures are faster on these small cases; removing prediction simplifies the design and is not evidence that prediction is useless. Their different candidate pools and master frequencies prevent attribution to prediction alone.
