# Report storyboard

The report is an explanation of a reproducible study, not a leaderboard.

1. Begin with the one research question and distinguish optimization evidence from statistical evidence. A reader should know what the method selects and what its numerical certificate says before seeing a result table.
2. Walk through one oracle/master cycle using a short algorithm and a one-coordinate analytic figure. Explain the risk variable, transport mass, and coordinate multiplier in words.
3. Explain how the local literature led to the evaluation: Xie for independent error and nonlinear alternatives; Wang for power/null calibration and FDP/NDP; Swaroop for the shared downstream training split; Bertsimas for computational ablations. State adaptation boundaries.
4. Describe each synthetic mechanism in familiar distributional terms, the complete split diagram, and real-data provenance. Show exactly which data each fitted quantity can see.
5. Introduce metrics by question, with denominators and risk normalization. Equality-test rejection and individual classification error have separate definitions and panels.
6. Present correctness and computational evidence first. A timeout is visible, methods receive equal budgets, and a stronger oracle need not be faster. State this before presenting speed results.
7. Pair testing power with null rejection in the next figure. Then show FDP/NDP and discuss equal-marginal alternatives. Avoid a single weighted score that hides these differences.
8. Present common-head deployment and stress behavior, distinguishing finite stress distributions from the certified Wasserstein supremum. Explain radius sensitivity with a fixed primary radius.
9. Present two real datasets using a compact table and variable-frequency panel. Use repeated-split SD, not independent-sample CI claims.
10. End with supported conclusions and limits: sample size, dimension, time limits, failures, shared-head evaluation, baseline reproduction choices, and no unearned novelty claim. Put detailed results, code map, settings, equations, and checks in appendices.

Layout: readable 11pt English, generous line spacing, restrained blue headings, math adjacent to its explanation. Each figure answers one question, uses consistent method colors, includes uncertainty/repetition units in the caption, and is saved as editable vector PDF/SVG. Tables use short column names explained immediately above, light rules and explicit units. No dense dictionary of unexplained acronyms on the opening pages.
