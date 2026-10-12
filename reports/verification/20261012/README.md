# 12 October revision

The current algorithm is `certified_global_cuts_v1` in package 0.7.0. Each
selected binary subset receives a complete inner solve; a global integer
master then chooses the next subset from the stored tangents and safe
fractions. The current theory contains Appendices A--D, with no GP or
Appendix E. Historical GP implementations remain explicit controls.

`no_prediction/` is the sealed four-procedure comparison on sixteen existing
configurations and four datasets, with three rotated timing repetitions.
It is a retrospective ablation, not independent confirmation or a test of
GP alone. It reuses hash-checked enumeration references. All 192 runs passed.
The current fraction procedure needs 446 completed evaluations versus 601
with tangents alone; the sums of per-configuration median times are 18.509
and 15.672 seconds, respectively. Fewer inner evaluations did not mean
lower total time on these small cases.

`clarity/` retains the earlier appendix-only and report-accuracy audits from
this same session. Those source/PDF hashes refer to an intermediate revision
and are superseded by `publication_audit.json` for the final artifacts.
Frozen statistical records from the earlier study were not modified or
relabeled as results of the new search procedure.

The final English-only Overleaf upload is recorded separately in
`overleaf_publication_audit.json`. The earlier bilingual cloud project is
retained as a backup.
