# Equivalent core refactor evidence

These files copy the independent temporary audit without moving or altering its originals. No frozen research record was modified. This is an equivalence check, not a performance experiment.

- `regression.log` / `regression.xml`: 75 focused tests passed; the log retains one Clarabel accuracy warning. Coverage includes cached-model failure guards, LP/cone certificates, global cuts, original transport references, master retries, and ratio/enumeration.
- `equivalence.json`: 24 before/after logistic cone models have identical canonical A, b, c, and cone dimensions; 15 critical LP/repair/master/retry/enumeration functions have unchanged ASTs; eight invalid-option errors match; CLI bytes are unchanged.
- `check_equivalence.py` and `*_before.py`: the audit procedure and original source snapshots. The new source is the canonical repository source at the hashes below.
- `cli_master.json` / `cli_ratio.json`: both legacy CLI checks returned optimal support [1,5]; the tangent run also completed enumeration.
- `core_mapping.md`: concise mathematical-to-function explanation.
- `core_point_audit.md`: separate read-only analysis of the existing same-point dual-face tangent selection; no new algorithm or benchmark.

Frozen refactor source SHA256:

| File | SHA256 |
|---|---|
| oracle.py | c056a4fdfa6fae866d027ae5a3575b02e7cafc515f58f22323d699a149f4ef9a |
| solver.py | 7a3a436724b831ddce5ca29396f9cdf28043f082cb97fecf62ea25f9132f4b48 |
| cli.py (unchanged) | 11700822149a51d777b7ba508b3a84e813e9bfc5bf59e0e002f95f52ee71495b |

Verified function starts in these sources: oracle `envelope`:21, `_repair_primal`:194, `_primal_lp`:228, `_dual_lp`:295, `_primal_cone_model`:429, `_dual_cone_model`:496, `_repair_dual`:576, `solve`:661; solver `_master_impl`:63, `_milp_with_retry`:82, `_ratio_master`:125, `solve`:324, `enumerate_supports`:349; CLI `main`:12.
