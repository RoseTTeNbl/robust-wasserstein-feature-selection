Provenance: `development.json` and the confirmation files were produced from the working tree on
top of commit `0eb179a`, before the study code (`src/robusthw/experiments/study20.py`,
`configs/study20.json`, `scripts/run_study20.py`) and the retry in `selection._master` (HiGHS
scheduler started with another thread count) were committed; they were committed together with
these results. The retry only affects runs that previously stopped with `solver_error`. `develop.log`
and `confirm.log` are committed despite the `*.log` ignore rule; the confirmation exited with status 0.
The two outer-approximation runs that stopped at the 600 s limit (d = 3) are kept in
`computation.json` with status `time_limit`.
