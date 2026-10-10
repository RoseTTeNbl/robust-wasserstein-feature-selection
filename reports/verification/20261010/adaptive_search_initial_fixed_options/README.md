# Aborted initial validation attempt

This directory preserves 23 completed matched cases from an initial attempt.
The attempt was interrupted before completion after discovering that reducing
the requested oracle tolerance did not yet change the LP/conic solver options.

These rows are **excluded** from the final comparison. The complete validation
was restarted with a fresh algorithm source snapshot in the sibling
`adaptive_search/` directory after that issue was corrected. Nothing in this
directory replaces the frozen historical study.
