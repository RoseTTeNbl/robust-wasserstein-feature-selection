# Canonical theory and code normalization

At an evaluated support z_s, the code stores tau>0, mu_code>=0 and h_code=entropy with source rows summing to tau*q and tau+mu_code^T z_s=1. Its retained bound is ell=h_code-margin and its original gradient is -h_code*mu_code.

The paper uses mu_paper=mu_code/tau and h_paper=h_code/tau. Then d_s=1+mu_paper^T z_s=1/tau,

- gradient_paper = -h_paper*mu_paper/d_s^2 = -h_code*mu_code;
- safe numerator_paper = [ell]+*d_s = [ell]+/tau <= h_code/tau = h_paper;
- F_paper(z) = ([ell]+/tau)/(1+mu_code^T z/tau) = [ell]+/(tau+mu_code^T z), exactly the code's stored fraction;
- eta+mu_paper^T b>=safe numerator_paper multiplies by tau to tau*eta+mu_code^T b>=[ell]+, exactly _ratio_master's row;
- mu_paper^T z>=safe numerator_paper/U-1 multiplies by tau to mu_code^T z>=[ell]+/U-tau, exactly its implied exclusion row.

Only same-attempt values are combined; old gradient never changes to -[ell]+*mu_code. The theorem permits zero numerator, for which the code simply omits the uninformative fractional row.

The full-budget condition is justified in Appendix C by coordinatewise nonincreasing lower functions and the original test's ability to ignore added coordinates. The master upper bound uses the true incumbent's risk and therefore retains an actual feasible witness. Its shared products add D variables, not D variables per certificate.

Conditional count/optimality statements match the fixed absolute tolerance branch. Relative tolerance permits earlier stopping at epsilon+epsilon_rel*|U|, as the algorithm guide explicitly states. Unresolved inner or master accuracy, time limits and solver failures return their true statuses and available valid intervals.
