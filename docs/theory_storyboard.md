# Structure of the formal methods report

Revised on 7 October 2026. The report presents a continuous argument from the robust testing objective to a certified global selection algorithm. The main text states the models and results needed to understand that argument; the appendices supply their complete derivations.

## Main argument

1. **Problem and assumptions.** Define the class distributions, Wasserstein balls, coordinate budget and loss. Fix the sign convention and risk normalization before introducing the constant-test benchmark.
2. **Exact finite testing.** State the finite convex formulation and explain what its test scores, loss intercepts and transport prices represent. State exactness under the declared losses, positive radii and full-space L1 cost.
3. **Logical gates and dual certificates.** Derive the price bound from the constant-test risk, then introduce binary-exact box and shared-triangle gates. Explain their fixed certificate domain. Replace the fixed allowance by the objective itself, normalize the resulting dual and obtain the fractional certificate representation. State integer exactness, convexity and fractional dominance precisely.
4. **Global cuts and algorithm.** Differentiate a touching certificate to obtain a global affine lower bound. Present the integer master and a complete loop that updates the best feasible test and global bounds. State the exact finite-termination guarantee and distinguish it from numerical termination.
5. **Scope.** Identify the population-calibration question and the difference between a stronger fractional oracle and faster computation. Refer empirical conclusions to the separate experiment report.

## Appendices

- **A:** projection and lifting, finite adversarial transport, and the whole-space test extension establishing exactness.
- **B:** attainment, relative Slater conditions and boundary cases.
- **C:** two-way binary equivalence of fixed gates and their conjugate-based duals.
- **D:** objective-scaled gate, normalization, homogeneity, dominance and direct certificate validity.
- **E:** global bounds, finite termination, numerical certificates, cut selection and a scalar example.
- **F:** optional master retaining entire fractional certificates and its product relaxation.

## Writing and layout contract

- Introduce notation before use and explain why each reformulation is needed. Use consistent score signs, risk scales and class indices across the main text and appendices.
- State the assumptions of a result where the result is used. Keep technical calculations in appendices without omitting a step needed to follow the main argument.
- Distinguish feasible certificates, optimal certificates and adversarial distributions. A numerical certificate uses its own objective value.
- Use standard theorem, equation, algorithm, table and reference numbering. Use black serif text, consistent mathematical fonts and restrained spacing; allow natural pagination and prevent floats from appearing before their introduction.
- Keep `note/main.tex` and `note/main.md` mathematically synchronized. Build intermediate files only in a task-specific system TEMP directory.
- The editorial revision changes presentation, not the experiment records or algorithm implementation. Previous sources and PDFs remain recoverable from the archive and Git history.
