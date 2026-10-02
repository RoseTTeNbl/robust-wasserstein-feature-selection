# Storyboard for the English group-meeting note

Planned before rewriting the note, 3 October 2026. The note explains one path from the research question to a certified algorithm. All detailed derivations follow the main presentation in appendices. The source is the verified 2 October theory; the wording and ordering are redesigned for a reader who has not seen the preceding conversation.

## The six-page main presentation

1. **The question and the quantity we minimize.** Introduce two classes, observing at most d coordinates, and class-conditional Wasserstein uncertainty. Define a positive score as evidence for class 0. Explain the sum-risk convention and the distinction between randomized half-hinge risk and logistic surrogate risk. Show only the original objective and constant-test benchmark. End by asking how one support evaluation can guide other supports.
2. **What a fixed support gives us.** Define the pooled target points and coordinate distances before writing the finite convex test. Explain the loss intercept and transportation price in words and identify the resulting feasible-test upper bound. State the finite-LP/whole-space-extension justification in two sentences; give Appendix A as the complete derivation. Introduce the coordinate-price block and prove binary equivalence with a short saturation argument. End with the explicit connection to the reference paper's logical constraint.
3. **The reference paper gives the first algorithm.** Use the natural constant-test bound to define a shared triangle domain. Introduce mass allocation, overlap, budget allocation, and residual coordinate price before presenting the fixed certificate domain. Show the paper-type affine dual and its global cut. Keep the conjugate computation in Appendix B. Explain why the same certificate remains feasible when the support changes.
4. **Our problem-specific strengthening.** Replace the constant risk allowance by the objective actually being optimized. Show the original-objective-scaled gate, its normalized convex dual, and the rational certificate representation. Explain the normalization in words and with one line. State integer exactness and stronger fractional lower bounds together, without interrupting the story with isolated proposition boxes. Full proofs are in Appendices C and D.
5. **The cut becomes a global search algorithm.** Derive the one-line cut from a touching convex certificate and normalized multipliers. Show the integer master, then the feasible-test UB and master LB. Give an explicit loop whose oracle reoptimizes all continuous variables at every candidate. Explain finite termination in words, numerical non-tight certificates, root cuts, and optional dual-face cut selection. Separate fractional lower-bound work from integer feasible-test upper bounds.
6. **A hand example and the scope of the result.** Derive the scalar risk curve for two atoms at distance two. Compare the box, triangle, and objective-scaled values at integer and fractional points in a small table. Close with the exact theoretical claims, limitations, and what experiments must establish. Keep novelty phrased as a problem-specific extension, not a literature-first claim.

## Appendix plan

A. Projection/lifting for the selected L1 metric; finite adversarial LP and its dual; sandwich argument; logistic endpoint extension and half-hinge extension; local overlap functions; no infinite-dimensional minimax swap.

B. Convex partial value function; grouped logical form; sign-consistent Fenchel/conjugate calculation for a nonnegative triangle; expansion into the fixed certificate domain; full box comparison.

C. Integer exactness of the objective-scaled gate; Lagrangian grouped by eta, a, u, lambda, w; normalized dual; positive-scale unscaling and zero-scale case; homogeneity; convexity; dominance; valid tangent for optimal or merely feasible certificates.

D. Feasibility, attainment and relative Slater; independent direct certificate inequality; precise global bounds and exact finite termination; core-point cut selection; zero-radius boundary and numerical caveats; details of the hand example.

## Notation and readability rules

- z is always the selection vector; V is always the original binary risk; R_B is the fixed-triangle extension; R is the objective-scaled extension.
- a is a loss intercept, lambda is a class transportation price, w_j is its active coordinate block, and eta is a risk epigraph variable.
- Gamma is a certificate mass allocation; p is received class mass; H is residual confusion risk. Expanded certificates need not themselves satisfy the original transportation budget.
- beta allocates a class budget; mu is one residual price per coordinate. Hats denote the normalized dual variables only. Tau is the common mass scale.
- Every displayed model is preceded by its purpose and followed by its interpretation. Detailed elimination is moved to the appendix, not omitted.
- Positive radii, fixed feature scales, and full ambient space are stated before being used. Boundary details follow the main story.
- The main presentation has manual page breaks in TeX so each page carries one stage. Use readable serif English text, uncluttered equations, a single accent color, and generous but controlled whitespace.
- This file and the rewritten note contain no experimental results; those belong in the separate report assembled after the frozen protocol is executed.
