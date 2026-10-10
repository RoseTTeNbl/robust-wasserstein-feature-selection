# Proof-readability revision

This revision reorganizes the English and Chinese theory appendices around their proof objectives. Every main-text sentence, theorem, equation and algorithm step remains unchanged.

A explains the two directions of exact finite reduction and motivates the whole-space score construction. B connects variable elimination, strong duality and mass normalization to the denominator and global cuts. C follows the algorithm from accurate cuts to separation, counting and termination. D shares one scalar-minimization setup and one boundary argument. E derives prediction by removing the part explained by observations, then obtains the incremental factor by block multiplication.

`review.json` identifies the changed sources and records independent mathematical review. `markdown_conversion.json` verifies complete preservation in the English reading copy and checks all numbered references against the compiled source. `visual_qa.json` identifies the final PDFs and records inspection of both complete rendered documents.

Earlier algorithm and experiment validations remain versioned evidence for their recorded sources. This writing revision does not rerun or revise those experiments. Production scripts, candidate drafts and render assets were kept in the task's system TEMP directory.
