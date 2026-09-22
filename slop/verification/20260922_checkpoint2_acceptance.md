# Checkpoint 2 verification

— PI/OpenAI

Observed from saved logs:
- `20260922_checkpoint2-final-focused-proof2.log`: `102 passed in 4198.00s (1:09:58)`.
- `20260922_checkpoint2-smoke-proof-final.log`: `62 passed in 47.58s`. Earlier chat said 63; the log is authoritative.
- `20260922_checkpoint2-full-proof-final.log`: partial full-suite run stopped on the human's instruction. It contains failure markers; it is not passing full-suite evidence. Full-suite completion is not required for this checkpoint under the updated instruction.

Scope: signed calibration and transfer, fixed extraction layers/readout/skip, reduced VJP intermediates, JSON-stable KL specification, signed cache identity and validation. Focused integration and tiny-model runtime proofs above support this commit, not a claim that every repository fixture or checkpoint-3 path has been migrated.

Checkpoint 3 still needs random replicates, signed report semantics, corrected counts and budget, strict fake CLI reuse, and provider metadata/probe validation before paid dispatch.
