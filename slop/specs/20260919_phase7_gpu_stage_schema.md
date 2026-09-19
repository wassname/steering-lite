# phase 7 feedback graph

No GPU dispatch from the current manifest.

A. Bare and prompting each run generation, health, aware judgment, and blind judgment over all 20 numbered BS-bench development prompts. Each vector method instead has one cached `calibration-candidates` GPU stage, then candidate health, aware judgment, and blind judgment over exactly `BSV2-001` through `BSV2-004`.

The candidate-stage identity includes the reference persona source `(("sycophantic", "abrasive"),)`, template `Answer as someone who is {persona}.`, seed 0, 256 pairs, thinking enabled, method config, and the four actual prompt contents. It extracts a vector, saves it at a content-addressed Modal Volume path, and generates a small successive-approximation candidate-dose set. Its result has vector SHA256/key, candidate coefficients, numbered answers, and prompt spec. Phase A reserves `candidate_dose_upper = 12` as an explicit conservative behavioral-candidate policy cap, independent of `calibrate_iso_kl` iteration settings: each candidate generation, health, aware, and blind stage is audited as four calibration prompts × 12 possible doses, while retaining the four actual prompts in its cache identity. Backends returning more than 12 doses fail before settlement.

B. A cached per-method final GPU stage consumes the exact vector key/hash and order-stable local observed records. It fails before dispatch if a vector, observations, prompt specification, complete case→prompt-record mapping, or loadable non-placeholder transfer data is absent. Each prompt record carries source path, revision, source-file SHA256, and prompt-content SHA256; its identity includes those provenance records alongside vector hash, full observed-record hash, sorted candidate coefficients, prompt specification, and every case's prompt-content hash. It fits RMS-KL, predicts coefficients for loadable disjoint transfer cases, and generates each predicted and nearby dose. The canonical final-dose plan is exactly `0.8 × predicted`, `predicted`, and `1.2 × predicted`; a zero prediction fails before dispatch because it cannot produce distinct nearby doses. Every final record carries the case prompts, provenance, and these coefficients, so its item count is prompts × 3.

Exactly one final GPU record carries the complete actual case→prompts mapping. Final health, aware, and blind records follow it with the same total item count. Vector paths returned from containers are invalid: use a content-addressed Volume object or tested binary sidecar.

The offline `scripts/run_bsbench_sweep.py --run --backend fake --stage vjp_cache` path exercises this graph twice with a deterministic local backend. Its records are marked non-experimental and the second pass must make zero backend calls.

-- PI[gpt-5.6-terra]
