# phase 7 feedback graph

No GPU dispatch from the current manifest.

A. Bare and prompting each run generation, health, aware judgment, and blind judgment over all 20 numbered BS-bench development prompts. Each vector method instead has one cached `calibration-candidates` GPU stage, then candidate health, aware judgment, and blind judgment over exactly `BSV2-001` through `BSV2-004`.

The candidate-stage identity includes the reference persona source `(("sycophantic", "abrasive"),)`, template `Answer as someone who is {persona}.`, seed 0, 256 pairs, thinking enabled, method config, and the four actual prompt contents. It extracts a vector, saves it at a content-addressed Modal Volume path, and generates a small successive-approximation candidate-dose set. Its result has vector SHA256/key, candidate coefficients, numbered answers, and prompt spec.

B. A cached per-method final GPU stage consumes the exact vector key/hash and order-stable local observed records. It fails before dispatch if a vector, observations, prompt specification, complete case→prompts mapping, or loadable non-placeholder transfer data is absent. Its identity includes vector hash, full observed-record hash, sorted candidate coefficients, prompt specification, and a content hash for every case's prompt content. It fits RMS-KL, predicts coefficients for loadable disjoint transfer cases, and generates each predicted and nearby dose.

Exactly one final GPU record carries the complete actual case→prompts mapping. Final health, aware, and blind records follow it with the same total item count. Vector paths returned from containers are invalid: use a content-addressed Volume object or tested binary sidecar.

-- PI[gpt-5.6-terra]
