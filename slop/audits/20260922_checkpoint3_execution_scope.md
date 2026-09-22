# Corrected execution scope

Recorded by PI/OpenAI. Supersedes the ten-random-seed, all-case judge plan.

Parent decision, 2026-09-22 10:55:44 UTC:

> Random region uses seeds 0..4 (five seeds, the predeclared minimum for a region), not our arbitrary ten-seed expansion. Keep one `random` condition and seed-level points.

> Candidate calibration: generate both signs and retain aware AB/BA×passes0/1 so each candidate dose has the required score. Do NOT run blind discovery on candidate calibration; it is not used to choose the health boundary or establish final faithfulness.

> Final behavioral judging covers only the 20-question evaluation cohort. The four disjoint cases test RMS-KL transfer using measured KL/search/health/generations; do not spend judge calls on them.

> Blind AB/BA×pass0 remains on every final 20-question evaluation point, plus persona validation.

> Preserve the 3-attempt retry behavior, but reserve one attempt immediately before each attempt. Preflight expected work is one attempt/request plus one largest affected-stage retry reserve; the ledger hard cap blocks later retry reservations.

Implementation interpretation:
- Ten vector replicates: five random seeds and five nonrandom methods. Each has a candidate GPU dispatch and a final GPU dispatch: 20 total. Eight condition names remain.
- Each candidate dispatch has at most 12 magnitudes × two sides × four calibration questions. Four aware requests per item; no blind candidate requests.
- Each final dispatch generates six signed doses across 28 prompts (20 evaluation, eight disjoint transfer): 168 generated items. Only 120 evaluation items are behaviorally judged, with four aware and two blind requests each.
- Transfer raw generations, health, target RMS, signed predictions and search histories remain stored. This scope does not claim behavioral transfer scores.
- The bound is one-attempt work plus one largest-stage reserve, not a guarantee of completing all possible retries. Each retry must independently fit under the remaining hard ledger limit, including the $2 external commitment.
- The completed external smoke is conservatively recorded at its original $2 reservation upper, linked to its saved completion summary; this is not an invoice or newly inferred actual cost.
- Directed intended effect is steered minus baseline in the disposition actually requested. The plot's signed-axis value negates this delta only for -C.

Latest execution evidence is `slop/verification/20260922_endpoint-priced-preflight.json`: eligible endpoint prices are $0.03/M input and $1/M output, enforced by `provider.max_price`. The GPU timeout is enforced at 44 minutes. This replaces the earlier aggregate-price preflight; methods, five random seeds, prompts, signed doses, and judge token caps are unchanged. See `20260922_v4_probe_failure.md` for the provider diagnosis, successful six-request probe, and timeout risk. Renderer changes are deferred until cached outcome evidence exists.
