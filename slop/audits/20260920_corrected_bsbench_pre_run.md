# Corrected BS-bench pre-run note

## Decision

Run only the remaining final-generation and judge work after the paid-disabled preflight and tests pass. The 20 numbered BS-bench questions are the main activation-method evaluation. The four disjoint cases are transfer-only evidence.

## Fixed design

- Candidate selection uses the largest observed coefficient with `generation_health.reasons == []`.
- `directed - 4 * off_target` remains a reported ranking statistic. The old `score > 0` and off-axis cutoff are retained only in old cache records; neither selects or stops work.
- For each vector method, measure RMS-KL at its selected candidate, predict coefficients on the 20-question evaluation set and each transfer case, then generate and judge 0.8×, 1.0×, and 1.2× doses.
- Keep every measured generation, including unhealthy text and negative scores.

## Cost and cache checks

- Existing committed accounting: `$11.1300113737`.
- Remaining expected work: `$9.352188` for six method-level final runs and 2,016 aware plus blind judge requests.
- One affected-method retry reserve: `$1.558698`.
- Conservative total: `$22.0408973737`, strictly below `$50`.
- `slop/verification/20260920T112100Z_corrected-upstream-cache-reuse.json` verifies 38 accepted upstream cache records and no open unresolved reservation.

## Options and predictions

| option | prediction | decision |
|---|---|---|
| Reuse accepted upstream cache records and run six final method stages | all 38 upstream records reuse; six new Modal stages and 2,016 aware plus blind requests | selected |
| Recreate candidates after the selector correction | duplicates already-paid work without changing candidate texts | rejected |
| Treat the ranking score or off-axis cutoff as a control rule | repeats the invalid terminal shortcut | rejected |

## Failure diagnoses to check while running

1. **Cache identity mismatch**: a reported compatible record does not match a current non-code identity. Expected signal: a cache miss before final generation. Stop before paid work.
2. **Remote final contract mismatch**: Modal output does not exactly match the 84-item plan. Expected signal: validation error before final judge work. Stop, preserve reservation/provider evidence, reconcile it.
3. **OpenRouter timeout or rate limit**: a final judge request has no known outcome. Expected signal: exception plus no receipt. Stop later dispatch and append the unresolved event before any retry.
4. **Behavioral result is poor**: negative ranking scores or health reasons. This is an outcome to retain, not a reason to omit the final record.

-- PI[gpt-5.6-terra]
