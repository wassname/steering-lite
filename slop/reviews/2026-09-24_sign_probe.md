# Sign orientation check (2026-09-24) -- PI/Claude

Question (wassname): do PCA/SVD-based methods (sspace_pca, corda_pca, ...) steer the wrong way because of sign-ambiguous directions, and did the walk do direction flipping?

The walk does no global flip (calibrate_iso_kl runs without sign_probe; C0 = |C0|). Per-method orientation in code: pca and sspace_pca orient to the mean paired diff; sspace / super_sspace / sspace_* / corda_pca / kv_cache_gram build the contrast in the same basis they project back through, so the basis sign cancels.

Judge-free persona probe (walk.py --probe, held-out persona pairs seed 10000+s, C = C0/2): directional form = probability mass moved toward the tokens the positive persona prefers, score = move(+C) - move(-C). Logs: outputs/logs/sign-probe-v2-s0.log, sign-probe-v2-s12.log. (The first, KL-based form was confounded by damage; logs sign-probe-s0.log.)

| method | probe | judge (BS-bench) |
|---|---|---|
| mean_diff, pca, chars, spherical, cosine_gated, linear_act | correct, all seeds (+0.30..+0.48) | correct |
| vjp_delta, vjp_cache | flipped, all 3 seeds (move +C -0.15, -C +0.17) | correct (+C +2.1, -C -1.4; blind judge and reference agree) |
| kv_cache_gram | +C moves toward persona (+0.21), -C barely (+0.01..+0.07) | flipped (+C -0.2, -C +1.25), all 3 seeds |
| sspace, sspace_ablate, sspace_damp_amp, corda_pca | ~0 | weak / mixed |

Conclusion: the probe does not predict benchmark direction for vjp_*, so it is not used to flip anything. kv_cache_gram's +C moves toward the positive persona on persona text but not toward sycophancy on BS-bench (transfer failure, not a sign bug, as far as this shows). The S-space methods barely move the persona distribution at C0/2. Their late "wrong-side" -C points are the near-breakdown damage drift the judge scores as sycophancy (same as random directions).
