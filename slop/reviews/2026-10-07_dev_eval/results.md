# Dev evaluation for new methods (2026-10-07)

PI/OpenAI. Requested by wassname: "a small dev version I can test new methods on".

`just dev METHOD` runs the method on Qwen3.5-9B with seed 0 on the dev cohort (every 5th question, 20 questions, plus their 20 controls). It then judges and renders `outputs/bsbench/results/v5-9b-dev/` (http://localhost:8081/v5-9b-dev/index.html). `just dev-results [METHOD]` re-renders without GPU work.

Comparison methods cost nothing extra: for the dev cohort, `data.walk_certificates` also reads the finished full walks (seed 0, random seeds 0-19), sliced to the 20 dev questions. A dev walk of the same method and seed replaces the sliced full walk. Judge ratings are keyed by request content, so the sliced walks reuse all existing ratings.

Check, from `dev_results.log`:

> JUDGE_CACHE_CHECK bsb required=72885 cached=72885 missing=0
> JUDGE_CACHE_CHECK blind required=1878 cached=1243 missing=635
> UAT_PASS

The 635 new blind ratings were small requests (the earlier 47 cost $0.0026). `--show cache_mean_diff` forced it onto the plot beside the top 5 (`plot.png`).

Limits:
- Doses of sliced full walks were chosen by the 100-question stop rule; a dev walk stops on 20 questions.
- 1 seed and 20 questions give wide intervals. Dev ranks differ from full ranks (dev: vjp_resid +0.17, cosine_gated −0.03, vjp_value −0.07; full: vjp_resid −0.03, cosine_gated −0.20, mean_diff −0.20). Use it to see whether a method moves, not to rank it.
- The GPU time and cost of `just dev` on a new method are not yet measured.
