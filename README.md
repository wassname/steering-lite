# steering-lite

When we steer a model, we want to change one thing without changing everything else. We might want less sycophancy, for example, while keeping its answers to ordinary factual questions the same.

steering-lite does this by changing the model's hidden activations during inference, without retraining. Give it pairs of prompts showing opposite behaviours, extract a steering vector, and apply it while the model generates. How well that works depends on the method and the strength of the steer.

The code is meant to be easy to change: one file per method, starting with [mean_diff.py](src/steering_lite/variants/mean_diff.py). It is a sister project of [lora-lite](https://github.com/wassname/lora-lite), for activation steering rather than adapter fine-tuning.

[Try it](#quickstart) · [Results](#results) · [Value maps](https://github.com/wassname/moral-maps#can-we-steer-these-values)

## Quickstart

From a local checkout, install with `uv pip install -e ".[hf-test]"`. The example uses a CUDA GPU.

```python
import torch
import steering_lite as sl
from steering_lite import Vector
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16,
).cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

pos = ["I want to be helpful and honest.", "I will tell the truth."]
neg = ["I will deceive you.", "I will lie to you."]

v = Vector.train(model, tok, pos, neg, sl.MeanDiffC()).calibrate(model, tok)

inputs = tok("Tell me about yourself.", return_tensors="pt").to(model.device)
with v(model):
    out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))

v.save("honesty.safetensors")
v2 = Vector.load("honesty.safetensors")
```

This shows the API; two prompt pairs are not an evaluation of honesty. Vectors can be scaled with `v * 0.5`, and compatible vectors can be added with `v + v2`.

## How strongly should we steer?

How do we compare methods when one might be strong and one weak? These are calibration questions.

We measure how much steering changes the next-token distribution using KL divergence. Calibration finds a coefficient that reaches the requested divergence budget on a set of prompts, then stores it in the vector. The default `Vector.calibrate` target is 1 nat using the root-mean-square token KL over short, 50-token rollouts. This makes intervention strength more comparable; it does not guarantee that the model remains useful.

```python
v.calibrate(model, tok, target_kl=1.0, target_stat="kl_rms")
```

## Results

<!-- Results section drafted by PI/claude-opus 2026-09-28. Numbers copied by script from `just results` output:
outputs/bsbench/results/{full,27b-full,olmo-full}/index.md (2026-09-28). Needs wassname's review. -->

We test sycophancy with petergpt's [Bullshit Benchmark v2](https://github.com/petergpt/bullshit-benchmark): 100 questions with a false premise. A sycophantic answer accepts the premise. A candid answer says what is wrong with it.

We steer in both directions: +C toward sycophantic, -C toward abrasive. For each method we increase the steering strength until the answers break (unfinished, looping or repeated). A judge model, Jev, rates each answer on two things: how much it accepts the premise, and how damaged it is.

The score is the change we want minus the damage, at the best strength, in the weaker of the two directions. Higher is better. `random` (a random direction) shows what noise scores.

![Pareto plot, Qwen3.5-4B](assets/bsbench_qwen3.5-4b_full.png)

The plot shows the 5 best methods on Qwen3.5-4B. Left is -C, right is +C, and lower means more damage. Each line stops at the last strength that still gave coherent answers.

The table shows every method we tested. Brackets are 90% bootstrap intervals. Most methods were tested only on the small model.

| method | Qwen3.5-4B score↑ | Qwen3.5-27B score↑ | OLMo-2-32B score↑ |
| --- | ---: | ---: | ---: |
| vjp_cache | **+1.14** [+0.75, +1.56] | +0.34 [+0.09, +0.69] | -0.20 [-0.38, -0.10] |
| chars | +0.88 [+0.49, +1.23] | +0.43 [+0.01, +1.11] | +0.13 [-0.04, +0.34] |
| linear_act | +0.71 [+0.39, +1.01] |  |  |
| vjp_delta | +0.66 [+0.39, +1.14] | -0.01 [-0.13, +0.25] | -0.05 [-0.16, +0.03] |
| spherical | +0.54 [+0.10, +1.01] |  |  |
| directional_ablation | +0.47 [+0.20, +0.84] |  |  |
| mean_diff | +0.37 [+0.15, +0.78] | **+0.91** [+0.60, +1.33] | **+0.21** [+0.04, +0.50] |
| topk_clusters | +0.33 [+0.09, +0.63] |  |  |
| corda_pca | +0.26 [-0.02, +0.72] |  |  |
| cosine_gated | +0.14 [-0.03, +0.49] |  |  |
| query_steer | +0.10 [-0.15, +0.41] |  |  |
| sspace_ablate | +0.07 [-0.09, +0.36] |  |  |
| super_sspace | +0.06 [-0.13, +0.33] |  |  |
| sspace | +0.01 [-0.08, +0.27] |  |  |
| sspace_pca | -0.07 [-0.28, +0.16] |  |  |
| *random* | -0.07 [-0.22, +0.13] | -0.05 [-0.12, +0.14] | -0.10 [-0.15, -0.01] |
| pca | -0.12 [-0.30, +0.21] |  |  |
| sspace_damp_amp | -0.14 [-0.32, +0.14] |  |  |
| kv_cache_gram | -0.25 [-0.50, -0.12] |  |  |
| *prompting* | — | — | — |
| *prompting_engineered* | — | — | — |

Seeds: 3 per method on the Qwen models, 1 on OLMo. `—`: the prompting baselines have no score, because one of their directions failed the coherence or damage check. These results are exploratory.

### Larger models

<!-- PI/Claude 2026-09-29, numbers from outputs/bsbench/results/{full,27b-full,olmo-full}/index.md. Needs wassname's review. -->

Four methods also ran on Qwen3.5-27B and OLMo-2-0325-32B-Instruct. The score can be low for two reasons: the method does not steer, or the model already gives the target answer. Bare Qwen3.5-27B already rejects 69 of the 100 false premises, against 35 for the 4B model ([by_stance.md](slop/reviews/2026-09-28_judged_by_stance/by_stance.md)), so there is little left to steer toward candour. The second number for each model divides the on-axis change by the room left (how far the bare answers could still move toward that side), in the weaker direction.

| method | 4B score↑ | 4B on ÷ room↑ | 27B score↑ | 27B on ÷ room↑ | OLMo score↑ | OLMo on ÷ room↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vjp_cache | **+1.14** [+0.75, +1.56] | **+0.40** | +0.34 [+0.09, +0.69] | +0.33 | -0.20 [-0.38, -0.10] | +0.00 |
| chars | +0.88 [+0.49, +1.23] | +0.35 | +0.43 [+0.01, +1.11] | +0.36 | +0.13 [-0.04, +0.34] | +0.23 |
| vjp_delta | +0.66 [+0.39, +1.14] | +0.24 | -0.01 [-0.13, +0.25] | +0.13 | -0.05 [-0.16, +0.03] | +0.03 |
| mean_diff | +0.37 [+0.15, +0.78] | +0.14 | **+0.91** [+0.60, +1.33] | **+0.64** | **+0.21** [+0.04, +0.50] | **+0.27** |
| *random* | -0.07 [-0.22, +0.13] | +0.01 | -0.05 [-0.12, +0.14] | +0.02 | -0.10 [-0.15, -0.01] | +0.01 |

Within each model, the two columns rank these four methods in the same order, so the room correction does not change which of them looks best there. Between models the order changes: mean_diff is 4th of 4 on the 4B model and 1st on both larger models. On Qwen3.5-27B the VJP methods keep much of their effect per unit of room (vjp_cache +0.40 → +0.33). On OLMo they barely steer (on ÷ room +0.00 and +0.03, random +0.01), while mean_diff and chars still do. The [research journal](RESEARCH_JOURNAL.md) has the checks for a bug (none found) and extra `vjp_delta` runs with other settings.

Each plot shows up to 5 best-scoring methods on that model (all 4 on the larger models), so the 4B plot above does not include mean_diff (7th there). The axis ranges differ between plots: compare the order of the curves, not their lengths.

![Pareto plot, Qwen3.5-27B](assets/bsbench_qwen3.5-27b_full.png)

![Pareto plot, OLMo-2-0325-32B-Instruct](assets/bsbench_olmo-2-32b_full.png)

To reproduce:

```bash
just sweep full    # steering runs on Modal, all 100 questions (`dev` = 20 questions)
just results full  # judge with Jev (needs OPENROUTER_API_KEY in .env), then the table and plot
```

The outputs go to `outputs/bsbench/results/<cohort>/`. [`calibration.py`](scripts/bsbench/calibration.py) and [`cost.py`](scripts/bsbench/cost.py) are extra checks: where each run breaks down, and what a run costs.

## Methods and debugging

Each implementation includes its own math and references in [the variants directory](src/steering_lite/variants). Start with [mean difference](src/steering_lite/variants/mean_diff.py) or [PCA](src/steering_lite/variants/pca.py). Newer variants are [VJP delta](src/steering_lite/variants/vjp_delta.py), [VJP cache](src/steering_lite/variants/vjp_cache.py), [KV-cache Gram](src/steering_lite/variants/kv_cache_gram.py) and [query steering](src/steering_lite/variants/query_steer.py). [S-space](src/steering_lite/variants/sspace.py) also supports `gate="signed"`. [Random](src/steering_lite/variants/random.py) is an evaluation-only null baseline.

The repo also includes clustering, gated and SVD-space methods, directional ablation, spherical steering, CHaRS, Linear-AcT, and angular steering.

See also [weight-steering](https://github.com/wassname/weight-steering), [IBM AISteer360](https://github.com/IBM/AISteer360), and [repeng](https://github.com/vgel/repeng).

## Citation

```bibtex
@misc{wassname2026steeringlite,
  title = {steering-lite},
  author = {Michael J Clark},
  year = {2026},
  url = {https://github.com/wassname/steering-lite}
}
```
