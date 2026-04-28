# steering-lite

Hackable forward-hook activation steering. One file per method. ~600 LoC.

Sister project of [lora-lite](https://github.com/wassname/lora-lite). Same hackable
research-code aesthetic (einops, jaxtyping, fail-fast), but for **activation steering** instead of
adapter fine-tuning. Verbs are repeng-style (`train` -> `attach` -> `detach`).

## Quickstart

```python
import torch, steering_lite as sl
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.bfloat16)
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

pos = ["I want to be helpful and honest.", "I will tell the truth."]
neg = ["I will deceive you.", "I will lie to you."]

cfg = sl.MeanDiffConfig(layers=(15,), coeff=2.0)
vectors = sl.train(model, tok, pos, neg, cfg)
sl.attach(model, cfg, vectors)
out = model.generate(**tok("Tell me about yourself.", return_tensors="pt"), max_new_tokens=64)
sl.detach(model)
```

## Methods

| Method            | File                                                | Paper                                                          |
| ----------------- | --------------------------------------------------- | -------------------------------------------------------------- |
| Mean-diff (CAA)   | [mean_diff.py](src/steering_lite/variants/mean_diff.py)         | [Panickssery+ 2023](https://arxiv.org/abs/2312.06681)          |
| PCA               | [pca.py](src/steering_lite/variants/pca.py)                     | [Zou+ 2023 RepE](https://arxiv.org/abs/2310.01405)             |
| Top-k clusters    | [topk_clusters.py](src/steering_lite/variants/topk_clusters.py) | -                                                              |
| Cosine-gated      | [cosine_gated.py](src/steering_lite/variants/cosine_gated.py)   | CAST-inspired soft gate, [Lee+ 2024](https://arxiv.org/abs/2409.05907) |
| S-space (SVD)     | [sspace.py](src/steering_lite/variants/sspace.py)               | internal activation-diff SVD baseline                         |
| Spherical (slerp) | [spherical.py](src/steering_lite/variants/spherical.py)         | ungated core of [Spherical Steering](https://arxiv.org/abs/2602.08169) |
| CHaRS             | [chars.py](src/steering_lite/variants/chars.py)                 | [Abdullaev+ 2026](https://arxiv.org/abs/2603.02237)            |
| Linear-AcT        | [linear_act.py](src/steering_lite/variants/linear_act.py)       | [Rodriguez+ 2025](https://openreview.net/forum?id=l2zFn6TIQi)  |
| Angular Steering  | [angular_steering.py](src/steering_lite/variants/angular_steering.py) | [Vu+ 2025](https://arxiv.org/abs/2510.26243)              |

## Eval

`scripts/daily_dilemmas_benchmark.py` measures **surgical informedness** on
[Daily Dilemmas](https://github.com/wassname/AntiPaSTO3) (1360 moral scenarios, 31 values):
effect on target value minus mean leakage to non-target values.

```sh
just bench Qwen/Qwen3.5-0.8B mean_diff 2.0
```

### Calibrated results

Qwen3.5-0.8B, target=`honesty`, layers 7..18 (12/24, ~30-80% depth), seed=0,
n_train=n_eval=32. Eval uses **guided CoT**: 32 think tokens under steering,
then force `</think>\nMy choice:` and read log P(Yes) - log P(No)
([gist](https://gist.github.com/wassname/733c568cd29c2a402be4442d6a061899)).
Coeffs from [iso-KL calibration](src/steering_lite/calibrate.py)
at KL_p95=1.0 nat (greedy, T=20, N_calib=4) on thinking-prefix probes so all
methods compare at equal distributional budget. Baseline target logratio = +0.312.

| method        | coeff | target_lr_steered | target_effect Δ | leakage_mean Δ | surgical_informedness |
| ------------- | ----: | ----------------: | --------------: | -------------: | --------------------: |
| mean_diff     | 0.149 |            +0.625 |          +0.312 |         +0.115 |                +0.198 |
| sspace        | 0.148 |            +0.531 |          +0.219 |         +0.089 |                +0.130 |
| topk_clusters | 0.239 |            -0.313 |          -0.625 |         -0.714 |                +0.088 |
| spherical     | 0.050 |            +0.312 |          +0.000 |         -0.057 |                +0.057 |
| pca           | 0.284 |            -0.000 |          -0.313 |         -0.000 |                -0.313 |
| cosine_gated  | 1.968 |            -0.469 |          -0.782 |         +0.292 |                -1.074 |

Δ = steered − baseline Yes/No logratio. `surgical_informedness = target_effect - leakage_mean`,
so positive means the method shifts target-tagged actions more than non-target ones at
matched KL. Reproduce:
`bash scripts/bench_v8_guided.sh && uv run python scripts/aggregate_bench.py outputs/daily_dilemmas/v8_guided`.

## Functional test

A single test runs the **full pipeline at tiny scale** (same code path as the bench, just smaller
args + `tiny-random-LlamaForCausalLM`):

```sh
just smoke
```

Asserts per method: extracted vectors are non-zero AND target effect > non-target effect AND
save/load round-trip preserves generation.

## Future

- [weight-steering](https://github.com/wassname/weight-steering)
- per-method calibration sweep (open question)
- MoE support, multi-token aggregation


See also

- [IBM/AISteer360 - an extensible library for general purpose steering of LLMs. ](https://github.com/IBM/AISteer360)
- [a hackable PCA steering library](https://github.com/vgel/repeng)

# Citation

```bibtex
@misc{wassname2026steeringlite,
  title = {steering-lite},
  author = {Michael J Clark},
  year = {2026},
  url = {https://github.com/wassname/steering-lite}
}
```