# steering-lite

Hackable forward-hook activation steering. One file per method. ~600 LoC.

Sister project of [lora-lite](https://github.com/wassname/lora-lite). Same hackable
research-code aesthetic (einops, jaxtyping, fail-fast), but for **activation steering** instead of
adapter fine-tuning. Verbs are repeng-style (`train` -> `attach` -> `detach`).

## Quickstart

```python
import torch, steering_lite as sl
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", torch_dtype=torch.bfloat16)
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

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
| Cosine-gated      | [cosine_gated.py](src/steering_lite/variants/cosine_gated.py)   | [Lee+ 2024 CAST](https://arxiv.org/abs/2409.05907)             |
| S-space (SVD)     | [sspace.py](src/steering_lite/variants/sspace.py)               | [ssteer-eval-aware](https://github.com/wassname/ssteer-eval-aware) |
| Spherical (slerp) | [spherical.py](src/steering_lite/variants/spherical.py)         | [chili-lab/Spherical-Steering](https://github.com/chili-lab/Spherical-Steering) |

## Eval

`scripts/daily_dilemmas_benchmark.py` measures **surgical informedness** on
[Daily Dilemmas](https://github.com/wassname/AntiPaSTO3) (1360 moral scenarios, 31 values):
effect on target value minus mean leakage to non-target values.

```sh
just bench Qwen/Qwen3-0.6B-Base mean_diff 2.0
```

### Calibrated results

Qwen3.5-0.8B, target=`honesty`, layer=4, seed=0, n_train=n_eval=32. Coeffs from
[iso-KL calibration](src/steering_lite/calibrate.py) at KL_p95=1.0 nat (greedy,
T=20, N_calib=4) so all methods compare at equal distributional budget.

| method        | coeff | target_effect | leakage_mean | surgical_informedness |
| ------------- | ----: | ------------: | -----------: | --------------------: |
| baseline      | 0.000 |        +0.000 |       +0.000 |                +0.000 |
| cosine_gated  | 6.440 |        -0.040 |       -0.191 |                +0.152 |
| mean_diff     | 0.910 |        -0.170 |       -0.299 |                +0.129 |
| spherical     | 0.262 |        -0.176 |       -0.301 |                +0.125 |
| sspace        | 0.879 |        -0.187 |       -0.298 |                +0.111 |
| topk_clusters | 0.928 |        -0.186 |       -0.270 |                +0.084 |
| pca           | 0.827 |        -0.258 |       -0.281 |                +0.023 |

`surgical_informedness = target_effect - leakage_mean`. Higher = method moves the
target value more than it moves the (mean of 7) non-target values, at matched KL.
Reproduce: `bash scripts/bench_l4_calibrated.sh && uv run python scripts/aggregate_bench_l4.py`.

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