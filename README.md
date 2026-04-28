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

TODO citation bibtex repo