# steering-lite

Hackable forward-hook activation steering. One file per method. ~600 LoC.
Canonical example: [mean_diff.py](src/steering_lite/variants/mean_diff.py).

Sister project of [lora-lite](https://github.com/wassname/lora-lite). Same hackable
research-code aesthetic (einops, jaxtyping, fail-fast), but for **activation steering** instead of
adapter fine-tuning.

## Quickstart

```python
import torch, steering_lite as sl
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

pos = ["I want to be helpful and honest.", "I will tell the truth."]
neg = ["I will deceive you.", "I will lie to you."]

v = sl.train(model, tok, pos, neg, sl.MeanDiffC())

with v(model, C=2.0):
    out = model.generate(**tok("Tell me about yourself.", return_tensors="pt"), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
# Human: Tell me about yourself. 
# AI: I try to be honest and straightforward. I don't always
# succeed, but I genuinely care about giving accurate, useful answers rather than
# what people want to hear.

v.save("honesty.safetensors")
v2 = sl.Vector.load("honesty.safetensors")

combined = v + v2   # add vectors
scaled   = v * 0.5  # scale vector
```

## Calibration

TODO read humanizer skill and summarize from here https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7

`calibrate_iso_kl` finds the coefficient `C` so that `KL(steered || base)` hits a target
(usually 1.0 nat), measured over the first 20 greedy-decoded tokens. This makes coefficients
comparable across methods: a coefficient calibrated to KL=1.0 produces the same amount of
distributional shift regardless of whether it came from `mean_diff` or `sspace`.

FIXME we've mean to calibrate as a verb or part of extraction / train
TODO consider train or extracte?

```python
coeff, history = sl.calibrate_iso_kl(
    model, prompts, cfg, v.state,
    target_kl=1.0, target_stat="kl_p95",
    T=20, eos_id=tok.eos_token_id, pad_id=tok.pad_token_id, device="cuda",
)
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

We run two evals to see how steering changes a model revealed preferences. 

- tinymfc - tiny moral vignettes
- [kellycyy/AIRiskDilemmas](https://huggingface.co/datasets/kellycyy/AIRiskDilemmas)


```sh
just bench Qwen/Qwen3-0.6B Truthfulness mean_diff 2.0
```

### Surgical Informedness (SI)

SI measures whether steering actually flips the model's *preferences* rather than
just adding noise. For a method at coeff `+c`:

- `fix` = P(+c flips a baseline-wrong row to right)
- `broke` = P(+c breaks a baseline-right row)

For the mirrored coeff `-c`:

- `flip` = P(-c flips a baseline-right row wrong)
- `counter` = P(-c breaks a baseline-wrong row)

```
SI = mean(fix - 2*broke, flip - 2*counter) * min(pmass_pos, pmass_neg)^2 * 100
```

### Results

Leaderboard: Qwen/Qwen3-0.6B, target=`Truthfulness`, AIRiskDilemmas, layers mid 30-80%, seed=0.

TODO Results pending re-run on AIRisk eval.

Reproduce:

```sh
just bench-readme Qwen/Qwen3-0.6B Truthfulness
just summarize outputs/airisk_dilemmas/readme_latest/full Truthfulness
```

## Future

- [weight-steering](https://github.com/wassname/weight-steering)
- per-method calibration sweep (open question)
- MoE support, multi-token aggregation


See also

- [IBM/AISteer360 - an extensible library for general purpose steering of LLMs. ](https://github.com/IBM/AISteer360)
- [vgel/repen - a hackable PCA steering library](https://github.com/vgel/repeng)

# Citation

```bibtex
@misc{wassname2026steeringlite,
  title = {steering-lite},
  author = {Michael J Clark},
  year = {2026},
  url = {https://github.com/wassname/steering-lite}
}
```
