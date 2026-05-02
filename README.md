# steering-lite

Hackable forward-hook activation steering. One file per method. ~600 LoC.
Canonical example: [mean_diff.py](src/steering_lite/variants/mean_diff.py).

Sister project of [lora-lite](https://github.com/wassname/lora-lite). Same hackable
research-code aesthetic (einops, jaxtyping, fail-fast), but for **activation steering** instead of
adapter fine-tuning.

## Quickstart

```python
import torch, steering_lite as sl
from steering_lite import Vector
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

pos = ["I want to be helpful and honest.", "I will tell the truth."]
neg = ["I will deceive you.", "I will lie to you."]

v = Vector.train(model, tok, pos, neg, sl.MeanDiffC(coeff=2.0)) .calibrate(model, tok)

with v(model):
    out = model.generate(**tok("Tell me about yourself.", return_tensors="pt"), max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
# Human: Tell me about yourself.
# AI: I try to be honest and straightforward. I don't always
# succeed, but I genuinely care about giving accurate, useful answers rather than
# what people want to hear.

v.save("honesty.safetensors")
v2 = Vector.load("honesty.safetensors")

combined = v + v2   # add vectors
scaled   = v * 0.5  # scale vector
```

## Calibration

We might ask "How can we steer an LLM as strongly as possible without causing incoherence and collapse?"

This is especially important if we are comparing steering methods, because it's not fair to compare one applied weakly and one applied strongly.

More formally, we can consider that steering is an intervention with side effects. Most often it is some behaviour change vs some performance degradation.


Why 50 tokens? An LLM trajectory is like a car on the road. A small nudge changes lanes;
a big nudge crashes you off course. Most of the divergence between steered and base
happens in the first ~50 tokens. Past there both models mostly self correct, so a cheap 50-token measurement predicts long-horizon coherence.

For a fuller explination see [here](https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7).

`v.calibrate(...)` picks a coefficient `C` so that `KL(steered || base)` hits a target (default 1.0 nat) over the first 50 greedy-decoded tokens, then bakes that `C` into the returned `Vector`.

For each candidate `C`: we greedy sample 50 tokens and record per-token distribution shift
`KL(p_C || p_0)`. We then search for the calibraiton factor where 95% of the tokens have less than our target shift. This is a cheap proxy for "steering as much as possible without causing incoherence or collapse". It also gives us a common KL budget across methods, so we can compare them more fairly.

```python
v = Vector.train(model, tok, pos, neg, sl.MeanDiffC()) \
          .calibrate(model, tok, target_kl=1.0)

# v.cfg.coeff is now the calibrated value. Same 1-nat budget across all methods,
# so leaderboard rows are directly comparable.
with v(model):
    ...
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

[tinymfv](https://github.com/wassname/tinymfv) — tiny moral-foundation vignettes,
guided CoT, 64 think tokens then forced JSON-bool answer.

Pipeline: extract on persona-branching contrastive pairs (POS/NEG share suffix,
differ only in system persona), iso-KL calibrate every method to the same KL
budget, then eval. Each stage emits one `logger.info` demo trace (decoded
prompt + generation, special tokens visible) for tokenizer/format debugging.

```sh
just sweep Qwen/Qwen3-0.6B
```

### Results

Leaderboard: Qwen/Qwen3-0.6B, layers mid 25-75%, seed=0, target_kl=1.0, vignettes=airisk.

TODO results pending sweep run.

TODO add example trace 

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
