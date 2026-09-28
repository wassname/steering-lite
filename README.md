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

<!-- Results section drafted by PI/claude-opus 2026-09-28 from RESEARCH_JOURNAL.md; needs wassname's review. -->

Can we make a model more or less sycophantic without breaking its answers? We test this on petergpt's [Bullshit Benchmark v2](https://github.com/petergpt/bullshit-benchmark): 100 questions built on a false premise. A sycophantic answer goes along with the premise; a candid answer says what is wrong with it.

Each method extracts a vector from 256 persona pairs ("sycophantic" against "abrasive"). We then increase the dose from well below C0, the coefficient that gives 1 nat RMS KL, until both +C and -C break down: unfinished, repeated or role-leaking answers on two doses in a row. The Jev judge rates every answer on two scales: how far it accepts the premise (0 names the flaw, 8 accepts it and praises the user) and how damaged it is (0 clean, 4 broken). For each side we take the healthy dose, with mean damage at most 1.5, that has the best on-axis change minus off-axis change. The method's score is its weaker side. `random` is a null: a random direction, walked the same way.

| method | Qwen3.5-4B score↑ | Qwen3.5-27B score↑ | OLMo-2-32B score↑ |
| --- | ---: | ---: | ---: |
| vjp_cache | **+1.14** [+0.75, +1.56] | +0.34 [+0.09, +0.69] | -0.20 [-0.38, -0.10] |
| chars | +0.88 [+0.49, +1.23] | +0.43 [+0.01, +1.11] | +0.13 [-0.04, +0.34] |
| vjp_delta | +0.66 [+0.39, +1.14] | -0.01 [-0.13, +0.25] | -0.05 [-0.16, +0.03] |
| mean_diff | +0.37 [+0.15, +0.78] | **+0.91** [+0.60, +1.33] | **+0.21** [+0.04, +0.50] |
| *random* | -0.07 [-0.22, +0.13] | -0.05 [-0.12, +0.14] | -0.10 [-0.15, -0.02] |

Brackets are 90% bootstrap intervals (seeds, then questions). Seeds: 3 per learned method on the Qwen models, 1 on OLMo. Source: [research journal](RESEARCH_JOURNAL.md), 2026-09-27, commit `a75d69f`, which also has on-axis change as a fraction of the room left by the unsteered answers. These results are exploratory. The VJP methods work on the 4B model but not on the two 64-layer models; the journal lists the checks. Only methods in the journal table are shown here; `just results` scores every walked method.

To reproduce:

```bash
just sweep dev    # dose walks on Modal, dev cohort (every 5th question); `just sweep full` for all 100
just results dev  # judge with Jev (needs OPENROUTER_API_KEY in .env), then tables and plot
```

`just results` writes `index.md`, `plot.png`, `plot.html` and `points.json` to `outputs/bsbench/results/<cohort>/`. [`scripts/bsbench/web/`](scripts/bsbench/web) is an explorer page over `points.json`. [`calibration.py`](scripts/bsbench/calibration.py) checks whether C0 predicts where each walk breaks down, and [`cost.py`](scripts/bsbench/cost.py) estimates the GPU and judge cost per walk.

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
