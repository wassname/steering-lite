# steering-lite

<!-- Generated from README.qmd by `just docs`; edit README.qmd, not this file. -->

steering-lite is a collection of steering methods, kept hackable and easy to evaluate. Steering changes a language model's behaviour by editing its hidden activations while it generates, without retraining ([Turner *et al.*, 2023](<https://arxiv.org/abs/2308.10248>)); ([Panickssery *et al.*, 2024](<https://arxiv.org/abs/2312.06681>)).

When we steer a model, we want to change one thing without changing everything else. We might want less sycophancy, for example, while keeping its answers to ordinary factual questions the same.

Give it pairs of prompts showing opposite behaviours, extract a steering vector, and apply it while the model generates. How well that works depends on the method and the strength of the steer.

The code is meant to be easy to change: one file per method, starting with [mean_diff.py](src/steering_lite/variants/mean_diff.py). It is a sister project of [lora-lite](https://github.com/wassname/lora-lite), for activation steering rather than adapter fine-tuning.

[Results](#results) · [Try it](#quickstart) · [Add a method](#add-a-method) · [Value maps](https://github.com/wassname/moral-maps#can-we-steer-these-values)

## Results

We steer Qwen3.5-9B on the 100 BullshitBench v2 questions ([petergpt, 2026](<https://github.com/petergpt/bullshit-benchmark>)), which all contain a nonsense premise. +C steers toward going along with the nonsense, −C toward pushing back. Each method gets 3 seeds and a sweep from weak to strong (Figure 1). Table 1 gives each method's best dose.

<img src="assets/bsbench/qwen3.5-9b.png" data-fig-alt="Plot of steering Qwen3.5-9B on BullshitBench v2. The x-axis is the change in premise acceptance compared with the unsteered answer: left means the model pushes back on the nonsense, right means it goes along with it. The y-axis is how much else in the answer changed, from 0 at the top to 1.6 at the bottom. Each line is one method, swept from weak steering near the unsteered answer at the top to strong steering further down. Most plotted methods reach about +1.8 on the right before other change passes 1.2. On the left only VJP-resid gets far, to about -1.0 at other change 1.0; the rest stay near 0. Grey bands show 20 random directions. Stars show prompting: +0.5 on the right; on the left its pushback is cancelled because it also calls legitimate questions nonsense." alt="Plot of steering Qwen3.5-9B on BullshitBench v2. The x-axis is the change in premise acceptance compared with the unsteered answer: left means the model pushes back on the nonsense, right means it goes along with it. The y-axis is how much else in the answer changed, from 0 at the top to 1.6 at the bottom. Each line is one method, swept from weak steering near the unsteered answer at the top to strong steering further down. Most plotted methods reach about +1.8 on the right before other change passes 1.2. On the left only VJP-resid gets far, to about -1.0 at other change 1.0; the rest stay near 0. Grey bands show 20 random directions. Stars show prompting: +0.5 on the right; on the left its pushback is cancelled because it also calls legitimate questions nonsense." />

Figure 1: Steering Qwen3.5-9B on BullshitBench v2: each line is one method swept from weak to strong. Left is pushing back on the nonsense, right is going along with it; lower means more else changed.

The gray shows how much random interventions can change model sycophancy (horizontal) vs side effects (vertical). The sweeps show that as we increase the dose of a steering intervention, it gets stronger effects and side effects, until it breaks down.

I compare to prompting, which I suspect is better if the model wants to change behaviour as instructed, and worse if it doesn't.

Table 1: Best dose per method, Qwen3.5-9B on BullshitBench v2. One row per method, sorted by score (higher is better). An LLM judge compares each steered answer with the unsteered one. It rates the change in premise acceptance (−3..+3) and how much everything else changed (0..4); the table shows means over questions and seeds. Each side (+C, −C) uses the method's best dose: the largest premise change minus other change, with other change at most 1.5 in every seed. The score is the lower of the two sides' premise change minus other change, so it is negative when other change outweighs premise change. −C pushback is the premise change with its sign flipped, reduced by 3 × the rise in legit rejected: the share of 100 sensible control questions, one per benchmark question, that the −C answer calls nonsense (unsteered: 3%). Random has no control questions. — means no dose passed in every seed. [more columns](slop/reviews/2026-10-07_cache_mean_diff/table.md) · [setup details](slop/specs/20261006_bsbench_eval_frozen.md) · [earlier evaluations](slop/research/20261007_historical_bsbench_results/README.md)

| method | score↑ (90% CI) | −C pushback↑ | −C other↓ | +C goes along↑ | +C other↓ | legit rejected↓ |
|----|---:|---:|---:|---:|---:|---:|
| [vjp_resid](src/steering_lite/variants/vjp_resid.py) | **-0.03 (-0.17, +0.15)** | **+1.00** | 1.03 | +1.88 | 1.14 | 12% |
| [cosine_gated](src/steering_lite/variants/cosine_gated.py) | -0.20 (-0.26, -0.15) | -0.01 | 0.20 | +1.81 | 1.23 | 3% |
| [mean_diff](src/steering_lite/variants/mean_diff.py) | -0.20 (-0.26, -0.15) | +0.03 | 0.24 | +1.84 | 1.24 | 3% |
| [cache_mean_diff](src/steering_lite/variants/cache_mean_diff.py) | -0.21 (-0.26, -0.17) | +0.01 | 0.20 | -0.01 | 0.20 | 3% |
| [sspace_pool](src/steering_lite/variants/sspace_pool.py) | -0.21 (-0.27, -0.14) | +0.05 | 0.25 | +1.71 | 1.14 | 3% |
| [vjp_value](src/steering_lite/variants/vjp_value.py) | -0.22 (-0.34, -0.05) | +0.59 | 0.81 | +1.90 | 1.19 | 4% |
| [linear_act](src/steering_lite/variants/linear_act.py) | -0.24 (-0.29, -0.19) | +0.05 | 0.29 | +1.83 | 1.31 | 3% |
| [topk_clusters](src/steering_lite/variants/topk_clusters.py) | -0.25 (-0.29, -0.18) | +0.03 | 0.28 | **+1.95** | 1.29 | 3% |
| [sspace_pca](src/steering_lite/variants/sspace_pca.py) | -0.25 (-0.32, -0.18) | +0.02 | 0.27 | +1.62 | 1.24 | 3% |
| [sspace_scale](src/steering_lite/variants/sspace_scale.py) | -0.27 (-0.36, -0.22) | -0.01 | 0.27 | +0.02 | 0.28 | 3% |
| [query_steer](src/steering_lite/variants/query_steer.py) | -0.29 (-0.39, -0.25) | +0.02 | 0.30 | +0.09 | 0.38 | 3% |
| [value_gram](src/steering_lite/variants/value_gram.py) | -0.30 (-0.41, -0.25) | -0.04 | 0.26 | +0.01 | 0.30 | 3% |
| [chars](src/steering_lite/variants/chars.py) | -0.30 (-0.37, -0.23) | +0.02 | 0.32 | +1.84 | 1.25 | 3% |
| [sspace](src/steering_lite/variants/sspace.py) | -0.31 (-0.43, -0.22) | -0.05 | 0.26 | +0.03 | 0.26 | 3% |
| *[random](src/steering_lite/variants/random.py)* | -0.31 (-0.38, -0.25) | -0.02 | 0.29 | +1.02 | 1.22 | — |
| [pca](src/steering_lite/variants/pca.py) | -0.32 (-0.43, -0.21) | -0.01 | 0.30 | +1.66 | 1.28 | 3% |
| [corda_pca](src/steering_lite/variants/corda_pca.py) | -0.40 (-0.54, -0.29) | -0.07 | 0.34 | +0.00 | 0.28 | 3% |
| [spherical](src/steering_lite/variants/spherical.py) | -0.46 (-0.56, -0.30) | +0.50 | 0.96 | +1.86 | 1.31 | 7% |
| [sink_split](src/steering_lite/variants/sink_split.py) | -0.47 (-0.57, -0.41) | -0.04 | 0.43 | +0.02 | 0.45 | 3% |
| [sink_split_resid](src/steering_lite/variants/sink_split.py) | -0.49 (-0.55, -0.38) | -0.04 | 0.45 | +1.90 | 1.28 | 2% |
| [directional_ablation](src/steering_lite/variants/directional_ablation.py) | -0.51 (-0.61, -0.37) | +0.28 | 0.79 | +1.45 | 1.44 | 4% |
| [sspace_ablate](src/steering_lite/variants/sspace_ablate.py) | -0.63 (-0.75, -0.58) | -0.04 | 0.57 | -0.06 | 0.57 | 3% |
| *[prompting](scripts/bsbench/walk.py)* | -0.94 (-1.22, -0.68) | +0.07 | 1.01 | +0.48 | 0.85 | 48% |
| [angular_steering](src/steering_lite/variants/angular_steering.py) | — | — | — | — | — | — |

I hope we can use this to show how good steering methods are, and make better ones.

## Quickstart

From a local checkout, install with `uv pip install -e ".[hf-test]"`. The example uses a CUDA GPU.

``` python
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

``` python
v.calibrate(model, tok, target_kl=1.0, target_stat="kl_rms")
```

## Add a method

Write one file in [the variants directory](src/steering_lite/variants), register it, and add its name to `METHODS` in `tests/test_pipeline.py`. Then test it in three steps:

``` bash
just check            # tiny random models on CPU: catches crashes, scores mean nothing
just dev my_method    # Qwen3.5-9B, 1 seed, 20 questions, compared with every finished method
just sweep my_method  # the full run: 3 seeds, 100 questions; then `just pull` and `just results`
```

`just dev` and `just sweep` need [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just), pnpm, a Modal account and `OPENROUTER_API_KEY` in `.env`. They cost GPU time and judge credits. Finished answers and ratings are cached and reused.

## Methods

Each file includes its math and paper references. Start with [mean difference](src/steering_lite/variants/mean_diff.py) or [PCA](src/steering_lite/variants/pca.py).

## See also

[weight-steering](https://github.com/wassname/weight-steering), [IBM AISteer360](https://github.com/IBM/AISteer360), and [repeng](https://github.com/vgel/repeng).

## Citation

``` bibtex
@misc{wassname2026steeringlite,
  title = {steering-lite},
  author = {Michael J Clark},
  year = {2026},
  url = {https://github.com/wassname/steering-lite}
}
```
