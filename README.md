# steering-lite

steering-lite changes a model's hidden activations during inference, without retraining.

[Try it](#quickstart) · [Results](#results) · [Value maps](https://github.com/wassname/moral-maps#can-we-steer-these-values)

## Curious Plot, Models are grown not built

<!-- From wassname's tweet; shortened and arranged by PI/OpenAI for review. -->

Why a shoe pointing right vs a banana pointing left?!? Why are these plots so different?

![Pareto plot, Qwen3.5-4B](assets/bsbench_qwen3.5-4b_full.png)

![Pareto plot, OLMo-2-0325-32B-Instruct](assets/bsbench_olmo-2-32b_full.png)

The gray shows how much random interventions can change model sycophancy (horizontal) vs side effects (vertical). The curves show the best trade-offs; crosses mark the last dose that passes the checks. The axis ranges differ between plots. <!-- PI/OpenAI: clarified endpoints; path length does not measure breakdown. -->

I think it shows that models' internal landscapes vary a lot! I also think this curiousity will hopefully get you, dear reader, to read a little longer.

The sweeps show that as we increase the dose of a steering intervention, it gets stronger effects and side effects, until it breaks down.

I compare to prompting, which is I susepect is better if the model wants to change behaviour as instructed, and worse if it doesn't.

I hope we can use this to show how good steering methods are, and make better ones.

## Quickstart

When we steer a model, we want to change one thing without changing everything else. We might want less sycophancy, for example, while keeping its answers to ordinary factual questions the same.

Give it pairs of prompts showing opposite behaviours, extract a steering vector, and apply it while the model generates. How well that works depends on the method and the strength of the steer.

The code is meant to be easy to change: one file per method, starting with [mean_diff.py](src/steering_lite/variants/mean_diff.py). It is a sister project of [lora-lite](https://github.com/wassname/lora-lite), for activation steering rather than adapter fine-tuning.

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

### Eval v2: skeptical axis, with a sound-question check (Qwen3.5-4B)

<!-- PI/OpenAI 2026-10-04. Table generated from outputs/bsbench/results/{v2-everywhere,v2-user}/index.md. Needs wassname's review. -->

Changes from v1 (further down):

- **Check questions.** Every BS-bench question has a [twin](data/bsbench/sound_twins_v1.jsonl): the same question with the made-up concept replaced by a real one from the same field. Every dose also answers the twins, and Jev rates whether each answer wrongly rejects the legitimate question ("false pushback"). This is reported, not used to filter or score.
- **Axis.** −C is *skeptical* (v1: abrasive), picked by a [mean_diff screen](slop/reviews/2026-10-03_eval_v2/pole_screen.md) of skeptical, abrasive, candid and accurate. Each answer's change is weighted by Jev's probability that it answers the question asked.

The score is unchanged from v1: on-axis change minus damage at each side's best dose, weaker side, with only Jev damage ≤ 1.5 of 4 counting as coherent. One seed per method (random: 5 directions per mode). "−C pushback" is the change toward rejecting the nonsense at the scored −C dose, on Jev's 0–8 scale. "user turn" adds the vector only while the model reads the user's message. Prompts act only on the prompt, so their rows are the same in both columns.

| method | everywhere score↑ | −C pushback, false pushback | user-turn score↑ | −C pushback, false pushback |
| --- | ---: | ---: | ---: | ---: |
| [vjp_resid](src/steering_lite/variants/vjp_resid.py) | +1.05 [+0.70, +1.43] | +1.37, +14 pp | +1.36 [+1.00, +1.64] | +2.06, +63 pp |
| [chars](src/steering_lite/variants/chars.py) | +1.01 [+0.62, +1.14] | +1.38, +15 pp | +0.12 [-0.02, +0.29] | +0.35, -0 pp |
| [linear_act](src/steering_lite/variants/linear_act.py) | +0.92 [+0.58, +1.09] | +1.25, +16 pp | +0.34 [+0.06, +0.51] | +0.68, +2 pp |
| [vjp_value](src/steering_lite/variants/vjp_value.py) | +0.87 [+0.53, +1.15] | +1.17, +7 pp | +1.22 [+0.74, +1.33] | +1.75, +14 pp |
| [mean_diff](src/steering_lite/variants/mean_diff.py) | +0.64 [+0.28, +0.97] | +1.04, +8 pp | +0.11 [-0.05, +0.33] | +0.27, +0 pp |
| *random* | +0.23 [-0.17, +0.41] | +0.35, +0 pp | +0.38 [-0.34, +0.46] | +1.44, +62 pp |
| [sspace_scale](src/steering_lite/variants/sspace_scale.py) | +0.01 [-0.18, +0.35] | +1.03, +7 pp | -0.25 [-0.50, -0.01] | +1.61, +38 pp |
| [corda_pca](src/steering_lite/variants/corda_pca.py) | -0.76 [-1.06, -0.56] | -0.62, -1 pp | -1.06 [-1.36, -0.74] | -0.89, -1 pp |
| *prompting* | — | +1.70, +26 pp | — | +1.70, +26 pp |
| *prompting_engineered* | — | +1.92, +17 pp | — | +1.92, +17 pp |

The plain and engineered prompts have no score because their +C (sycophantic) prompts are rated incoherent.

![Eval v2, steering everywhere, Qwen3.5-4B](assets/bsbench_qwen3.5-4b_v2_everywhere.png)

The second chart shows each −C sweep as pushback gained on the nonsense questions against false pushback gained on the twins.

![Eval v2, −C pushback against false pushback](assets/bsbench_qwen3.5-4b_v2_discrimination.png)

What the twins show (one seed, one model):

- Unsteered, the 4B rejects 41 of 100 nonsense questions and wrongly rejects 1 of 100 twins, so it can push back for the right reason on part of the benchmark.
- Steering adds at most about 14 nonsense questions that flip to rejection while their twin is still answered. Beyond that the extra rejections are blanket: at the user-turn vjp_resid scored dose, 35 flip but 63 of 100 twins are wrongly rejected.
- The user-turn scores for vjp_resid and vjp_value are higher than steering everywhere, but come with +63 and +14 pp false pushback. Random directions on the user turn also reach 1.44 levels of −C pushback, with +62 pp, so that part of the user-turn effect is not specific to the steering direction.
- corda_pca moves the wrong way on both sides with this axis (it scored +0.26 on v1).

Notes: [fresh-eyes review](slop/reviews/2026-10-04_fresh_eyes/review.md) (written while a 5 pp false-pushback cap was in the scoring; the cap was removed), [manual read](slop/reviews/2026-10-03_eval_v2/main_read.md), [journal](RESEARCH_JOURNAL.md). Interactive pages: `outputs/bsbench/results/v2-everywhere/index.html`, `v2-user/index.html`.

### Eval v1: abrasive axis, no control set

<!-- Results section drafted by PI/claude-opus 2026-09-28. Numbers copied by script from `just results` output:
outputs/bsbench/results/{full,27b-full,olmo-full}/index.md. Qwen3.5-4B updated by PI/OpenAI 2026-09-30; larger models retain the 2026-09-28 results. Needs wassname's review. -->

We test sycophancy with petergpt's [Bullshit Benchmark v2](https://github.com/petergpt/bullshit-benchmark): 100 questions with a false premise. A sycophantic answer accepts the premise. A candid answer says what is wrong with it.

We steer in both directions: +C toward sycophantic, -C toward abrasive. For each method we increase the steering strength until the answers break (unfinished, looping or repeated). A judge model, Jev, rates each answer on two things: how much it accepts the premise, and how damaged it is.

The score is the change we want minus the damage, at the best strength, in the weaker of the two directions. Higher is better. `random` (a random direction) shows what noise scores.

The Qwen3.5-4B plot at the top shows its 5 best methods. Left is -C, right is +C, and lower means more damage. Each line joins the best trade-offs; a cross marks the last strength that passes the checks. The grey regions show random directions at each dose, using seeds admissible in both signs and their median damage. Light to dark: p90 (roughly the 10th–90th percentiles), p75 (25th–75th), and p50 (median), each smoothed and filled back to zero change. They are reference envelopes, not confidence regions or 90%/75%/50% coverage regions. <!-- PI/OpenAI -->

The stars are prompt baselines: they ask the model to act a certain way, and leave its internals alone. "prompt" is a one-line instruction ("Answer as someone who is sycophantic."), and "eng. prompt" is a longer system prompt of the kind [AxBench](https://arxiv.org/abs/2501.17148) found hard to beat ("You are a language model designed to embody sycophantic values...", [walk.py](scripts/bsbench/walk.py#L55)). So they are not strictly comparable with the lines. We include them because people already use prompts and have a feel for how strong they are, which gives a familiar reference point for the steering results. On 4B, the prompt −C and engineered-prompt +C stars fail the checks and are omitted. <!-- PI/OpenAI -->

The table shows every method we tested. Brackets are 90% bootstrap intervals. Most methods were tested only on the small model.

| method | Qwen3.5-4B score↑ | Qwen3.5-27B score↑ | OLMo-2-32B score↑ |
| --- | ---: | ---: | ---: |
| [vjp_value](src/steering_lite/variants/vjp_value.py) | **+1.14** [+0.73, +1.56] | +0.34 [+0.09, +0.69] | -0.20 [-0.38, -0.10] |
| [chars](src/steering_lite/variants/chars.py) | +0.88 [+0.49, +1.23] | +0.43 [+0.01, +1.11] | +0.13 [-0.04, +0.34] |
| [linear_act](src/steering_lite/variants/linear_act.py) | +0.71 [+0.42, +1.02] |  |  |
| [sink_split_resid](src/steering_lite/variants/sink_split.py) | +0.70 [+0.44, +1.16] |  |  |
| [vjp_resid](src/steering_lite/variants/vjp_resid.py) | +0.66 [+0.38, +1.13] | -0.01 [-0.13, +0.25] | -0.05 [-0.16, +0.03] |
| [spherical](src/steering_lite/variants/spherical.py) | +0.54 [+0.09, +1.01] |  |  |
| [directional_ablation](src/steering_lite/variants/directional_ablation.py) | +0.47 [+0.20, +0.84] |  |  |
| [mean_diff](src/steering_lite/variants/mean_diff.py) | +0.37 [+0.14, +0.78] | **+0.91** [+0.60, +1.33] | **+0.21** [+0.04, +0.50] |
| [sink_split](src/steering_lite/variants/sink_split.py) | +0.34 [+0.09, +0.68] |  |  |
| [topk_clusters](src/steering_lite/variants/topk_clusters.py) | +0.33 [+0.10, +0.64] |  |  |
| [corda_pca](src/steering_lite/variants/corda_pca.py) | +0.26 [-0.02, +0.72] |  |  |
| [cosine_gated](src/steering_lite/variants/cosine_gated.py) | +0.14 [-0.03, +0.49] |  |  |
| [query_steer](src/steering_lite/variants/query_steer.py) | +0.10 [-0.12, +0.41] |  |  |
| [sspace_ablate](src/steering_lite/variants/sspace_ablate.py) | +0.07 [-0.08, +0.35] |  |  |
| [sspace_pool](src/steering_lite/variants/sspace_pool.py) | +0.06 [-0.13, +0.33] |  |  |
| [sspace](src/steering_lite/variants/sspace.py) | +0.01 [-0.09, +0.25] |  |  |
| [sspace_pca](src/steering_lite/variants/sspace_pca.py) | -0.07 [-0.28, +0.16] |  |  |
| *[random](src/steering_lite/variants/random.py)* | -0.07 [-0.23, +0.14] | -0.05 [-0.12, +0.14] | -0.10 [-0.15, -0.01] |
| [pca](src/steering_lite/variants/pca.py) | -0.12 [-0.28, +0.19] |  |  |
| [sspace_scale](src/steering_lite/variants/sspace_scale.py) | -0.14 [-0.30, +0.16] |  |  |
| [value_gram](src/steering_lite/variants/value_gram.py) | -0.25 [-0.49, -0.13] |  |  |
| *[prompting](scripts/bsbench/walk.py#L62)* | — | — | — |
| *[prompting_engineered](scripts/bsbench/walk.py#L55)* | — | — | — |

Seeds: 3 per learned method on the Qwen models, 1 on OLMo. The 4B random reference uses seeds 0–10. Extra cached seeds are excluded from these reports. `—`: the prompting baselines have no score, because one of their directions failed the coherence or damage check. These results are exploratory.

The attention-sink method with a residual vector (`sink_split_resid`) ranks fourth on 4B. Its score exceeds mean_diff by +0.33, with a paired 90% interval of [+0.08, +0.63]; the weaker, premise-rejecting direction sets both scores. Attention-only (`sink_split`) is not distinguishable from mean_diff in that comparison. These intervals reselect doses on the evaluated questions; the full set includes the dev questions. Zero dose is approximately, not exactly, the bare model, and we have no matched random-attention-plus-residual control. [Paired comparison](slop/reviews/2026-09-29_svdkv/full-comparison.md), [answer samples](slop/reviews/2026-09-29_svdkv/full-examples.md). <!-- PI/OpenAI -->

### Larger models

<!-- PI/Claude 2026-09-29, numbers from outputs/bsbench/results/{full,27b-full,olmo-full}/index.md. Needs wassname's review. -->

Four methods also ran on Qwen3.5-27B and OLMo-2-0325-32B-Instruct. The score can be low for two reasons: the method does not steer, or the model already gives the target answer. Bare Qwen3.5-27B already rejects 69 of the 100 false premises, against 35 for the 4B model ([by_stance.md](slop/reviews/2026-09-28_judged_by_stance/by_stance.md)), so there is little left to steer toward candour. The second number for each model divides the on-axis change by the room left (how far the bare answers could still move toward that side), in the weaker direction.

| method | 4B score↑ | 4B on ÷ room↑ | 27B score↑ | 27B on ÷ room↑ | OLMo score↑ | OLMo on ÷ room↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [vjp_value](src/steering_lite/variants/vjp_value.py) | **+1.14** [+0.73, +1.56] | **+0.40** | +0.34 [+0.09, +0.69] | +0.33 | -0.20 [-0.38, -0.10] | +0.00 |
| [chars](src/steering_lite/variants/chars.py) | +0.88 [+0.49, +1.23] | +0.35 | +0.43 [+0.01, +1.11] | +0.36 | +0.13 [-0.04, +0.34] | +0.23 |
| [vjp_resid](src/steering_lite/variants/vjp_resid.py) | +0.66 [+0.38, +1.13] | +0.24 | -0.01 [-0.13, +0.25] | +0.13 | -0.05 [-0.16, +0.03] | +0.03 |
| [mean_diff](src/steering_lite/variants/mean_diff.py) | +0.37 [+0.14, +0.78] | +0.14 | **+0.91** [+0.60, +1.33] | **+0.64** | **+0.21** [+0.04, +0.50] | **+0.27** |
| *[random](src/steering_lite/variants/random.py)* | -0.07 [-0.23, +0.14] | +0.01 | -0.05 [-0.12, +0.14] | +0.02 | -0.10 [-0.15, -0.01] | +0.01 |

Within each model, the two columns rank these four methods in the same order, so the room correction does not change which of them looks best there. Between models the order changes: mean_diff is 4th of 4 on the 4B model and 1st on both larger models. On Qwen3.5-27B the VJP methods keep much of their effect per unit of room (vjp_value +0.40 → +0.33). On OLMo they barely steer (on ÷ room +0.00 and +0.03, random +0.01), while mean_diff and chars still do. The [research journal](RESEARCH_JOURNAL.md) has the checks for a bug (none found) and extra `vjp_resid` runs with other settings.

Each plot shows up to 5 best-scoring methods on that model (all 4 on the larger models), so the 4B plot above does not include mean_diff (8th there). The axis ranges differ between plots: compare the order of the curves, not their lengths.

![Pareto plot, Qwen3.5-27B](assets/bsbench_qwen3.5-27b_full.png)

To run the benchmark you need [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just), pnpm, a Modal account (`uv run --extra benchmark modal setup`) and `OPENROUTER_API_KEY` in `.env`.

```bash
just check                                   # tiny CPU smoke tests of every method and of the walk
just sweep dev my_method 0 0                 # one method, seed 0, 20 questions (+ random seed 0), on Modal
just results dev                             # judge with Jev, then table, plot and page
just sweep full mean_diff,chars 0,1,2 0,1,2  # 100 questions, 3 seeds, as in the table above
just results full
```

To add a method, copy [mean_diff.py](src/steering_lite/variants/mean_diff.py), register it (see [AGENTS.md](AGENTS.md)), then run `just smoke-bsbench my_method` before the dev sweep.

The outputs go to `outputs/bsbench/results/<cohort>/`. [`calibration.py`](scripts/bsbench/calibration.py) and [`cost.py`](scripts/bsbench/cost.py) are extra checks: where each run breaks down, and what a run costs.

## Methods and debugging

Each implementation includes its own math and references in [the variants directory](src/steering_lite/variants). Start with [mean difference](src/steering_lite/variants/mean_diff.py) or [PCA](src/steering_lite/variants/pca.py). Newer variants are [VJP residual](src/steering_lite/variants/vjp_resid.py), [VJP value](src/steering_lite/variants/vjp_value.py), [Value Gram](src/steering_lite/variants/value_gram.py) and [query steering](src/steering_lite/variants/query_steer.py). [S-space](src/steering_lite/variants/sspace.py) also supports `gate="signed"`. [Random](src/steering_lite/variants/random.py) is an evaluation-only null baseline.

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
