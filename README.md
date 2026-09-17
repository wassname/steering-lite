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

The older results below used the 95th percentile of token KL at a target of 0.50 nats instead. See [calibration code](src/steering_lite/calibrate.py) and the [longer explanation](https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7).

## Results

Can we make a model treat disobedience to authority as less morally significant, while giving more weight to care? We test this with [moralmaps](https://github.com/wassname/moral-maps): short stories where the model chooses which moral concern is involved. We measure intended changes and changes to the other concerns.

We want steering to have a precise, bidirectional effect: pushing one way should increase the target concept, and pushing the other way should decrease it, without changing unrelated answers. We measure this with *steering selectivity*, comparing the logprobs in the two steering directions:

$$
\text{selectivity} = \text{intended movement} - 0.1\,\text{unintended movement}.
$$

Moving the target the right way earns credit; moving it the wrong way loses credit. Side effects count at one tenth the weight. Logprobs let us see small changes even when the chosen answer stays the same. [Scoring function](https://github.com/wassname/moral-maps/blob/main/src/moralmaps/metrics.py#L80).

Here are the saved Qwen3-4B results. Higher selectivity is better; `on` and `off` show its intended and unintended movement.

| method | selectivity↑ | on↑ | off↓ | 95% interval |
| --- | ---: | ---: | ---: | :--- |
| pca[+] | **+2.12** | **+2.17** | 0.54 | [+1.67,+2.60] |
| sspace_pca[+] | +1.54 | +1.60 | 0.66 | [+1.02,+2.12] |
| corda_pca[+] | +1.50 | +1.71 | 2.12 | [+1.01,+1.98] |
| sspace_signed[-] | +1.49 | +1.58 | 0.89 | [+1.00,+1.97] |
| topk_clusters[-] | +0.31 | +0.35 | 0.45 | [-0.03,+0.63] |
| super_sspace[-] | +0.24 | +0.28 | 0.38 | [-0.02,+0.50] |
| sspace_damp_amp[+] | +0.17 | +0.25 | 0.75 | [-0.24,+0.58] |
| mean_diff[-] | +0.10 | +0.24 | 1.37 | [-0.29,+0.51] |
| cosine_gated[+] | -0.13 | -0.09 | 0.42 | [-0.40,+0.13] |
| directional_ablation[-] | -0.14 | -0.14 | **0.09** | [-0.56,+0.27] |
| spherical[-] | -0.49 | -0.41 | 0.82 | [-0.86,-0.10] |
| sspace_ablate[-] | -0.56 | -0.41 | 1.52 | [-1.08,-0.08] |
| sspace[-] | -0.72 | -0.52 | 1.99 | [-1.36,-0.11] |
| *prompt_only* | -1.80 | -1.66 | 1.40 | [-2.30,-1.30] |

These values are exploratory. The run used 132 classic vignettes, 256 persona-branching pairs, layers 7-27, and a 256-token thinking budget. It ran on 2026-07-16 as `82d4c8319de5` with code `514b97e`, calibrated at `0.5 kl_p95`, and used 2,000 row-bootstrap samples. The table was rescored by `bba61e6`. It predates the matched-pair correction `055bd94`; do not treat its ranking as a corrected comparison. `random` is an equal-KL evaluation null, but has no saved result and is not in this table.

`on` is the mean signed change toward Authority-down and Care-up; `off` is the mean absolute change on the other foundations. Both use centered logprobs: each answer's logprob minus the mean across answers. The maintained scorer calls `moralmaps.gated_selectivity`; `tests/test_results_seam.py` checks that input seam.

To produce a new table with the current code:

```bash
just sweep Qwen/Qwen3-4B outputs/tinymfv_sweep_4b
just results outputs/tinymfv_sweep_4b
```

## Methods and debugging

Each implementation includes its own math and references in [the variants directory](src/steering_lite/variants). Start with [mean difference](src/steering_lite/variants/mean_diff.py) or [PCA](src/steering_lite/variants/pca.py). The new variants are [S-space PCA](src/steering_lite/variants/sspace_pca.py) and [CorDA PCA](src/steering_lite/variants/corda_pca.py). [S-space](src/steering_lite/variants/sspace.py) also supports `gate="signed"`. [Random](src/steering_lite/variants/random.py) is an evaluation-only null baseline.

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
