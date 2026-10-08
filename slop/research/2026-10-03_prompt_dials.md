# Continuous prompt strength: candidates and limits

Research date: 2026-10-03. Recommendation: compare two-branch logit guidance with constant attention-logit bias. Both expose a continuous scalar without scaling token embeddings. A guaranteed monotone BullshitBench judge score was **not found**. Increasing a mathematical intervention is different from increasing a generated behaviour.

## Ranked fit for this benchmark

Ranking is my inference. Cost counts cached forwards per generated token, not measured wall time. HF attention compatibility needs checking.

| Rank / method | Mechanism and scalar | Monotonicity evidence | High-strength failure; cost; model + prompt only? |
|---|---|---|---|
| 1. CFG / CAD / Context Steering [1–3] | Same model with persona vs neutral/no-persona system prompt; mix logits as z₀+s(z₁−z₀). s=0 neutral, s=1 ordinary persona, s>1 amplified. CFG γ=s; CAD α=s−1; CoS λ=s−1. | Continuous logit contrast; CoS has graded personalization examples; CAD improves knowledge-conflict scores with higher α. CFG system-prompt preference peaks at γ=3, so global behavioural monotonicity fails. | Lost user-query relevance, degenerate text; ≈2 forwards and two KV caches, shared weights. Yes. HF provides CFG logits processors; a paired-cache generation loop also works. |
| 2. InstABoost [4] | Add B to instruction-key attention logits in every head/layer; multiplier M=exp(B), baseline B=0. | Instruction/noninstruction attention odds increase by exp(B) for fixed queries/keys. Paper's rule-system theory is not a proof of monotone LLM judge scores. | Suppresses needed task context; 1 forward plus attention-mask edits. Yes; no profiling. HF additive-mask implementation; fused-kernel compatibility must be checked. |
| 3. Prompt-pair ActAdd [7] | Extract residual difference h(prompt+)−h(prompt−), then inject c times it at a fixed layer/alignment. | Paper explicitly describes continuous weighting; a coefficient-to-judge monotonicity guarantee was not found. | Excess intervention can change unrelated content (inference); exact high-c failure threshold not found. 1 forward plus additions; two extraction prefills once. Yes, needs a prompt pair. Already a steering-vector method, rather than a pure prompt baseline. |
| 4. SpotLight [5] | When span attention is below target ψ, add log(ψ/current) to span logits. Increase ψ∈(0,1). | Refusal increases with ψ on WildJailbreak; IFEval is relatively flat. Local attention response is ordered, not a semantic guarantee. | Over-refusal and incoherence; 1 forward plus attention measurement/editing, possible loss of cache/kernel optimizations. Yes; no profiling; paper describes HF hooks. |
| 5. PASTA [6] | Downweight non-highlighted attention by α, renormalize selected heads. Use strength −log α; baseline α=1. | Fig. 3(c) reports robustness, not an ordered behavioural response. Local attention odds are ordered. | α=0 removes essential context; steering all heads can hurt performance. 1 forward plus mask edits. No training, but published head selection needs labelled profiling examples; fixed heads need only prompt/model but depart from that protocol. |
| 6. Persona statement budget [8] | Append k persona-consistent statements to the system prefix. | Published curves explicitly sometimes nonmonotone. Discrete, not smooth. | Saturation/reversal; high-strength fluency failure not found. 1 forward, longer prefill/KV context. Needs a statement bank, no training. |
| 7. Soft-prefix interpolation / Compel analogue [9–10] | Interpolate equal-shaped prefixes E(s)=(1−s)E₀+sE₁; Compel blends diffusion text conditioning. | Untuned LM persona monotonicity not found; Compel explicitly warns of unexpected blends. | Off-distribution interpolates (inference); Compel's warning is about images, not LMs. 1 forward with prefix overhead. Untuned interpolation: yes with two aligned prompts; trained soft prompts: no. |
| 8. Trained continuous control (CIE) [10] | Fine-tune the LM to accept an interpolated low/high control embedding. | More reliable response-length control; evidence for arbitrary persona monotonicity not found. | High-strength persona failure not found. 1 forward after training. No: requires fine-tuning and control-labelled data. |
| 9. DExperts / original contrastive decoding [11–12] | DExperts adds α(z_expert−z_anti); original CD subtracts a smaller model under a plausibility mask. | DExperts reports smooth attribute/fluency tradeoffs. Original CD's α is a vocabulary cutoff, not prompt intensity. | DExperts fluency cost; unconstrained contrasts favour improbable tokens. Base +2 expert forwards (often smaller), or base +1 amateur. No: needs auxiliary models; DExperts experts are attribute-trained. Prompt-only analogue reduces to rank 1. |

## Primary evidence (verbatim)

Authors' own papers/documentation, not independent replications. Mathematical notation is rendered in Markdown.

[1] Sanchez et al., *Stay on topic with Classifier-Free Guidance* (2023), paper, §3.4 and Fig. 5: https://arxiv.org/html/2306.17806v1#S3.SS4

> Our results in Figure 5 shows compelling evidence that CFG emphasized the difference between c and c̄ more than sampling with c alone. There is a clear peak at γ=3 with 75% of system-prompt following preference over γ=1 and undegraded user-prompt relevance (52%).

An optimum, not global monotonicity; Fig. 5 reports degraded user relevance at γ≥4.

[2] Shi et al., *Trusting Your Evidence: Hallucinate Less with Context-aware Decoding* (2023 preprint), paper, §4.2: https://arxiv.org/html/2305.14739v1#S4.SS2

> Across all three datasets, we find λ=0.5 consistently provide robust improvements over regular decoding. Further increasing the value of α yields additional improvement in tasks involving knowledge conflicts.

The source switches symbols; it measures contextual factuality, not persona intensity.

[3] He et al., *Context Steering: Controllable Personalization at Inference Time* (2024 preprint), paper, §2.2: https://arxiv.org/html/2405.01768v3#S2.SS2

> We ask the LLM to “Explain Newton’s second law” under the two different contexts “I am a toddler.” and “I got a D- in elementary school science.” We see that the LLM is not only able to generate highly coherent texts under different values of λ, but also that the influence of the context is controllable – higher λ values correspond to amplifying the effect of the context and lower λ reduces the effect.

Qualitative examples. Appendix D/Table 4: “Observed issues include concatenating words together, generating blobs of foreign language, and outputting random texts.”

[4] Guardieiro et al., *Instruction Following by Principled Boosting Attention of Large Language Models* (2025; revised 2026), paper, §1: https://arxiv.org/html/2506.13734v3#S1

> We analyze boosting attention to the instruction by adding a bias to the pre-softmax attention and show that it systematically increases the influence of instruction rules, making it exponentially harder for competing context to override instruction-consistent updates. At the same time, the framework predicts a suppression regime in which excessive boosting downweights benign competing rules so strongly that necessary task details fail to activate. This yields an instruction over-focus failure mode that reduces relevance and degrades generation quality.

Theory under Logicbreaks sparse-rule assumptions; “monotonicity” there means retaining facts, not dose response.

[5] Venkateswaran et al., *Spotlight Your Instructions* (2025; revised 2026), paper, §3.4/Fig. 6: https://arxiv.org/html/2505.12025v2#S3.SS4

> On IFEval, the performance remains relatively stable across different proportions. In contrast, on WildJailbreak, as ψ_target increases, refusal accuracy improves on both models, indicating stronger alignment with the safety instructions. But this also leads to a notable drop in benign accuracy.

A safety/refusal sweep, not sycophancy. §2.2 shows extreme-target incoherence.

[6] Zhang et al., *Tell Your Model Where to Attend* (2023), paper, §5.3/Fig. 3(c): https://arxiv.org/html/2311.02262#S5.SS3

> The results indicate that PASTA is fairly robust to this hyperparameter; in practice, we fix it as 0.01. Notice that setting α to zero should be avoided, as this leads to the complete removal of other crucial contexts at the steered heads, resulting in performance degeneration.

Robustness to α does not establish a useful strength response.

[7] Turner et al., *Steering Language Models With Activation Engineering*, paper, “Activation engineering vs prompt engineering”: https://arxiv.org/html/2308.10248v5

> Activation additions can be continuously weighted, while prompts are discrete – a token is either present, or not. To more intensely steer the model to generate wedding-related text, our method does not require any edit to the prompt, but instead just increasing the injection coefficient.

Mechanism claim. Appendix H varies dimensions, not coefficient c, and reports nonmonotonicity.

[8] Miehling et al., *Evaluating the Prompt Steerability of Large Language Models* (2024/2025), paper, §4: https://arxiv.org/html/2411.12405v2#S4

> Generally, a larger steering budget k (more steering statements) yields a more steered model. Interestingly, as seen in Figs. 4 (e), (f), the trend is not always monotonic.

[9] Damian0815/Compel maintainers, repo documentation, “Blend”: https://github.com/damian0815/compel/blob/main/doc/syntax.md#blend

> Note the weights `(1, 0.8)`. Blending breaks some of the assumptions about how the text encoding is supposed to function, so your blends may not come out like you expect. Chaning the weights can have a dramatic effect on how the different parts of the prompt are interpreted, and therefore how the resulting image turns out.

[10] Samuel et al., *CIE* (EMNLP 2025), paper abstract: https://aclanthology.org/2025.emnlp-main.189/

> Through a case study in controlling the precise response-length of generations, we demonstrate how an LM can be finetuned to expect a control vector that is interpolated between a “low” and a “high” token embedding.Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal.

This supports training the model to interpret interpolation, not expecting an untouched model to do so.

[11] Liu et al., *DExperts* (2021), paper, Appendix D.1: https://arxiv.org/html/2105.03023v2#A4.SS1

> Figure 8 shows the relationship between output toxicity and fluency for different values of α in our method. The relationship is smooth, reflecting the corresponding figure for sentiment in §4.3.

[12] Li et al., *Contrastive Decoding* (2022/2023), paper, §3.2: https://arxiv.org/html/2210.15097#S3.SS2

> Larger α entails more aggressive truncation, keeping only high probability tokens, whereas smaller α allows tokens of lower probabilities to be generated. We set α=0.1 throughout the paper.

## What `generative-computing/steerability` implements

Read README, controls and evaluation tutorial at commit `ecf8dd120a44cd87cb8b5bdaf09674fe9e17c743`. It has method-specific strength parameters, not a shared monotonicity guarantee. Code observations:

- `ContrastiveGuidance`: `base_weight=gamma`, auxiliary prompt-variant weight `−(gamma−1)` implements CFG/CAD. Important cost caveat: `PromptVariantSource.logprobs()` decodes, transforms, retokenizes and forwards the entire growing prefix each step; it does not maintain an auxiliary KV cache. Its current implementation is more expensive than the two-cached-branch estimate above. https://github.com/generative-computing/steerability/blob/ecf8dd120a44cd87cb8b5bdaf09674fe9e17c743/steerability/algorithms/output_control/common/logit_sources.py#L150
- PASTA study uses `scale_position="include"`, with emphasis `alpha=100` corresponding to the paper's α=0.01. Sweeps strict Split-IFEval prompt accuracy against response reward-model quality, after separate head profiling. `ActAdd.multiplier` is paper coefficient c; its positional injection can occur entirely during prefill. https://github.com/generative-computing/steerability/blob/ecf8dd120a44cd87cb8b5bdaf09674fe9e17c743/examples/notebooks/studies/instruction_following/instruction_following.ipynb

The maintainers' evaluation tutorial states:

> This facilitates the evaluation of both the target behavior of a pipeline (did instruction following ability improve?) and its off-target effects (degradation in math ability, coding ability, general knowledge, etc.).

Source: repo documentation, opening paragraph: https://github.com/generative-computing/steerability/blob/ecf8dd120a44cd87cb8b5bdaf09674fe9e17c743/docs/tutorials/evaluate_steering_pipelines.md

Inspect evaluates whole pipelines; constructor-parameter sweeps yield sensitivity/tradeoff plots. Another study measures positional answer bias. No universal steerability score.

## Decision and discriminating test

My inference: sweep cached guidance at s=0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3. Both branches receive the same question and continuation; only the persona instruction differs. Also test cheaper InstABoost after verifying mask coverage on Qwen3.5's hybrid architecture.

Mathematical observation: fixed-history guidance increases expected log-likelihood contrast (derivative = contrast variance); fixed-query/key attention bias increases span odds. Autoregressive histories diverge, so neither guarantees ordered judge scores. Report paired adjacent-dose reversals and all behaviour-change/Jev-damage points, without fitting away reversals.

Own proposal: choose full-persona generation with probability s, neutral otherwise, once per answer. Expected fixed-judge score interpolates exactly between endpoints. Cost: one generation. This guarantees ordered population means if endpoints are ordered, but mixes weak/strong answers rather than intermediate-intensity answers.

Epistemic summary: CFG/CAD/CoS share arithmetic, not independent confirmation. Behavioural evidence is task-specific and contains reversals. A judged Qwen3.5 sycophancy sweep must establish any usable ordered interval.

Signed: PI/OpenAI research subagent (exact model identifier was not exposed in this session).
