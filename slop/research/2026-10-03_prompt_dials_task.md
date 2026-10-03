Research task (from wassname, via PI/OpenAI, 2026-10-03). You are a research subagent. Do not edit any file except the output file named below. No GPU jobs, no credentials.

wassname asked: "do a search for ways people dial prompts up and down monotonically. pormpt prefix. look in sterrabiltiy for ideas too https://github.com/generative-computing/steerability/"

Context: in /workspace/2026/lite/steering-lite-bsbench we tried to dial a persona prompt ("Answer as someone who is sycophantic.") by multiplying its token embeddings by a gain g. Result on Qwen3.5-4B, BullshitBench 100 questions (premise change, + = more accepting of false premises): g=0.0625 +1.05, g=0.094 +3.33, g=0.5 +3.40, g=1 +3.50, g=3 +2.94, g=4 -0.47, g=6..16 about 0. So it is a switch near g=0.07 and garbles above g=3; not a usable monotone dial. We want a dial whose effect grows smoothly and monotonically with one scalar, so it can be swept like a steering vector dose and compared on a Pareto plot (behaviour change vs side effects).

Find methods that give a monotone strength knob for an instruction / persona / system prompt, e.g. (leads, verify, find others): classifier-free guidance on LM logits (Sanchez et al. 2023), context-aware decoding (Shi et al. 2023), contrastive decoding variants, attention reweighting toward prompt tokens (PASTA and successors), prompt weighting as in diffusion (compel-style), soft-prompt / prefix interpolation, ActAdd-style activation addition from a prompt pair, persona or instruction "intensity" papers, and the generative-computing/steerability repo (read its README/code: how does it define and vary steering strength, what goals/dimensions does it measure?).

For each method: one-line mechanism, the scalar knob, evidence that the effect is monotone in the knob (quote the source, with URL and section), known failure modes at high strength, compute cost per token, and whether it needs only the model plus the prompt (no training). Rank by fit for our use (cheap, monotone, judge-evaluable, HF transformers).

Quote primary sources verbatim with a sentence of context, attribute them, and say what kind of source each is (paper, repo README, blog). Separate observations from your inference. Say "not found" rather than guessing.

Write about 1-2 pages of markdown to: /workspace/2026/lite/steering-lite-bsbench/slop/research/2026-10-03_prompt_dials.md
Sign it with your model name.
