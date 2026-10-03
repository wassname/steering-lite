You are a fresh-eyes reviewer of an ML result. Repo: /workspace/2026/lite/steering-lite-bsbench. Do not edit any file except your output file.

Read: README.md section "### Eval v2" (lines ~76-117), RESEARCH_JOURNAL.md last 2 entries, slop/reviews/2026-10-03_eval_v2/{pilot_read,main_read,pole_screen}.md.
Data: outputs/bsbench/results/v2-everywhere/points.json and v2-user/points.json (each point: method, seed, C, side, effect, steered_damage, admissible, false_pushback, false_pushback_bare, questions[...]), scoring code scripts/bsbench/results.py (build_points, method_curve, pareto_score), judge prompts scripts/bsbench/judge.py (control_request, audit_request).

Tasks, in order:
1. For each numeric claim in the README eval v2 section, verify it against points.json or index.md. List any mismatch with file:line.
2. Look for bugs or misconceptions that would change the conclusions: e.g. on-target weighting applied to bare vs steered asymmetrically, false pushback cap computed per seed vs per mean, twins misaligned with bench questions, sign conventions (effect negative = rejection on -C), the pole screen being selected on the same data it is reported on (selection bias), prompt rows identical across views.
3. Read 10 random twin answers at the vjp_resid everywhere scored -C dose and say whether Jev's false_pushback looks right (quote).
4. Give calibrated probabilities for the 3 main conclusions: (a) user-turn gains were mostly contrarianism; (b) skeptical is a better -C pole than accurate for vector methods; (c) steering beats prompting on -C under the cap.

Write the review to slop/reviews/2026-10-04_fresh_eyes/review.md (sign it "— PI/Sol"). Be concise, quote evidence, separate observation from inference.
