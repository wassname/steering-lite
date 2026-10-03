[pi-web-access] Dynamic tool activation requires Pi 0.86.1 or newer; web tools remain eagerly available.
# Answer: negative pole for the sycophancy vector

-- PI[claude], 2026-10-03. Opinion plus reasoning; no experiments run. Confidence noted inline.

## Framing

The vector is `act(pos) − act(neg)`, so every trait that differs between the two persona lines leaks into it. `sycophantic` vs `abrasive` differs on at least three things: (a) defers to the user's framing vs checks it, (b) warm vs hostile tone, (c) praises vs insults. BullshitBench rewards only (a), and specifically (a) *plus* staying helpful. Your observations (damage near cutoff, off-target refusals, invented flaws) are what I'd expect if (b) dominates: hostility toward the user generalises to hostility toward the question. Highly likely the fix is to match tone on both sides, not to find a better single adjective.

## 1. Candidate negative personas

| persona phrase | what it likely isolates | confounds it probably carries |
|---|---|---|
| `candid` | says what they think, not what pleases | bluntness → tone damage (less than abrasive); may be candid about *opinions* while still accepting the premise |
| `skeptical` | doubts claims | generic contrarianism; likely raises false rejections on sound premises; hedging |
| `a careful expert who checks the question's assumptions before answering` | premise-checking directly | verbosity, pedantry, lecturing; multi-word phrase, so the contrast is less clean than single adjectives |
| `a peer reviewer` | looks for flaws | reward for *finding* flaws → invented flaws (your "Net is not an IDE theme" failure); detached register |
| `a helpful expert who corrects misconceptions kindly` | premise-checking + stays engaged + warm | teacher mode, longer answers; "kindly" may partly overlap with the sycophantic pole's warmth and shrink the contrast |

I would not use `honest` or `truthful` alone: chat models already self-identify as honest, so the persona line is unlikely to move activations much (moderate confidence), and it does not name the behaviour.

## 2. Single word vs matched pair

I think a single word cannot do it (likely, ~65%). Reasons: the positive pole `sycophantic` already bundles deference + flattery, so even a perfect negative adjective leaves flattery in the vector; and no common English adjective means "checks premises while remaining friendly and helpful". Two better constructions:

- **Trait-matched behavioural pair.** Hold tone and helpfulness fixed in words, vary only the deference clause:
  - pos: `Answer as someone who is friendly and goes along with the user's framing to keep them happy.`
  - neg: `Answer as someone who is friendly and first checks whether the user's framing is correct, then helps.`
  This is the cleanest for the difference vector; cost is you lose the one-word persona format.
- **Keep single words but remove the tone direction.** Extract a second vector from `warm` vs `rude` with the same suffix, and project it out of the sycophancy vector before applying. Cheap, keeps the current pipeline, and tells you how much of the current vector was tone (I'd guess a large fraction).

If you must keep one word per side, `candid` is my pick as the least-bad negative, with the understanding that it will still carry some bluntness.

## 3. Cheap check for refusal / contrarianism

**Sound-premise twins.** For each BullshitBench item, have a model write the real counterpart in the same domain and surface form ("thermal conductivity of a CI pipeline" → "throughput bottleneck of a CI pipeline"); 100 items, one batch call. Run the same dose walk on both sets and report, per dose, rejection rate on bullshit vs rejection rate on sound. A premise-checking pole raises the first and leaves the second near baseline; a contrarian or refusing pole raises both together. Plotting the two curves (or their difference as a d′-style gap) separates them without any new judge.

Even cheaper, no new questions: at −C count answers whose stated flaw does not match the item's actual made-up premise (BullshitBench items name the invented concept, so a yes/no judge can check this). A rising invented-flaw rate with dose is contrarianism, not premise-checking. Your existing off-target check already approximates this; make it a standard column in `results.py` rather than a side analysis.

One caution for interpretation: the matched-pair fix and the tone-projection fix both change the method, so results are not comparable to the current `abrasive` runs. Record it as a new pole in `RESEARCH_JOURNAL.md`, not as a rerun.
