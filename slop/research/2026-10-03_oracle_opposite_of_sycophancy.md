# Opposite of sycophancy, not refusal: oracle panel summary

PI/OpenAI, 2026-10-03. Brief: `2026-10-03_oracle_opposite_of_sycophancy_brief.md` (neutral, no preferred answer given). Independent answers, no tools:

- `2026-10-03_oracle_openai-codex_gpt-6-astra_medium.md` (Astra)
- `2026-10-03_oracle_openrouter_anthropic_claude-fable-5.1_medium.md` (Fable)
- `2026-10-03_oracle_github-copilot_gemini-3.8-flash.md` (Gemini; it signed itself "PI/Sonnet")
- Grok 4.7 (github-copilot) returned no text twice (first run tried to use tools; second, with an explicit no-tools note, wrote only the pi startup banner). Not counted.

## Where all three agree

1. A single negative word probably cannot isolate premise-checking. Fable: "I think a single word cannot do it (likely, ~65%)"; Astra: "I would not expect it to isolate this distinction reliably"; Gemini: "A single word is unlikely to cleanly isolate the epistemic axis."
2. The fix is to hold tone and helpfulness fixed on both sides and vary only whether the person goes along with or checks the user's framing. Fable's pair:
   > pos: `Answer as someone who is friendly and goes along with the user's framing to keep them happy.`
   > neg: `Answer as someone who is friendly and first checks whether the user's framing is correct, then helps.`
3. The cheap check is sound-premise twins: minimally edited versions of BullshitBench questions with a real premise; a premise-checking pole answers them, a contrarian or refusing pole rejects them. Astra: score "correct discrimination", "contrarianism" and "refusal/disengagement" separately, "Keep refusals in" the denominator.

## Differences

- Single-word candidates if the format must stay: Fable `candid`; Astra `polite and intellectually independent`, `truthful and tactfully corrective`; Gemini `rigorous and reality-grounded`, `scrupulous about factual accuracy`. All three flag `skeptical` as risking generic contrarianism.
- Astra: "sycophancy is not identical to false-premise acceptance"; name the direction "agreement over premise accuracy".
- Fable adds a pipeline-compatible option: extract a `warm` vs `rude` tone vector and project it out of the sycophancy vector, which also measures how much of the current vector is tone ("I'd guess a large fraction").

## My read (PI/OpenAI)

The trait-matched behavioural pair plus sound-premise twins is the consensus; I think it probable that `abrasive` puts a tone direction into every current -C result. Not tested.
