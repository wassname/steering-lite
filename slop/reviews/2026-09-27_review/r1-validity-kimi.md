Review complete. Verified the pipeline end-to-end (results.py, judge.py, the three slop scripts, compare assets); cache-key contract and INTENDED sign conventions check out. Findings below.

---

**P1 — "score ÷ room" divides the off-axis penalty by premise room, breaking the calibrated 1:1 tradeoff, and is mislabeled "fraction of room used".**
`results.py:140-144`: `room_score` computes `(directed(point) - OFF_WEIGHT * point["off_axis"]) / point["room"]`. The on-axis term and `room` are premise levels (0–8); `off_axis` is |Δdamage| on the damage scale (0–4, `judge.py:36`: "MAX_DAMAGE = 1.5 # of 4"). Dividing the damage term by premise room makes the *effective* off-axis weight `1/room`, which varies per side and per model — the deliberately set `OFF_WEIGHT = 1.0` ("1:1", `results.py:31`) no longer holds in the normalized score, so the 4B-vs-27B `score_room` comparison in `compare.py`/`compare.md` is not on a common on:off scale. Separately, the label "fraction of the available room used" (`results.py` tables() footer; `main.jsx:169` lede; `compare.py` right-panel title) is only true when off_axis = 0: since per-question `on ≤ room` but the off term is subtracted after, a method using 100% of on-axis room with any damage scores < 1, and `random` shows "fractions" of −0.02/−0.03 (`compare.md`). Simpler and honest: `on/room − off` (or report `on/room` alone), and rename.

**P2 — room denominator can vanish; min-over-sides silently shifts side selection.**
`results.py:105-107`: for −C, `room = mean(bare_premise)`; `compare.md:5` states "27B bare answers already reject more", i.e. 27B −C room is small by design. A bootstrap draw whose chosen bare answers all score 0.0 gives `room() == 0.0` → `ZeroDivisionError` at `results.py:144` (rare, but unguarded); near-zero rooms inflate that side's ratio, so the min over sides flips to the other side — the "weaker side" being scored differs between raw and normalized score and between 4B and 27B. Undisclosed in compare.md.

**P2 — judge.py CONCEPTS provenance wording.**
`judge.py:70-72`: "Named from 3,640 unanchored free-text phrases … quoted phrases are the judge's own words." The phrases came from `deepseek/deepseek-v4-flash-0731` (`freetext.py:17`), not from Jev, whom `judge.py:1` names as "The judge". The request instruction ("phrases a judge used") is accurate; the comment is not. Count (3,640) matches `cluster.md:1`.

**P2 — jlens_words.py measures Δh on the bare answer's tokens.**
`jlens_words.py:36-38`: teacher-forces the steered model on `prompt + bare answer`. For methods whose effect is a different answer (fabricates, different_advice), the residual delta is read at positions of text the steered model would not generate, so the J-lens words may not describe actual steered generations. Disclosed in the docstring, so report-only.

Notes: no second source of truth for room — `compare.py` and the web table both read `score_room`/`ci_room` from points.json (good); only the prose definition is triplicated (results.py footer, main.jsx lede, compare.py docstring). `key()` hashes the full request (`judge.py:92-94`), so the CONCEPTS rename forces cache misses and the `blind_summary` assert fails fast — consistent with house rules. INTENDED (+C→accepts_premise, −C→rejects_premise) matches `directed()` signs. Natural-label pipeline (freetext → cluster → CONCEPTS) is sound: unanchored prompt, verbatim quotes, deterministic sampling.

Merge verdict: OK with notes