# Repository cleanup and cache-method evaluation

Author: PI/OpenAI. User requests: "there's a new kv cache steering right? run that too?" and "make this 9b the main plot. clean uyp readme. audit what's commited. semantic paths group and explain things etc".

- [x] goal: evaluate `cache_mean_diff` in the same main 9B report (self-verified; see method evidence below)
  - Integrated the four reviewed commits from `feat/cache-mean-diff` into this development branch; no merge into literal main.
  - Run real library/cache/benchmark smoke, then three seeds with controls and default lower-dose samples.
  - Failure mode: scoring only prefill logits misses a one-shot cache edit. Check continuation logits and hybrid-model generation before paying for runs.
  - Deliverable: judged method row and selectable curve; retain failures if any instead of silently excluding them.
- [x] goal: make the current 9B result the clear README entry point (self-verified; editorial diff left for review)
  - Move results above quickstart; remove earlier incompatible score tables from the front page, preserving historical prose in an explicitly historical file.
  - Make maintained commands reproduce this model/setup; state method registration steps and the lower-dose default.
  - Failure mode: README shows 9B but its example commands still run generic-pair 4B.
  - Deliverable: short README with one main figure, a source map and matching commands.
- [x] goal: audit tracked content and organize paths by purpose (bounded inventory/link audit, not a complete code or security audit)
  - Inventory tracked files, sizes, entry points, machine-only paths and stale result assets; preserve unrelated dirty/untracked files.
  - Group current figures separately from historical figures; keep dated research evidence under slop rather than mixing it with maintained scripts.
  - Failure mode: moving a file silently breaks imports, links or an old evidence reference.
  - Deliverable: inventory and decisions in this directory; link checks and smoke/UAT output.

## Verification

Success: main README figure is the verified 9B PNG; repeatable commands resolve to the same setup; cache method passes real tiny-model checks and is judged with three seeds.
Likely failure: new cache method fails calibration or exceeds GPU memory; inspect the actual error and fix before rerunning.
Subtle failure: low-dose dots exist only in a scratch script, or score prefill rather than cache-affected continuation. Production smoke and report-marker checks must exercise both.

No push, no wholesale branch merge, no deletion of uncommitted user work. README rearrangement remains reviewable as a working-tree diff.

## Observed inventory and changes

`tracked_before.txt` inventories HEAD `fb1fd14`. `inventory_summary.txt` reports 383 slop files (39,006,054 bytes), 36 source files, 18 benchmark/web files, 7 figures, and the pinned `docs/vendor/vjp-steering` submodule. `.pi/goals/` holds persistent goals and review evidence; it is retained. No tracked `.local/`, `.env`, installed dependencies, model outputs or key files matched the filename check.

`credential_patterns.log`: "Checked 485 committed blobs against 4 credential patterns" and "Findings=0". This checks common OpenRouter/GitHub token forms, private-key headers and AWS access IDs at HEAD only. It is not a scan of all history or arbitrary secrets, and not a security certification.

Changes:
- README initially reduced from 269 to 138 lines (139 after adding the cache-method result) by moving existing historical sections verbatim to `slop/research/20261007_historical_bsbench_results/README.md`, with relative links adjusted. The current 9B result now appears before quickstart.
- Current figure moved to `assets/bsbench/qwen3.5-9b.png`. References and scratch refresh scripts follow the move. Six older figure paths remain unchanged because archived reviews cite them; they are no longer competing main figures in README.
- `just sweep METHOD` now explicitly uses 9B, nonsense-question pairs, three seeds and controls; random is a separate recipe. `just results` explicitly renders the same main report and copies its verified figure into README assets. Previous recipes silently defaulted to 4B/generic pairs despite the 9B README; `just_dry_run.log` records the corrected commands.
- `.local/` and `.pi/subagents/` are ignored; research evidence stays tracked. Unrelated dirty logs and untracked research were left alone.
- The pre-low-dose comparison snapshot shrank from 3,647,528 to 352,175 bytes by retaining only provenance, summaries and the point metrics used for comparison. `baseline_recheck.log` still reports `REPORT_PASS old point metrics unchanged`. Full original content remains in git history and the machine-only prior report.
- Angular's selected empty curve now produces an explicit explanation of the cutoff instead of a silent toggle. Browser introduction now describes behavioral change rather than equating the off-axis cutoff with broken answers.

Verification: `links.log` reports 24 current and 50 historical local file targets exist, and the README figure equals the generated 9B PNG. `uat.log` passes, including the empty-selection explanation and measured marker checks. The maintained method and plotting changes are integrated. Cache-method smoke passed: 68 library tests, 11 typed cache tests, and the real tiny-model walk with four lower-dose additions (`slop/reviews/2026-10-07_cache_mean_diff/`). The three-seed sweep and maintained `just results` completed. `slop/reviews/2026-10-07_cache_mean_diff/verify_report.log` verifies 116 method points, both common-seed curves and lower doses; `judge_report.log` reports `JUDGE_COMPLETE missing=0`. The method produces little directed premise change in this setup; its high rank comes from near-inactivity. Evidence, costs and limitations are in that directory's `results.md`. Maintained recipes/UI fixes are committed as `bef462f`; README rearrangement and figure-path moves remain an editorial working-tree diff.

Retained limitations: 12 previously observed full-BEARTYPE annotation failures outside the cache-specific smoke are not repaired here. No complete scientific audit of every benchmark answer or method, and no full-history secret scan, is claimed. Raw evidence is retained rather than removed merely to make the tree smaller.
