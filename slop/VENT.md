
## 2026-09-24 exit interview: pi-goals harness on bsbench-v3 -- PI/Claude

- helped: the goals file survived two compactions. Each goal's discriminator ("same sign as the README table", "walk passes the predicted dose") told me what to check next without re-reading the chat. The loop line "a finished run or a negative result is not done, check the port against the reference first" stopped me from reporting the first dev table as a result; the random-seed overlap with the reference and the stop-rule difference both came out of that check.
- helped: "record your assumption in the goals file and continue" made me pick a budget assumption instead of stalling on a question.
- friction: the loop prompt re-sends the whole goals file (~3k tokens) on every wake, and it fires while long jobs run, so I produced several near-duplicate status messages. My own scheduled checks then fired after the work they asked for was already done. A "nothing changed since last wake, keep waiting" path would cut this.
- friction: I have not called CompleteGoal yet. Writing quoted evidence per goal is fine, but I was unsure whether a goal with one open sub-question (random -C sign) should go to the judge or wait, so I waited. A "partial / blocked on X" state would help.
- misfire: the long-comment hook flagged a judge prompt string in blind.py as a comment wall.

## 2026-09-27 — something outside the container SIGTERMs big processes (PI/Claude)
results.py kept dying with exit 143 after ~10 s while loading the 800 MB Jev cache. strace: `SIGTERM {si_code=SI_USER, si_pid=0, si_uid=65534}` = sent from outside the container's user namespace (host/incus), not a session in here; cgroup memory.events oom_kill 0, no earlyoom/oomd. Fix that worked: judge.cached() drops Jev's echoed "legend" (60% of each record), peak RSS now 0.97 GB. If exit 143 shows up again on a memory-heavy step: strace -e signal=SIGTERM first, then shrink RSS; do not just retry.

## 2026-09-30 -- Self-verify references were not installed (PI/OpenAI)

`/home/code/.pi/agent/skills/self-verify/` contains only `SKILL.md`. Its required `references/boundary-probing.md` and `references/checklist.md` both returned ENOENT. The entrypoint checklist is usable, but the linked procedures are unavailable; restore the audited upstream references rather than assuming they were read. For the prompt sweep I checked the actual zero/identity/maximum-gain cases, raw coverage and explicitly missing causal/held-out controls instead.
- 2026-10-03 PI/OpenAI: steering-lite-bsbench worktree index.lock went stale 3 times today (0-byte, no holder). Short-lived 'git status --porcelain=2' / 'git diff HEAD --numstat' processes run every few seconds from something outside this session (footer/herdr?); one killed mid-refresh would leave the lock. Removed only after confirming no holder.

## 2026-10-03 `pi -p` in a background process hangs forever -- PI/OpenAI
A chained `pi --model X --no-tools --no-session -p "..."` sat 3 h with 2 s CPU under the process tool. Same call with `< /dev/null` returned in 28 s. Likely pi waits on an open non-TTY stdin. Always add `< /dev/null` (and a `timeout`). The earlier ENOTEMPTY/clone crash was a separate race on first install of gotgenes/pi-anthropic-auth; that dir now exists.
