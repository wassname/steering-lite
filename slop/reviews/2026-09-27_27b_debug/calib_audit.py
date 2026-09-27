"""Is the dose walk's breakdown boundary real, and is the useful region inside it? (PI/Claude, 2026-09-27)

Per walk side (all methods, all seeds, 4B and 27B):
  n_pre   admissible rungs (health ok, before boundary, Jev damage <= 1.5)
  n_post  rungs at/after the first unhealthy one        -> need both > 0: a pre and a post
  edge    best dose (max on - off over admissible) is the LAST admissible rung before the first unhealthy one:
          the true optimum may sit in the unsampled gap to the next rung
  false   rungs flagged unhealthy by the regex health check while Jev damage < 1.0 (answers look fine to the judge):
          the walk may have stopped on a formatting artefact (e.g. 'unfinished' = no final . ! ? " ) )
Run: python slop/reviews/2026-09-27_27b_debug/calib_audit.py > slop/reviews/2026-09-27_27b_debug/calib_audit.md
"""
import collections, glob, json

for model, res, pat in (("4B", "full", "Qwen--Qwen3.5-4B"), ("27B", "27b-full", "Qwen--Qwen3.5-27B"), ("OLMo-32B", "olmo-full", "allenai--OLMo-2-0325-32B-Instruct")):
    md = glob.glob(f"outputs/bsbench/{pat}-g*")[0]
    pts = {(p["method"], p["seed"], p["C"], p["side"]): p for p in json.load(open(f"outputs/bsbench/results/{res}/points.json"))["points"]}
    tot = collections.Counter()
    false_rows, lines = [], []
    for path in sorted(glob.glob(f"{md}/walks/*_full.json")):
        c = json.load(open(path))
        if c["method"].startswith("prompting"):
            continue
        rungs = sorted(c["rungs"], key=lambda r: r["coefficient"])
        for side in ("-C", "+C"):
            rows = []
            for r in rungs:
                p = pts[(c["method"], c["seed"], r["coefficient"], side)]
                st = r[side]["stats"]
                on = p["effect"] if side == "+C" else -p["effect"]
                rows.append(dict(C=r["coefficient"], reasons=r[side]["breakdown_reasons"], ok=p["admissible"], on=on, off=p["off_axis"],
                                 dmg=p["steered_damage"], unf=st["unfinished"] / st["answers"], words=st["mean_words"], ans=r[side]["answers"]))
            first_bad = next(i for i, x in enumerate(rows) if x["reasons"])
            adm = [i for i, x in enumerate(rows) if x["ok"]]
            best = max(adm, key=lambda i: rows[i]["on"] - rows[i]["off"])
            edge = best == max(i for i in adm if i < first_bad)
            n_pre, n_post = sum(i < first_bad for i in adm), len(rows) - first_bad
            fb = rows[first_bad]
            falses = [x for x in rows if x["reasons"] and x["dmg"] < 1.0]
            for x in falses:
                false_rows.append((c["method"], c["seed"], side, x))
            tot["sides"] += 1; tot["edge"] += edge; tot["no_post"] += n_post == 0; tot["false"] += bool(falses)
            lines.append(f"| {c['method']} | {c['seed']} | {side} | {n_pre} | {n_post} | {'EDGE' if edge else ''} | {rows[best]['C']:.3g} | {rows[best]['on']:+.2f} | "
                         f"{fb['C']:.3g}: {','.join(fb['reasons'])} dmg {fb['dmg']:.2f} unf {fb['unf']:.0%} | {len(falses)} |")
    print(f"\n## {model}: {tot['sides']} walk sides; best at edge {tot['edge']}; no post-boundary rung {tot['no_post']}; sides with a false flag {tot['false']}\n")
    print("| method | seed | side | n_pre | n_post | edge | best C | best on | first unhealthy rung | false flags |\n|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(lines))
    if false_rows:
        print(f"\n### {model}: flagged unhealthy but Jev damage < 1.0\n\n| method | seed | side | C | reasons | Jev dmg | unfinished | mean words | on |\n|---|---|---|---|---|---|---|---|---|")
        for m, s, side, x in false_rows:
            print(f"| {m} | {s} | {side} | {x['C']:.3g} | {','.join(x['reasons'])} | {x['dmg']:.2f} | {x['unf']:.0%} | {x['words']:.0f} | {x['on']:+.2f} |")
