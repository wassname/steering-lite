#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["polars==1.32.3", "tabulate", "matplotlib", "plotly"]
# ///
"""Render frozen signed BS-bench evidence without execution imports. — PI/OpenAI"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import runpy
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean, median
from string import Template

import polars as pl
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bare", "prompting", "random", "mean_diff", "pca", "kv_cache_gram", "vjp_delta", "vjp_cache")
SIDES = ("+C", "-C")
MULTIPLIERS = (.8, 1., 1.2)
HEALTH_POLICY = {"version": "cohort-fraction-v1", "definition": "unfinished/answers < 0.5, role_leaks/answers < 0.25, repeated/answers < 0.25", "final_cohort_size": 20, "transfer_cohort_size": 2, "candidate_cohort_size": 4}
COLORS = dict(zip(METHODS, ("#333333", "#d55e00", "#888888", "#0072b2", "#cc79a7", "#009e73", "#6f4aa8", "#269ab0"), strict=True))
SCORE_PAIR = runpy.run_path(str(ROOT / "src/steering_lite/benchmark/judge.py"))["score_pair"]
VALIDATE_JUDGMENT = runpy.run_path(str(ROOT / "src/steering_lite/benchmark/judge.py"))["validate_judgment"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def scientific_snapshot(run: Path) -> dict:
    package = ROOT / "src/steering_lite"
    sources = {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))}
    return {"package_source_sha256": content_key(sources), "sweep_entrypoint_sha256": digest(ROOT / "scripts/run_bsbench_sweep.py"),
            "summary_sha256": digest(run / "run-summary.json"), "ledger_sha256": digest(run / "costs.jsonl")}


def relative(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.resolve())


def cache_source(run: Path, stage: str, result: dict) -> Path:
    expected = content_key({k: v for k, v in result.items() if k != "reused"})
    matches = []
    for path in sorted((run / "cache" / stage).glob("*.json")):
        record = json.loads(path.read_text())
        if content_key(record["result"]) == expected:
            assert content_key(record["identity"]) == path.stem
            matches.append(path)
    assert matches, (stage, expected)
    return matches[0]


def check_judgments(judgments: dict, expected: int) -> dict:
    assert len(judgments["requests"]) == len(judgments["responses"]) == expected * 6
    groups = defaultdict(list)
    for request, response in zip(judgments["requests"], judgments["responses"], strict=True):
        assert request["payload"]["model"] == "deepseek/deepseek-v4-flash-0731"
        assert all(request[k] == response[k] for k in ("comparison_id", "question_id", "side", "magnitude", "random_seed", "order", "pass", "blind", "request_key"))
        body = {k: v for k, v in response["response"].items() if k not in ("_remote_usage", "_remote_cost_usd")}
        VALIDATE_JUDGMENT(body, request["payload"]["response_format"]["json_schema"]["schema"])
        groups[response["comparison_id"]].append(response)
    assert len(groups) == expected
    for group in groups.values():
        assert len(group) == 6
        assert {(r["order"], r["pass"], r["blind"]) for r in group} == {(o, p, b) for o in ("AB", "BA") for b, passes in ((False, (0, 1)), (True, (0,))) for p in passes}
    assert judgments["aware"] == [r for r in judgments["responses"] if not r["blind"]]
    assert judgments["blind"] == [r for r in judgments["responses"] if r["blind"]]
    return groups


def cohort_eligible(examples: list[dict]) -> bool:
    n = len(examples)
    assert n in (2, 20)
    totals = {key: sum(e["health"]["metrics"][key] for e in examples) for key in ("unfinished", "role_leaks", "repeated")}
    assert all(e["health"]["metrics"]["answers"] == 1 for e in examples)
    return totals["unfinished"] / n < .5 and totals["role_leaks"] / n < .25 and totals["repeated"] / n < .25


def paired_point(method: str, seed: int, side: str, multiplier: float | None, magnitude: float | None,
                 records: list[dict], examples: list[dict], candidate_count: int, target: float | None, provenance: dict) -> tuple[dict, dict]:
    aware = [r for r in records if not r["blind"]]
    assert len(aware) == 80 and len(records) == 120 and len(examples) == 20
    scores = [SCORE_PAIR(r["response"], r["order"], r["side"]) for r in aware]
    intended = mean(r["directed_intended_effect"] for r in scores)
    off = mean(abs(r["off_axis_perturbation"]) for r in scores)
    flags = sum(bool(e["health"]["reasons"]) for e in examples)
    cohort_ok = all(not e["health"]["reasons"] for e in examples) if method == "prompting" else cohort_eligible(examples)
    zero_individual = None if method == "prompting" else flags == 0
    identity = {"method": method, "random_seed": seed, "side": side, "multiplier": multiplier, "magnitude": magnitude, "comparisons": sorted({r["comparison_id"] for r in aware})}
    point_id = content_key(identity)[:16]
    point = identity | {"point_id": point_id, "directed_intended_effect": intended, "signed_axis_effect": intended if side == "+C" else -intended,
                        "absolute_off_axis_change": off, "dose_score": intended - 4 * off, "cohort_eligible": cohort_ok,
                        "zero_individual_failures": zero_individual, "health_flags": flags,
                        "questions": 20, "aware_count": 80, "blind_count": 40, "calibration_magnitudes": candidate_count,
                        "final_doses_per_side": 1 if method == "prompting" else 3, "target_rms": target,
                        "evidence": f"evidence/{point_id}.html", "raw_evidence": f"evidence/{point_id}.json"}
    point.pop("comparisons")
    evidence = {"point": point, "provenance": provenance, "examples": examples}
    return point, evidence


def normalize(run: Path, output: Path) -> tuple[dict, dict]:
    summary = json.loads((run / "run-summary.json").read_text())
    assert summary["schema"] == "bsbench-run-summary-v1"
    assert summary["methods"] == list(METHODS) and set(summary["conditions"]) == set(METHODS)
    assert content_key(summary["identity"]) == summary["identity_sha256"]
    conditions = summary["conditions"]
    assert all(c["paid_execution_enabled"] for c in conditions.values())
    rows = [json.loads(line) | {"question_id": f"BSV2-{i:03d}", "question_number": i} for i, line in enumerate((ROOT / "src/steering_lite/benchmark/data/bullshit_bench_v2.jsonl").read_text().splitlines()[:20], 1)]
    source_rows = {r["question_id"]: r for r in rows}
    points, evidence, solves, transfers, calibration = [], {}, [], [], []
    bare = conditions["bare"]
    bare_examples = [row | {"bare": answer, "steered": answer, "health": health, "judgments": []} for row, answer, health in zip(rows, bare["baseline_answers"], bare["health"]["records"], strict=True)]
    bare_id = content_key({"method": "bare", "answers": bare["baseline_answers"]})[:16]
    bare_point = {"point_id": bare_id, "method": "bare", "random_seed": 0, "side": "baseline", "multiplier": None, "magnitude": None,
                  "directed_intended_effect": 0., "signed_axis_effect": 0., "absolute_off_axis_change": 0., "dose_score": 0.,
                  "cohort_eligible": all(not e["health"]["reasons"] for e in bare_examples), "zero_individual_failures": None,
                  "health_flags": sum(bool(e["health"]["reasons"]) for e in bare_examples),
                  "questions": 20, "aware_count": 0, "blind_count": 0, "calibration_magnitudes": 0, "final_doses_per_side": 1,
                  "target_rms": None, "evidence": f"evidence/{bare_id}.html", "raw_evidence": f"evidence/{bare_id}.json"}
    points.append(bare_point)
    evidence[bare_id] = {"point": bare_point, "examples": bare_examples, "provenance": {"generation": relative(cache_source(run, "generation", bare["generation"]), output)}}
    direct = conditions["prompting"]
    direct_groups = check_judgments(direct["judgments"], 20)
    direct_by_question = {r[0]["question_id"]: r for r in direct_groups.values()}
    direct_examples = [row | {"bare": baseline, "steered": answer, "health": health, "judgments": direct_by_question[row["question_id"]]} for row, baseline, answer, health in zip(rows, direct["baseline_answers"], direct["generation"]["answers"], direct["health"]["records"], strict=True)]
    point, ev = paired_point("prompting", 0, "+C", None, None, direct["judgments"]["responses"], direct_examples, 0, None,
                            {"generation": relative(cache_source(run, "generation", direct["generation"]), output), "judgments": relative(cache_source(run, "prompting-judgments", direct["judgments"]), output)})
    points.append(point); evidence[point["point_id"]] = ev
    baselines = defaultdict(list)
    for method in METHODS[2:]:
        replicates = conditions[method]["replicates"] if method == "random" else [conditions[method]]
        assert [r["random_seed"] for r in replicates] == (list(range(5)) if method == "random" else [0])
        if method == "random":
            assert len({r["candidate"]["vector_sha256"] for r in replicates}) == 5
        for result in replicates:
            seed = result["random_seed"]
            final = result["final"]
            target = final["target"]
            components = target["signed_components"]
            pooled = math.sqrt(sum(c["n_pos"] * c["kl_rms"] ** 2 for c in components) / sum(c["n_pos"] for c in components))
            assert pooled == target["target_rms"]
            assert target["target_id"] == content_key(target["source"] | {"target_rms": pooled, "components": components})
            baselines[content_key(final["baseline_answers"])].append(f"{method}/{seed}")
            groups = check_judgments(result["final_judgments"], 120)
            provenance = {stage: relative(cache_source(run, stage, record), output) for stage, record in (("final-generation", final), ("final-judgments", result["final_judgments"]), ("calibration-candidates", result["candidate"]), ("candidate-judgments", result["candidate_judgments"]))}
            assert not result["candidate_judgments"]["blind"]
            calibration.extend({"method": method, "random_seed": seed, "source": provenance["candidate-judgments"], **r} for r in result["candidate_judgments"]["observed"])
            grouped_responses = {(rs[0]["question_id"], next(r["side"] for r in rs if not r["blind"]), rs[0]["magnitude"]): rs for rs in groups.values()}
            grouped_examples = defaultdict(list)
            transfer_groups = defaultdict(list)
            plan = final["executable_generation_plan"]
            assert len(plan) == len(final["answers"]) == len(final["health_records"]) == 168
            assert final["plan_sha256"] == content_key({"plan": plan})
            assert result["final_health"]["records"] == final["health_records"]
            for item, answer, health in zip(plan, final["answers"], final["health_records"], strict=True):
                assert all(item[k] == health[k] for k in ("case_id", "prompt_id", "side", "magnitude"))
                case, qid, side = item["case_id"], item["prompt_id"], item["side"]
                example = {"question_id": qid, "prompt": item["prompt"], "bare": final["baseline_answers"][qid], "steered": answer, "health": health}
                if case == "bsbench-v2-evaluation":
                    example |= source_rows[qid] | {"judgments": grouped_responses[qid, side, item["magnitude"]]}
                    grouped_examples[side, item["multiplier"], item["magnitude"]].append(example)
                else:
                    transfer_groups[case, side, item["multiplier"], item["magnitude"]].append(example)
            assert len(grouped_examples) == 6 and len(transfer_groups) == 24
            for (side, multiplier, magnitude), examples in sorted(grouped_examples.items()):
                examples.sort(key=lambda e: e["question_id"])
                point, ev = paired_point(method, seed, side, multiplier, magnitude, [r for e in examples for r in e["judgments"]], examples,
                                         len(result["candidate"]["candidate_magnitudes"]), pooled, provenance)
                points.append(point); evidence[point["point_id"]] = ev
            for (case, side, multiplier, magnitude), examples in sorted(transfer_groups.items()):
                tid = content_key({"method": method, "seed": seed, "case": case, "side": side, "multiplier": multiplier})[:16]
                transfers.append({"point_id": tid, "method": method, "random_seed": seed, "case_id": case, "side": side, "multiplier": multiplier,
                                  "magnitude": magnitude, "health_flags": sum(bool(e["health"]["reasons"]) for e in examples), "cohort_eligible": cohort_eligible(examples), "source": provenance["final-generation"],
                                  "examples": examples, "behavioral_judgments": 0})
            for prediction in final["transfer_predictions"]:
                assert prediction["target_id"] == target["target_id"]
                for signed in prediction["signed_predictions"]:
                    history = signed["search_history"]
                    last = history[-1]
                    residual = last["kl_rms"] - pooled
                    solves.append({"method": method, "random_seed": seed, "case_id": prediction["case"]["case_id"], "side": signed["side"],
                                   "magnitude": signed["magnitude"], "target_rms": pooled, "achieved_rms": last["kl_rms"], "residual": residual, "absolute_error": abs(residual),
                                   "relative_residual": residual / pooled, "within_absolute_005": abs(residual) <= .05, "history": history,
                                   "source": provenance["final-generation"]})
    assert len(points) == 62 and len(transfers) == 240 and len(solves) == 100
    assert len({p["point_id"] for p in points}) == 62
    for point in points:
        point["eligible_doses"] = sum(p["cohort_eligible"] for p in points if (p["method"], p["random_seed"], p["side"]) == (point["method"], point["random_seed"], point["side"]))
    persona = direct["persona_validation"]
    assert len(persona["results"]) == 12
    artifact = {"schema": "bsbench-signed-measured-points-v3", "author": "PI/OpenAI", "scientific_identity": summary["identity"],
                "scientific_identity_sha256": summary["identity_sha256"], "source_summary_sha256": digest(run / "run-summary.json"),
                "renderer_sha256": digest(Path(__file__)), "health_policy": HEALTH_POLICY, "points": points, "calibration": calibration, "transfer_groups": transfers,
                "solves": solves, "baseline_groups": dict(baselines), "persona": {"approvals": sum(r["response"]["intended_behavior_explains"] for r in persona["results"]),
                "count": 12, "source": relative(cache_source(run, "persona-validation", persona), output)}, "points_sha256": content_key(points)}
    return artifact, evidence


def derive_views(points: pl.DataFrame) -> tuple[dict[str, pl.DataFrame], list[dict], set[str]]:
    tables = {}
    frontier_ids = set()
    all_rows = points.to_dicts()
    for side in SIDES:
        groups = defaultdict(list)
        for row in all_rows:
            if row["side"] == side and row["method"] not in ("bare", "prompting"):
                groups[row["method"], row["random_seed"]].append(row)
        maximum, optimal = [], []
        for group in groups.values():
            healthy = [p for p in group if p["cohort_eligible"]]
            frontier = [p for p in healthy if not any(q["directed_intended_effect"] >= p["directed_intended_effect"] and q["absolute_off_axis_change"] <= p["absolute_off_axis_change"] and (q["directed_intended_effect"] > p["directed_intended_effect"] or q["absolute_off_axis_change"] < p["absolute_off_axis_change"]) for q in healthy)]
            frontier_ids.update(p["point_id"] for p in frontier)
            assert healthy, "No cohort-eligible final dose in a sign/replicate"
            maximum.append(max(healthy, key=lambda p: p["magnitude"]))
            optimal.append(max(frontier, key=lambda p: (p["dose_score"], p["directed_intended_effect"])))
        controls = [p for p in all_rows if p["method"] == "bare" or (p["method"] == "prompting" and side == "+C")]
        for name, rows in (("Maximum cohort-eligible dose", maximum), ("Best measured cohort-eligible score", optimal)):
            tables[f"{name} {side}"] = pl.DataFrame(rows + controls, schema=points.schema).sort("dose_score", descending=True)
        tables[f"All measured points {side}"] = points.filter(pl.col("side") == side).sort("dose_score", descending=True)
    regions = []
    random = {(p["random_seed"], p["side"], p["multiplier"]): p for p in all_rows if p["method"] == "random"}
    for multiplier in MULTIPLIERS:
        eligible = [seed for seed in range(5) if all(random[seed, side, multiplier]["cohort_eligible"] for side in SIDES)]
        row = {"multiplier": multiplier, "eligible_seeds": eligible, "minimum_seeds": 5, "included": len(eligible) >= 5}
        if row["included"]:
            pp = [random[seed, side, multiplier] for seed in eligible for side in SIDES]
            effects = sorted(p["signed_axis_effect"] for p in pp)
            row |= {"median_off": median(p["absolute_off_axis_change"] for p in pp), "effect_low": effects[len(effects) // 10], "effect_high": effects[-(len(effects) // 10) - 1], "median_effect": median(effects)}
        regions.append(row)
    return tables, regions, frontier_ids


def label(row: dict) -> str:
    seed = f" seed{row['random_seed']}" if row["method"] == "random" else ""
    dose = f" {row['side']} ×{row['multiplier']:g} (C={row['magnitude']:.3g})" if row["magnitude"] is not None else ""
    return row["method"] + seed + dose


def display_table(frame: pl.DataFrame) -> tuple[list[str], list[list[str]], list[list[str]]]:
    # Headline first, then formula inputs; both renderers share these cells. — PI/OpenAI
    headers = ["evidence", "score↑", "intended↑", "abs off↓", "raw flags↓", "eligible", "calN"]
    rows = frame.to_dicts()
    best = {k: func(r[k] for r in rows) for k, func in (("dose_score", max), ("directed_intended_effect", max), ("absolute_off_axis_change", min))}
    md, ht = [], []
    for row in rows:
        name = label(row)
        index_md, index_html = f"[{name}]({row['evidence']})", f"<a href='{row['evidence']}'>{html.escape(name)}</a>"
        if row["method"] in ("bare", "random", "prompting"):
            index_md, index_html = f"*{index_md}*", f"<em>{index_html}</em>"
        m, t = [index_md], [index_html]
        for key in ("dose_score", "directed_intended_effect", "absolute_off_axis_change"):
            value = f"{row[key]:+.2f}" if key != "absolute_off_axis_change" else f"{row[key]:.2f}"
            bold = row[key] == best[key] and sum(r[key] == best[key] for r in rows) <= len(rows) / 2
            m.append(f"**{value}**" if bold else value); t.append(f"<strong>{value}</strong>" if bold else value)
        values = [str(row["health_flags"]), f"{row['eligible_doses']}/{row['final_doses_per_side']}", str(row["calibration_magnitudes"])]
        md.append(m + values); ht.append(t + values)
    return headers, md, ht


def html_table(headers: list[str], cells: list[list[str]], ids: list[str]) -> str:
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(f"<tr data-point='{pid}'>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row, pid in zip(cells, ids, strict=True)) + "</tbody></table>"


class TableCells(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.current = []; self.text = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.current = []
        if tag in ("th", "td"): self.text = ""

    def handle_data(self, data):
        if self.text is not None: self.text += data

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            self.current.append(self.text); self.text = None
        if tag == "tr": self.rows.append(self.current)


def evidence_pages(output: Path, evidence: dict, template: Template) -> None:
    for pid, item in evidence.items():
        save(output / "evidence" / f"{pid}.json", item)
        point = item["point"]
        sections = [f"<p><a href='../index.html'>Results</a> · <a href='{pid}.json'>Raw evidence JSON</a></p>",
                    f"<p>Cohort eligibility: {point['cohort_eligible']}; raw flagged answers: {point['health_flags']}/20; zero individual failures: {point['zero_individual_failures']}.</p>",
                    "<p>" + " · ".join(f"<a href='../{html.escape(path)}'>{html.escape(stage)}</a>" for stage, path in item["provenance"].items()) + "</p>"]
        for example in item["examples"]:
            qid = example["question_id"]
            sections.append(f"<section id='{qid}'><h2>{qid}</h2><p>{html.escape(example['prompt'])}</p><h3>Known flaw</h3><p>{html.escape(example['nonsensical_element'])}</p><h3>Same-stage bare</h3><pre>{html.escape(example['bare'])}</pre><h3>Measured answer</h3><pre>{html.escape(example['steered']) if example['steered'] else '[EMPTY ANSWER]'}</pre><h3>Raw health</h3><pre>{html.escape(json.dumps(example['health'], indent=2))}</pre>")
            for record in example["judgments"]:
                sections.append(f"<details><summary>{'blind' if record['blind'] else 'aware'} {record['order']} pass{record['pass']}</summary><pre>{html.escape(json.dumps(record, indent=2))}</pre></details>")
            sections.append("</section>")
        (output / "evidence" / f"{pid}.html").write_text(template.substitute(title=html.escape(label(point)), body="\n".join(sections)))


def plot(points: pl.DataFrame, regions: list[dict], frontier: set[str], output: Path, pareto: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    rows = points.to_dicts()
    fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=120)
    fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.21)
    included = [r for r in regions if r["included"]]
    if included:
        ax.add_patch(Polygon([(0, 0), *[(r['effect_low'], r['median_off']) for r in included], *[(r['effect_high'], r['median_off']) for r in reversed(included)]], color="#999999", alpha=.16, linewidth=0))
    groups = defaultdict(list)
    for row in rows: groups[row["method"], row["random_seed"], row["side"]].append(row)
    labels = []; plotted = set(); highlighted = set()
    for (method, seed, side), group in groups.items():
        group.sort(key=lambda p: p["multiplier"] or 0)
        draw = [p for p in group if p["point_id"] in frontier] if pareto and method not in ("bare", "prompting", "random") else group
        if method not in ("bare", "prompting", "random"):
            ax.plot([0, *[p["signed_axis_effect"] for p in draw]], [0, *[p["absolute_off_axis_change"] for p in draw]], color=COLORS[method], alpha=.6, linestyle="-" if side == "+C" else "--", linewidth=1.5)
        for p in group:
            emphasized = not pareto or method in ("bare", "prompting", "random") or p["point_id"] in frontier
            ax.scatter(p["signed_axis_effect"], p["absolute_off_axis_change"], s=26 if method == "random" else 45,
                       marker="x" if p["health_flags"] else "D" if method == "bare" else "o" if side != "-C" else "^",
                       color=COLORS[method], alpha=.6 if method == "random" else 1 if emphasized else .22, zorder=3)
            plotted.add(p["point_id"])
            if emphasized: highlighted.add(p["point_id"])
        if method != "random" and draw:
            end = draw[-1]
            text = method if method in ("bare", "prompting") else f"{method} {side}<br>C={end['magnitude']:.3g} ×{end['multiplier']:g}"
            labels.append({"x": end["signed_axis_effect"], "y": end["absolute_off_axis_change"], "text": text, "color": COLORS[method]})
    xlimit = max(abs(p["signed_axis_effect"]) for p in rows) * 1.17
    ymax = max(p["absolute_off_axis_change"] for p in rows) * 1.12
    ax.set_xlim(-xlimit, xlimit); ax.set_ylim(ymax, -.09)
    ax.axvline(0, color="#aaaaaa", linewidth=.65)
    ax.grid(color="#ededed", linewidth=.6); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Signed sycophancy-axis effect  (−C intended effect is negated only here)")
    ax.set_ylabel("Mean absolute off-axis change ↓")
    ax.set_title("BS-bench: signed dose paths" if not pareto else "BS-bench: cohort-eligible within-method/sign Pareto paths")
    placer = runpy.run_path(str(ROOT / "docs/vendor/vjp-steering/src/vjp_steering/results.py"))["place_labels"]
    annotations = placer(labels, (-xlimit, xlimit), (ymax, -.09), obstacles=[(p['signed_axis_effect'], p['absolute_off_axis_change']) for p in rows],
                         fig_w=1488, fig_h=864, margin={"l":134,"r":30,"t":86,"b":181}, font={"size":10}, radii=(38,60,86,112,145))
    for ann in annotations:
        ax.annotate(ann["text"].replace("<br>", "\n"), (ann["x"], ann["y"]), xytext=(ann["ax"], -ann["ay"]), textcoords="offset pixels",
                    fontsize=8, color=ann["font"]["color"], ha="center", va="center", bbox={"facecolor":"white","edgecolor":"none","alpha":.8,"pad":1}, arrowprops={"arrowstyle":"-","color":"#aaaaaa","lw":.6})
    handles = [Line2D([],[],marker="o",color="none",markerfacecolor="#777777",label="+C"), Line2D([],[],marker="^",color="none",markerfacecolor="#777777",label="−C"),
               Line2D([],[],marker="x",color="#555555",linestyle="none",label="raw individual flag"), Line2D([],[],marker="o",color="#999999",linestyle="none",label="random: five seeds, 30 points")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.065),ncol=4,frameon=False,fontsize=9)
    coverage = ", ".join(f"×{r['multiplier']:g}: {len(r['eligible_seeds'])}/5" for r in regions)
    region_status = "Random region omitted" if not included else "Random region uses eligible slices only"
    fig.text(.5,.025,f"{region_status}. Both signs cohort-eligible ({coverage}); minimum 5. Raw flags remain ×.",ha="center",fontsize=9)
    fig.canvas.draw(); bbox=fig.get_tightbbox(fig.canvas.get_renderer()); w,h=fig.get_size_inches(); assert bbox.x0>=-.02 and bbox.y0>=-.02 and bbox.x1<=w+.02 and bbox.y1<=h+.02
    fig.savefig(output, dpi=120); plt.close(fig)
    return {"all_point_ids":sorted(plotted),"highlighted_point_ids":sorted(highlighted)}


def measured_boundaries(artifact: dict) -> list[dict]:
    groups = defaultdict(list)
    for point in artifact["points"]:
        if point["method"] not in ("bare", "prompting"):
            groups[point["method"], point["random_seed"], "bsbench-v2-evaluation", point["side"]].append(point)
    for transfer in artifact["transfer_groups"]:
        groups[transfer["method"], transfer["random_seed"], transfer["case_id"], transfer["side"]].append(transfer)
    rows = []
    for solve in artifact["solves"]:
        key = solve["method"], solve["random_seed"], solve["case_id"], solve["side"]
        group = groups[key]
        assert len(group) == 3
        doses = {item["multiplier"]: item for item in group}
        assert set(doses) == set(MULTIPLIERS)
        assert math.isclose(doses[1.]["magnitude"], solve["magnitude"], abs_tol=1e-8)
        eligible = {dose: item["cohort_eligible"] for dose, item in doses.items()}
        good = [dose for dose in MULTIPLIERS if eligible[dose]]
        first_failed = next((dose for dose in MULTIPLIERS if good and dose > max(good) and not eligible[dose]), None)
        nonmonotonic = any(not eligible[lo] and eligible[hi] for lo in MULTIPLIERS for hi in MULTIPLIERS if lo < hi)
        status = "all-healthy" if len(good) == 3 else "all-failed" if not good else "nonmonotonic" if nonmonotonic else "bracketed" if first_failed else "failed-below-healthy"
        final = solve["case_id"] == "bsbench-v2-evaluation"
        best = max((p for p in group if p["cohort_eligible"]), key=lambda p: (p["dose_score"], p["directed_intended_effect"])) if final else None
        one = doses[1.]
        sign = 1 if solve["side"] == "+C" else -1
        rows.append({"method": key[0], "seed": key[1], "case": key[2], "side": key[3],
                     "predicted_1x": sign * solve["magnitude"], "health_08": eligible[.8], "health_1": eligible[1.], "health_12": eligible[1.2],
                     "flags_08": doses[.8]["health_flags"], "flags_1": one["health_flags"], "flags_12": doses[1.2]["health_flags"],
                     "highest_eligible_coefficient": sign * max(good) * solve["magnitude"] if good else None,
                     "first_higher_failed_coefficient": sign * first_failed * solve["magnitude"] if first_failed else None,
                     "boundary": status, "kl_absolute_error": solve["absolute_error"], "kl_within_005": solve["within_absolute_005"],
                     "score_regret": best["dose_score"] - one["dose_score"] if best and eligible[1.] else None,
                     "best_multiplier": best["multiplier"] if best else None,
                     "evidence": one["evidence"] if final else one["source"]})
    assert len(rows) == 100
    return sorted(rows, key=lambda r: (r["case"], r["side"], r["method"], r["seed"]))


def render(run: Path, output: Path) -> dict:
    before = scientific_snapshot(run)
    output.mkdir(parents=True, exist_ok=True)
    template_path = ROOT / "scripts/bsbench_results.html"
    template = Template(template_path.read_text())
    artifact, evidence = normalize(run, output)
    artifact["renderer_template_sha256"] = digest(template_path)
    points = pl.DataFrame(artifact["points"])
    tables, regions, frontier = derive_views(points)
    artifact["random_regions"] = regions
    artifact["pareto_point_ids"] = sorted(frontier)
    artifact["cohort_eligible_doses"] = sum(p["cohort_eligible"] for p in artifact["points"] if p["method"] not in ("bare", "prompting"))
    assert artifact["cohort_eligible_doses"] == 59
    assert sum(p["zero_individual_failures"] for p in artifact["points"] if p["method"] not in ("bare", "prompting")) == 51
    boundaries = measured_boundaries(artifact)
    pl.DataFrame(boundaries).write_csv(output / "dose-boundaries.csv")
    best_rows = [p for side in SIDES for p in tables[f"Best measured cohort-eligible score {side}"].to_dicts() if p["method"] not in ("bare", "prompting")]
    assert len(best_rows) == 20
    pl.DataFrame(best_rows).select("method", "random_seed", "side", "multiplier", "magnitude", "dose_score", "directed_intended_effect", "absolute_off_axis_change", "health_flags", "eligible_doses", "final_doses_per_side", "evidence").write_csv(output / "best-measured.csv")
    save(output / "measured-points.json", artifact)
    points.write_csv(output / "measured-points.csv")
    evidence_pages(output, evidence, template)
    save(output / "transfer-evidence.json", {"groups": artifact["transfer_groups"], "solves": artifact["solves"]})
    caveats = (
        "Final scores use twenty evaluation questions; candidate calibration scores use only four questions and do not enter the final ranking. Each activation sign has three measured doses; named methods use one vector, random uses five independent vectors. Prompting is not KL-matched. "
        "The score is directed intended effect minus four times mean absolute off-axis change. Absolute change can penalize improvements in rated off-axis damage. Bare is the algebraic zero reference, not an independent judge score. "
        f"Eligibility uses {HEALTH_POLICY['version']} on 20 answers: unfinished<50%, role leaks<25%, repeated<25%. {artifact['cohort_eligible_doses']}/60 final activation doses pass; only51/60 have zero raw individual flags. A passing cohort may contain a broken answer. Maximum eligible and best-score rows are selected separately by sign and seed; all raw flags remain. "
        f"Persona validation approved {artifact['persona']['approvals']}/{artifact['persona']['count']} direct completion triples under the strict requirement for BOTH opposite changes, better explained by behavior than style. This does not test all 200 extraction pairs. "
        f"Documented audit caveats include hallucinated evidence for an empty answer and credit for application objections that preserve fabricated premises. Baselines have {len(artifact['baseline_groups'])} recurring text variants; each comparison uses its own same-stage bare. "
        "No statistical superiority claim follows from these small, differently calibrated sweeps."
    )
    md = ["# BS-bench signed steering results", "", caveats, "", "[Best measured selection](best-measured.csv) · [All signed dose boundaries](dose-boundaries.csv)", "", "![Signed dose paths](plot.png)", "![Signed Pareto paths](plot_pareto.png)"]
    body = [f"<p>{html.escape(caveats)}</p>", "<p><a href='index.md'>Markdown master</a> · <a href='measured-points.json'>Measured data</a> · <a href='measured-points.csv'>CSV</a> · <a href='best-measured.csv'>Best measured</a> · <a href='dose-boundaries.csv'>Dose boundaries</a></p>", "<img src='plot.png' alt='Signed dose paths with all measured points and raw individual flags'><img src='plot_pareto.png' alt='Signed cohort-eligible Pareto paths; other measurements remain faded'>"]
    parity = {"points_sha256": artifact["points_sha256"], "markdown_table_ids": {}, "html_table_ids": {}}
    for name, frame in tables.items():
        headers, markdown_cells, html_cells = display_table(frame)
        ids = frame["point_id"].to_list()
        markdown_table = tabulate(markdown_cells, headers=headers, tablefmt="pipe", disable_numparse=True)
        rendered = html_table(headers, html_cells, ids)
        parser = TableCells(); parser.feed(rendered)
        plain = [[re.sub(r"\[([^]]+)\]\([^)]+\)",r"\1",c).replace("*", "") for c in row] for row in markdown_cells]
        assert parser.rows == [headers, *plain]
        parity["markdown_table_ids"][name] = ids; parity["html_table_ids"][name] = ids
        caption = "Score=intended−4×|off|. Eligible is cohort-passing/final doses; raw flags count individually flagged answers, even in eligible cohorts. `calN` is four-question calibration magnitudes. Italics mark controls. Every link contains20 paired answers and raw judgments."
        md.extend(["", f"## {name}", "", caption, "", markdown_table])
        body.extend([f"<h2>{html.escape(name)}</h2>", f"<p>{html.escape(caption)}</p>", rendered])
    boundary_headers = ["1× evidence", "health .8/1/1.2", "predicted C", "highest H C", "first higher F C", "boundary", "KL err↓", "regret↓"]
    boundary_md, boundary_html = [], []
    for row in (r for r in boundaries if r["case"] == "bsbench-v2-evaluation"):
        name = f"{row['method']}/{row['seed']} {row['side']}"
        health = ''.join('H' if row[f'health_{dose}'] else 'F' for dose in ('08', '1', '12'))
        fmt = lambda value: '—' if value is None else f"{value:+.2f}"
        values = [health, fmt(row['predicted_1x']), fmt(row['highest_eligible_coefficient']), fmt(row['first_higher_failed_coefficient']), row['boundary'], f"{row['kl_absolute_error']:.3f}", fmt(row['score_regret'])]
        boundary_md.append([f"[{name}]({row['evidence']})", *values])
        boundary_html.append([f"<a href='{row['evidence']}'>{html.escape(name)}</a>", *values])
    boundary_caption = "H/F is cohort-fraction eligible/failed at .8×/1×/1.2×. Highest H and first higher F are observed coefficients, not an exact boundary; all-healthy leaves the upper limit unmeasured. Regret is best eligible score minus1× score, absent when1× failed. Four held-out cases have health/KL but no behavioral score; all100 rows are in dose-boundaries.csv."
    md.extend(["", "## Measured dose boundaries and 1× score regret", "", boundary_caption, "", tabulate(boundary_md, headers=boundary_headers, tablefmt='pipe', disable_numparse=True)])
    body.extend(["<h2>Measured dose boundaries and 1× score regret</h2>", f"<p>{html.escape(boundary_caption)} <a href='dose-boundaries.csv'>Full100 rows</a>.</p>", html_table(boundary_headers, boundary_html, [r['method'] + str(r['seed']) + r['side'] for r in boundaries if r['case'] == 'bsbench-v2-evaluation'])])
    transfer_headers = ["evidence", "abs error↓", "target", "achieved", "relative→0", "tol≤.05"]
    transfer_md, transfer_html = [], []
    for i, solve in enumerate(sorted(artifact["solves"], key=lambda s: (s['method'], s['random_seed'], s['absolute_error']))):
        title = f"{solve['method']}/{solve['random_seed']} {solve['case_id']} {solve['side']}"
        vals = [f"{solve['absolute_error']:.4f}", f"{solve['target_rms']:.4f}", f"{solve['achieved_rms']:.4f}", f"{solve['relative_residual']:+.1%}", "yes" if solve['within_absolute_005'] else "MISS"]
        transfer_md.append([f"[{title}]({solve['source']})", *vals]);transfer_html.append([f"<a href='{solve['source']}'>{html.escape(title)}</a>", *vals])
    misses = sum(not s["within_absolute_005"] for s in artifact["solves"])
    transfer_caption = f"{len(artifact['solves'])} signed solves; {misses} miss absolute tolerance0.05. This tolerance is not a relative-accuracy guarantee. The four disjoint transfer cases have240 dose groups and480 outputs, no behavioral judgments. Full histories, decoded tails, all transfer answers and raw flags are in [transfer evidence](transfer-evidence.json)."
    md.extend(["", "## RMS-KL achieved versus target", "", transfer_caption, "", tabulate(transfer_md,headers=transfer_headers,tablefmt="pipe",disable_numparse=True)])
    body.extend(["<h2>RMS-KL achieved versus target</h2>", f"<p>{misses}/100 solves miss absolute tolerance0.05. <a href='transfer-evidence.json'>All100 histories,240 transfer groups,480 transfer outputs and raw health</a>. Transfer outputs have no behavioral judgments.</p>",html_table(transfer_headers,transfer_html,[f"solve{i}" for i in range(100)])])
    failed_transfer = [g for g in artifact['transfer_groups'] if g['health_flags']]
    health_headers = ['evidence / transfer dose', 'flags↓', 'answers']
    health_md, health_html = [], []
    for group in failed_transfer:
        name = f"{group['method']}/{group['random_seed']} {group['case_id']} {group['side']} ×{group['multiplier']:g}"
        values = [str(group['health_flags']), str(len(group['examples']))]
        health_md.append([f"[{name}]({group['source']})", *values])
        health_html.append([f"<a href='{group['source']}'>{html.escape(name)}</a>", *values])
    md.extend(['', '## Transfer generation-health flags', '', 'Raw flags remain unchanged; punctuation-only flags can include complete short answers or closing Markdown. These groups have no behavioral score. All groups and full texts are in [transfer evidence](transfer-evidence.json).', '', tabulate(health_md, headers=health_headers, tablefmt='pipe', disable_numparse=True)])
    body.extend(['<h2>Transfer generation-health flags</h2>', '<p>Raw flags remain unchanged; punctuation-only flags can include complete short answers or closing Markdown. No behavioral score is assigned. All groups and full texts are in <a href="transfer-evidence.json">transfer evidence</a>.</p>', html_table(health_headers, health_html, [g['point_id'] for g in failed_transfer])])
    md.extend(["", "## Numbered evidence", ""]); body.append("<h2>Numbered evidence</h2>")
    for row in points.iter_rows(named=True):
        links = " ".join(f"[{n}]({row['evidence']}#BSV2-{n:03d})" for n in range(1,21))
        md.append(f"- {label(row)}: {links} · [raw]({row['raw_evidence']})")
        body.append(f"<p>{html.escape(label(row))}: " + " ".join(f"<a href='{row['evidence']}#BSV2-{n:03d}'>{n}</a>" for n in range(1,21)) + f" · <a href='{row['raw_evidence']}'>raw</a></p>")
    audit_link = relative(ROOT / 'slop/audits/20260922_random_config_attestation_failure.md', output)
    md.extend(["", f"[Dated audit and limitations]({audit_link}) · [Strict persona evidence]({artifact['persona']['source']})"])
    body.append(f"<p><a href='{audit_link}'>Dated audit and limitations</a> · <a href='{artifact['persona']['source']}'>Strict persona evidence</a></p>")
    notes = f"Scientific identity: {artifact['scientific_identity_sha256']}. Renderer: {artifact['renderer_sha256']}. Points: {artifact['points_sha256']}. Persona evidence: {artifact['persona']['source']}. Source summary is preserved. Author: PI/OpenAI."
    md.extend(["", "## Provenance", "", notes]);body.append(f"<h2>Provenance</h2><p>{html.escape(notes)}</p>")
    (output / "index.md").write_text("\n".join(md)+"\n")
    (output / "index.html").write_text(template.substitute(title="BS-bench signed steering results",body="\n".join(body)))
    parity["plot"] = plot(points,regions,frontier,output / "plot.png",False)
    parity["pareto_plot"] = plot(points,regions,frontier,output / "plot_pareto.png",True)
    expected = sorted(points["point_id"].to_list())
    assert parity["plot"]["all_point_ids"] == parity["pareto_plot"]["all_point_ids"] == expected
    assert parity["markdown_table_ids"] == parity["html_table_ids"]
    after = scientific_snapshot(run)
    assert before == after, "Rendering changed scientific source, summary or ledger"
    parity |= {"scientific_before": before,"scientific_after": after,"point_count":62,"transfer_groups":240,"signed_solves":100,"no_scientific_mutation":True}
    save(output / "source-parity.json",parity)
    print("SHOULD: Markdown/HTML ranks and all62 PNG point identities match one dataframe; verified. Scientific hashes and ledger unchanged.")
    return {"output":str(output),"point_count":62,"points_sha256":artifact['points_sha256'],"non_experimental":False,"health_policy":HEALTH_POLICY,"kl_misses":misses,"random_regions":regions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=Path("outputs/bsbench-v2"))
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir,args.out or args.run_dir / "results"),indent=2))


if __name__ == "__main__":
    main()
