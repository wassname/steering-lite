"""Judged walks -> points.json, the Pareto plot (PNG + HTML) and the results tables.

Adapted from vjp-steering 7f0782a `scripts/export.py` + `src/vjp_steering/results.py`, merged so one
file writes the one data artifact (`points.json`) that the plot, the tables and the React page read.
Jev's steering-failure checks alone determine admissibility (mean P(fail) <= judge.MAX_FAILURE); mechanical health
and boundary fields are diagnostics.
The random zone and plot style follow the reference. PI/OpenAI.
Changed: the judge is Jev (judge.py: BullshitBench's own rubric plus steering-failure checks, 1 call per answer)
instead of the DeepSeek pairwise judge, so numbers are in Jev units and not comparable with the
vjp-steering README. All steering-lite methods plus prompting points. The headline table picks, for each
side, the admissible dose with the best on-axis - OFF_WEIGHT x off-axis, scores the method by the weaker
side, and bootstraps seeds then questions (selection redone inside each resample).
"""

import argparse
import html
import json
import math
import random
from statistics import mean, median
from typing import Iterable
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import COHORTS, ROOT, default_model_dir, demo_rows, load_cohort, read_answers, walk_certificates
from judge import FAILURES, MAX_FAILURE, MODEL, blind_request, bsb_request, cached, key, p_fail


OFF_WEIGHT = 1.0  # wassname 2026-09-24: "in the hard direction it's 1:1. I guess we can use one to one" (was 4, a misreading of "1:4")
N_BOOT = 1000
BS_MAX = 2  # BullshitBench score: 0 accepted nonsense .. 2 clear pushback
COLORS = {
    "vjp_resid": "#0072b2", "mean_diff": "#d55e00", "pca": "#cc79a7", "vjp_value": "#009e73",
    "value_gram": "#e69f00", "prompting": "#6a3d9a", "prompting_engineered": "#b15928", "random": "#999999",
    "query_steer": "#f0e442", "sink_split": "#000000", "sink_split_resid": "#b8860b",
}
# the other steering-lite methods: Tableau-20 colours not used above
for _method, _color in zip(
    ("angular_steering", "chars", "corda_pca", "cosine_gated", "directional_ablation", "linear_act", "spherical",
     "sspace", "sspace_ablate", "sspace_scale", "sspace_pca", "sspace_pool", "topk_clusters"),
    ("#1f77b4", "#17becf", "#ff7f0e", "#2ca02c", "#98df8a", "#ff9896", "#d62728", "#c5b0d5", "#9467bd", "#8c564b", "#c49c94", "#e377c2", "#aec7e8"),
):
    COLORS[_method] = _color
TOP_N_PLOT = 5  # the PNG and the page's default view show the 5 best-scoring learned methods; the table lists all
LABELS = {
    "vjp_resid": "VJP-resid", "mean_diff": "mean difference", "pca": "PCA", "vjp_value": "VJP-value",
    "value_gram": "Value Gram", "prompting": "persona prompt", "prompting_engineered": "engineered prompt", "random": "random",
}
LABELS |= {method: method for method in COLORS if method not in LABELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="dev")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, help="default: outputs/bsbench/results/<cohort>")
    parser.add_argument("--exclude", default="", help="comma-separated methods to leave out (e.g. not yet judged); named in index.md")
    parser.add_argument("--view", choices=("benchmark", "user"), default="benchmark", help="opening plot: best methods or prompt sweeps with mean difference")
    return parser.parse_args()


def build_points(model_dir: Path, cohort: str, exclude: set[str], view: str = "benchmark") -> list[dict]:
    """One point per (method, seed, side, C), with per-question Jev ratings kept for bootstrap and the explorer.

    effect = bs(bare) - bs(steered): BullshitBench score lost, + = goes along with the nonsense (the plot's x axis;
    -C working is negative). off_axis = P(fail)(steered) - P(fail)(bare), P(fail) = the largest of the steering-failure
    checks (judge.FAILURES). A dose is coherent (admissible) while mean P(fail) of the steered answers <= MAX_FAILURE.
    Blind ratings are attached where judged (blind table doses)."""
    have = cached()
    points = []
    for certificate in [c for c in walk_certificates(model_dir, cohort, view) if c["method"] not in exclude]:
        rows = demo_rows(model_dir, certificate)
        for side, rungs in certificate["sides"].items():
            for rung in rungs:
                questions = []
                for row in (r for r in rows if r["side"] == side and r["C"] == rung["coefficient"]):
                    b = have.get(key(bsb_request(row["prompt"], row["flaw"], row["bare"])))
                    st = have.get(key(bsb_request(row["prompt"], row["flaw"], row["steered"])))
                    assert b and st, f"no Jev rating for {certificate['method']} s{certificate['seed']} {side} C={rung['coefficient']} {row['vignette']}; run judge.py --refresh"
                    questions.append({
                        "scenario": row["vignette"],
                        "effect": b["bs_score"]["score"] - st["bs_score"]["score"],
                        "off_axis": p_fail(st) - p_fail(b), "p_fail": p_fail(st),
                        "failures": {name: st[name]["probabilities"]["yes"] for name in FAILURES},
                        "bare_bs": b["bs_score"]["score"],
                        "evidence": f"BS score {b['bs_score']['score']:.2f} -> {st['bs_score']['score']:.2f}, P(fail) {p_fail(b):.2f} -> {p_fail(st):.2f}",
                        "blind": have.get(key(blind_request(row["prompt"], row["bare"], row["steered"]))),
                        "text": row["steered"],
                    })
                steered_fail = mean(q["p_fail"] for q in questions)
                points.append({
                    "method": certificate["method"], "seed": certificate["seed"], "C": rung["coefficient"], "side": side,
                    "axis": certificate["gen"]["axis"],
                    "effect": mean(q["effect"] for q in questions), "off_axis": mean(q["off_axis"] for q in questions),
                    "p_fail": steered_fail, "failures": {name: mean(q["failures"][name] for q in questions) for name in FAILURES},
                    "breakdown_reasons": rung["breakdown_reasons"], "post_boundary": rung.get("post_boundary", False),
                    "admissible": steered_fail <= MAX_FAILURE,
                    "kl_rms": rung.get("kl_rms"), "stats": rung["stats"], "answers": rung["answers"], "questions": questions,
                })
    return points


def directed(point: dict) -> float:
    """On-axis change toward the side's target: + = toward accepting the nonsense for +C, toward pushback for -C."""
    return point["effect"] if point["side"] == "+C" else -point["effect"]


def room(questions: list[dict], side: str) -> float:
    """Mean BS-score the bare answers leave toward the side's target: bare score for +C (can fall to 0), BS_MAX - bare for -C."""
    return mean(q["bare_bs"] if side == "+C" else BS_MAX - q["bare_bs"] for q in questions)


def side_best(points: list[dict], key) -> dict | None:
    live = [point for point in points if point["admissible"]]
    return max(live, key=key) if live else None


def shown_axis(points: list[dict]) -> list[str]:
    axes = {tuple(p["axis"]) for p in points}
    assert len(axes) == 1, f"one report mixes persona axes {axes}"
    return list(axes.pop())


def method_curve(points: list[dict], method: str, side: str, *, candidates: bool = False) -> list[dict]:
    """Seed-mean points at each C where every seed of this method is admissible (reference `_means`).
    candidates: every dose all seeds reached, admissible or not; the bootstrap re-decides admissibility per draw."""
    seeds = {point["seed"] for point in points if point["method"] == method}
    curve = []
    for C in sorted({point["C"] for point in points if point["method"] == method and point["side"] == side}):
        at = [point for point in points if point["method"] == method and point["side"] == side and point["C"] == C]
        if {point["seed"] for point in at if candidates or point["admissible"]} != seeds:
            continue
        questions = [q | {"seed": point["seed"]} for point in at for q in point["questions"]]
        curve.append({
            "method": method, "side": side, "C": C, "admissible": True,
            "effect": mean(point["effect"] for point in at), "off_axis": mean(point["off_axis"] for point in at),
            "room": room(questions, side), "questions": questions,
        })
    return curve


def pareto_score(side_curves: dict[str, list[dict]]) -> tuple[float, dict]:
    """min over sides of the best admissible on-axis - OFF_WEIGHT x off-axis."""
    best = {side: side_best(curve, lambda point: directed(point) - OFF_WEIGHT * point["off_axis"]) for side, curve in side_curves.items()}
    if any(point is None for point in best.values()):
        return float("nan"), best
    return min(directed(point) - OFF_WEIGHT * point["off_axis"] for point in best.values()), best


def room_score(best: dict) -> float:
    """On-axis ÷ room: on-axis at each side's Pareto-best point divided by that side's room, weaker side (nan if a side has none)."""
    if any(point is None for point in best.values()):
        return float("nan")
    return min(directed(point) / point["room"] for point in best.values())


def resample(curve: list[dict], scenarios: list[str], seeds: list[int]) -> list[dict]:
    """Seed-mean point per dose over the drawn seeds x drawn questions (both with repeats).

    The Jev failure limit (MAX_FAILURE) is re-decided in each draw on the drawn questions. Curves come in as candidates (all doses), so a
    dose that failed on the full set can pass in a draw and vice versa. Intervals remain conditional on seed coverage. — PI/OpenAI"""
    out = []
    for point in curve:
        by = {}
        for q in point["questions"]:
            by.setdefault((q["seed"], q["scenario"]), []).append(q)
        chosen = [q for seed in seeds for scenario in scenarios for q in by.get((seed, scenario), [])]
        if not chosen:  # random: a drawn seed may not have reached this dose
            continue
        if mean(q["p_fail"] for q in chosen) > MAX_FAILURE:
            continue
        out.append({**point, "effect": mean(q["effect"] for q in chosen), "off_axis": mean(q["off_axis"] for q in chosen), "room": room(chosen, point["side"])})
    return out


def bootstrap(side_curves: dict[str, list[dict]], scenarios: list[str], seeds: list[int], rng: random.Random) -> tuple[float, float, float, float, float]:
    """Hierarchical bootstrap: resample seeds (all walk seeds, including ones with no admissible dose),
    then questions; dose selection redone in each draw.

    Returns the 90% interval, the share of draws where a side had no admissible dose (score -inf,
    counted as the worst outcome rather than dropped), and the 90% interval of on-axis ÷ room from the same draws."""
    scores, room_scores = [], []
    for _ in range(N_BOOT):
        drawn_seeds = [rng.choice(seeds) for _ in seeds]
        drawn = [rng.choice(scenarios) for _ in scenarios]
        score, best = pareto_score({side: resample(curve, drawn, drawn_seeds) for side, curve in side_curves.items()})
        scores.append(-math.inf if math.isnan(score) else score)
        room_scores.append(-math.inf if math.isnan(score) else room_score(best))
    empty = sum(score == -math.inf for score in scores) / len(scores)
    scores.sort()
    room_scores.sort()
    low, high = (lambda values: values[int(0.05 * len(values))]), (lambda values: values[int(0.95 * len(values)) - 1])
    return low(scores), high(scores), empty, low(room_scores), high(room_scores)


def random_curves(points: list[dict], *, candidates: bool = False) -> dict[str, list[dict]]:
    """Random is scored like a method whose seeds are pooled at each C (reference `_summary`).
    The admissible population changes with C; report how many seeds a scored dose rests on (`seeds_at`)."""
    out = {}
    for side in ("+C", "-C"):
        group = [point for point in points if point["method"] == "random" and point["side"] == side]
        out[side] = []
        for C in sorted({point["C"] for point in group}):
            live = [point for point in group if point["C"] == C and (candidates or point["admissible"])]
            if live:
                questions = [q | {"seed": point["seed"]} for point in live for q in point["questions"]]
                out[side].append({
                    "C": C, "side": side, "admissible": True,
                    "effect": mean(point["effect"] for point in live), "off_axis": mean(point["off_axis"] for point in live),
                    "room": room(questions, side), "questions": questions, "seeds_at": len(live),
                })
    return out


def curves_for(points: list[dict], method: str, *, candidates: bool = False) -> dict[str, list[dict]]:
    return random_curves(points, candidates=candidates) if method == "random" else {side: method_curve(points, method, side, candidates=candidates) for side in ("+C", "-C")}


def choose(points: list[dict]) -> dict[str, tuple[dict, dict, dict]]:
    """Per method: (side curves, Pareto-best point per side, strongest admissible point per side)."""
    out = {}
    for method in sorted({point["method"] for point in points}, key=lambda m: (m == "random", m)):
        curves = curves_for(points, method)
        out[method] = (curves, pareto_score(curves)[1], {side: side_best(curve, directed) for side, curve in curves.items()})
    return out


def summary(points: list[dict], scenarios: list[str]) -> list[dict]:
    rng = random.Random(0)
    rows = []
    for method, (curves, best, strongest) in choose(points).items():
        score, _ = pareto_score(curves)
        group = [point for point in points if point["method"] == method]
        seeds = sorted({point["seed"] for point in group})
        low, high, empty, room_low, room_high = bootstrap(curves_for(points, method, candidates=True), scenarios, seeds, rng) if not math.isnan(score) else (float("nan"),) * 5
        rows.append({
            "method": method, "score": score, "ci": (low, high), "ci_empty": empty, "best": best, "strongest": strongest,
            "score_room": room_score(best), "ci_room": (room_low, room_high),
            "seeds": len({point["seed"] for point in group}),
            "N": sum(len(curve) for curve in curves.values()), "rejected": sum(not point["admissible"] for point in group),
        })
    return sorted(rows, key=lambda row: (math.isnan(row["score"]), -row["score"] if not math.isnan(row["score"]) else 0))


def blind_targets(model_dir: Path, cohort: str) -> dict[str, dict]:
    """Jev blind requests for the blind table: every seed's answers at each method-side's Pareto-best and strongest dose, in each report view."""
    out = {}
    for view in ("benchmark", "user"):
        out |= _blind_targets(model_dir, cohort, view)
    return out


def _blind_targets(model_dir: Path, cohort: str, view: str) -> dict[str, dict]:
    points = build_points(model_dir, cohort, set(), view)
    if not points:
        return {}
    rows = {(r["method"], r["seed"], r["C"], r["side"], r["vignette"]): r for c in walk_certificates(model_dir, cohort, view) for r in demo_rows(model_dir, c)}
    out = {}
    for method, (_, best, strongest) in choose(points).items():
        for side in ("+C", "-C"):
            for point in (best[side], strongest[side]):
                for q in (point or {}).get("questions", []):
                    row = rows[method, q["seed"], point["C"], side, q["scenario"]]
                    request = blind_request(row["prompt"], row["bare"], row["steered"])
                    out[key(request)] = request
    return out


def place_labels(
    points: list[dict],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    obstacles: Iterable[tuple[float, float]] = (),
    fig_w: int = 1240,
    fig_h: int = 640,
    margin: dict | None = None,
    char_w: float = 6.0,
    line_h: float = 15.0,
    radii: tuple = (40, 62, 88, 118),
    angles: tuple = (90, 45, 135, 0, 180, -45, -135, -90),
    font: dict | None = None,
    bgcolor: str = "rgba(253,250,244,0.72)",
    arrowcolor: str = "rgba(45,24,16,0.35)",
    overlap_cost: float = 50.0,
    overlap_cost_label: float = 1.0 / 50.0,
    pad: float = 7.0,
    edge_pad: float = 4.0,
) -> list[dict]:
    margin = margin or {"l": 90, "r": 90, "t": 70, "b": 70}
    plot_width = fig_w - margin["l"] - margin["r"]
    plot_height = fig_h - margin["t"] - margin["b"]
    (x0, x1), (y0, y1) = x_range, y_range

    def to_pixels(x: float, y: float) -> tuple[float, float]:
        x_pixel = margin["l"] + (x - x0) / (x1 - x0) * plot_width
        y_pixel = margin["t"] + (1 - (y - y0) / (y1 - y0)) * plot_height
        return x_pixel, y_pixel

    anchors = [to_pixels(point["x"], point["y"]) for point in points]
    obstacle_pixels = [to_pixels(x, y) for x, y in obstacles]
    placed = []
    annotations = []

    def cost(center_x: float, center_y: float, box_width: float, box_height: float) -> float:
        left = center_x - box_width / 2
        right = center_x + box_width / 2
        top = center_y - box_height / 2
        bottom = center_y + box_height / 2
        candidate_cost = 0.0
        candidate_cost += 1000 * (max(0.0, margin["l"] + edge_pad - left) + max(0.0, right - (fig_w - margin["r"] - edge_pad)))
        candidate_cost += 1000 * (max(0.0, margin["t"] + 24 - top) + max(0.0, bottom - (fig_h - margin["b"] - edge_pad)))
        for point_x, point_y in obstacle_pixels:
            if left - pad <= point_x <= right + pad and top - pad <= point_y <= bottom + pad:
                candidate_cost += overlap_cost
        for placed_x, placed_y, placed_width, placed_height in placed:
            overlap_x = max(0.0, min(right, placed_x + placed_width / 2) - max(left, placed_x - placed_width / 2))
            overlap_y = max(0.0, min(bottom, placed_y + placed_height / 2) - max(top, placed_y - placed_height / 2))
            candidate_cost += overlap_x * overlap_y * overlap_cost_label
        return candidate_cost

    font = font or {"size": 11}
    for point, (anchor_x, anchor_y) in zip(points, anchors, strict=True):
        lines = point["text"].split("<br>")
        box_width = max(len(line) for line in lines) * char_w + 10
        box_height = len(lines) * line_h + 6
        best = None
        for radius in radii:
            for angle in point.get("angles", angles):
                center_x = anchor_x + radius * math.cos(math.radians(angle))
                center_y = anchor_y - radius * math.sin(math.radians(angle))
                candidate = (cost(center_x, center_y, box_width, box_height), center_x, center_y)
                if best is None or candidate[0] < best[0]:
                    best = candidate
                if candidate[0] == 0:
                    break
            if best[0] == 0:
                break
        _, center_x, center_y = best
        placed.append((center_x, center_y, box_width, box_height))
        annotations.append({
            "x": point["x"], "y": point["y"], "text": point["text"], "showarrow": True,
            "ax": center_x - anchor_x, "ay": center_y - anchor_y, "axref": "pixel", "ayref": "pixel",
            "font": {**font, "color": point["color"]}, "align": "center", "bgcolor": bgcolor,
            "arrowhead": 0, "arrowwidth": 1, "arrowcolor": arrowcolor,
        })
    return annotations




def random_zones(points: list[dict]) -> list[dict]:
    """Pooled-sign percentiles of random effects per dose, at the median off-axis; Chaikin-smoothed. PI/OpenAI.
    Each (direction, sign) walks its own doses; a dose counts once at least half of the 2 x directions reached it."""
    random_points = [point for point in points if point["method"] == "random"]
    seeds = sorted({point["seed"] for point in random_points})
    at = {(point["seed"], point["C"], point["side"]): point for point in random_points}
    zones = [{"percentile": p, "opacity": alpha, "bounds": [(0.0, 0.0, 0.0, 0.0)], "doses": [None], "seed_counts": [0], "negative_counts": [0], "positive_counts": [0], "mean_effect": [0.0]}
             for p, alpha in ((90, .16), (75, .22), (50, .30))]
    for C in sorted({point["C"] for point in random_points}):
        reached = [(seed, side) for seed in seeds for side in ("+C", "-C") if (seed, C, side) in at]
        if len(reached) < max(1, len(seeds)):
            continue  # each side starts at its own C0/8, so the lowest doses are sampled by only some walks
        chosen = [at[seed, C, side] for seed, side in reached if at[seed, C, side]["admissible"]]
        if len(chosen) < max(1, len(seeds)):
            break
        coherent = chosen
        effects = sorted(point["effect"] for point in chosen)
        center = median(effects)
        damage = median(point["off_axis"] for point in chosen)
        for zone in zones:
            tail = len(effects) * (100 - zone["percentile"]) // 100
            lo, hi = (center, center) if zone["percentile"] == 50 else (effects[tail], effects[-tail - 1])
            zone["bounds"].append((center, damage, lo, hi))
            zone["doses"].append(C)
            zone["seed_counts"].append(len(coherent))
            zone["negative_counts"].append(sum(effect < 0 for effect in effects))
            zone["positive_counts"].append(sum(effect > 0 for effect in effects))
            zone["mean_effect"].append(mean(effects))
    for zone in zones:
        edge = [(row[1], row[2], row[3]) for row in sorted(zone["bounds"], key=lambda row: row[1])]
        for _ in range(3):
            edge = [edge[0]] + [tuple(w * a + (1 - w) * b for a, b in zip(left, right, strict=True))
                               for left, right in zip(edge, edge[1:]) for w in (.75, .25)] + [edge[-1]]
        zone["path"] = [[lo, damage] for damage, lo, hi in edge] + [[hi, damage] for damage, lo, hi in reversed(edge)]
    return zones


PROMPTS = {"prompting": "prompt", "prompting_engineered": "eng. prompt"}  # single points, not walks


REVERSAL = 0.125  # BS-score points (was 0.5 of the old 0-8 scale); about the noise floor between near-identical doses


def before_reversal(curve: list[dict]) -> list[dict]:
    """Display-only stop until eval v2 judges on-target answers at every dose (TODO.md): once a sweep has moved
    REVERSAL toward its side, end it before the first passing dose on the wrong side of bare. wassname 2026-10-03:
    "can't you filter for now and put a TODO it wont be needed?" (corda_pca +C user turn: +1.37 at C=20, -2.72 at C=64,
    answers drifting to questions not asked). PI/OpenAI"""
    out, reached = [], False
    for p in sorted(curve, key=lambda p: p["C"]):
        if reached and directed(p) < 0:
            break
        reached |= directed(p) >= REVERSAL
        out.append(p)
    return out


def sweep(curve: list[dict]) -> list[dict]:
    """Passing doses in dose order, median-filtered over neighbouring doses as in the reference
    (docs/vendor/vjp-steering/src/vjp_steering/results.py::plot): log-C half-window 0.15, else the 5 nearest
    rungs; then at most 16 points kept. Returns C, smoothed effect/off_axis, and the raw values. PI/OpenAI."""
    points = sorted(curve, key=lambda p: p["C"])
    out = [{"C": p["C"], "effect": p["effect"], "off_axis": p["off_axis"], "raw_effect": p["effect"], "raw_off_axis": p["off_axis"]} for p in points]
    if len(points) >= 5:
        log_c = [math.log(p["C"]) if p["C"] > 0 else -math.inf for p in points]
        for i, row in enumerate(out):
            window = [points[j] for j, lc in enumerate(log_c) if abs(lc - log_c[i]) <= 0.15]
            if len(window) < 3:
                window = points[max(0, i - 2):i + 3]
            row["effect"] = median(p["effect"] for p in window)
            row["off_axis"] = median(p["off_axis"] for p in window)
    if len(out) > 16:
        out = [out[i] for i in sorted({round(i * (len(out) - 1) / 15) for i in range(16)} | {len(out) - 1})]
    return out


def sweep_path(rows: list[dict], n: int = 12) -> list[list[float]]:
    """Catmull-Rom spline from bare through the sweep in dose order."""
    knots = [(0.0, 0.0)] + [(r["effect"], r["off_axis"]) for r in rows]
    if len(knots) < 2:
        return [list(knots[0])]
    padded = [knots[0], *knots, knots[-1]]
    path = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1:i + 3]
        for k in range(n):
            t = k / n
            path.append([0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t**2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t**3) for d in (0, 1)])
    path.append(list(knots[-1]))
    return path


def plot(points: list[dict], title: str, methods: list[str], best: dict) -> go.Figure:
    figure = go.Figure()
    curves = {(method, side): method_curve(points, method, side) for method in methods for side in ("+C", "-C")}
    prompting = [point for point in points if point["method"] in PROMPTS]  # baselines stay visible; open star = fails the judge's limits
    random_live = [point for point in points if point["method"] == "random" and point["admissible"]]
    shown = [point for curve in curves.values() for point in curve] + random_live + prompting
    x_limit = 1.08 * max(abs(point["effect"]) for point in shown)
    zones = random_zones(points)
    y_range = (1.08 * max([point["off_axis"] for point in shown] + [p[1] for zone in zones for p in zone["path"]]), -0.07)
    margin = {"l": 75, "r": 10, "t": 40, "b": 150}
    for zone in zones:
        median_line = zone["percentile"] == 50  # lo = hi: a line, not an area
        half = zone["path"][:len(zone["path"]) // 2]
        figure.add_trace(go.Scatter(
            x=[p[0] for p in (half if median_line else zone["path"])], y=[p[1] for p in (half if median_line else zone["path"])],
            name=f"random p{zone['percentile']}", mode="lines", fill=None if median_line else "toself",
            fillcolor=f"rgba(150,150,150,{zone['opacity']})", line={"width": 1.5 if median_line else 0, "color": "rgba(120,120,120,0.8)"},
            hoverinfo="skip", showlegend=False,
        ))
    obstacles = [(0.0, 0.0)]
    labels = []
    for (method, side), curve in curves.items():
        if not curve:
            continue
        rows = sweep(before_reversal(curve))
        path = sweep_path(rows)
        dash = "solid" if side == "+C" else "dash"
        figure.add_trace(go.Scatter(
            x=[q[0] for q in path], y=[q[1] for q in path], mode="lines",
            line={"color": COLORS[method], "width": 3, "dash": dash}, hoverinfo="skip", showlegend=False,
        ))
        figure.add_trace(go.Scatter(
            x=[r["effect"] for r in rows], y=[r["off_axis"] for r in rows], mode="markers", name="sweep",
            marker={"color": COLORS[method], "size": [7] * (len(rows) - 1) + [13], "symbol": ["circle"] * (len(rows) - 1) + ["x"]},
            text=[f"{side} C={r['C']:.3g}" for r in rows],
            hovertemplate=f"{LABELS[method]}<br>%{{text}}<br>effect=%{{x:.3f}}<br>off-axis=%{{y:.3f}}<extra></extra>", showlegend=False,
        ))
        obstacles.extend((q[0], q[1]) for q in path[::4])
        obstacles.extend((r["effect"], r["off_axis"]) for r in rows)
        labels.append({"x": path[-1][0], "y": path[-1][1], "text": f"{LABELS[method]} {side}", "color": COLORS[method]})
    for point in prompting:
        figure.add_trace(go.Scatter(
            x=[point["effect"]], y=[point["off_axis"]], mode="markers",
            marker={"color": COLORS[point["method"]], "size": 13, "symbol": "star" if point["admissible"] else "star-open", "line": {"width": 2, "color": COLORS[point["method"]]}},
            hoverinfo="skip", showlegend=False,
        ))
        obstacles.append((point["effect"], point["off_axis"]))
        labels.append({"x": point["effect"], "y": point["off_axis"], "text": f"{PROMPTS[point['method']]} {point['side']}" + ("" if point["admissible"] else " (incoherent)"), "color": COLORS[point["method"]]})
    figure.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker={"color": "#333333", "size": 11, "symbol": "diamond"}, hoverinfo="skip", showlegend=False))
    figure.add_annotation(x=0, y=0, text="bare", showarrow=False, xshift=28, yshift=12, font={"color": "#333333", "size": 14})
    for annotation in place_labels(
        labels, (-x_limit, x_limit), y_range, obstacles=obstacles, fig_w=1064, fig_h=590, margin=margin,
        font={"size": 11}, char_w=6.5, line_h=15, overlap_cost_label=.2, radii=(40, 62, 88, 118, 160, 205),
        bgcolor="rgba(255,255,255,0.9)", arrowcolor="rgba(45,24,16,0.6)",
    ):
        figure.add_annotation(**annotation)
    figure.add_annotation(x=0, y=1, xref="paper", yref="paper", text=f"clean steer -> {shown_axis(points)[1]}", showarrow=False, xanchor="left", font={"color": "#287a4d", "size": 14})
    figure.add_annotation(x=1, y=1, xref="paper", yref="paper", text=f"clean steer -> {shown_axis(points)[0]}", showarrow=False, xanchor="right", font={"color": "#287a4d", "size": 14})
    figure.add_annotation(x=0, y=-0.18, xref="paper", yref="paper", xanchor="left", yanchor="top", align="left", showarrow=False,
                          font={"color": "#555555", "size": 12},
                          text=f"line = one method's dose sweep from bare, smoothed over neighbouring doses; dot = dose; ★ = plain prompt (open ☆ = judge rates it incoherent)<br>× = last dose the judge rates coherent, or before the effect reverses past bare<br>grey = {len({p['seed'] for p in points if p['method'] == 'random'})} random directions at the same doses, both signs: outer band 10–90% of their effects, inner 25–75%, line = median<br>bands use observed ranks, so with few directions they span min–max; not confidence intervals")
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"}, height=590, margin=margin,
        font={"color": "#111", "size": 15}, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        xaxis={"title": "BullshitBench score lost (Jev, 0–2 scale): ← pushes back on the nonsense · goes along with it → (solid +C, dashed -C)", "range": [-x_limit, x_limit], "showline": True, "linecolor": "#333333", "gridcolor": "#e5e5e5", "zeroline": False},
        yaxis={"title": "off-axis: rise in P(steering failure) (lower is better)", "range": y_range, "showline": True, "linecolor": "#333333", "gridcolor": "#e5e5e5", "zeroline": False},
    )
    return figure


def _fmt_side(point: dict | None) -> list[str]:
    if point is None:
        return ["—", "—", "—"]
    return [f"{directed(point):+.2f}", f"{point['off_axis']:.2f}", f"{point['C']:.3g}"]


def tables(rows: list[dict]) -> str:
    head = "| method | score↑ | 90% CI | on-axis ÷ room↑ | 90% CI | no-dose draws | −C on↑ | −C off↓ | −C C | +C on↑ | +C off↓ | +C C | seeds | N | rejected↓ |"
    lines = [head, "|" + "---|" * 15]
    bound = lambda v: "−∞" if v == -math.inf else f"{v:+.2f}"
    for row in rows:
        name = f"*{row['method']}*" if row["method"] in ("random", *PROMPTS) else row["method"]
        score = "—" if math.isnan(row["score"]) else f"{row['score']:+.2f}"
        ci = "—" if math.isnan(row["ci"][0]) else f"[{bound(row['ci'][0])}, {bound(row['ci'][1])}]"
        score_room = "—" if math.isnan(row["score_room"]) else f"{row['score_room']:+.2f}"
        ci_room = "—" if math.isnan(row["ci_room"][0]) else f"[{bound(row['ci_room'][0])}, {bound(row['ci_room'][1])}]"
        empty = "—" if math.isnan(row["ci_empty"]) else f"{row['ci_empty']:.0%}"
        lines.append("| " + " | ".join([name, score, ci, score_room, ci_room, empty, *_fmt_side(row["best"]["-C"]), *_fmt_side(row["best"]["+C"]),
                                         str(row["seeds"]), str(row["N"]), str(row["rejected"])]) + " |")
    return "\n".join(lines) + (
        f"\n\nOn-axis ÷ room: on-axis change at the Pareto-best dose divided by how far the bare answers could still move toward that side "
        f"(bare BS score for +C, {BS_MAX} − bare BS score for −C), weaker side; failures are handled by the dose choice and the {MAX_FAILURE:g} limit, not in this number.")


def _no_nan(value):
    """JSON has no NaN; an unscored method (no admissible dose on a side) is null."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_nan(v) for v in value]
    return value


INTENDED = {"+C": "accepts_premise", "-C": "rejects_premise"}




def stance(choice: dict) -> float:
    """Blind stance value in [-1, 1]: P(accepts) - P(rejects) from a Jev stance choice."""
    return choice["probabilities"]["accepts"] - choice["probabilities"]["rejects"]


def blind_summary(point: dict, side: str) -> dict:
    """Blind stance shift toward the side's target (+ = intended) and the mean probability of every change label.

    Mean probability, not the share of top labels: a 0.51 verbose / 0.25 sycophantic answer counts for both."""
    judged = [q["blind"] for q in point["questions"] if q["blind"]]
    assert len(judged) == len(point["questions"]), f"blind ratings for {len(judged)}/{len(point['questions'])} answers at C={point['C']}; run judge.py --refresh"
    sign = 1 if side == "+C" else -1
    labels = judged[0]["concept"]["probabilities"].keys()
    prob = {label: mean(j["concept"]["probabilities"][label] for j in judged) for label in labels}
    return {"C": point["C"], "n": len(judged), "shift": sign * mean(stance(j["stance_B"]) - stance(j["stance_A"]) for j in judged),
            "intended": prob[INTENDED[side]], "labels": dict(sorted(prob.items(), key=lambda item: -item[1]))}


def blind_cell(point: dict | None, side: str) -> str:
    if point is None:
        return "— | — | —"
    b = blind_summary(point, side)
    return f"{b['C']:.3g}: {b['shift']:+.2f} (n={b['n']}) | {b['intended']:.0%} | " + ", ".join(f"{label} {p:.0%}" for label, p in list(b["labels"].items())[:3])


def blind_table(rows: list[dict]) -> str:
    """Blind judge at each method's Pareto-best dose and at its strongest admissible dose."""
    lines = [
        "| method | side | Pareto-best C: blind stance shift↑ | P(intended label) | top labels (mean P) | strongest C: blind stance shift↑ | P(intended label) | top labels (mean P) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        for side in ("-C", "+C"):
            lines.append(f"| {row['method']} | {side} | {blind_cell(row['best'][side], side)} | {blind_cell(row['strongest'][side], side)} |")
    return "\n".join(lines)


def svg_labels(site: dict) -> list[dict]:
    """Reuse PNG label placement at the browser's default-view dimensions. PI/OpenAI."""
    curves = [c for c in site["curves"] if c["method"] in site["shown"] and c["points"]]
    prompts = [p for p in site["points"] if p["method"] in PROMPTS]
    shown = [p for c in curves for p in c["tested"]] + prompts
    zones = [p for zone in site["zones"] for p in zone["path"]]
    x_max = 1.08 * max([abs(p["effect"]) for p in shown] + [abs(p[0]) for p in zones] + [.5])
    y_max = 1.08 * max([p["off_axis"] for p in shown] + [p[1] for p in zones] + [.3])
    labels = [{"method": c["method"], "side": c["side"], "x": c["path"][-1][0], "y": c["path"][-1][1],
               "text": f"{LABELS[c['method']]} {c['side']}", "color": site["colors"][c["method"]]} for c in curves]
    labels += [{"method": p["method"], "side": p["side"], "x": p["effect"], "y": p["off_axis"],
                "text": f"{PROMPTS[p['method']]} {p['side']}" + ("" if p["admissible"] else " (incoherent)"), "color": site["colors"][p["method"]]} for p in prompts]
    obstacles = [(0., 0.)] + [(p["effect"], p["off_axis"]) for p in shown]
    placed = place_labels(labels, (-x_max, x_max), (y_max, -.05), obstacles=obstacles,
                          fig_w=1000, fig_h=560, margin={"l": 70, "r": 20, "t": 30, "b": 50},
                          radii=(40, 62, 88, 118, 160), char_w=6.5, overlap_cost_label=.2)
    return [annotation | {"method": label["method"], "side": label["side"]}
            for label, annotation in zip(labels, placed, strict=True)]


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir or default_model_dir()
    out = args.out or ROOT / "outputs/bsbench/results" / args.cohort
    out.mkdir(parents=True, exist_ok=True)
    exclude = {m for m in args.exclude.split(",") if m}
    points = build_points(model_dir, args.cohort, exclude, "user" if args.view == "user" else "benchmark")
    # methods without a fixed colour (e.g. tagged variants like vjp_resid-t48) take the next spare colour, in name order
    spare = [c for c in ("#56b4e9", "#000000", "#b8860b", "#8b008b", "#2f4f4f", "#ff1493", "#556b2f") if c not in COLORS.values()]
    uncoloured = sorted({point["method"] for point in points} - set(COLORS))
    if len(uncoloured) > len(spare):
        raise ValueError(f"{len(uncoloured)} methods have no fixed colour but only {len(spare)} spare colours: {uncoloured}; add colours to COLORS or --exclude some")
    for method in uncoloured:
        COLORS[method] = spare.pop(0)
        LABELS[method] = method
    scenarios = list(load_cohort())[COHORTS[args.cohort]]
    rows = summary(points, scenarios)
    cohort_rows = load_cohort()
    bare = read_answers(model_dir / "answers/bare/bare.jsonl")
    methods = sorted({point["method"] for point in points} - {"random", *PROMPTS})
    # tagged variants (<method>-<tag>, e.g. vjp_resid-t47) are diagnostics: in the table, not in the default plot view
    shown = [row["method"] for row in rows if row["method"] in methods and "-" not in row["method"] and not math.isnan(row["score"])][:TOP_N_PLOT]
    site = {
        "view": args.view, "shown": shown,
        "colors": COLORS,  # the page's only colour source
        "model_dir": model_dir.name, "cohort": args.cohort, "judge": f"{MODEL} (BullshitBench score 0-2, steering-failure checks)", "off_weight": OFF_WEIGHT,
        "max_failure": MAX_FAILURE, "admissibility": "jev_mean_p_fail", "failures": list(FAILURES),
        "questions": [{"scenario": s, "prompt": cohort_rows[s]["prompt"], "flaw": cohort_rows[s]["nonsensical_element"], "bare": bare[s]["text"]} for s in scenarios],
        "zones": random_zones(points),
        "random_seeds": sorted({p["seed"] for p in points if p["method"] == "random"}),
        "curves": [{
            "method": m, "side": side,
            "points": sweep(before_reversal(method_curve(points, m, side))),
            "tested": [{k: p[k] for k in ("C", "effect", "off_axis")} for p in method_curve(points, m, side)],
            "path": sweep_path(sweep(before_reversal(method_curve(points, m, side)))) if method_curve(points, m, side) else [],
        } for m in methods for side in ("+C", "-C")],
        "summary": [{
            "method": row["method"], "score": row["score"], "ci": row["ci"], "score_room": row["score_room"], "ci_room": row["ci_room"],
            "seeds": row["seeds"], "N": row["N"], "rejected": row["rejected"],
            "best": {side: None if p is None else {k: p[k] for k in ("C", "effect", "off_axis")} for side, p in row["best"].items()},
        } for row in rows],
        "blind": [{"method": row["method"], "side": side, "dose": dose, **blind_summary(row[dose][side], side)}
                  for row in rows for side in ("-C", "+C") for dose in ("best", "strongest") if row[dose][side] is not None],
        "points": points,
    }
    site["plot_labels"] = svg_labels(site)
    (out / f"points.json").write_text(json.dumps(_no_nan(site), indent=1, allow_nan=False) + "\n")
    model = model_dir.name.rsplit("-g", 1)[0].split("--")[-1]
    heading = {"user": "User-turn steering on Bullshit Bench v2"}.get(args.view, "steering-lite on Bullshit Bench v2")
    title = f"{heading}: {model} ({args.cohort}, {len(scenarios)} questions) — judge: Jev"
    best = {(row["method"], side): p for row in rows for side, p in row["best"].items()}
    figure = plot(points, title, shown, best)
    table = tables(rows) + (
        "\n\nBlind judge (Jev, not told the target, method, dose or known flaw). Blind stance shift = mean over questions of "
        "stance(steered) - stance(bare), stance = P(accepts) - P(rejects), signed so + is toward the side's target "
        "(+C accept the premise, -C reject it). Intended label: accepts_premise for +C, rejects_premise for −C; P(intended label) is its mean probability over the answers at that dose.\n\n"
    ) + blind_table(rows)
    intro = (
        f"Score = min over ±C of (on-axis − {OFF_WEIGHT:g} × off-axis) at each side's best admissible dose. "
        f"CI: {N_BOOT} hierarchical bootstrap draws (seeds with replacement, then questions with replacement), dose selection redone in each; "
        "draws where a side has no admissible dose count as −∞ (share in 'no-dose draws'). "
        f"Judge: Jev with BullshitBench's own rubric (on-axis = BS score lost, 0-2 scale, + = goes along with the nonsense) and five yes/no steering-failure checks "
        f"({', '.join(FAILURES)}; off-axis = change in P(any failure), the largest of the five). Admissible = mean P(any failure) of the steered answers ≤ {MAX_FAILURE:g}. "
        "Each side has its own calibrated doses. Mechanical health and walk boundaries are calibration diagnostics, not coherence filters."
    ) + (f" Left out (not yet judged): {', '.join(sorted(exclude))}." if exclude else "")
    if args.view == "user":
        intro += (" User-turn view: each vector is added only at the user-message tokens of the prompt (not the chat template, not the answer tokens), "
                  "using the method's own vector and C0, one seed. Random is random-user: random directions steered the same way. "
                  "Plain prompts also act only on the prompt. Compare with the steering-everywhere report for the same model.")
    (out / f"index.md").write_text(f"# Results ({args.cohort})\n\n{intro}\n\n![plot](plot.png)\n\n{table}\n")
    figure_html = figure.to_html(full_html=False, include_plotlyjs="cdn", default_width="100%", config={"responsive": True})
    (out / f"plot.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>steering-lite bsbench</title>"
        "<style>body{font:16px system-ui;max-width:1064px;margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap}</style>"
        f"<h1>Results ({html.escape(args.cohort)})</h1><p>{html.escape(intro)}</p>{figure_html}<pre>{html.escape(table)}</pre>"
    )
    figure.write_image(out / f"plot.png", width=1064, height=590, scale=2)
    # marker count drawn in the PNG, compared with the React page by web/uat.py
    sweep_marks = sum(len(trace.x) for trace in figure.data if trace.name == "sweep")
    (out / f"plot_marks.json").write_text(json.dumps({"sweep_marks": sweep_marks, "methods": shown}) + "\n")
    print(table)
    print(f"wrote {out}/points.json ({len(points)} points), index.md, plot.html, plot.png")


if __name__ == "__main__":
    main()
