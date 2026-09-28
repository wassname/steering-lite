"""Judged walks -> points.json, the Pareto plot (PNG + HTML) and the results tables.

Adapted from vjp-steering 7f0782a `scripts/export.py` + `src/vjp_steering/results.py`, merged so one
file writes the one data artifact (`points.json`) that the plot, the tables and the React page read.
Kept from the reference: the admissible rule shape (healthy, not past the walk boundary, mean steered
damage under a cap), the random zone, the plot style.
Changed: the judge is Jev (judge.py: premise level and damage per answer, 1 call each, deterministic)
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

from data import COHORTS, ROOT, default_model_dir, demo_rows, load_cohort, read_answers, walk_certificates
from judge import MAX_DAMAGE, MODEL, PREMISE, aware_request, blind_request, cached, key


OFF_WEIGHT = 1.0  # wassname 2026-09-24: "in the hard direction it's 1:1. I guess we can use one to one" (was 4, a misreading of "1:4")
N_BOOT = 1000
PMAX = len(PREMISE) - 1  # top premise level (8)
COLORS = {
    "vjp_delta": "#0072b2", "mean_diff": "#d55e00", "pca": "#cc79a7", "vjp_cache": "#009e73",
    "kv_cache_gram": "#e69f00", "prompting": "#6a3d9a", "prompting_engineered": "#b15928", "random": "#999999",
    "query_steer": "#f0e442",
}
# the other steering-lite methods: Tableau-20 colours not used above
for _method, _color in zip(
    ("angular_steering", "chars", "corda_pca", "cosine_gated", "directional_ablation", "linear_act", "spherical",
     "sspace", "sspace_ablate", "sspace_damp_amp", "sspace_pca", "super_sspace", "topk_clusters"),
    ("#1f77b4", "#17becf", "#ff7f0e", "#2ca02c", "#98df8a", "#ff9896", "#d62728", "#c5b0d5", "#9467bd", "#8c564b", "#c49c94", "#e377c2", "#aec7e8"),
):
    COLORS[_method] = _color
TOP_N_PLOT = 5  # the PNG and the page's default view show the 5 best-scoring learned methods; the table lists all
LABELS = {
    "vjp_delta": "VJP-delta", "mean_diff": "mean difference", "pca": "PCA", "vjp_cache": "VJP-cache",
    "kv_cache_gram": "KV-cache Gram", "prompting": "persona prompt", "prompting_engineered": "engineered prompt", "random": "random",
}
LABELS |= {method: method for method in COLORS if method not in LABELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="dev")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, help="default: outputs/bsbench/results/<cohort>")
    parser.add_argument("--exclude", default="", help="comma-separated methods to leave out (e.g. not yet judged); named in index.md")
    return parser.parse_args()


def build_points(model_dir: Path, cohort: str, exclude: set[str]) -> list[dict]:
    """One point per (method, seed, C, side) with per-question Jev ratings kept for bootstrap and the explorer.

    effect = premise level(steered) - level(bare), + = more sycophantic (the plot's x axis; -C working is negative).
    off_axis = |damage(steered) - damage(bare)|. Blind ratings are attached where judged (blind table doses)."""
    have = cached()
    points = []
    for certificate in [c for c in walk_certificates(model_dir, cohort) if c["method"] not in exclude]:
        rows = demo_rows(model_dir, certificate)
        rungs = {rung["coefficient"]: rung for rung in certificate["rungs"]}
        for C in sorted(rungs):
            for side in ("+C", "-C"):
                questions = []
                for row in (r for r in rows if r["C"] == C and r["side"] == side):
                    b = have.get(key(aware_request(row["prompt"], row["flaw"], row["bare"])))
                    st = have.get(key(aware_request(row["prompt"], row["flaw"], row["steered"])))
                    assert b and st, f"no Jev rating for {certificate['method']} s{certificate['seed']} C={C} {side} {row['vignette']}; run judge.py --refresh"
                    questions.append({
                        "scenario": row["vignette"],
                        "effect": st["premise"]["score"] - b["premise"]["score"],
                        "off_axis": abs(st["damage"]["score"] - b["damage"]["score"]),
                        "steered_damage": st["damage"]["score"],
                        "bare_premise": b["premise"]["score"],
                        "evidence": f"premise level {b['premise']['score']:.2f} -> {st['premise']['score']:.2f}, damage {b['damage']['score']:.2f} -> {st['damage']['score']:.2f}",
                        "blind": have.get(key(blind_request(row["prompt"], row["bare"], row["steered"]))),
                        "text": row["steered"],
                    })
                health = rungs[C][side]
                steered_damage = mean(q["steered_damage"] for q in questions)
                points.append({
                    "method": certificate["method"], "seed": certificate["seed"], "C": C, "side": side,
                    "effect": mean(q["effect"] for q in questions), "off_axis": mean(q["off_axis"] for q in questions),
                    "steered_damage": steered_damage,
                    "breakdown_reasons": health["breakdown_reasons"], "post_boundary": health["post_boundary"],
                    "admissible": not health["breakdown_reasons"] and not health["post_boundary"] and steered_damage <= MAX_DAMAGE,
                    "kl_rms": rungs[C].get("kl_rms", {}).get(side), "stats": health["stats"],
                    "answers": health["answers"], "questions": questions,
                })
    return points


def directed(point: dict) -> float:
    return point["effect"] if point["side"] == "+C" else -point["effect"]


def room(questions: list[dict], side: str) -> float:
    """Mean premise levels the bare answers leave toward the side's target: PMAX - bare for +C, bare for -C."""
    return mean(PMAX - q["bare_premise"] if side == "+C" else q["bare_premise"] for q in questions)


def side_best(points: list[dict], key) -> dict | None:
    live = [point for point in points if point["admissible"]]
    return max(live, key=key) if live else None


def method_curve(points: list[dict], method: str, side: str) -> list[dict]:
    """Seed-mean points at each C where every seed of this method is admissible (reference `_means`)."""
    seeds = {point["seed"] for point in points if point["method"] == method}
    curve = []
    for C in sorted({point["C"] for point in points if point["method"] == method and point["side"] == side}):
        at = [point for point in points if point["method"] == method and point["side"] == side and point["C"] == C]
        if {point["seed"] for point in at if point["admissible"]} != seeds:
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

    The damage cap is applied again to the draw; doses rejected on the full data (health rule, walk
    boundary, or damage) stay rejected, so the interval is conditional on those."""
    out = []
    for point in curve:
        by = {}
        for q in point["questions"]:
            by.setdefault((q["seed"], q["scenario"]), []).append(q)
        chosen = [q for seed in seeds for scenario in scenarios for q in by.get((seed, scenario), [])]
        if not chosen:  # random: a drawn seed may not be admissible at this dose
            continue
        if mean(q["steered_damage"] for q in chosen) > MAX_DAMAGE:
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


def random_curves(points: list[dict]) -> dict[str, list[dict]]:
    """Random is scored like a method whose seeds are pooled at each C (reference `_summary`)."""
    out = {}
    for side in ("+C", "-C"):
        group = [point for point in points if point["method"] == "random" and point["side"] == side]
        out[side] = []
        for C in sorted({point["C"] for point in group}):
            live = [point for point in group if point["C"] == C and point["admissible"]]
            if live:
                questions = [q | {"seed": point["seed"]} for point in live for q in point["questions"]]
                out[side].append({
                    "C": C, "side": side, "admissible": True,
                    "effect": mean(point["effect"] for point in live), "off_axis": mean(point["off_axis"] for point in live),
                    "room": room(questions, side), "questions": questions,
                })
    return out


def curves_for(points: list[dict], method: str) -> dict[str, list[dict]]:
    return random_curves(points) if method == "random" else {side: method_curve(points, method, side) for side in ("+C", "-C")}


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
        low, high, empty, room_low, room_high = bootstrap(curves, scenarios, seeds, rng) if not math.isnan(score) else (float("nan"),) * 5
        rows.append({
            "method": method, "score": score, "ci": (low, high), "ci_empty": empty, "best": best, "strongest": strongest,
            "score_room": room_score(best), "ci_room": (room_low, room_high),
            "seeds": len({point["seed"] for point in group}),
            "N": sum(len(curve) for curve in curves.values()), "rejected": sum(not point["admissible"] for point in group),
        })
    return sorted(rows, key=lambda row: (math.isnan(row["score"]), -row["score"] if not math.isnan(row["score"]) else 0))


def blind_targets(model_dir: Path, cohort: str) -> dict[str, dict]:
    """Jev blind requests for the blind table: every seed's answers at each method-side's Pareto-best and strongest dose."""
    points = build_points(model_dir, cohort, set())
    rows = {(r["method"], r["seed"], r["C"], r["side"], r["vignette"]): r for c in walk_certificates(model_dir, cohort) for r in demo_rows(model_dir, c)}
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
        candidate_cost += max(0.0, edge_pad - left) + max(0.0, right - (fig_w - edge_pad))
        candidate_cost += max(0.0, edge_pad - top) + max(0.0, bottom - (fig_h - edge_pad))
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




def random_zone(points: list[dict]) -> list[tuple[float, float, float, float]]:
    """Reference cone: per C, median and 10/90% effect over seeds coherent in both signs, until fewer than half are."""
    random_points = [point for point in points if point["method"] == "random"]
    seeds = sorted({point["seed"] for point in random_points})
    at = {(point["seed"], point["C"], point["side"]): point for point in random_points}
    cone = [(0.0, 0.0, 0.0, 0.0)]
    for C in sorted({point["C"] for point in random_points}):
        if len({seed for seed in seeds if (seed, C, "+C") in at}) < max(1, len(seeds) // 2):
            continue  # seeds start at their own C0/8, so the lowest doses are sampled by only some seeds
        coherent = [seed for seed in seeds if all((seed, C, side) in at and at[seed, C, side]["admissible"] for side in ("+C", "-C"))]
        if len(coherent) < max(1, len(seeds) // 2):
            break
        chosen = [at[seed, C, side] for seed in coherent for side in ("+C", "-C")]
        effects = sorted(point["effect"] for point in chosen)
        cone.append((median(effects), median(point["off_axis"] for point in chosen), effects[len(effects) // 10], effects[-(len(effects) // 10) - 1]))
    return cone


PROMPTS = {"prompting": "prompt", "prompting_engineered": "eng. prompt"}  # single points, not walks


def frontier(curve: list[dict]) -> list[dict]:
    """Pareto points of one walk in on-axis order, then the last coherent dose (always kept, the x).

    A point stays if no other point has at least its on-axis gain with less damage. If the last
    coherent dose is not itself on the frontier, the line takes one straight step back to it."""
    if not curve:
        return []
    end = curve[-1]
    kept = [
        p for p in curve
        if p is not end and not any(q is not p and directed(q) >= directed(p) and q["off_axis"] < p["off_axis"] for q in curve)
    ]
    return sorted(kept, key=directed) + [end]


def smooth_path(support: list[dict], side: str, n: int = 40) -> list[list[float]]:
    """Monotone cubic (Fritsch-Carlson PCHIP) of damage over on-axis gain, pinned at bare and at the end.

    Monotone interpolation cannot overshoot, so the drawn line stays between its support points."""
    sign = 1.0 if side == "+C" else -1.0
    end = support[-1]
    ts, ys = [0.0], [0.0]
    for p in support[:-1]:
        if directed(p) > ts[-1]:
            ts.append(directed(p)); ys.append(p["off_axis"])
    if directed(end) > ts[-1]:  # the end continues the monotone frontier
        ts.append(directed(end)); ys.append(end["off_axis"])
        end = None
    if len(ts) < 2:  # no forward progress on this side: straight line to the end
        return [[0.0, 0.0], [support[-1]["effect"], support[-1]["off_axis"]]]
    tail = [] if end is None else [[end["effect"], end["off_axis"]]]  # one straight step back to the last coherent dose
    h = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(len(h))]
    m = [d[0]] + [0.0 if d[i - 1] * d[i] <= 0 else 3 * (h[i - 1] + h[i]) / ((2 * h[i] + h[i - 1]) / d[i - 1] + (h[i] + 2 * h[i - 1]) / d[i]) for i in range(1, len(d))] + [d[-1]]
    path = []
    for i in range(len(h)):
        for k in range(n):
            u = k / n
            h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
            y = h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]
            path.append([sign * (ts[i] + u * h[i]), y])
    path.append([sign * ts[-1], ys[-1]])
    return path + tail


def plot(points: list[dict], title: str, methods: list[str], best: dict) -> go.Figure:
    figure = go.Figure()
    curves = {(method, side): method_curve(points, method, side) for method in methods for side in ("+C", "-C")}
    prompting = [point for point in points if point["method"] in PROMPTS]
    random_live = [point for point in points if point["method"] == "random" and point["admissible"]]
    shown = [point for curve in curves.values() for point in curve] + random_live + prompting
    x_limit = 1.08 * max(abs(point["effect"]) for point in shown)
    y_range = (1.08 * max(point["off_axis"] for point in shown), -0.07)
    margin = {"l": 75, "r": 10, "t": 40, "b": 58}
    cone = random_zone(points)
    figure.add_trace(go.Scatter(
        x=[p[2] for p in cone] + [p[3] for p in reversed(cone)], y=[p[1] for p in cone] + [p[1] for p in reversed(cone)],
        fill="toself", fillcolor="rgba(150,150,150,0.22)", line={"color": "rgba(150,150,150,0)", "width": 0},
        line_shape="spline", line_smoothing=0.8, hoverinfo="skip", showlegend=False,
    ))
    obstacles = [(0.0, 0.0)]
    labels = []
    for (method, side), curve in curves.items():
        if not curve:
            continue
        path = smooth_path(frontier(curve), side)
        dash = "solid" if side == "+C" else "dash"
        figure.add_trace(go.Scatter(
            x=[q[0] for q in path], y=[q[1] for q in path], mode="lines",
            line={"color": COLORS[method], "width": 3, "dash": dash}, hoverinfo="skip", showlegend=False,
        ))
        support = frontier(curve)  # the points the line is fitted to; last = last coherent dose (x)
        figure.add_trace(go.Scatter(
            x=[p["effect"] for p in support], y=[p["off_axis"] for p in support], mode="markers", name="frontier",
            marker={"color": COLORS[method], "size": [8] * (len(support) - 1) + [13], "symbol": ["circle"] * (len(support) - 1) + ["x"]},
            text=[f"{side} C={p['C']:.3g}" for p in support],
            hovertemplate=f"{LABELS[method]}<br>%{{text}}<br>effect=%{{x:.3f}}<br>damage=%{{y:.3f}}<extra></extra>", showlegend=False,
        ))
        if best.get((method, side)) is not None:  # dose that sets the score: ring
            b = best[method, side]
            figure.add_trace(go.Scatter(
                x=[b["effect"]], y=[b["off_axis"]], mode="markers", hoverinfo="skip", showlegend=False,
                marker={"color": COLORS[method], "size": 22, "symbol": "circle-open", "line": {"width": 3}},  # open symbols draw in marker.color
            ))
        obstacles.extend((q[0], q[1]) for q in path[::4])
        obstacles.extend((p["effect"], p["off_axis"]) for p in curve)
        labels.append({"x": curve[-1]["effect"], "y": curve[-1]["off_axis"], "text": f"{LABELS[method]} {side}", "color": COLORS[method]})
    for point in prompting:
        figure.add_trace(go.Scatter(
            x=[point["effect"]], y=[point["off_axis"]], mode="markers",
            marker={"color": COLORS[point["method"]], "size": 13, "symbol": "star"}, hoverinfo="skip", showlegend=False,
        ))
        obstacles.append((point["effect"], point["off_axis"]))
        labels.append({"x": point["effect"], "y": point["off_axis"], "text": f"{PROMPTS[point['method']]} {point['side']}", "color": COLORS[point["method"]]})
    figure.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker={"color": "#333333", "size": 11, "symbol": "diamond"}, hoverinfo="skip", showlegend=False))
    figure.add_annotation(x=0, y=0, text="bare", showarrow=False, xshift=28, yshift=12, font={"color": "#333333", "size": 14})
    for annotation in place_labels(
        labels, (-x_limit, x_limit), y_range, obstacles=obstacles, fig_w=1064, fig_h=590, margin=margin,
        font={"size": 13}, bgcolor="rgba(255,255,255,0.9)", arrowcolor="rgba(45,24,16,0.6)",
    ):
        figure.add_annotation(**annotation)
    if len(cone) > 1:
        figure.add_annotation(x=cone[-1][0], y=cone[-1][1] / 2, text="null zone of<br>random directions", showarrow=False, font={"color": "#666666", "size": 13})
    figure.add_annotation(x=0, y=1, xref="paper", yref="paper", text="clean steer -> abrasive", showarrow=False, xanchor="left", font={"color": "#287a4d", "size": 14})
    figure.add_annotation(x=1, y=1, xref="paper", yref="paper", text="clean steer -> sycophantic", showarrow=False, xanchor="right", font={"color": "#287a4d", "size": 14})
    figure.add_annotation(x=0.005, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top", yshift=-34, align="left", showarrow=False,
                          font={"color": "#555555", "size": 12},
                          text="dot = Pareto point · ring = dose that sets the score<br>× = last coherent dose · ★ = prompt baseline")
    figure.add_annotation(x=0.5, y=0, xref="paper", yref="paper", text="mostly side effects", showarrow=False, yshift=18, font={"color": "#c44e52", "size": 14})
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"}, height=590, margin=margin,
        font={"color": "#111", "size": 15}, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        xaxis={"title": "Jev on-axis change: premise level (solid +C, dashed -C)", "range": [-x_limit, x_limit], "showline": True, "linecolor": "#333333", "gridcolor": "#e5e5e5", "zeroline": False},
        yaxis={"title": "off-axis damage (lower is better)", "range": y_range, "showline": True, "linecolor": "#333333", "gridcolor": "#e5e5e5", "zeroline": False},
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
        f"({PMAX} − bare level for +C, bare level for −C), weaker side; damage is handled by the dose choice and the {MAX_DAMAGE:g} cap, not in this number.")


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


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir or default_model_dir()
    out = args.out or ROOT / "outputs/bsbench/results" / args.cohort
    out.mkdir(parents=True, exist_ok=True)
    exclude = {m for m in args.exclude.split(",") if m}
    points = build_points(model_dir, args.cohort, exclude)
    # methods without a fixed colour (e.g. tagged variants like vjp_delta-t48) take the next spare colour, in name order
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
    shown = [row["method"] for row in rows if row["method"] in methods and not math.isnan(row["score"])][:TOP_N_PLOT]
    site = {
        "shown": shown,
        "colors": COLORS,  # the page's only colour source
        "model_dir": model_dir.name, "cohort": args.cohort, "judge": f"{MODEL} (premise level 0-8, damage 0-4)", "off_weight": OFF_WEIGHT,
        "questions": [{"scenario": s, "prompt": cohort_rows[s]["prompt"], "flaw": cohort_rows[s]["nonsensical_element"], "bare": bare[s]["text"]} for s in scenarios],
        "zone": random_zone(points),
        "curves": [{
            "method": m, "side": side,
            "points": [{k: p[k] for k in ("C", "effect", "off_axis")} for p in frontier(method_curve(points, m, side))],
            "path": smooth_path(frontier(method_curve(points, m, side)), side) if method_curve(points, m, side) else [],
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
    (out / f"points.json").write_text(json.dumps(_no_nan(site), indent=1, allow_nan=False) + "\n")
    model = model_dir.name.rsplit("-g", 1)[0].split("--")[-1]
    title = f"steering-lite on Bullshit Bench v2: {model} ({args.cohort}, {len(scenarios)} questions) — judge: Jev"
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
        f"Judge: Jev (on-axis = change in premise level 0-8, off-axis = |change in damage level 0-4|). Admissible = healthy answers, not past the walk boundary, mean steered damage ≤ {MAX_DAMAGE:g} of 4."
    ) + (f" Left out (not yet judged): {', '.join(sorted(exclude))}." if exclude else "")
    (out / f"index.md").write_text(f"# Results ({args.cohort})\n\n{intro}\n\n![plot](plot.png)\n\n{table}\n")
    figure_html = figure.to_html(full_html=False, include_plotlyjs="cdn", default_width="100%", config={"responsive": True})
    (out / f"plot.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>steering-lite bsbench</title>"
        "<style>body{font:16px system-ui;max-width:1064px;margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap}</style>"
        f"<h1>Results ({html.escape(args.cohort)})</h1><p>{html.escape(intro)}</p>{figure_html}<pre>{html.escape(table)}</pre>"
    )
    figure.write_image(out / f"plot.png", width=1064, height=590, scale=2)
    # marker count drawn in the PNG, compared with the React page by web/uat.py
    frontier_marks = sum(len(trace.x) for trace in figure.data if trace.name == "frontier")
    (out / f"plot_marks.json").write_text(json.dumps({"frontier_marks": frontier_marks, "methods": shown}) + "\n")
    print(table)
    print(f"wrote {out}/points.json ({len(points)} points), index.md, plot.html, plot.png")


if __name__ == "__main__":
    main()
