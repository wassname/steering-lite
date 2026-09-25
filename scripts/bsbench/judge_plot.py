"""Judge noise, seen: DeepSeek vs Jev Pareto frontiers on the same walks, with bootstrap spread.

Left figure (judge_frontiers.png): one panel per method (top 8 by DeepSeek + random). Each judge's
frontier is drawn bold; faint lines are the frontier recomputed on N_SPAGHETTI hierarchical bootstrap
draws (seeds, then questions), so the width of each bundle is the inferred uncertainty. Jev uses its
own admissible doses (damage <= 1.5 of 4) and is rescaled to DeepSeek units (std matching over the
same answers, judge_compare.jev_scale), so the two bundles share axes.
Right figure (judge_scores.png): method score DeepSeek (x) vs Jev in DeepSeek units (y), 90% CI bars.

    python judge_plot.py --cohort full
"""

import argparse
import random

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from judge import COHORTS, default_model_dir, load_cohort
from judge_compare import JEV_MAX_DAMAGE, curves_for, draw_score, jev_scale
from results import COLORS, PROMPTS, ROOT, build_points, directed, frontier, jev_points, pareto_score, resample

N_SPAGHETTI = 30
N_DRAWS = 300
JUDGES = {"DeepSeek": "#1f4e9c", "Jev (DeepSeek units)": "#e07b00"}


def draw_frontier(figure, curve: list[dict], color: str, width: float, opacity: float, row: int, col: int, name: str, legend: bool) -> None:
    pts = frontier(curve)
    if not pts:
        return
    xs = [0.0] + [p["effect"] for p in pts]
    ys = [0.0] + [p["off_axis"] for p in pts]
    figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": width}, opacity=opacity,
                                name=name, legendgroup=name, showlegend=legend, hoverinfo="skip"), row=row, col=col)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="full")
    args = parser.parse_args()
    model_dir = default_model_dir()
    out = ROOT / "outputs/bsbench/results" / args.cohort
    ds = build_points(model_dir, args.cohort, set())
    jev = jev_points(ds, model_dir)
    jev = [p | {"admissible": not p["breakdown_reasons"] and not p["post_boundary"] and p["jev_damage"] <= JEV_MAX_DAMAGE} for p in jev]
    s_on, s_off = jev_scale(ds, jev)
    jev = [p | {"effect": p["effect"] * s_on, "off_axis": p["off_axis"] * s_off,
                "questions": [q | {"effect": q["effect"] * s_on, "off_axis": q["off_axis"] * s_off} for q in p["questions"]]} for p in jev]
    views = {"DeepSeek": ds, "Jev (DeepSeek units)": jev}
    scenarios = list(load_cohort())[COHORTS[args.cohort]]
    methods = sorted({p["method"] for p in ds} - set(PROMPTS))
    seeds = {m: sorted({p["seed"] for p in ds if p["method"] == m}) for m in methods}
    curves = {v: {m: curves_for(pts, m) for m in methods} for v, pts in views.items()}
    score = {v: {m: pareto_score(curves[v][m])[0] for m in methods} for v in views}

    # scores with paired 90% CIs
    ci = {v: {} for v in views}
    for m in methods:
        for v in views:
            draws = sorted(draw_score(curves[v][m], seeds[m], scenarios, random.Random(f"{d}-{m}")) for d in range(N_DRAWS))
            ci[v][m] = (draws[int(0.05 * N_DRAWS)], draws[int(0.95 * N_DRAWS) - 1])
    fig = go.Figure()
    lo = min(min(c[0] for c in ci[v].values()) for v in views) - 0.1
    hi = max(max(c[1] for c in ci[v].values()) for v in views) + 0.1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": "#999", "dash": "dot"}, name="same score", hoverinfo="skip"))
    for m in methods:
        x, y = score["DeepSeek"][m], score["Jev (DeepSeek units)"][m]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[m], textposition="top right", textfont={"size": 11, "color": COLORS.get(m, "#555")},
            marker={"size": 9, "color": COLORS.get(m, "#555")}, showlegend=False,
            error_x={"type": "data", "symmetric": False, "array": [ci["DeepSeek"][m][1] - x], "arrayminus": [x - ci["DeepSeek"][m][0]], "thickness": 1},
            error_y={"type": "data", "symmetric": False, "array": [ci["Jev (DeepSeek units)"][m][1] - y], "arrayminus": [y - ci["Jev (DeepSeek units)"][m][0]], "thickness": 1},
        ))
    fig.update_layout(title=f"Method score under each judge ({args.cohort}, {len(scenarios)} questions; bars = 90% bootstrap CI)",
                      xaxis_title="DeepSeek score (min over ±C of on − off)", yaxis_title=f"Jev score, DeepSeek units (on ×{s_on:.2f}, off ×{s_off:.2f})",
                      template="simple_white", width=900, height=820)
    fig.update_xaxes(range=[lo, hi]); fig.update_yaxes(range=[lo, hi], scaleanchor="x")
    fig.write_image(out / "judge_scores.png", scale=2)

    # frontier small multiples with bootstrap spaghetti
    shown = sorted([m for m in methods if m != "random"], key=lambda m: -score["DeepSeek"][m])[:8] + ["random"]
    grid = make_subplots(rows=3, cols=3, subplot_titles=[f"{m}  (DS {score['DeepSeek'][m]:+.2f}, Jev {score['Jev (DeepSeek units)'][m]:+.2f})" for m in shown],
                         shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.04, vertical_spacing=0.07)
    for i, m in enumerate(shown):
        row, col = i // 3 + 1, i % 3 + 1
        for v, color in JUDGES.items():
            for d in range(N_SPAGHETTI):
                rng = random.Random(f"spaghetti-{d}-{m}")
                drawn_seeds = [rng.choice(seeds[m]) for _ in seeds[m]]
                drawn = [rng.choice(scenarios) for _ in scenarios]
                for side, curve in curves[v][m].items():
                    draw_frontier(grid, resample(curve, drawn, drawn_seeds), color, 1, 0.12, row, col, v, False)
            for side, curve in curves[v][m].items():
                draw_frontier(grid, curve, color, 2.5, 1.0, row, col, v, i == 0 and side == "+C")
                best = pareto_score(curves[v][m])[1][side]
                if best is not None:
                    grid.add_trace(go.Scatter(x=[best["effect"]], y=[best["off_axis"]], mode="markers", showlegend=False, hoverinfo="skip",
                                              marker={"symbol": "circle-open", "size": 11, "color": color, "line": {"width": 2}}), row=row, col=col)
    grid.update_yaxes(autorange="reversed", title_text="off-axis damage", col=1)
    grid.update_xaxes(title_text="on-axis change (−C left, +C right)", row=3)
    grid.update_layout(title=f"Pareto frontiers, DeepSeek vs Jev, same walks ({args.cohort}). Bold = estimate, faint = {N_SPAGHETTI} bootstrap draws (seeds, questions), ring = dose that sets the score",
                       template="simple_white", width=1500, height=1150, legend={"orientation": "h", "y": 1.06, "x": 0.5, "xanchor": "center"}, font={"size": 12})
    grid.write_image(out / "judge_frontiers.png", scale=2)
    print(f"wrote {out}/judge_scores.png, judge_frontiers.png; Jev scale on x{s_on:.2f} off x{s_off:.2f}")


if __name__ == "__main__":
    main()
