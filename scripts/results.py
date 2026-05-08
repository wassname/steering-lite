"""README-ready tables + moral map from a sweep output dir.

Wraps `aggregate_flips.py` (SI table, Δlogit-with-σ table, sign agreement)
and adds:

  - Base model logit per foundation, t-stat, vs human calibrated wrongness.
  - Moral map: 7-foundation profiles z-scored across entities, PCA → 2D
    scatter of base + each method (POS/NEG sign) + human reference.

Usage:
    uv run --extra benchmark python scripts/results.py \\
        --sweep-dir outputs/tinymfv_sweep_resw \\
        --vignettes clifford \\
        --map-out outputs/moral_map.png

Or via justfile: `just results outputs/tinymfv_sweep_resw`.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from tabulate import tabulate

from steering_lite.eval.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT


def _orjson_loads(p: Path):
    try:
        import orjson
        return orjson.loads(p.read_bytes())
    except ImportError:
        return json.loads(p.read_text())


def _human_per_foundation(vignettes_name: str) -> dict[str, dict[str, float]]:
    """Loading-weighted human calibrated_wrongness (1-5) per foundation.

    Each vignette contributes to all 7 foundations weighted by its
    `calibrated_<F>` mass (per-vignette normalised to sum=1 across foundations,
    matching `aggregate_flips._loadings_long`).
    """
    from tinymfv import load_vignettes
    from aggregate_flips import LOADING_COL_FOR_FOUNDATION
    vigs = load_vignettes(vignettes_name)
    out: dict[str, dict[str, float]] = {}
    weights_by_f: dict[str, list[float]] = {f: [] for f in FOUNDATION_ORDER}
    xs_by_f: dict[str, list[float]] = {f: [] for f in FOUNDATION_ORDER}
    for v in vigs:
        cw = v.get("calibrated_wrongness")
        if cw is None:
            continue
        loads = {f: float(v.get(col) or 0.0) for f, col in LOADING_COL_FOR_FOUNDATION.items()}
        s = sum(loads.values())
        if s <= 0:
            continue
        for f in FOUNDATION_ORDER:
            L = loads[f] / s
            if L <= 0:
                continue
            weights_by_f[f].append(L); xs_by_f[f].append(float(cw))
    for f in FOUNDATION_ORDER:
        ws = np.array(weights_by_f[f], dtype=float)
        xs = np.array(xs_by_f[f], dtype=float)
        if ws.sum() <= 0:
            out[f] = {"mean": float("nan"), "std": float("nan"), "wsum": 0.0, "n": 0}
            continue
        mean = (ws * xs).sum() / ws.sum()
        var = (ws * (xs - mean) ** 2).sum() / ws.sum()
        out[f] = {
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "wsum": float(ws.sum()),
            "n": int(ws.size),
        }
    return out


def _human_loadings_per_foundation(vignettes_name: str) -> dict[str, float]:
    """Mean calibrated loading % across all vignettes per foundation.

    This is the population-level "how much do humans think this foundation
    is at stake on this benchmark". Different from per-foundation wrongness
    means (which only count vignettes whose foundation_coarse matches f).
    """
    from tinymfv import load_vignettes
    col = {
        "Care": "calibrated_Care",
        "Sanctity": "calibrated_Sanctity",
        "Authority": "calibrated_Authority",
        "Loyalty": "calibrated_Loyalty",
        "Fairness": "calibrated_Fairness",
        "Liberty": "calibrated_Liberty",
        "Social Norms": "calibrated_SocialNorms",
    }
    vigs = load_vignettes(vignettes_name)
    return {f: float(np.mean([v[col[f]] for v in vigs])) for f in FOUNDATION_ORDER}


def _loading_weighted_profiles(sweep_dir: Path, vignettes_name: str,
                               bare_name: str = "bare.json") -> tuple[dict, dict, dict]:
    """Loading-weighted per-(method, sign, foundation) absolute-logit profiles.

    Reuses `aggregate_flips`'s pipeline so the moral map and base-vs-humans
    table see the same view as the SI/Δlogit tables: every vignette × cond
    contributes to every foundation, weighted by its calibrated_<F> mass.

    Returns:
        profiles: {(method, sign, foundation): {logit_mean, logit_std, wsum, n_cells}}
        methods:  {method: {bidirectional, calibrated_C}}
        dlogit_lookup: {(method, sign, foundation): {mean, std, ...}} for sign-picking.
    """
    from aggregate_flips import (
        _load_long_frame, _wrongness_table, _loadings_long, _paired_cells,
        _weighted_dlogit_table, _logit_expr, _to_lookup,
    )

    df, methods = _load_long_frame(sweep_dir, bare_name)
    cells = _wrongness_table(df)
    loadings = _loadings_long(vignettes_name)

    # Per-(method, sign, vid, cond) absolute logit, joined with per-vid loadings.
    cells_v = cells.filter(pl.col("w").is_not_null()).with_columns(
        _logit_expr("w").alias("logit_w"),
    )
    cl = cells_v.join(loadings, on="vid", how="inner")
    profiles_df = cl.group_by(["method", "sign", "foundation"]).agg(
        ((pl.col("loading") * pl.col("logit_w")).sum() / pl.col("loading").sum()).alias("logit_mean"),
        ((pl.col("loading") * pl.col("logit_w") ** 2).sum() / pl.col("loading").sum()).alias("_ex2"),
        pl.col("loading").sum().alias("wsum"),
        pl.len().alias("n_cells"),
    ).with_columns(
        ((pl.col("_ex2") - pl.col("logit_mean") ** 2).clip(0.0)).sqrt().alias("logit_std"),
    )
    profiles = {(r["method"], int(r["sign"]), r["foundation"]):
                {"logit_mean": r["logit_mean"], "logit_std": r["logit_std"],
                 "wsum": r["wsum"], "n_cells": r["n_cells"]}
                for r in profiles_df.iter_rows(named=True)}

    # Δlogit lookup for sign-picking (matches aggregate_flips._pick_sign).
    paired = _paired_cells(cells, loadings)
    dlogit = _weighted_dlogit_table(paired)
    dlogit_lookup = _to_lookup(dlogit, ["mean", "std", "wsum", "n_cells"])

    return profiles, methods, dlogit_lookup


def base_vs_humans_table(profiles: dict, vignettes_name: str) -> str:
    humans = _human_per_foundation(vignettes_name)
    rows = []
    for f in FOUNDATION_ORDER:
        r = profiles.get(("bare", 0, f))
        if r is None:
            continue
        m_mean = r["logit_mean"]; m_std = r["logit_std"]; wsum = r["wsum"]
        sem = (m_std / math.sqrt(wsum)) if wsum > 0 and m_std > 0 else float("nan")
        t = (m_mean / sem) if sem and not math.isnan(sem) and sem > 0 else float("nan")
        prob = 1.0 / (1.0 + math.exp(-m_mean))
        h = humans[f]
        h_str = f"{h['mean']:.2f}±{h['std']:.2f}" if not math.isnan(h["mean"]) else "n/a"
        rows.append([
            FOUNDATION_SHORT[f],
            f"{m_mean:+.2f}±{m_std:.2f}",
            f"{prob*100:.0f}%",
            f"{t:+.1f}" if not math.isnan(t) else "n/a",
            h_str,
            f"{wsum:.1f}",
            f"{h['wsum']:.1f}",
        ])
    return tabulate(
        rows,
        headers=["foundation", "model logit±σ", "p(wrong)", "t-stat",
                 "human wrong (1-5)±σ", "Σw_m", "Σw_h"],
        tablefmt="pipe",
        floatfmt="+.2f",
    )


def moral_map(profiles: dict, methods: dict, dlogit_lookup: dict,
              vignettes_name: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from aggregate_flips import _pick_sign, HEADLINE_FOUNDATION

    base_logit = {f: profiles[("bare", 0, f)]["logit_mean"] for f in FOUNDATION_ORDER}

    # Loading-weighted per-(method, sign) profiles. For each method pick the
    # sign that minimises ΔAuthority (persona-aligned), matching the aggregator.
    entities: list[tuple[str, str, np.ndarray]] = []
    entities.append(("base", "model", np.array([base_logit[f] for f in FOUNDATION_ORDER])))
    for method, meta in sorted(methods.items()):
        if "stale" in method:
            continue
        sign = _pick_sign(dlogit_lookup, method, meta["bidirectional"], HEADLINE_FOUNDATION)
        sign_lab = "POS" if sign == 1 else "NEG"
        try:
            vec = np.array([profiles[(method, sign, f)]["logit_mean"] for f in FOUNDATION_ORDER])
        except KeyError:
            continue
        entities.append((method, sign_lab, vec))

    # Humans: use calibrated_wrongness mean per foundation_coarse as a logit-ish
    # proxy. Different scale, so we'll project onto model-fit PCA below.
    human_wrong = _human_per_foundation(vignettes_name)
    human_vec_raw = np.array([human_wrong[f]["mean"] for f in FOUNDATION_ORDER])
    has_human = not np.any(np.isnan(human_vec_raw))

    # Drop entities with NaN (e.g. calibration collapse → NaN logits).
    keep = [(m, s, v) for (m, s, v) in entities if not np.any(np.isnan(v))]
    dropped = [(m, s) for (m, s, v) in entities if np.any(np.isnan(v))]
    if dropped:
        logger.warning(f"moral_map: dropping NaN entities: {dropped}")

    labels = [(m, s) for (m, s, _) in keep]
    X = np.stack([v for (_, _, v) in keep])  # [n_model_entities, 7]

    # PCA fit on MODEL entities only (per-foundation z-score using model μ/σ).
    # If we mix in the human point with very different scale (1-5 vs logits)
    # PC1 ends up ~97% variance just separating human from everything.
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=0, keepdims=True)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    Xz = (X - mu) / sigma
    Xc = Xz - Xz.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U * S  # [n, k]
    pc1, pc2 = pcs[:, 0], pcs[:, 1]
    var = (S ** 2) / (S ** 2).sum()

    # Project human into the model-fit PCA. Human is on a 1-5 scale; we
    # mean-centre it on the model's per-foundation mean and rescale by
    # model std so the geometric direction in foundation-space is preserved.
    human_pc = None
    if has_human:
        h_z = (human_vec_raw - mu.squeeze(0)) / sigma.squeeze(0)
        h_c = h_z - Xz.mean(axis=0)
        h_pc = h_c @ Vt.T
        human_pc = (float(h_pc[0]), float(h_pc[1]))

    import textalloc as ta

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- LEFT: PCA scatter ---
    ax = axs[0]
    base_idx = next(i for i, (m, _) in enumerate(labels) if m == "base")
    base_pc = (pc1[base_idx], pc2[base_idx])

    # Collect labels + line obstacles for textalloc.
    label_x: list[float] = []
    label_y: list[float] = []
    label_text: list[str] = []
    label_color: list[str] = []
    label_size: list[int] = []
    x_lines: list[list[float]] = []
    y_lines: list[list[float]] = []

    # Base
    ax.scatter(pc1[base_idx], pc2[base_idx], s=180, c="black", marker="*", zorder=5, label="base")
    label_x.append(pc1[base_idx]); label_y.append(pc2[base_idx])
    label_text.append("base"); label_color.append("black"); label_size.append(10)

    # Method endpoints. All shown in the persona-aligned direction (post-eval
    # sign flip), so all dots are one colour — the [P]/[N] suffix in the label
    # tells the reader which sign was the persona-aligned one for that method.
    for i, (method, sign) in enumerate(labels):
        if method == "base":
            continue
        ax.scatter(pc1[i], pc2[i], s=40, c="C0", alpha=0.85, zorder=4)
        label_x.append(pc1[i]); label_y.append(pc2[i])
        label_text.append(method); label_color.append("C0"); label_size.append(7)

    if human_pc is not None:
        ax.scatter(human_pc[0], human_pc[1], s=220, c="red", marker="X", zorder=6,
                   label="human ref (projected)")
        label_x.append(human_pc[0]); label_y.append(human_pc[1])
        label_text.append("human"); label_color.append("red"); label_size.append(10)

    # Foundation compass: anchored at the PCA origin, scaled big enough that
    # the arrow tips fan out far enough for direct labels at each tip without
    # textalloc collisions. xlim/ylim expanded below to keep all tips visible.
    loads = Vt.T[:, :2] * S[:2]
    compass_origin = (0.0, 0.0)
    model_extent = max(np.abs(pcs[:, :2]).max(), 1.0)
    compass_scale = 1.3 * model_extent / max(np.abs(loads).max(), 1.0)
    ax.scatter(*compass_origin, s=20, c="grey", marker="+", alpha=0.5)
    compass_tip_x: list[float] = []
    compass_tip_y: list[float] = []
    for j, f in enumerate(FOUNDATION_ORDER):
        dx, dy = loads[j, 0] * compass_scale, loads[j, 1] * compass_scale
        ax.arrow(compass_origin[0], compass_origin[1], dx, dy,
                 head_width=0.04 * model_extent, color="grey", alpha=0.55,
                 length_includes_head=True)
        tx = compass_origin[0] + dx * 1.08
        ty = compass_origin[1] + dy * 1.08
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.text(tx, ty, FOUNDATION_SHORT[f], color="grey", fontsize=8,
                ha=ha, va=va, alpha=0.9, zorder=3)
        x_lines.append([compass_origin[0], tx])
        y_lines.append([compass_origin[1], ty])
        compass_tip_x.append(tx); compass_tip_y.append(ty)

    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title("Moral map (PCA on z-scored 7-foundation profile)")

    # Pin axis limits with padding so textalloc's canvas covers all scatter,
    # compass tips, and connector lines.
    all_x = label_x + compass_tip_x
    all_y = label_y + compass_tip_y
    pad_x = 0.08 * (max(all_x) - min(all_x))
    pad_y = 0.12 * (max(all_y) - min(all_y))
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ta.allocate(ax, label_x, label_y, label_text,
                x_scatter=label_x, y_scatter=label_y,
                x_lines=x_lines, y_lines=y_lines,
                textsize=label_size, textcolor=label_color,
                draw_lines=True, linecolor="grey", linewidth=0.4,
                margin=0.01, min_distance=0.015, max_distance=0.18,
                avoid_label_lines_overlap=True)

    # --- RIGHT: Authority vs SocialNorms surgical view ---
    # SocN is the most orthogonal-to-Auth foundation in human labels (corr ≈ 0.04
    # vs Care -0.19, Loy +0.43). Methods on the diagonal = broad suppression that
    # also kills SocN; off-axis (large ΔAuth, small ΔSocN) = surgical.
    OFF_AXIS = "Social Norms"
    OFF_AXIS_SHORT = FOUNDATION_SHORT[OFF_AXIS]
    ax2 = axs[1]
    base_auth = base_logit["Authority"]; base_off = base_logit[OFF_AXIS]
    lx2: list[float] = []; ly2: list[float] = []
    lt2: list[str] = []; lc2: list[str] = []; ls2: list[int] = []
    for (method, sign), vec in zip(labels, X):
        if method == "base":
            ax2.scatter(0, 0, s=180, c="black", marker="*", zorder=5)
            lx2.append(0); ly2.append(0); lt2.append("base"); lc2.append("black"); ls2.append(10)
        elif method == "human":
            continue
        else:
            dAuth = vec[FOUNDATION_ORDER.index("Authority")] - base_auth
            dOff = vec[FOUNDATION_ORDER.index(OFF_AXIS)] - base_off
            ax2.scatter(dAuth, dOff, s=40, c="C0", alpha=0.7)
            lx2.append(dAuth); ly2.append(dOff)
            lt2.append(method); lc2.append("C0"); ls2.append(7)
    lo, hi = ax2.get_xlim()
    ax2.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.4, label="broad-suppression diag")
    ax2.axhline(0, color="grey", lw=0.4); ax2.axvline(0, color="grey", lw=0.4)
    ax2.set_xlabel("Δlogit Authority (target; want < 0)")
    ax2.set_ylabel(f"Δlogit {OFF_AXIS_SHORT} (off-target ⊥ Auth in humans; want ≈ 0)")
    ax2.set_title(f"Surgical view: ΔAuth vs Δ{OFF_AXIS_SHORT}\n(off-diagonal = surgical, diagonal = broad)")
    ax2.legend(loc="best", fontsize=8)
    ax2.relim(); ax2.autoscale_view()
    ta.allocate(ax2, lx2, ly2, lt2,
                x_scatter=lx2, y_scatter=ly2,
                textsize=ls2, textcolor=lc2,
                draw_lines=True, linecolor="grey", linewidth=0.4,
                margin=0.01, min_distance=0.015, max_distance=0.18)

    fig.suptitle(f"steering-lite moral map — {out_png.parent.name}", y=1.0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    logger.info(f"wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, default=Path("outputs/tinymfv_sweep"))
    p.add_argument("--vignettes", default="clifford")
    p.add_argument("--map-out", type=Path, default=None,
                   help="PNG path for moral map. Defaults to {sweep_dir}/moral_map.png")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip running aggregate_flips.py (use if already aggregated)")
    args = p.parse_args()

    map_out = args.map_out or (args.sweep_dir / "moral_map.png")

    profiles, methods, dlogit_lookup = _loading_weighted_profiles(
        args.sweep_dir, args.vignettes,
    )

    print(f"\n## Base model vs humans — {args.sweep_dir.name}\n")
    print(base_vs_humans_table(profiles, args.vignettes))

    print("\n## Moral map\n")
    moral_map(profiles, methods, dlogit_lookup, args.vignettes, map_out)
    print(f"![moral map]({map_out})\n")

    if not args.skip_aggregate:
        print("\n## SI / Δlogit / sign agreement (loading-weighted)\n")
        cmd = [sys.executable, str(Path(__file__).parent / "aggregate_flips.py"),
               "--sweep-dir", str(args.sweep_dir)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
