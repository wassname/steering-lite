"""README-ready tables + moral map from a sweep output dir.

Wraps `aggregate_flips.py` (SI table, Δlogit-with-σ table, sign agreement)
and adds:

  - Base model logit per foundation, t-stat, vs human calibrated wrongness.
  - Moral map: 7-foundation profiles z-scored across entities, PCA → 2D
    scatter of base + each method (POS/NEG sign) + human reference.

Usage:
    uv run --extra benchmark python scripts/results.py \\
        --sweep-dir outputs/tinymfv_sweep_resw \\
        --vignettes airisk \\
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
    """Human calibrated_wrongness (1-5 scale) grouped by foundation_coarse."""
    from tinymfv import load_vignettes
    vigs = load_vignettes(vignettes_name)
    by_f: dict[str, list[float]] = {f: [] for f in FOUNDATION_ORDER}
    # Foundation_coarse uses different names; map them
    for v in vigs:
        fc = v.get("foundation_coarse", "")
        for f in FOUNDATION_ORDER:
            if fc.lower() == f.lower() or fc.lower() == f.lower().replace(" ", ""):
                by_f[f].append(float(v["calibrated_wrongness"]))
                break
    out = {}
    for f in FOUNDATION_ORDER:
        xs = np.array(by_f[f], dtype=float)
        out[f] = {
            "mean": float(xs.mean()) if xs.size else float("nan"),
            "std": float(xs.std(ddof=1)) if xs.size > 1 else float("nan"),
            "n": int(xs.size),
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


def base_vs_humans_table(sweep_dir: Path, vignettes_name: str) -> str:
    bare = _orjson_loads(sweep_dir / "bare.json")
    model = bare["absolute_logit_per_foundation"]
    humans = _human_per_foundation(vignettes_name)

    rows = []
    for f in FOUNDATION_ORDER:
        m = model[f]
        h = humans[f]
        sem = (m["std"] / math.sqrt(m["n"])) if m["n"] > 0 and m["std"] else float("nan")
        t = (m["mean"] / sem) if sem and not math.isnan(sem) and sem > 0 else float("nan")
        prob = 1.0 / (1.0 + math.exp(-m["mean"]))
        h_str = f"{h['mean']:.2f}±{h['std']:.2f}" if not math.isnan(h["mean"]) else "n/a"
        rows.append([
            FOUNDATION_SHORT[f],
            f"{m['mean']:+.2f}±{m['std']:.2f}",
            f"{prob*100:.0f}%",
            f"{t:+.1f}" if not math.isnan(t) else "n/a",
            h_str,
            m["n"],
            h["n"],
        ])
    return tabulate(
        rows,
        headers=["foundation", "model logit±σ", "p(wrong)", "t-stat", "human wrong (1-5)±σ", "n_m", "n_h"],
        tablefmt="pipe",
        floatfmt="+.2f",
    )


def _profile_for_method(d: dict, base_logit: dict[str, float], sign: str) -> dict[str, float] | None:
    sub = d.get("pos") if sign == "+" else d.get("neg")
    if sub is None:
        return None
    dl = sub.get("dlogit_per_foundation", {})
    out = {}
    for f in FOUNDATION_ORDER:
        if f not in dl or dl[f].get("mean") is None:
            return None
        out[f] = base_logit[f] + float(dl[f]["mean"])
    return out


def _zscore(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0, keepdims=True)
    sigma = M.std(axis=0, ddof=0, keepdims=True)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    return (M - mu) / sigma


def moral_map(sweep_dir: Path, vignettes_name: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bare = _orjson_loads(sweep_dir / "bare.json")
    base_logit = {f: float(bare["absolute_logit_per_foundation"][f]["mean"]) for f in FOUNDATION_ORDER}

    # Per-method profiles (POS+NEG separately when bidirectional).
    entities: list[tuple[str, str, np.ndarray]] = []
    entities.append(("base", "model", np.array([base_logit[f] for f in FOUNDATION_ORDER])))

    # Post-eval sign flip: same logic as aggregate_flips._pick_sign — pick the
    # sign that gives smaller ΔAuthority (persona wants Auth↓). For
    # unidirectional baselines (prompt_only, repeng) only POS exists.
    auth_idx = FOUNDATION_ORDER.index("Authority")
    skip = {"bare.json"}
    for jp in sorted(sweep_dir.glob("*.json")):
        if jp.name in skip or "stale" in jp.name or jp.name == "runs.jsonl":
            continue
        try:
            d = _orjson_loads(jp)
        except Exception:
            continue
        method = jp.stem
        candidates = []
        for sign, lab in [("+", "POS"), ("-", "NEG")]:
            prof = _profile_for_method(d, base_logit, sign)
            if prof is None:
                continue
            vec = np.array([prof[f] for f in FOUNDATION_ORDER])
            candidates.append((sign, lab, vec))
        if not candidates:
            continue
        # pick sign minimising ΔAuth = vec[auth] − base_auth
        base_auth = base_logit["Authority"]
        best = min(candidates, key=lambda c: c[2][auth_idx] - base_auth)
        entities.append((method, best[1], best[2]))

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

    # Steering: line from base★ to each method endpoint
    for i, (method, sign) in enumerate(labels):
        if method == "base":
            continue
        color = "C0" if sign == "POS" else "C3"
        ax.plot([base_pc[0], pc1[i]], [base_pc[1], pc2[i]],
                color=color, alpha=0.35, lw=0.7, zorder=2)
        ax.scatter(pc1[i], pc2[i], s=40, c=color, alpha=0.85, zorder=4)
        x_lines.append([base_pc[0], pc1[i]])
        y_lines.append([base_pc[1], pc2[i]])
        label_x.append(pc1[i]); label_y.append(pc2[i])
        label_text.append(f"{method}[{sign[0]}]"); label_color.append(color); label_size.append(7)

    if human_pc is not None:
        ax.scatter(human_pc[0], human_pc[1], s=220, c="red", marker="X", zorder=6,
                   label="human ref (projected)")
        label_x.append(human_pc[0]); label_y.append(human_pc[1])
        label_text.append("human"); label_color.append("red"); label_size.append(10)

    # Foundation compass: fixed reference frame, anchored in the emptiest quadrant.
    loads = Vt.T[:, :2] * S[:2]
    if human_pc is not None:
        compass_origin = (0.5 * (human_pc[0] + pc1.mean()), human_pc[1])
    else:
        compass_origin = (pc1.max() + 1.0, pc2.max())
    compass_scale = 0.35 * max(np.abs(pcs[:, :2]).max(), 1.0) / max(np.abs(loads).max(), 1.0)
    for j, f in enumerate(FOUNDATION_ORDER):
        dx, dy = loads[j, 0] * compass_scale, loads[j, 1] * compass_scale
        ax.arrow(compass_origin[0], compass_origin[1], dx, dy,
                 head_width=0.08, color="grey", alpha=0.55, length_includes_head=True)
        x_lines.append([compass_origin[0], compass_origin[0] + dx])
        y_lines.append([compass_origin[1], compass_origin[1] + dy])
        # tip position is the label anchor; textalloc finds a free spot near it
        label_x.append(compass_origin[0] + dx)
        label_y.append(compass_origin[1] + dy)
        label_text.append(FOUNDATION_SHORT[f])
        label_color.append("grey"); label_size.append(9)
    ax.scatter(*compass_origin, s=20, c="grey", marker="+", alpha=0.5)

    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title("Moral map (PCA on z-scored 7-foundation profile)")

    # Pin axis limits with padding so textalloc's canvas covers all scatter+lines.
    pad_x = 0.08 * (max(label_x) - min(label_x))
    pad_y = 0.12 * (max(label_y) - min(label_y))
    ax.set_xlim(min(label_x) - pad_x, max(label_x) + pad_x)
    ax.set_ylim(min(label_y) - pad_y, max(label_y) + pad_y)
    ta.allocate(ax, label_x, label_y, label_text,
                x_scatter=label_x, y_scatter=label_y,
                x_lines=x_lines, y_lines=y_lines,
                textsize=label_size, textcolor=label_color,
                draw_lines=True, linecolor="grey", linewidth=0.4,
                margin=0.01, min_distance=0.015, max_distance=0.18,
                avoid_label_lines_overlap=True)

    # --- RIGHT: Authority vs Care surgical view ---
    ax2 = axs[1]
    base_auth = base_logit["Authority"]; base_care = base_logit["Care"]
    lx2: list[float] = []; ly2: list[float] = []
    lt2: list[str] = []; lc2: list[str] = []; ls2: list[int] = []
    for (method, sign), vec in zip(labels, X):
        if method == "base":
            ax2.scatter(0, 0, s=180, c="black", marker="*", zorder=5)
            lx2.append(0); ly2.append(0); lt2.append("base"); lc2.append("black"); ls2.append(10)
        elif method == "human":
            continue
        else:
            color = "C0" if sign == "POS" else "C3"
            dAuth = vec[FOUNDATION_ORDER.index("Authority")] - base_auth
            dCare = vec[FOUNDATION_ORDER.index("Care")] - base_care
            ax2.scatter(dAuth, dCare, s=40, c=color, alpha=0.7)
            lx2.append(dAuth); ly2.append(dCare)
            lt2.append(f"{method}[{sign[0]}]"); lc2.append(color); ls2.append(7)
    lo, hi = ax2.get_xlim()
    ax2.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.4, label="broad-suppression diag")
    ax2.axhline(0, color="grey", lw=0.4); ax2.axvline(0, color="grey", lw=0.4)
    ax2.set_xlabel("ΔlogitAuthority (target axis; want < 0)")
    ax2.set_ylabel("ΔlogitCare (off-target; want ≈ 0)")
    ax2.set_title("Surgical view: ΔAuth vs ΔCare\n(diagonal = broad suppression, below = surgical)")
    ax2.legend(loc="best", fontsize=8)
    ax2.relim(); ax2.autoscale_view()
    ta.allocate(ax2, lx2, ly2, lt2,
                x_scatter=lx2, y_scatter=ly2,
                textsize=ls2, textcolor=lc2,
                draw_lines=True, linecolor="grey", linewidth=0.4,
                margin=0.01, min_distance=0.015, max_distance=0.18)

    fig.suptitle(f"steering-lite moral map — {sweep_dir.name}", y=1.0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    logger.info(f"wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, default=Path("outputs/tinymfv_sweep"))
    p.add_argument("--vignettes", default="airisk")
    p.add_argument("--map-out", type=Path, default=None,
                   help="PNG path for moral map. Defaults to {sweep_dir}/moral_map.png")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip running aggregate_flips.py (use if already aggregated)")
    args = p.parse_args()

    map_out = args.map_out or (args.sweep_dir / "moral_map.png")

    print(f"\n## Base model vs humans — {args.sweep_dir.name}\n")
    print(base_vs_humans_table(args.sweep_dir, args.vignettes))

    print("\n## Moral map\n")
    moral_map(args.sweep_dir, args.vignettes, map_out)
    print(f"![moral map]({map_out})\n")

    if not args.skip_aggregate:
        print("\n## SI / Δlogit / sign agreement (loading-weighted)\n")
        cmd = [sys.executable, str(Path(__file__).parent / "aggregate_flips.py"),
               "--sweep-dir", str(args.sweep_dir)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
