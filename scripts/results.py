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

    skip = {"bare.json"}
    for jp in sorted(sweep_dir.glob("*.json")):
        if jp.name in skip or "stale" in jp.name or jp.name == "runs.jsonl":
            continue
        try:
            d = _orjson_loads(jp)
        except Exception:
            continue
        method = jp.stem
        for sign, lab in [("+", "POS"), ("-", "NEG")]:
            prof = _profile_for_method(d, base_logit, sign)
            if prof is None:
                continue
            vec = np.array([prof[f] for f in FOUNDATION_ORDER])
            entities.append((method, lab, vec))

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

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None
        logger.warning("adjustText not installed; labels will overlap. pip install adjustText")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- LEFT: PCA scatter ---
    ax = axs[0]
    base_idx = next(i for i, (m, _) in enumerate(labels) if m == "base")
    base_pc = (pc1[base_idx], pc2[base_idx])
    texts = []
    for i, (method, sign) in enumerate(labels):
        if method == "base":
            ax.scatter(pc1[i], pc2[i], s=180, c="black", marker="*", zorder=5, label="base")
            texts.append(ax.text(pc1[i], pc2[i], "base", fontsize=10, weight="bold"))
        else:
            color = "C0" if sign == "POS" else "C3"
            ax.scatter(pc1[i], pc2[i], s=40, c=color, alpha=0.7)
            texts.append(ax.text(pc1[i], pc2[i], f"{method}[{sign[0]}]",
                                 fontsize=7, color=color, alpha=0.9))
    if human_pc is not None:
        ax.scatter(human_pc[0], human_pc[1], s=220, c="red", marker="X", zorder=6,
                   label="human ref (projected)")
        texts.append(ax.text(human_pc[0], human_pc[1], "human",
                             fontsize=10, weight="bold", color="red"))
    # Loadings arrows for foundations, anchored at BASE position (more intuitive
    # than the centroid-anchored biplot convention: "from where the model is now,
    # increasing X moves you this way").
    loads = Vt.T[:, :2] * S[:2]
    scale = 0.45 * max(np.abs(pcs[:, :2]).max(), 1.0) / max(np.abs(loads).max(), 1.0)
    foundation_texts = []
    for j, f in enumerate(FOUNDATION_ORDER):
        dx, dy = loads[j, 0] * scale, loads[j, 1] * scale
        ax.arrow(base_pc[0], base_pc[1], dx, dy,
                 head_width=0.05, color="grey", alpha=0.5, length_includes_head=True)
        foundation_texts.append(ax.text(base_pc[0] + dx * 1.08, base_pc[1] + dy * 1.08,
                                        FOUNDATION_SHORT[f],
                                        fontsize=9, color="grey", weight="bold",
                                        ha="center", va="center"))
    if adjust_text is not None:
        adjust_text(texts, ax=ax, expand_points=(1.2, 1.4), expand_text=(1.1, 1.2),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.4, alpha=0.5))
    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title("Moral map (PCA on z-scored 7-foundation profile)")

    # --- RIGHT: Authority vs Care surgical view ---
    # X = ΔAuthority (target axis), Y = ΔCare (off-target). Below diagonal = surgical.
    ax2 = axs[1]
    base_auth = base_logit["Authority"]; base_care = base_logit["Care"]
    texts2 = []
    for (method, sign), vec in zip(labels, X):
        if method == "base":
            ax2.scatter(0, 0, s=180, c="black", marker="*", zorder=5)
            texts2.append(ax2.text(0, 0, "base", fontsize=10, weight="bold"))
        elif method == "human":
            continue
        else:
            color = "C0" if sign == "POS" else "C3"
            dAuth = vec[FOUNDATION_ORDER.index("Authority")] - base_auth
            dCare = vec[FOUNDATION_ORDER.index("Care")] - base_care
            ax2.scatter(dAuth, dCare, s=40, c=color, alpha=0.7)
            texts2.append(ax2.text(dAuth, dCare, f"{method}[{sign[0]}]",
                                   fontsize=7, color=color, alpha=0.9))
    lo, hi = ax2.get_xlim()
    ax2.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.4, label="broad-suppression diag")
    ax2.axhline(0, color="grey", lw=0.4); ax2.axvline(0, color="grey", lw=0.4)
    ax2.set_xlabel("ΔlogitAuthority (target axis; want < 0)")
    ax2.set_ylabel("ΔlogitCare (off-target; want ≈ 0)")
    ax2.set_title("Surgical view: ΔAuth vs ΔCare\n(diagonal = broad suppression, below = surgical)")
    ax2.legend(loc="best", fontsize=8)
    if adjust_text is not None:
        adjust_text(texts2, ax=ax2, expand_points=(1.2, 1.4), expand_text=(1.1, 1.2),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.4, alpha=0.5))

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
