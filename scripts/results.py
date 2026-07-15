"""README-ready tables + moral map from a sweep output dir.

Headline metric lives in ONE place, `moralmaps.metrics` (imported, not
reforked): `gated_selectivity` (primary, continuous) + `si_flips` (secondary,
behavioral). This script feeds them the per-method `raw_logratios`
(per-(vid,cond) clr(score)[f]) + `mean_pmass_allowed` each sweep JSON stores,
with persona-aligned sign selection.

Emits:
  - Base model clr per foundation, t-stat, vs human calibrated wrongness.
  - Gated-selectivity table (sel_gated, on, off, coherence, si_flips, CI)
    + Δclr-with-σ table.
  - Moral map: 7-foundation absolute-clr profiles, PCA → 2D scatter of
    base + each method (persona-aligned sign) + human reference.

Usage:
    uv run --extra benchmark python scripts/results.py \\
        --sweep-dir outputs/tinymfv_sweep_4b_fc --vignettes classic

Or via justfile: `just results outputs/tinymfv_sweep_4b_fc`.

History: the flip-based SI (`si_per_foundation`) was retired 2026-07 in favour
of the continuous `gated_selectivity` shared with j-steer -- clr's 0 is not a
decision boundary, so per-foundation flip counts across it were meaningless.
-- Claude
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from loguru import logger
from tabulate import tabulate

from moralmaps import gated_selectivity, si_flips, OFF_WEIGHT
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, FOUNDATION_SHORT,
    baseline_clr_per_foundation, dclr_per_foundation,
)

# On-axis intent for the shared metric, lowercase probe names as in raw_logratios(=clr).
INTENT = {"authority": -1, "care": +1}  # persona: Auth↓, Care↑

# vignette calibrated_<F> loading columns (human attribution %) for the
# base-vs-humans reference table + moral-map human point.
LOADING_COL_FOR_FOUNDATION = {
    "Care": "calibrated_Care", "Sanctity": "calibrated_Sanctity",
    "Authority": "calibrated_Authority", "Loyalty": "calibrated_Loyalty",
    "Fairness": "calibrated_Fairness", "Liberty": "calibrated_Liberty",
    "Social Norms": "calibrated_SocialNorms",
}


def _orjson_loads(p: Path):
    try:
        import orjson
        return orjson.loads(p.read_bytes())
    except ImportError:
        return json.loads(p.read_text())


# ── load sweep dir → canonical per-method measurements ────────────────────────

def _pmass(report: dict) -> float:
    """mean_pmass_allowed (coherence) for a sweep-JSON sub-report; NaN if absent (pre-metric sweeps)."""
    v = report.get("mean_pmass_allowed")
    return float(v) if v is not None else float("nan")


def _load_sweep(sweep_dir: Path, bare_name: str = "bare.json") -> tuple[dict, dict]:
    """Returns (bare_report, methods) where methods[m] carries the persona-aligned
    sign, Δclr dicts, the shared gated_selectivity + si_flips, abs-clr profiles."""
    bare = _orjson_loads(sweep_dir / bare_name)
    base_abs = baseline_clr_per_foundation(bare)
    bare_pmass = _pmass(bare)

    methods: dict[str, dict] = {}
    for f in sorted(sweep_dir.glob("*.json")):
        if f.name == bare_name:
            continue
        d = _orjson_loads(f)
        method = f.stem
        if "pos" in d and "neg" in d:
            pos, neg = d["pos"], d["neg"]
            dl_pos = dclr_per_foundation(bare, pos)
            dl_neg = dclr_per_foundation(bare, neg)
            # Persona-aligned direction = the one that moves ΔAuth most downward.
            a_pos = dl_pos["Authority"]["mean"]
            a_neg = dl_neg["Authority"]["mean"]
            if math.isnan(a_neg) or (not math.isnan(a_pos) and a_pos <= a_neg):
                sign, aligned, opp, dl = +1, pos, neg, dl_pos
            else:
                sign, aligned, opp, dl = -1, neg, pos, dl_neg
            # Shared metric: on = signed Δ(aligned − opp) on intent axes, off = the rest.
            sel = gated_selectivity(
                aligned["raw_logratios"], opp["raw_logratios"], INTENT,
                pmass_pos=_pmass(aligned), pmass_neg=_pmass(opp), pmass_base=bare_pmass)
            sif = si_flips(aligned["raw_logratios"], opp["raw_logratios"], INTENT)
            methods[method] = {
                "bidirectional": True, "sign": sign,
                "calibrated_C": d.get("calibrated_C"),
                "dclr": dl, "sel": sel, "si_flips": sif["si_flips"],
                "abs_clr": {fo: base_abs[fo]["mean"] + dl[fo]["mean"]
                              for fo in FOUNDATION_ORDER},
            }
        else:  # single-direction baseline (prompt_only): contrast against bare.
            dl = dclr_per_foundation(bare, d)
            sel = gated_selectivity(
                d["raw_logratios"], bare["raw_logratios"], INTENT,
                pmass_pos=_pmass(d), pmass_neg=bare_pmass, pmass_base=bare_pmass)
            sif = si_flips(d["raw_logratios"], bare["raw_logratios"], INTENT)
            methods[method] = {
                "bidirectional": False, "sign": +1,
                "calibrated_C": d.get("coeff") or d.get("calibrated_C"),
                "dclr": dl, "sel": sel, "si_flips": sif["si_flips"],
                "abs_clr": {fo: base_abs[fo]["mean"] + dl[fo]["mean"]
                              for fo in FOUNDATION_ORDER},
            }
    return bare, methods


# ── human reference (loading-weighted, unchanged by the refactor) ─────────────

def _human_per_foundation(vignettes_name: str) -> dict[str, dict[str, float]]:
    """Loading-weighted human calibrated_wrongness (1-5) per foundation."""
    from moralmaps import load_vignettes
    vigs = load_vignettes(vignettes_name)
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
            if L > 0:
                weights_by_f[f].append(L); xs_by_f[f].append(float(cw))
    out: dict[str, dict[str, float]] = {}
    for f in FOUNDATION_ORDER:
        ws = np.array(weights_by_f[f], dtype=float)
        xs = np.array(xs_by_f[f], dtype=float)
        if ws.sum() <= 0:
            out[f] = {"mean": float("nan"), "std": float("nan"), "wsum": 0.0, "n": 0}
            continue
        mean = (ws * xs).sum() / ws.sum()
        var = (ws * (xs - mean) ** 2).sum() / ws.sum()
        out[f] = {"mean": float(mean), "std": float(np.sqrt(var)),
                  "wsum": float(ws.sum()), "n": int(ws.size)}
    return out


def base_vs_humans_table(bare: dict, vignettes_name: str) -> str:
    base_abs = baseline_clr_per_foundation(bare)
    humans = _human_per_foundation(vignettes_name)
    rows = []
    for f in FOUNDATION_ORDER:
        r = base_abs[f]
        m_mean, m_std, n = r["mean"], r["std"], r["n"]
        sem = r.get("sem", float("nan"))
        t = (m_mean / sem) if sem and not math.isnan(sem) and sem > 0 else float("nan")
        prob = 1.0 / (1.0 + math.exp(-m_mean)) if not math.isnan(m_mean) else float("nan")
        h = humans[f]
        h_str = f"{h['mean']:.2f}±{h['std']:.2f}" if not math.isnan(h["mean"]) else "n/a"
        rows.append([
            FOUNDATION_SHORT[f], f"{m_mean:+.2f}±{m_std:.2f}",
            f"{prob*100:.0f}%", f"{t:+.1f}" if not math.isnan(t) else "n/a",
            h_str, f"{n}", f"{h['wsum']:.1f}",
        ])
    return tabulate(rows, headers=["foundation", "model clr±σ", "p(wrong)",
                                   "t-stat", "human wrong (1-5)±σ", "n", "Σw_h"],
                    tablefmt="pipe", floatfmt="+.2f")


# ── SI / Δclr / sign tables (canonical foundations source) ──────────────────

def _fmt(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{digits}f}"


def print_tables(methods: dict) -> None:
    def tag(m):
        meta = methods[m]
        return f"[{'+' if meta['sign'] == 1 else '-'}]" if meta["bidirectional"] else "[ ]"

    # --- Gated selectivity table (shared moralmaps.metrics) ---
    sel_rows = []
    for m in methods:
        s = methods[m]["sel"]
        sel_rows.append([f"{m}{tag(m)}", s["sel_gated"], s["on"], s["off"],
                         s["coherence"], methods[m]["si_flips"],
                         f"[{s['ci_lo']:+.2f},{s['ci_hi']:+.2f}]"])
    sel_rows.sort(key=lambda r: (-1e9 if (isinstance(r[1], float) and math.isnan(r[1])) else r[1]),
                  reverse=True)
    sel_rows = [[r[0], _fmt(r[1]), _fmt(r[2]), _fmt(r[3]), _fmt(r[4], 3), _fmt(r[5]), r[6]]
                for r in sel_rows]
    print("\n## Gated selectivity (headline; shared with j-steer via moralmaps.metrics)\n")
    print(tabulate(sel_rows, headers=["method", "sel_gated", "on", "off", "coh", "si_flips", "CI95"],
                   tablefmt="pipe"))
    print(f"\nsel_gated = (on − {OFF_WEIGHT}·off)·coh²; on = mean signed Δclr on {list(INTENT)} "
          "(Auth↓,Care↑), off = mean|Δclr| over the other 5, coh = min(1, min-arm pmass / base). "
          "si_flips = signed argmax pick-rate change (behavioral, bounded). CI95 = 2000× row bootstrap.")

    # --- Δclr table ---
    dl_rows = []
    for m in methods:
        meta = methods[m]
        ax = -meta["dclr"]["Authority"]["mean"]
        row = [f"{m}{tag(m)}", _fmt(ax)]
        for fo in FOUNDATION_ORDER:
            r = meta["dclr"][fo]
            row.append(f"{r['mean']:+.2f}±{r['std']:.2f}" if not math.isnan(r["mean"]) else "n/a")
        dl_rows.append((ax, row))
    dl_rows.sort(key=lambda t: (-1e9 if math.isnan(t[0]) else -t[0]))
    print(f"\n## Δclr per foundation (mean ± σ, sorted by axis_Δ = −ΔAuth)\n")
    print(tabulate([r for _, r in dl_rows],
                   headers=["method", "axis_Δ(Auth)"] + [f"Δ{FOUNDATION_SHORT[f]}" for f in FOUNDATION_ORDER],
                   tablefmt="pipe"))
    print("Δclr > 0 = wrongness went UP. intent[Auth]=−1 wants ΔAuth<0; intent[Care]=+1 wants ΔCare>0.")


# ── moral map (fed from canonical absolute clr) ───────────────────────────────

def moral_map(bare: dict, methods: dict, vignettes_name: str, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textalloc as ta

    base_abs = baseline_clr_per_foundation(bare)
    base_clr = {f: base_abs[f]["mean"] for f in FOUNDATION_ORDER}

    entities: list[tuple[str, str, np.ndarray]] = [
        ("base", "model", np.array([base_clr[f] for f in FOUNDATION_ORDER]))
    ]
    for method, meta in sorted(methods.items()):
        sign_lab = "POS" if meta["sign"] == 1 else "NEG"
        vec = np.array([meta["abs_clr"][f] for f in FOUNDATION_ORDER])
        entities.append((method, sign_lab, vec))

    human_wrong = _human_per_foundation(vignettes_name)
    human_vec_raw = np.array([human_wrong[f]["mean"] for f in FOUNDATION_ORDER])
    has_human = not np.any(np.isnan(human_vec_raw))

    keep = [(m, s, v) for (m, s, v) in entities if not np.any(np.isnan(v))]
    dropped = [(m, s) for (m, s, v) in entities if np.any(np.isnan(v))]
    if dropped:
        logger.warning(f"moral_map: dropping NaN entities: {dropped}")

    labels = [(m, s) for (m, s, _) in keep]
    X = np.stack([v for (_, _, v) in keep])
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=0, keepdims=True)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    Xz = (X - mu) / sigma
    Xc = Xz - Xz.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U * S
    pc1, pc2 = pcs[:, 0], pcs[:, 1]
    var = (S ** 2) / (S ** 2).sum()

    human_pc = None
    if has_human:
        h_z = (human_vec_raw - mu.squeeze(0)) / sigma.squeeze(0)
        h_c = h_z - Xz.mean(axis=0)
        h_pc = h_c @ Vt.T
        human_pc = (float(h_pc[0]), float(h_pc[1]))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axs[0]
    base_idx = next(i for i, (m, _) in enumerate(labels) if m == "base")

    label_x: list[float] = []; label_y: list[float] = []
    label_text: list[str] = []; label_color: list[str] = []; label_size: list[int] = []
    x_lines: list[list[float]] = []; y_lines: list[list[float]] = []

    ax.scatter(pc1[base_idx], pc2[base_idx], s=180, c="black", marker="*", zorder=5, label="base")
    label_x.append(pc1[base_idx]); label_y.append(pc2[base_idx])
    label_text.append("base"); label_color.append("black"); label_size.append(10)
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

    loads = Vt.T[:, :2] * S[:2]
    model_extent = max(np.abs(pcs[:, :2]).max(), 1.0)
    compass_scale = 1.3 * model_extent / max(np.abs(loads).max(), 1.0)
    ax.scatter(0.0, 0.0, s=20, c="grey", marker="+", alpha=0.5)
    compass_tip_x: list[float] = []; compass_tip_y: list[float] = []
    for j, f in enumerate(FOUNDATION_ORDER):
        dx, dy = loads[j, 0] * compass_scale, loads[j, 1] * compass_scale
        ax.arrow(0.0, 0.0, dx, dy, head_width=0.04 * model_extent, color="grey",
                 alpha=0.55, length_includes_head=True)
        tx, ty = dx * 1.08, dy * 1.08
        ax.text(tx, ty, FOUNDATION_SHORT[f], color="grey", fontsize=8,
                ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top",
                alpha=0.9, zorder=3)
        x_lines.append([0.0, tx]); y_lines.append([0.0, ty])
        compass_tip_x.append(tx); compass_tip_y.append(ty)

    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title("Moral map (PCA on z-scored 7-foundation profile)")
    all_x = label_x + compass_tip_x; all_y = label_y + compass_tip_y
    pad_x = 0.08 * (max(all_x) - min(all_x)); pad_y = 0.12 * (max(all_y) - min(all_y))
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ta.allocate(ax, label_x, label_y, label_text, x_scatter=label_x, y_scatter=label_y,
                x_lines=x_lines, y_lines=y_lines, textsize=label_size, textcolor=label_color,
                draw_lines=True, linecolor="grey", linewidth=0.4,
                margin=0.01, min_distance=0.015, max_distance=0.18,
                avoid_label_lines_overlap=True)

    OFF_AXIS = "Social Norms"
    ax2 = axs[1]
    base_auth = base_clr["Authority"]; base_off = base_clr[OFF_AXIS]
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
            lx2.append(dAuth); ly2.append(dOff); lt2.append(method); lc2.append("C0"); ls2.append(7)
    lo, hi = ax2.get_xlim()
    ax2.plot([lo, hi], [lo, hi], "--", color="grey", alpha=0.4, label="broad-suppression diag")
    ax2.axhline(0, color="grey", lw=0.4); ax2.axvline(0, color="grey", lw=0.4)
    ax2.set_xlabel("Δclr Authority (target; want < 0)")
    ax2.set_ylabel(f"Δclr {FOUNDATION_SHORT[OFF_AXIS]} (off-target ⊥ Auth in humans; want ≈ 0)")
    ax2.set_title(f"Surgical view: ΔAuth vs Δ{FOUNDATION_SHORT[OFF_AXIS]}\n(off-diagonal = surgical, diagonal = broad)")
    ax2.legend(loc="best", fontsize=8)
    ax2.relim(); ax2.autoscale_view()
    ta.allocate(ax2, lx2, ly2, lt2, x_scatter=lx2, y_scatter=ly2, textsize=ls2, textcolor=lc2,
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
    p.add_argument("--vignettes", default="classic")
    p.add_argument("--map-out", type=Path, default=None)
    p.add_argument("--no-map", action="store_true", help="skip the moral-map PNG")
    args = p.parse_args()

    bare, methods = _load_sweep(args.sweep_dir)

    print(f"\n## Base model vs humans — {args.sweep_dir.name}\n")
    print(base_vs_humans_table(bare, args.vignettes))

    print_tables(methods)

    if not args.no_map:
        map_out = args.map_out or (args.sweep_dir / "moral_map.png")
        print("\n## Moral map\n")
        moral_map(bare, methods, args.vignettes, map_out)
        print(f"![moral map]({map_out})\n")


if __name__ == "__main__":
    main()
