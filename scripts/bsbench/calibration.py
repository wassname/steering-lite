"""Does C0 (the dose at 1 nat RMS KL on the calibration prompts) predict where each walk stops being coherent?

Per walk and side: the last healthy dose before the confirmed boundary (2 unhealthy rungs in a row),
its ratio to C0, and the RMS KL there. Rows for BS-bench (dev) and the held-out AlpacaEval set (ood),
plus, from points.json when it exists, the last judge-admissible dose (the health rule misses short
gibberish, e.g. vjp_delta +C C=1.26 "Heat shit fast like damn shit yeah.") and the Pareto-best dose.

    python calibration.py            # writes outputs/bsbench/results/calibration.md
"""

import json
from pathlib import Path
from statistics import median

from judge import default_model_dir

ROOT = Path(__file__).resolve().parents[2]


def last_healthy(walk: dict, side: str) -> dict | None:
    """Rung just before the unhealthy streak that set the boundary (streak = 2, so boundary - 2)."""
    boundary = walk["state"][side]["boundary"]
    if boundary is None:
        return None
    return walk["rungs"][boundary - 2] if boundary >= 2 else None


def rows_for(model_dir: Path, cohort: str) -> list[dict]:
    rows = []
    for path in sorted((model_dir / "walks").glob(f"*_{cohort}.json")):
        walk = json.loads(path.read_text())
        if walk["status"] != "COMPLETE" or "c0" not in walk:
            continue
        for side in ("+C", "-C"):
            rung = last_healthy(walk, side)
            rows.append({
                "method": walk["method"], "seed": walk["seed"], "cohort": cohort, "side": side, "c0": walk["c0"],
                "last_C": None if rung is None else rung["coefficient"],
                "kl": None if rung is None else rung["kl_rms"][side],
            })
    return rows


def main() -> None:
    model_dir = default_model_dir()
    rows = rows_for(model_dir, "dev") + rows_for(model_dir, "ood")
    best, admissible = {}, {}
    points_path = ROOT / "outputs/bsbench/results/dev/points.json"
    if points_path.exists():
        site = json.loads(points_path.read_text())
        for summary in site["summary"]:
            for side, point in summary["best"].items():
                if point:
                    best[summary["method"], side] = point["C"]
        for point in site["points"]:
            if point["admissible"]:
                key = (point["method"], point["side"])
                admissible.setdefault(key, []).append(point)
    lines = [
        "C0 = dose where RMS KL reaches 1 nat on the calibration prompts (walk starts at C0/8). "
        "last healthy = last dose before 2 unhealthy rungs in a row (health rule only, no judge). "
        "Random: median over seeds.", "",
        "| method | side | C0 | dev last healthy (×C0) | dev KL there | dev last admissible, judged (×C0) | ood last healthy (×C0) | ood KL there | dev Pareto-best C (×C0) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["side"]), []).append(row)

    def cell(group, cohort):
        picked = [r for r in group if r["cohort"] == cohort and r["last_C"] is not None]
        if not picked:
            return "—", "—"
        ratio = median(r["last_C"] / r["c0"] for r in picked)
        return f"{median(r['last_C'] for r in picked):.3g} ({ratio:.2f})", f"{median(r['kl'] for r in picked):.2f}"

    for (method, side), group in sorted(groups.items()):
        c0 = median(r["c0"] for r in group)
        dev, dev_kl = cell(group, "dev")
        ood, ood_kl = cell(group, "ood")
        pareto = best.get((method, side))
        pareto_cell = "—" if pareto is None else f"{pareto:.3g} ({pareto / c0:.2f})"
        # random: max admissible C per seed, then median; others have one seed
        per_seed = {}
        for point in admissible.get((method, side), []):
            per_seed[point["seed"]] = max(per_seed.get(point["seed"], 0.0), point["C"])
        judged = "—" if not per_seed else f"{median(per_seed.values()):.3g} ({median(per_seed.values()) / c0:.2f})"
        lines.append(f"| {method} | {side} | {c0:.3g} | {dev} | {dev_kl} | {judged} | {ood} | {ood_kl} | {pareto_cell} |")
    out = ROOT / "outputs/bsbench/results/calibration.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# RMS-KL calibration vs walk breakdown\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
