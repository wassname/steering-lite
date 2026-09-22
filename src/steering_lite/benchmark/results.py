"""Render one auditable BS-bench report from a completed full-sweep summary."""
from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from .cache import content_key, save_json
from .judge import score_pair
from .pipeline import METHODS
from .validation import comparison_id, disagreements


def _reasons(record: dict) -> list[str]:
    reasons = record.get("reasons", [])
    if not isinstance(reasons, list):
        raise ValueError("health records require a reasons list")
    return reasons


def _complete_pair(records: list[dict], *, comparison: str) -> tuple[list[dict], list[dict]]:
    aware = [record for record in records if not record["blind"]]
    blind = [record for record in records if record["blind"]]
    if len(aware) != 4 or len(blind) != 2:
        raise ValueError(f"comparison {comparison} needs two-pass AB/BA aware and one-pass AB/BA blind evidence")
    if {(record["order"], record["pass"]) for record in aware} != {("AB", 0), ("AB", 1), ("BA", 0), ("BA", 1)}:
        raise ValueError(f"comparison {comparison} needs both aware passes in AB and BA order")
    if {(record["order"], record["pass"]) for record in blind} != {("AB", 0), ("BA", 0)}:
        raise ValueError(f"comparison {comparison} needs one blind pass in AB and BA order")
    return aware, blind


def _point(
    *,
    method: str,
    phase: str,
    case_id: str,
    coefficient: float | None,
    comparisons: list[str],
    records_by_comparison: dict[str, list[dict]],
    health: list[dict],
) -> dict:
    aware = []
    blind = []
    for comparison in comparisons:
        comparison_aware, comparison_blind = _complete_pair(records_by_comparison[comparison], comparison=comparison)
        aware.extend(comparison_aware)
        blind.extend(comparison_blind)
    effects = [score_pair(record["response"], record["order"], record["side"]) for record in aware]
    directed_effect = sum(effect["effect"] for effect in effects) / len(effects)
    absolute_off_target_effect = sum(abs(effect["off_axis_perturbation"]) for effect in effects) / len(effects)
    point = {
        "method": method,
        "phase": phase,
        "case_id": case_id,
        "coefficient": coefficient,
        "directed_effect": directed_effect,
        "absolute_off_target_effect": absolute_off_target_effect,
        "dose_score": directed_effect - 4 * absolute_off_target_effect,
        "coherent": all(not _reasons(record) for record in health),
        "health": health,
        "aware": aware,
        "blind": blind,
        "paired_disagreements": disagreements([*aware, *blind]),
        "examples": [
            {"question_id": record["question_id"], "question_number": record["question_number"]}
            for record in aware[::2]
        ],
    }
    point["point_id"] = content_key({
        "method": method,
        "phase": phase,
        "case_id": case_id,
        "coefficient": coefficient,
        "comparisons": comparisons,
        "health": health,
    })[:16]
    return point


def _records_by_comparison(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["comparison_id"]].append(record)
    return grouped


def _candidate_points(method: str, condition: dict) -> list[dict]:
    judgment = condition.get("candidate_judgments")
    health_result = condition.get("candidate_health")
    if not isinstance(judgment, dict) or not isinstance(health_result, dict):
        raise ValueError(f"{method} is missing candidate judgments or health")
    aware_result = condition.get("candidate_aware")
    blind_result = condition.get("candidate_blind")
    if not isinstance(aware_result, dict) or not isinstance(blind_result, dict):
        raise ValueError(f"{method} is missing persisted candidate aware or blind records")
    aware = aware_result.get("records")
    blind = blind_result.get("records")
    if aware != judgment.get("aware") or blind != judgment.get("blind"):
        raise ValueError(f"{method} candidate judgment records disagree with persisted local stages")
    if not isinstance(aware, list) or not isinstance(blind, list):
        raise ValueError(f"{method} candidate judgments require record lists")
    all_records = [*aware, *blind]
    grouped = _records_by_comparison(all_records)
    health_by_coefficient = health_result.get("records")
    if not isinstance(health_by_coefficient, dict):
        raise ValueError(f"{method} candidate health must be keyed by coefficient")
    coefficients = condition["candidate"]["candidate_coefficients"]
    points = []
    for coefficient in coefficients:
        comparison_ids = [
            comparison_id({"question_id": question_id, "method": method, "coefficient": float(coefficient), "side": "+C"})
            for question_id in sorted({record["question_id"] for record in all_records})
        ]
        if any(comparison not in grouped for comparison in comparison_ids):
            raise ValueError(f"{method} candidate coefficient {coefficient} is missing judgment evidence")
        health = health_by_coefficient.get(str(float(coefficient)))
        if not isinstance(health, dict):
            raise ValueError(f"{method} candidate coefficient {coefficient} is missing health")
        points.append(_point(
            method=method,
            phase="candidate",
            case_id="calibration",
            coefficient=float(coefficient),
            comparisons=comparison_ids,
            records_by_comparison=grouped,
            health=[health],
        ))
    return points


def _final_points(method: str, condition: dict) -> list[dict]:
    final = condition.get("final")
    final_judgments = condition.get("final_judgments")
    final_health = condition.get("final_health")
    if not isinstance(final, dict) or not isinstance(final_judgments, dict) or not isinstance(final_health, dict):
        raise ValueError(f"{method} is missing final generation, judgments, or health")
    plan = final.get("executable_generation_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError(f"{method} final generation has no executable measured plan")
    health_records = final_health.get("records")
    if not isinstance(health_records, list):
        raise ValueError(f"{method} final health has no attributable records")
    health_by_key = {
        (record.get("case_id"), record.get("prompt_id"), float(record["coefficient"])): record
        for record in health_records
    }
    expected_health = {(item["case_id"], item["prompt_id"], float(item["coefficient"])) for item in plan}
    if set(health_by_key) != expected_health or len(health_by_key) != len(health_records):
        raise ValueError(f"{method} final health does not exactly cover the executable plan")
    aware_result = condition.get("final_aware")
    blind_result = condition.get("final_blind")
    if not isinstance(aware_result, dict) or not isinstance(blind_result, dict):
        raise ValueError(f"{method} is missing persisted final aware or blind records")
    aware = aware_result.get("records")
    blind = blind_result.get("records")
    if aware != final_judgments.get("aware") or blind != final_judgments.get("blind"):
        raise ValueError(f"{method} final judgment records disagree with persisted local stages")
    if not isinstance(aware, list) or not isinstance(blind, list):
        raise ValueError(f"{method} final judgments require record lists")
    grouped = _records_by_comparison([*aware, *blind])
    plan_groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for item in plan:
        plan_groups[item["case_id"], float(item["coefficient"])].append(item)
    points = []
    for (case_id, coefficient), items in sorted(plan_groups.items()):
        comparison_ids = [
            comparison_id({"question_id": item["prompt_id"], "method": method, "coefficient": coefficient, "side": "+C"})
            for item in items
        ]
        if any(comparison not in grouped for comparison in comparison_ids):
            raise ValueError(f"{method} final {case_id} coefficient {coefficient} is missing judgment evidence")
        points.append(_point(
            method=method,
            phase="final",
            case_id=case_id,
            coefficient=coefficient,
            comparisons=comparison_ids,
            records_by_comparison=grouped,
            health=[health_by_key[item["case_id"], item["prompt_id"], coefficient] for item in items],
        ))
    return points


def _direct_point(method: str, condition: dict) -> dict:
    if method == "bare":
        health = condition.get("health", {}).get("records")
        if not isinstance(health, list) or not health:
            raise ValueError("bare condition is missing identified health")
        point = {
            "method": "bare",
            "phase": "baseline",
            "case_id": "dev",
            "coefficient": None,
            "directed_effect": 0.0,
            "absolute_off_target_effect": 0.0,
            "dose_score": 0.0,
            "coherent": all(not _reasons(record) for record in health),
            "health": health,
            "aware": [],
            "blind": [],
            "paired_disagreements": [],
            "examples": [{"question_id": record["question_id"], "question_number": index + 1} for index, record in enumerate(health)],
        }
        point["point_id"] = content_key({"method": "bare", "health": health})[:16]
        return point
    judgments = condition.get("judgments")
    aware_result = condition.get("aware")
    blind_result = condition.get("blind")
    health = condition.get("health", {}).get("records")
    if not isinstance(judgments, dict) or not isinstance(aware_result, dict) or not isinstance(blind_result, dict) or not isinstance(health, list) or not health:
        raise ValueError("prompting condition is missing complete judgments or health")
    aware = aware_result.get("records")
    blind = blind_result.get("records")
    if aware != judgments.get("aware") or blind != judgments.get("blind"):
        raise ValueError("prompting judgments disagree with persisted local stages")
    if not isinstance(aware, list) or not isinstance(blind, list):
        raise ValueError("prompting judgments require record lists")
    grouped = _records_by_comparison([*aware, *blind])
    return _point(
        method="prompting",
        phase="direct",
        case_id="dev",
        coefficient=None,
        comparisons=sorted(grouped),
        records_by_comparison=grouped,
        health=health,
    )


def normalize_summary(summary: dict) -> dict:
    """Normalize one complete cached full sweep into the only report point source."""
    if summary.get("schema") != "bsbench-run-summary-v1":
        raise ValueError("results require a current full run summary")
    identity = summary.get("identity")
    if not isinstance(identity, dict) or summary.get("identity_sha256") != content_key(identity):
        raise ValueError("results require a complete matching run identity")
    if summary.get("methods") != list(METHODS) or identity.get("methods") != list(METHODS):
        raise ValueError("results require the complete canonical method order, not a recovery subset")
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(METHODS) or len(conditions) != len(METHODS):
        raise ValueError("results require exactly one current result for every canonical method")
    paid = {result.get("paid_execution_enabled") for result in conditions.values()}
    if len(paid) != 1 or not isinstance(next(iter(paid)), bool):
        raise ValueError("results refuse mixed or missing execution identities")
    points = [_direct_point("bare", conditions["bare"]), _direct_point("prompting", conditions["prompting"])]
    calibration_points = []
    transfer_points = []
    transfer_predictions = []
    for method in METHODS[2:]:
        calibration_points.extend(_candidate_points(method, conditions[method]))
        predictions = conditions[method].get("transfer_prediction", {}).get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 5:
            raise ValueError(f"{method} is missing evaluation and transfer RMS-KL predictions")
        transfer_predictions.extend(prediction for prediction in predictions if prediction["case"]["case_id"] != "bsbench-v2-evaluation")
        for point in _final_points(method, conditions[method]):
            if point["case_id"] == "bsbench-v2-evaluation":
                points.append(point)
            else:
                transfer_points.append(point)
    expected_methods = set(METHODS[2:])
    if {point["method"] for point in points if point["method"] not in {"bare", "prompting"}} != expected_methods:
        raise ValueError("results require completed numbered evaluation points for every activation method")
    artifact = {
        "schema": "bsbench-measured-points-v2",
        "run_identity": identity,
        "run_identity_sha256": summary["identity_sha256"],
        "non_experimental": not next(iter(paid)),
        "points": points,
        "calibration_points": calibration_points,
        "transfer_points": transfer_points,
        "transfer_predictions": transfer_predictions,
    }
    artifact["points_sha256"] = content_key({"points": points})
    return artifact


def _pareto(points: list[dict]) -> list[dict]:
    coherent = [point for point in points if point["coherent"]]
    return [
        point for point in coherent
        if not any(
            other["directed_effect"] >= point["directed_effect"]
            and other["absolute_off_target_effect"] <= point["absolute_off_target_effect"]
            and (other["directed_effect"] > point["directed_effect"] or other["absolute_off_target_effect"] < point["absolute_off_target_effect"])
            for other in coherent
            if other is not point
        )
    ]


def _group_points(points: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for point in points:
        groups[point["method"], point["phase"], point["case_id"]].append(point)
    return groups


def report_tables(points: list[dict]) -> tuple[list[dict], list[dict]]:
    maximum = []
    optimal = []
    for (method, phase, case_id), group in sorted(_group_points(points).items()):
        coherent = [point for point in group if point["coherent"]]
        if not coherent:
            maximum.append({"method": method, "phase": phase, "case_id": case_id, "status": "no coherent measured dose"})
            continue
        maximum.append(max(coherent, key=lambda point: (abs(point["coefficient"] or 0.0), point["directed_effect"])))
        frontier = _pareto(coherent)
        optimal.append(max(frontier, key=lambda point: (point["dose_score"], point["directed_effect"])))
    return maximum, optimal


def _dose_label(point: dict) -> str:
    coefficient = point["coefficient"]
    if coefficient is not None:
        return f"{coefficient:.3g}"
    if point["method"] in {"bare", "prompting"}:
        return point["method"]
    raise ValueError(f"missing dose for {point['method']} {point['phase']} {point['case_id']}")


def _table_rows(points: list[dict]) -> str:
    rows = []
    for point in points:
        if "point_id" not in point:
            rows.append("<tr><td>{method}</td><td>{phase}</td><td>{case_id}</td><td colspan='5'>{status}</td></tr>".format(**{key: html.escape(str(value)) for key, value in point.items()}))
            continue
        dose = _dose_label(point)
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td><a href='#point-{}'>{}</a></td><td>{:+.3f}</td><td>{:.3f}</td><td>{:+.3f}</td><td>{}</td></tr>".format(
                html.escape(point["method"]), html.escape(point["phase"]), html.escape(point["case_id"]), point["point_id"], dose,
                point["directed_effect"], point["absolute_off_target_effect"], point["dose_score"], "yes" if point["coherent"] else "no",
            )
        )
    return "".join(rows)


def _evidence(point: dict) -> str:
    return "<details id='point-{}'><summary>{} {} {} dose {}</summary><p>Examples: {}</p><h4>Complete target-aware evidence</h4><pre>{}</pre><h4>Blind descriptions</h4><pre>{}</pre><h4>Order disagreements</h4><pre>{}</pre><h4>Health</h4><pre>{}</pre></details>".format(
        point["point_id"],
        html.escape(point["method"]), html.escape(point["phase"]), html.escape(point["case_id"]),
        html.escape(_dose_label(point)),
        html.escape(", ".join(f"{item['question_number']}:{item['question_id']}" for item in point["examples"])),
        html.escape(json.dumps(point["aware"], indent=2, sort_keys=True)),
        html.escape(json.dumps(point["blind"], indent=2, sort_keys=True)),
        html.escape(json.dumps(point["paired_disagreements"], indent=2, sort_keys=True)),
        html.escape(json.dumps(point["health"], indent=2, sort_keys=True)),
    )


def _transfer_prediction_rows(predictions: list[dict]) -> str:
    rows = []
    for prediction in predictions:
        history = prediction["search_history"]
        last = history[-1] if history else {"status": "empty search history"}
        endpoint_kl = last.get("kl_rms") if isinstance(last, dict) else None
        error = "unreported" if not isinstance(endpoint_kl, (int, float)) else f"{endpoint_kl - prediction['target_rms']:+.6g}"
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{:.6g}</td><td>{:.6g}</td><td>{}</td><td><details><summary>last: <code>{}</code></summary><pre>{}</pre></details></td></tr>".format(
                html.escape(prediction["method"]),
                html.escape(prediction["case"]["case_id"]),
                prediction["target_rms"],
                prediction["predicted_coefficient"],
                error,
                html.escape(json.dumps(last, sort_keys=True)),
                html.escape(json.dumps(history, indent=2, sort_keys=True)),
            )
        )
    return "".join(rows)


def render_html(artifact: dict, maximum: list[dict], optimal: list[dict]) -> str:
    warning = "<p class='warning'>FAKE DATA — NON-EXPERIMENTAL. This report tests the artifact contract; it is not a benchmark result.</p>" if artifact["non_experimental"] else ""
    headers = "<tr><th>method</th><th>phase</th><th>case</th><th>dose</th><th>directed intended effect</th><th>|off-target effect|</th><th>1:4 score</th><th>coherent</th></tr>"
    details = "".join(_evidence(point) for point in artifact["points"])
    transfer_rows = _table_rows(artifact["transfer_points"])
    prediction_rows = _transfer_prediction_rows(artifact["transfer_predictions"])
    return """<!doctype html><meta charset='utf-8'><title>BS-bench results</title>
<style>body{{font:16px system-ui;max-width:1120px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.35rem .55rem;border-bottom:1px solid #ccc;text-align:left}}.warning{{background:#fff0d8;padding:.7rem;border-left:4px solid #d55e00}}pre{{overflow:auto;background:#f6f6f6;padding:.7rem;font-size:.8rem}}details{{margin:.5rem 0}}</style>
<h1>BS-bench measured results</h1>{warning}
<p>All tables and both plots read <code>measured-points.json</code> (SHA-256: <code>{hash}</code>). Points retain every measured dose. Directed intended effect is the paired AB/BA mean. The 1:4 score is directed effect minus four times mean absolute off-target effect.</p>
<h2>Maximum coherent dose</h2><table>{headers}{maximum}</table>
<h2>1:4-optimal Pareto point</h2><table>{headers}{optimal}</table>
<h2>Four-case RMS-KL transfer</h2><p>These rows are separate from the 20-question method comparison. Each case reports the measured 0.8×, 1.0×, and 1.2× predicted doses.</p><table>{headers}{transfer_rows}</table><h3>RMS-KL target and search endpoint</h3><table><tr><th>method</th><th>case</th><th>target RMS-KL</th><th>predicted coefficient</th><th>last search KL minus target</th><th>last search record</th></tr>{prediction_rows}</table>
<h2>20-question evaluation dose paths</h2><img src='plot.png' alt='Measured dose paths'>
<h2>Pareto view</h2><img src='plot_pareto.png' alt='Pareto dose paths'>
<p>The highlighted frontier is non-dominated within each method, phase, and case; faded marks remain measured points.</p>
<h2>Numbered evidence</h2>{details}
""".format(warning=warning, hash=artifact["points_sha256"], headers=headers, maximum=_table_rows(maximum), optimal=_table_rows(optimal), transfer_rows=transfer_rows, prediction_rows=prediction_rows, details=details)


def _plot(points: list[dict], output: Path, *, pareto: bool, non_experimental: bool) -> set[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    colors = {method: color for method, color in zip(METHODS, ("#222222", "#d55e00", "#777777", "#0072b2", "#cc79a7", "#009e73", "#6f4aa8", "#56b4e9"), strict=True)}
    fig, ax = plt.subplots(figsize=(10.64, 6.2))
    fig.subplots_adjust(left=.11, right=.98, top=.91, bottom=.27)
    random = [point for point in points if point["method"] == "random"]
    if len(random) >= 3:
        cloud = [(point["directed_effect"], point["absolute_off_target_effect"]) for point in random]
        hull = _convex_hull(cloud)
        ax.add_patch(Polygon(hull, closed=True, facecolor="#999999", edgecolor="none", alpha=.18, label="random measured region"))
    groups = _group_points(points)
    labeled_methods: set[str] = set()
    for (method, phase, case_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda point: (-1 if point["coefficient"] is None else point["coefficient"]))
        draw = _pareto(ordered) if pareto else ordered
        if pareto:
            ax.scatter(
                [point["directed_effect"] for point in ordered],
                [point["absolute_off_target_effect"] for point in ordered],
                color=colors[method], s=20, alpha=.18, marker="o",
            )
        x = [point["directed_effect"] for point in draw]
        y = [point["absolute_off_target_effect"] for point in draw]
        if method != "bare" and method != "random" and len(draw) > 1:
            ax.plot(x, y, color=colors[method], alpha=.9 if pareto else .45, linewidth=2.2 if pareto else 1.3)
        label = method if method not in labeled_methods else None
        coherent = [point for point in draw if point["coherent"]]
        incoherent = [point for point in draw if not point["coherent"]]
        if coherent:
            ax.scatter(
                [point["directed_effect"] for point in coherent],
                [point["absolute_off_target_effect"] for point in coherent],
                color=colors[method], s=38, alpha=.85, edgecolors="#222222", linewidths=.55,
                marker="o", label=label,
            )
            label = None
        if incoherent:
            ax.scatter(
                [point["directed_effect"] for point in incoherent],
                [point["absolute_off_target_effect"] for point in incoherent],
                color=colors[method], s=44, alpha=.95, linewidths=1.25,
                marker="x", label=label,
            )
        labeled_methods.add(method)
    title = "FAKE — non-experimental BS-bench Pareto frontiers" if pareto and non_experimental else "BS-bench Pareto frontiers" if pareto else "FAKE — non-experimental BS-bench measured points" if non_experimental else "BS-bench measured points"
    ax.set_title(title)
    ax.set_xlabel("directed intended effect")
    ax.set_ylabel("absolute off-target effect (lower is better)")
    ax.margins(x=.06, y=.08)
    ax.invert_yaxis()
    ax.axvline(0, color="#999999", linewidth=.7)
    ax.grid(color="#e5e5e5", linewidth=.7)
    handles, labels = ax.get_legend_handles_labels()
    if pareto:
        handles.append(Line2D([], [], color="#222222", linewidth=2.2))
        labels.append("within-method Pareto frontier")
    ax.legend(handles, labels, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -0.19), frameon=False)
    fig.savefig(output, dpi=150)
    return {point["point_id"] for point in points}


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return ordered

    def cross(origin, left, right):
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def render_report(run_dir: Path, output: Path) -> dict:
    summary_path = run_dir / "run-summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing full run summary: {summary_path}")
    artifact = normalize_summary(json.loads(summary_path.read_text()))
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "measured-points.json", artifact)
    maximum, optimal = report_tables(artifact["points"])
    table_ids = {point["point_id"] for point in [*maximum, *optimal] if "point_id" in point}
    plot_ids = _plot(artifact["points"], output / "plot.png", pareto=False, non_experimental=artifact["non_experimental"])
    pareto_ids = _plot(artifact["points"], output / "plot_pareto.png", pareto=True, non_experimental=artifact["non_experimental"])
    if plot_ids != {point["point_id"] for point in artifact["points"]} or pareto_ids != plot_ids or not table_ids <= plot_ids:
        raise ValueError("report plot/table points do not match the measured artifact")
    parity = {"schema": "bsbench-report-parity-v1", "points_sha256": artifact["points_sha256"], "artifact_point_ids": sorted(plot_ids), "plot_point_ids": sorted(plot_ids), "pareto_plot_point_ids": sorted(pareto_ids), "table_point_ids": sorted(table_ids)}
    save_json(output / "source-parity.json", parity)
    (output / "index.html").write_text(render_html(artifact, maximum, optimal))
    return {"artifact": artifact, "maximum": maximum, "optimal": optimal, "parity": parity}
