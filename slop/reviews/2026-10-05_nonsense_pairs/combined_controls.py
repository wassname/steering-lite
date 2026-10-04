"""One chart: -C pushback vs legitimate control questions called nonsense, generic vs nonsense-question extraction pairs.
PI/OpenAI 2026-10-05. uv run --extra benchmark python slop/reviews/2026-10-05_nonsense_pairs/combined_controls.py"""
import json
from statistics import mean
import plotly.graph_objects as go

R = "outputs/bsbench/results"
lines = [("vjp_resid", "v3-4b", "VJP-resid, generic pairs", "#0072b2", "dot"), ("vjp_resid", "v3-4b-nonsense-pairs", "VJP-resid, nonsense-question pairs", "#0072b2", "solid"),
         ("mean_diff", "v3-4b", "mean difference, generic pairs", "#d55e00", "dot"), ("mean_diff", "v3-4b-nonsense-pairs", "mean difference, nonsense-question pairs", "#d55e00", "solid")]
fig = go.Figure()
for method, report, name, color, dash in lines:
    pts = [p for p in json.load(open(f"{R}/{report}/points.json"))["points"] if p["method"] == method and p["side"] == "-C" and p["admissible"]]
    pts.sort(key=lambda p: p["C"])
    bare = 100 * mean(p["control_claims_bare"] for p in pts)
    fig.add_trace(go.Scatter(x=[0] + [-p["effect"] for p in pts], y=[bare] + [100 * p["control_claims"] for p in pts], mode="lines+markers", name=name,
                             text=[""] + [f"C={p['C']:.3g}" for p in pts], line={"color": color, "dash": dash, "width": 3}, marker={"size": 7, "color": color}))
prompt = next(p for p in json.load(open(f"{R}/v3-4b/points.json"))["points"] if p["method"] == "prompting" and p["side"] == "-C")
fig.add_trace(go.Scatter(x=[-prompt["effect"]], y=[100 * prompt["control_claims"]], mode="markers", name="plain prompt −C", marker={"color": "#6a3d9a", "size": 16, "symbol": "star"}))
fig.add_trace(go.Scatter(x=[0], y=[100 * prompt["control_claims_bare"]], mode="markers", name="bare", marker={"color": "#333", "size": 11, "symbol": "diamond"}))
fig.update_layout(template="plotly_white", height=560, title={"text": "Qwen3.5-4B, −C sweeps: pushback on nonsense vs rejecting legitimate questions", "x": 0.5},
                  xaxis={"title": "pushback gained on 100 BullshitBench v2 questions (BullshitBench score, 0–2; Jev)"},
                  yaxis={"title": "legitimate questions called nonsense (%)", "rangemode": "tozero"},
                  legend={"orientation": "h", "y": -0.2}, margin={"b": 150})
fig.write_image("slop/reviews/2026-10-05_nonsense_pairs/combined_controls.png", width=1064, height=600, scale=2)
fig.write_html("slop/reviews/2026-10-05_nonsense_pairs/combined_controls.html", include_plotlyjs="cdn")
print("ok")
