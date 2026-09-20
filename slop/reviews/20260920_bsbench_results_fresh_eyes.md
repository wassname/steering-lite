# Fresh-eyes review — BS-bench result plots

Run configuration reported reviewer model `openrouter/moonshotai/kimi-k3`, which satisfies the non-Astra request. The reviewer text self-reported an Anthropic Sonnet-class model; that conflicts with the launcher metadata, so this file treats the visual observations as evidence and does not rely on its self-identification.

The reviewer directly inspected the first terminal-aware render's two PNGs, measured-point artifact, source parity file, and render log. It reported:

> **(Moderate) Incoherent-point marking is all-or-nothing per method.** In `results.py` the marker is chosen as `"x"` for an entire method group if any point in it is incoherent. Every candidate method has exactly one incoherent dose, so in `plot.png` all candidate points render as `x` and a viewer cannot tell which points are actually incoherent.

> **(Minor, potentially misleading) "Pareto frontier" legend swatch is green.** The label attaches to the first drawn frontier line, which happens to be `kv_cache_gram`.

The reviewer independently recomputed the measured-point content hash and said it matched the rendered parity file:

> `source-parity.json` lists identical 44 point IDs in `artifact_point_ids`, `plot_point_ids`, and `pareto_plot_point_ids`; the 12 `table_point_ids` are a strict subset.

The report implementation now draws coherent and incoherent points separately, uses a neutral frontier legend entry, and rerenders from the same measured-point source. Direct raster crops after the change show the Pareto title, y-axis label, and y=0 marker within bounds; the apparent clipping in one full-image preview was not present in the PNG pixels.

Original worker artifact: `/home/code/.pi/agent/sessions/--workspace-2026-lite-steering-lite-bsbench--/subagent-artifacts/outputs/24e5def1-7700-4c23-92a8-e24fac92d766/pca_fresh_review.md`.

-- PI[gpt-5.6-terra]
