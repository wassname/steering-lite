"""Regression guard for the sweep -> results measurement seam.

`results.py` reads each sweep JSON's `raw_logratios` (= clr) + `mean_pmass_allowed`
and feeds the shared `moralmaps.metrics` headline (gated_selectivity + si_flips).
This exercises that path on synthetic sweep JSONs, no model, so the seam (and the
persona-aligned sign selection) can't silently rot again. -- Claude
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

PROBES = ["care", "fairness", "loyalty", "authority", "sanctity", "liberty", "social"]


def _lr(**overrides: float) -> dict[str, float]:
    """7-probe clr-ish dict; unset probes default to a mild negative value."""
    d = {p: -3.0 for p in PROBES}
    d.update(overrides)
    return d


_BASE_AUTH = [+1.5, -1.5, +2.0, -2.0]
_BASE_CARE = [+1.5, -1.5, +2.0, -2.0]


def _report(auth_shift: float, care_shift: float, pmass: float = 0.9) -> dict:
    """raw_logratios (clr) for 4 Authority + 4 Care vignettes, shifted from base, + coherence."""
    lr = {}
    for i, b in enumerate(_BASE_AUTH):
        lr[f"vid_auth{i}|other_violate"] = _lr(authority=b + auth_shift, care=-1.0)
    for i, b in enumerate(_BASE_CARE):
        lr[f"vid_care{i}|other_violate"] = _lr(care=b + care_shift, authority=-2.0)
    return {"raw_logratios": lr, "mean_margin": 1.5, "mean_pmass_allowed": pmass}


def _write_sweep(tmp: Path) -> None:
    (tmp / "bare.json").write_text(json.dumps(_report(0.0, 0.0)))
    # +C drives Authority DOWN (persona-aligned Auth↓) + Care UP; -C mirrors it.
    (tmp / "mean_diff.json").write_text(json.dumps({
        "calibrated_C": 1.0,
        "pos": _report(auth_shift=-3.5, care_shift=+3.5),
        "neg": _report(auth_shift=+3.5, care_shift=-3.5),
    }))
    (tmp / "repeng.json").write_text(json.dumps({
        **_report(auth_shift=-3.5, care_shift=+3.5),
        "coeff": 0.0,
    }))
    (tmp / "prompt_only.json").write_text(json.dumps({
        **_report(auth_shift=-1.0, care_shift=+1.0),
        "coeff": None,
    }))


def test_results_seam(tmp_path: Path) -> None:
    import results

    bare, methods = results._load_sweep(tmp_path)
    assert "mean_diff" in methods
    m = methods["mean_diff"]
    # Persona-aligned sign must be the one that moves ΔAuth down (= +C here).
    assert m["sign"] == +1
    sel = m["sel"]
    # Aligned=pos moves Auth down + Care up with zero off-axis collateral -> strong positive,
    # full coherence (pmass preserved), so sel_gated ~= on and clears its CI.
    assert not math.isnan(sel["sel_gated"]) and sel["sel_gated"] > 0
    assert abs(sel["off"]) < 1e-9
    assert sel["coherence"] == 1.0
    assert -1.0 <= m["si_flips"] <= 1.0
    assert methods["repeng"]["calibrated_C"] == 0.0
    assert methods["prompt_only"]["calibrated_C"] is None
    # Selectivity + Δclr tables must render without raising.
    results.print_tables(methods)


def test_missing_coherence_fails() -> None:
    import results

    with pytest.raises(KeyError, match="mean_pmass_allowed"):
        results._pmass({})


def test_base_table_does_not_invent_binary_probability(monkeypatch) -> None:
    import results

    humans = {
        f: {"mean": 3.0, "std": 0.5, "wsum": 4.0, "n": 4}
        for f in results.FOUNDATION_ORDER
    }
    monkeypatch.setattr(results, "_human_per_foundation", lambda _: humans)
    table = results.base_vs_humans_table(_report(0.0, 0.0), "unused")
    assert "p(wrong)" not in table


def test_clr_readout_seam() -> None:
    """Guard the score->clr conversion in the tinymfv wrapper (the OTHER seam).

    tinymfv commit 9237aa9 replaced one-vs-rest logit(p_f) with clr(score)_f to
    stop a single-category steer fabricating off-axis collateral (job 95). The
    wrapper must consume `score`, emit gauge-free clr, and keep sum=0. -- Claude
    """
    import numpy as np

    from moralmaps.guided import _DEFAULT_FORCED_FOUNDATIONS
    from steering_lite.eval.tinymfv import _per_row_to_multibool

    fs = list(_DEFAULT_FORCED_FOUNDATIONS)
    score = np.array([0.1, 0.0, -0.2, 3.0, -0.1, 0.05, -0.3])  # authority-dominant
    p = np.exp(score) / np.exp(score).sum()
    row = {"id": "vid1", "condition": "other_violate", "p": list(p),
           "score": list(score), "margin": 1.5}
    rl, _, _, _ = _per_row_to_multibool([row])
    d = rl["vid1|other_violate"]
    vals = np.array([d[f] for f in fs])
    assert abs(float(vals.sum())) < 1e-9          # clr is centered (sum = 0)
    assert fs[int(vals.argmax())] == "authority"  # steered category still wins
    # gauge invariance: a constant offset on score must not change clr.
    d2 = _per_row_to_multibool([dict(row, score=list(score + 5.0))])[0]["vid1|other_violate"]
    assert max(abs(d[f] - d2[f]) for f in fs) < 1e-9


@pytest.fixture(autouse=True)
def _make_sweep(tmp_path: Path):
    _write_sweep(tmp_path)
    yield
