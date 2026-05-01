import importlib.util
import sys
from pathlib import Path

from steering_lite.eval.airisk_dilemmas import AiriskPair, load_eval_rows
import steering_lite.eval.airisk_dilemmas as airisk

SPEC = importlib.util.spec_from_file_location(
    "airisk_dilemmas_benchmark",
    Path(__file__).resolve().parent.parent / "scripts" / "airisk_dilemmas_benchmark.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_load_eval_rows_symmetrizes_value_labels(monkeypatch):
    def fake_pairs():
        return [
            AiriskPair(
                dilemma_idx=0,
                dilemma="d0",
                action_1="Action 1: Tell the truth",
                action_2="Action 2: Lie to preserve access",
                values_action_1=["Truthfulness", "Protection"],
                values_action_2=["Protection", "Care"],
            )
        ]

    monkeypatch.setattr(airisk, "_load_pairs_raw", fake_pairs)
    rows = load_eval_rows(seed=0)

    assert len(rows) == 1
    assert rows[0].value_labels == {
        "Truthfulness": +1.0,
        "Care": -1.0,
    }


def test_aggregate_effects_uses_shared_finite_support_only():
    base_rows = [
        {"logratio": 1.0, "value_labels": {"Truthfulness": 1.0}},
        {"logratio": float("nan"), "value_labels": {"Truthfulness": 1.0}},
    ]
    steered_rows = [
        {"logratio": float("nan"), "value_labels": {"Truthfulness": 1.0}},
        {"logratio": 3.0, "value_labels": {"Truthfulness": 1.0}},
    ]

    effects = benchmark._aggregate_effects(base_rows, steered_rows, ["Truthfulness"])

    assert effects == {}
