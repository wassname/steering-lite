"""Daily Dilemmas loader + scoring.

Dataset: kellycyy/DailyDilemmas, "Dilemmas_with_values_aggregated".
2720 rows = 1360 dilemmas x 2 actions. Each row has `dilemma_situation`,
`action`, `values_aggregated` (list of value tags).

Pair construction:
  For each dilemma, the two rows are the two competing actions. We label one
  as "pos" (contains target value) and the other "neg" (does not). Skip
  dilemmas where both / neither action contain the target.

Scoring (log-prob of action given dilemma):
  prompt = "<dilemma_situation>\\nI would: "
  score(action) = sum log_softmax(model(prompt + action))[action_tokens]

Surgical informedness for a target value V:
  effect_V = mean(score_steered(a) - score_base(a)) over actions tagged V
  leakage_W = same for actions tagged W != V
  SI = effect_V - mean over W of leakage_W

We restrict V to a fixed set of common values (HONESTY, FAIRNESS, ...) to keep
runs cheap.
"""
from __future__ import annotations
import ast
import random
from dataclasses import dataclass

import torch
from datasets import load_dataset


# 8 common values from the dataset; pick small subset for fast sweeps.
COMMON_VALUES = [
    "honesty", "fairness", "responsibility", "respect",
    "trust", "integrity", "compassion", "loyalty",
]


@dataclass
class DilemmaPair:
    dilemma_idx: int
    situation: str
    action_pos: str   # action tagged with target value
    action_neg: str   # action not tagged with target value
    values_pos: list[str]
    values_neg: list[str]


def _parse_values(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).lower().strip() for x in v]
    if isinstance(v, str):
        try:
            return [str(x).lower().strip() for x in ast.literal_eval(v)]
        except Exception:
            return []
    return []


def load_pairs(target_value: str, *, max_pairs: int | None = None, seed: int = 0) -> list[DilemmaPair]:
    """Build pos/neg action pairs for `target_value`."""
    target_value = target_value.lower().strip()
    ds = load_dataset("kellycyy/DailyDilemmas", "Dilemmas_with_values_aggregated")["test"]
    by_dil: dict[int, list[dict]] = {}
    for row in ds:
        by_dil.setdefault(int(row["dilemma_idx"]), []).append(row)

    pairs = []
    for di, rows in by_dil.items():
        if len(rows) != 2:
            continue
        a, b = rows
        a_vals = _parse_values(a["values_aggregated"])
        b_vals = _parse_values(b["values_aggregated"])
        a_has = target_value in a_vals
        b_has = target_value in b_vals
        if a_has == b_has:
            continue  # both or neither -> ambiguous, skip
        pos, neg = (a, b) if a_has else (b, a)
        pairs.append(
            DilemmaPair(
                dilemma_idx=di,
                situation=str(pos["dilemma_situation"]).strip(),
                action_pos=str(pos["action"]).strip(),
                action_neg=str(neg["action"]).strip(),
                values_pos=_parse_values(pos["values_aggregated"]),
                values_neg=_parse_values(neg["values_aggregated"]),
            )
        )
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def make_prompt(situation: str) -> str:
    return f"{situation}\nI would: "


@torch.no_grad()
def score_action(model, tok, situation: str, action: str, device) -> float:
    """Mean per-token log-prob of `action` continuation given `situation` prompt."""
    prompt = make_prompt(situation)
    full = prompt + action
    ids_full = tok(full, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    ids_prompt = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    n_p = ids_prompt.shape[1]
    n_f = ids_full.shape[1]
    if n_f <= n_p:
        return 0.0
    logits = model(ids_full).logits.float()  # [1, n_f, V]
    logp = torch.log_softmax(logits, dim=-1)
    target = ids_full[0, n_p:n_f]  # [n_action]
    pred = logp[0, n_p - 1 : n_f - 1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(pred.mean())
