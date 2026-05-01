"""Eval harnesses: scoring loops on top of trained steering vectors.

- `airisk_dilemmas`: Action 1/2 MCQ on kellycyy/AIRiskDilemmas with guided-CoT scorer.
- `tinymfv`: thin adapter to the sibling tiny-mcf-vignettes package.
"""
from .airisk_dilemmas import (
    AiriskPair, AiriskEvalRow,
    all_value_classes,
    load_pairs, load_eval_rows,
    format_training_prompt, format_mcq, format_mcq_thinking,
    get_choice_ids, score_mcq, score_mcq_guided,
)

__all__ = [
    "AiriskPair", "AiriskEvalRow",
    "all_value_classes",
    "load_pairs", "load_eval_rows",
    "format_training_prompt", "format_mcq", "format_mcq_thinking",
    "get_choice_ids", "score_mcq", "score_mcq_guided",
]
