"""Non-paid BS-bench integration: real numbered cohort, cache, and tiny pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from steering_lite.benchmark.cache import cached_stage, reserve, settle
from steering_lite.benchmark.generation import DEV_SIZE, read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS, run_method


class TinyTokenizer:
    """Deterministic local tokenizer for the real model pipeline smoke."""

    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def _encode(self, text: str) -> list[int]:
        return [ord(char) % 30 + 1 for char in text][:48] or [1]

    def __call__(self, texts, *, return_tensors=None, padding=False, **_kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [self._encode(text) for text in texts]
        width = max(map(len, rows)) if padding else len(rows[0])
        ids = torch.tensor([
            ([self.pad_token_id] * (width - len(row)) + row if self.padding_side == "left" else row + [self.pad_token_id] * (width - len(row)))
            for row in rows
        ])
        return BatchEncoding({"input_ids": ids, "attention_mask": ids.ne(self.pad_token_id).long()})

    def apply_chat_template(self, messages, *, tokenize=False, **_kwargs):
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
        return self(text, return_tensors="pt")["input_ids"] if tokenize else text

    def batch_decode(self, rows, **_kwargs):
        return ["answer." for _ in rows]

    def encode(self, text: str):
        return self._encode(text)


def tiny_model():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=0,
    )
    return LlamaForCausalLM(config).eval(), TinyTokenizer()


def test_dev_cohort_is_numbered_stable_and_reorder_fails(monkeypatch):
    rows = read_dev_cohort()
    assert len(rows) == DEV_SIZE == 20
    assert [row["question_number"] for row in rows] == list(range(1, 21))
    assert [row["question_id"] for row in rows] == [f"BSV2-{n:03d}" for n in range(1, 21)]

    from steering_lite.benchmark import generation
    monkeypatch.setattr(generation, "_cohort_rows", lambda: list(reversed(rows)) + [{"scenario": f"other-{n}", "prompt": "x", "nonsensical_element": "x"} for n in range(80)])
    with pytest.raises(ValueError, match="changed order or content"):
        generation.read_dev_cohort()


def test_content_cache_reuses_only_identical_identity(tmp_path: Path):
    calls = []

    def compute():
        calls.append(len(calls))
        return {"call": len(calls)}

    base = dict(
        model={"id": "tiny", "revision": "a"},
        data={"sha256": "data-a", "rows": 20},
        method="mean_diff",
        config={"coeff": 0.2},
        prompts=["prompt-a"],
        compute=compute,
        code="code-a",
    )
    assert cached_stage(tmp_path, "generation", **base) == {"call": 1}
    assert cached_stage(tmp_path, "generation", **base) == {"call": 1}
    assert len(calls) == 1
    for changed in (
        {"model": {"id": "tiny", "revision": "b"}},
        {"data": {"sha256": "data-b", "rows": 20}},
        {"method": "pca"},
        {"config": {"coeff": 0.3}},
        {"prompts": ["prompt-b"]},
        {"code": "code-b"},
    ):
        expected_call = len(calls) + 1
        assert cached_stage(tmp_path, "generation", **(base | changed)) == {"call": expected_call}
    assert len(calls) == 7


def test_spending_ledger_reserves_and_settles_without_silent_overrun(tmp_path: Path):
    ledger = tmp_path / "costs.jsonl"
    first = reserve(ledger, "local-smoke", 2.0, limit_usd=3.0)
    settle(ledger, first, 1.0)
    second = reserve(ledger, "local-smoke", 2.0, limit_usd=3.0)
    with pytest.raises(RuntimeError, match="exceeds"):
        reserve(ledger, "over-budget", 0.1, limit_usd=3.0)
    settle(ledger, second, 2.0)


@pytest.mark.parametrize("method", METHODS)
def test_tiny_numbered_benchmark_uses_real_pipeline(method: str, tmp_path: Path):
    model, tokenizer = tiny_model()
    rows = read_dev_cohort()[:2]
    result = run_method(
        model,
        tokenizer,
        method,
        ["user: tell the truth\nassistant: yes"],
        ["user: flatter me\nassistant: yes"],
        vector_dir=tmp_path / "vectors",
        rows=rows,
        layers=(0,),
        target_layer=2,
        max_length=48,
        max_new_tokens=1,
    )
    assert result["method"] == method
    assert len(result["answers"]) == len(rows)
    if method not in {"bare", "prompting"}:
        assert (tmp_path / "vectors" / result["vector"]).exists()
        assert result["vector_sha256"]
