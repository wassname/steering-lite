"""BS-bench generation and health checks from vjp-steering efcd848/scripts/walk.py.

Adapted by PI/OpenAI; prompts, greedy decoding and health thresholds are retained.
"""

import hashlib
import json
import re
from pathlib import Path

import torch
from loguru import logger

from steering_lite import Vector
from steering_lite.calibrate import _ngram_rep


COHORT_SIZE = 100
DEV_SIZE = 20
DEV_SHA256 = "c220039523b581f6698e55c67842511c63f5f4e8ef93c3638e16d32000560ae9"


def _cohort_rows() -> list[dict]:
    path = Path(__file__).with_name("data").joinpath("bullshit_bench_v2.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != COHORT_SIZE:
        raise ValueError(f"BS-bench v2 expected {COHORT_SIZE} rows, got {len(rows)}")
    scenarios = [row["scenario"] for row in rows]
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("BS-bench v2 has duplicate scenario IDs")
    return rows


def cohort_identity(rows: list[dict]) -> dict[str, str | int]:
    payload = [[row["scenario"], row["prompt"]] for row in rows]
    return {
        "rows": len(rows),
        "sha256": hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
    }


def read_cohort(limit: int = DEV_SIZE, offset: int = 0) -> list[dict]:
    rows = _cohort_rows()
    if not (0 <= offset < COHORT_SIZE and 1 <= limit <= COHORT_SIZE - offset):
        raise ValueError(f"invalid cohort slice: offset={offset}, limit={limit}")
    selected = rows[offset:offset + limit]
    return [
        {"question_number": number, "question_id": f"BSV2-{number:03d}", **row}
        for number, row in enumerate(selected, offset + 1)
    ]


def read_dev_cohort() -> list[dict]:
    """Return the fixed, visibly numbered 20-question development cohort."""
    rows = read_cohort(DEV_SIZE)
    identity = cohort_identity(rows)
    if identity["sha256"] != DEV_SHA256:
        raise ValueError(
            "BS-bench v2 dev cohort changed order or content; update it deliberately with a new reviewed hash"
        )
    if [row["question_number"] for row in rows] != list(range(1, DEV_SIZE + 1)):
        raise ValueError("BS-bench v2 dev question numbering is not contiguous")
    return rows


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    answers = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start:start + batch_size], return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(next(model.parameters()).device)
        output = model.generate(
            **batch, do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
        )
        answers.extend(tokenizer.batch_decode(output[:, batch.input_ids.shape[1]:], skip_special_tokens=True))
        logger.info("generation {}/{}", min(start + batch_size, len(prompts)), len(prompts))
    return [answer.strip() for answer in answers]


PERSONAS = [("sycophantic", "abrasive")]


PERSONA_TEMPLATE = "Answer as someone who is {persona}."


ROLE_LEAK = re.compile(r"<\s*/?\s*think\s*>|^\s*(user|assistant|system)\s*$", re.I | re.M)


def vector_hash(vector: Vector) -> str:
    digest = hashlib.sha256()
    for kind, tree in (("shared", vector.shared), ("stacked", vector.stacked)):
        for layer, tensors in sorted(tree.items()):
            for name, tensor in sorted(tensors.items()):
                value = tensor.detach().contiguous().cpu()
                digest.update(f"{kind}:{layer}:{name}:{value.dtype}:{tuple(value.shape)}".encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def generation_inputs(
    tokenizer, rows: list[dict[str, str]], persona: str | None = None,
) -> list[str]:
    prefix = "" if persona is None else PERSONA_TEMPLATE.format(persona=persona) + "\n\n"
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + row["prompt"] + " Answer in 2 short sentences."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]


def worst_repetition(token_ids: list[int], window: int = 128) -> float:
    if len(token_ids) <= window:
        return _ngram_rep(token_ids)
    return max(
        _ngram_rep(token_ids[start : start + window])
        for start in range(0, len(token_ids) - window + 1, window // 4)
    )


def health(tokenizer, answers: list[str]) -> tuple[dict[str, float | int], list[str]]:
    unfinished = sum(not re.search(r"[.!?\")]$", answer) for answer in answers)
    role_leaks = sum(bool(ROLE_LEAK.search(answer)) for answer in answers)
    repetitions = [worst_repetition(tokenizer.encode(answer)) for answer in answers]
    repeated = sum(value > 0.5 for value in repetitions)
    n = len(answers)
    reasons = []
    if unfinished / n >= 0.5:
        reasons.append("unfinished")
    if role_leaks / n >= 0.25:
        reasons.append("role_leak")
    if repeated / n >= 0.25:
        reasons.append("repetition")
    return {
        "answers": n,
        "unfinished": unfinished,
        "role_leaks": role_leaks,
        "repeated": repeated,
        "max_repetition": max(repetitions),
        "mean_words": sum(len(answer.split()) for answer in answers) / n,
    }, reasons
