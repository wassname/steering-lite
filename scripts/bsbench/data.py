"""BS-bench v2 data: cohorts, walk certificates and the (bare, steered) answer rows the judge rates.

From vjp-steering 7f0782a `scripts/judge.py` (the judge-independent part).
"""

import hashlib
import json
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {"dev": slice(0, 100, 5), "full": slice(0, 100)}  # same as walk.py
REPORT_SEEDS = {"dev": range(1), "full": range(3)}
RANDOM_SEEDS = {"dev": range(32), "full": range(11)}  # denser dev reference; published full unchanged. PI/OpenAI
USER_RANDOM_SEEDS = {"dev": range(32), "full": range(50)}  # random-user: reference for user-positions steering
PROMPT_WALKS = ("prompting", "prompting_engineered", "prompting_scale", "prompting_engineered_scale")
# Eval v2 (2026-10-03): every dose also answers the sound-premise twins (control set); -C persona below.
EVAL_VERSION = 2
# (+C persona, -C persona). -C chosen 2026-10-03 by a mean_diff pole screen on 4B full (largest -C pushback with
# damage <= 1.5 and false pushback <= 5 pp), effect in premise levels, - = rejection: skeptical -0.78, abrasive -0.56,
# candid -0.40, accurate +0.02 (slightly toward acceptance).
# slop/reviews/2026-10-03_eval_v2/pole_screen.md
AXIS = ("sycophantic", "skeptical")
TWINS = ROOT / "data/bsbench/sound_twins_v1.jsonl"
# everything that defines an answer is in its cache key; change one and all answers regenerate in a new output dir
GEN = {"suffix": " Answer in 2 short sentences.", "enable_thinking": False, "do_sample": False, "max_new_tokens": 512,
       "axis": list(AXIS), "eval_version": EVAL_VERSION, "twins": TWINS.name}
GEN_KEY = hashlib.sha256(json.dumps(GEN, sort_keys=True).encode()).hexdigest()[:8]


def load_cohort() -> dict[str, dict]:
    rows = [json.loads(line) for line in (ROOT / "data/bsbench/bullshit_bench_v2.jsonl").open()]
    assert len(rows) == 100
    return {row["scenario"]: row for row in rows}


def load_twins() -> dict[str, dict]:
    """Sound-premise twin of each BS-bench question, keyed by the original scenario; all 100 are usable."""
    rows = [json.loads(line) for line in TWINS.open()]
    assert len(rows) == 100 and all(r["usable"] for r in rows)
    return {r["scenario"]: {"scenario": r["scenario"], "prompt": r["question"]} for r in rows}


def default_model_dir(model: str = "Qwen/Qwen3.5-4B") -> Path:
    """The current eval version's output dir; earlier versions live in other -g<key> dirs (pass --model-dir)."""
    path = ROOT / "outputs/bsbench" / f"{model.replace('/', '--')}-g{GEN_KEY}"
    assert (path / "walks").is_dir(), f"no walks for {model} at eval v{EVAL_VERSION} ({path})"
    return path


def walk_certificates(model_dir: Path, cohort: str, view: str = "benchmark") -> list[dict]:
    """COMPLETE walks at the published report seeds; extra cached seeds do not change the comparison. PI/OpenAI

    view: benchmark = steering everywhere; user = user-positions walks (renamed without -user) plus prompt
    walks, which also act only on the prompt; all = both populations, unrenamed (for the judge)."""
    certificates = []
    for path in sorted((model_dir / "walks").glob(f"*_{cohort}.json")):
        certificate = json.loads(path.read_text())
        user = certificate["method"].endswith("-user")
        if view == "benchmark" and user or view == "user" and not user and certificate["method"] not in PROMPT_WALKS:
            continue
        if certificate["status"] != "COMPLETE":
            logger.warning("skip {} status={}", path.name, certificate["status"])
            continue
        seeds = {"random": RANDOM_SEEDS, "random-user": USER_RANDOM_SEEDS}.get(certificate["method"], REPORT_SEEDS)[cohort]
        if certificate["seed"] not in seeds:
            logger.info("exclude {}: seed outside published {} report", path.name, cohort)
            continue
        if view == "user" and user:
            certificate["method"] = certificate["method"].removesuffix("-user")
        certificates.append(certificate)
    return certificates


def read_answers(path: Path) -> dict[str, dict]:
    return {record["scenario"]: record for record in map(json.loads, path.open())}


def demo_rows(model_dir: Path, certificate: dict) -> list[dict]:
    """One row per (rung, side, question): the bare and steered answers. From eval v2 also one row per sound twin
    (set="twin", flaw=None); v1 certificates have no twins."""
    cohort = load_cohort()
    scenarios = list(cohort)[COHORTS[certificate["cohort"]]]
    twin_set = certificate.get("eval_version", 1) >= 2  # v1 certificates predate the field and the twins
    bare = read_answers(model_dir / "answers/bare/bare.jsonl")
    twins = load_twins() if twin_set else {}
    bare_twins = read_answers(model_dir / "answers_twins/bare/bare.jsonl") if twin_set else {}
    rows = []
    for rung in certificate["rungs"]:
        for side in ("+C", "-C"):
            steered = read_answers(model_dir / rung[side]["answers"])
            steered_twins = read_answers(model_dir / rung[side]["twin_answers"]) if twin_set else {}
            for scenario in scenarios:
                assert bare[scenario]["prompt"] == steered[scenario]["prompt"] == cohort[scenario]["prompt"]
                common = {"method": certificate["method"], "seed": certificate["seed"], "C": rung["coefficient"], "side": side, "vignette": scenario}
                rows.append({**common, "set": "bench", "prompt": cohort[scenario]["prompt"], "flaw": cohort[scenario]["nonsensical_element"],
                             "bare": bare[scenario]["text"], "steered": steered[scenario]["text"]})
                if twin_set:
                    assert bare_twins[scenario]["prompt"] == steered_twins[scenario]["prompt"] == twins[scenario]["prompt"]
                    rows.append({**common, "set": "twin", "prompt": twins[scenario]["prompt"], "flaw": None,
                                 "bare": bare_twins[scenario]["text"], "steered": steered_twins[scenario]["text"]})
    return rows


def manifest(model_dir: Path, cohort: str) -> list[dict]:
    certificates = walk_certificates(model_dir, cohort, view="all")
    rows = [row for certificate in certificates for row in demo_rows(model_dir, certificate)]
    logger.info("manifest walks={} rows={}", len(certificates), len(rows))
    return rows
