"""BS-bench v2 data: cohorts, walk certificates and the (bare, steered) answer rows the judge rates.

From vjp-steering 7f0782a `scripts/judge.py` (the judge-independent part).
"""

import json
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {"dev": slice(0, 100, 5), "full": slice(0, 100)}  # same as walk.py
REPORT_SEEDS = {"dev": range(1), "full": range(3)}
RANDOM_SEEDS = {"dev": range(32), "full": range(11)}  # denser dev reference; published full unchanged. PI/OpenAI
USER_RANDOM_SEEDS = {"dev": range(32), "full": range(50)}  # random-user: reference for user-positions steering
PROMPT_WALKS = ("prompting", "prompting_engineered", "prompting_scale", "prompting_engineered_scale")


def load_cohort() -> dict[str, dict]:
    rows = [json.loads(line) for line in (ROOT / "data/bsbench/bullshit_bench_v2.jsonl").open()]
    assert len(rows) == 100
    return {row["scenario"]: row for row in rows}


def default_model_dir(model: str = "Qwen/Qwen3.5-4B") -> Path:
    dirs = [path for path in (ROOT / "outputs/bsbench").glob(f"{model.replace('/', '--')}-g*") if (path / "walks").is_dir()]
    assert len(dirs) == 1, f"expected one output dir for {model}, found {dirs}"
    return dirs[0]


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
    """One row per (rung, side, question): the bare and steered answers."""
    cohort = load_cohort()
    scenarios = list(cohort)[COHORTS[certificate["cohort"]]]
    bare = read_answers(model_dir / "answers/bare/bare.jsonl")
    rows = []
    for rung in certificate["rungs"]:
        for side in ("+C", "-C"):
            steered = read_answers(model_dir / rung[side]["answers"])
            for scenario in scenarios:
                assert bare[scenario]["prompt"] == steered[scenario]["prompt"] == cohort[scenario]["prompt"]
                rows.append({
                    "method": certificate["method"], "seed": certificate["seed"], "C": rung["coefficient"], "side": side,
                    "vignette": scenario, "prompt": cohort[scenario]["prompt"], "flaw": cohort[scenario]["nonsensical_element"],
                    "bare": bare[scenario]["text"], "steered": steered[scenario]["text"],
                })
    return rows


def manifest(model_dir: Path, cohort: str) -> list[dict]:
    certificates = walk_certificates(model_dir, cohort, view="all")
    rows = [row for certificate in certificates for row in demo_rows(model_dir, certificate)]
    logger.info("manifest walks={} rows={}", len(certificates), len(rows))
    return rows
