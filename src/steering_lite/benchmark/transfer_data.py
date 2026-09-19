"""Auditable, disjoint prompt records for RMS-KL transfer. — PI/OpenAI

The source files are vendored byte-for-byte from the recorded j-steer_pub revision.
They are checked at load time: changing source text or provenance is a deliberate cache
invalidation, never a silent prompt substitution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    dataset: str
    source_path: str
    source_revision: str
    source_sha256: str
    content_sha256: str

    def identity(self) -> dict[str, str]:
        return asdict(self)


BSBENCH_SOURCE_PATH = "data/bullshit_bench_v2.jsonl"
BSBENCH_SOURCE_REVISION = "5d60d53e5456bc23e8895f9aafe1a29478db7a96"
BSBENCH_SOURCE_SHA256 = "0c79ad8bbb3d35293ce9b8aec0660ce9b4d10b398b7854aff794d2a38e9f251c"
PAPER_NATIVE_SOURCE_PATH = "data/dev/paper_native_false_claim_agreement.json"
PAPER_NATIVE_SOURCE_REVISION = "8d32aa1fc7974740e97985337d65b041a4938e6a"
PAPER_NATIVE_SOURCE_SHA256 = "d345bde457f791c3caf20279d9e00b1629642204026557ffa8aa445f1d67878f"

TRANSFER_CASE_PROMPT_IDS = {
    "bsbench-v2-heldout-a": ("BSV2-021", "BSV2-022"),
    "bsbench-v2-heldout-b": ("BSV2-023", "BSV2-024"),
    "paper-native-false-claim-agreement-a": ("PNFCA-001", "PNFCA-002"),
    "paper-native-false-claim-agreement-b": ("PNFCA-003", "PNFCA-004"),
}


def _source_file(name: str) -> Path:
    return Path(__file__).with_name("data").joinpath(name)


def _verified_bytes(name: str, expected_sha256: str) -> bytes:
    value = _source_file(name).read_bytes()
    actual = hashlib.sha256(value).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"transfer source {name} SHA256 changed: {actual}")
    return value


def _record(prompt_id: str, prompt: str, dataset: str, source_path: str, source_revision: str, source_sha256: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        dataset=dataset,
        source_path=source_path,
        source_revision=source_revision,
        source_sha256=source_sha256,
        content_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def load_transfer_records() -> dict[str, tuple[PromptRecord, ...]]:
    """Load exact source prompts for all declared transfer cases, or fail before dispatch."""
    bsbench_rows = [json.loads(line) for line in _verified_bytes("bullshit_bench_v2.jsonl", BSBENCH_SOURCE_SHA256).splitlines()]
    if len(bsbench_rows) != 100:
        raise ValueError("BS-bench v2 transfer source must contain exactly 100 rows")
    bsbench = {
        f"BSV2-{number:03d}": _record(
            f"BSV2-{number:03d}", row["prompt"], "bsbench-v2-heldout",
            BSBENCH_SOURCE_PATH, BSBENCH_SOURCE_REVISION, BSBENCH_SOURCE_SHA256,
        )
        for number, row in enumerate(bsbench_rows, 1)
    }
    native_prompts = json.loads(_verified_bytes("paper_native_false_claim_agreement.json", PAPER_NATIVE_SOURCE_SHA256))
    if not isinstance(native_prompts, list) or not all(isinstance(prompt, str) and prompt for prompt in native_prompts):
        raise ValueError("paper-native false-claim agreement source must be non-empty prompt strings")
    native = {
        f"PNFCA-{number:03d}": _record(
            f"PNFCA-{number:03d}", prompt, "paper-native-false-claim-agreement",
            PAPER_NATIVE_SOURCE_PATH, PAPER_NATIVE_SOURCE_REVISION, PAPER_NATIVE_SOURCE_SHA256,
        )
        for number, prompt in enumerate(native_prompts, 1)
    }
    all_records = bsbench | native
    try:
        return {
            case_id: tuple(all_records[prompt_id] for prompt_id in prompt_ids)
            for case_id, prompt_ids in TRANSFER_CASE_PROMPT_IDS.items()
        }
    except KeyError as error:
        raise ValueError(f"declared transfer prompt is absent from its audited source: {error.args[0]}") from error


def transfer_provenance(records: tuple[PromptRecord, ...] | list[PromptRecord]) -> list[dict[str, str]]:
    return [record.identity() for record in records]


def transfer_records_identity(records_by_case: dict[str, tuple[PromptRecord, ...]]) -> dict[str, list[dict[str, str]]]:
    return {case_id: transfer_provenance(records_by_case[case_id]) for case_id in sorted(records_by_case)}
