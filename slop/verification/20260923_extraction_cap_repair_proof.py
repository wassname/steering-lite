"""Exercise the shared extraction preflight on the exact local 200-pair corpus. — PI/gpt-6-sol"""
import hashlib
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from steering_lite import MeanDiffC
from steering_lite.attach import _require_untruncated_prompts, train
from steering_lite.benchmark.sweep import persona_extraction_identity
from steering_lite.data.personas import make_persona_pairs

root = Path(__file__).resolve().parents[2]
snapshot = Path('/home/code/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
identity = persona_extraction_identity()
tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
assert tokenizer.truncation_side == 'right'
positive, negative = make_persona_pairs(
    tokenizer, n_pairs=identity['requested_pairs'], thinking=identity['thinking'],
    persona_pairs=[tuple(pair) for pair in identity['pairs']],
    template=identity['template'], seed=identity['seed'],
)
assert len(positive) == len(negative) == identity['actual_pairs'] == 200
start = time.monotonic()
try:
    _require_untruncated_prompts(tokenizer, positive, negative, 64)
except ValueError as exc:
    old_error = str(exc)
else:
    raise AssertionError('64-token real extraction should fail')
assert old_error == 'extraction prompt truncation: 259/400 exceed max_length=64 (longest=240)', old_error
try:
    train(None, tokenizer, positive, negative, MeanDiffC(layers=(0,)), max_length=64)
except ValueError as exc:
    assert str(exc) == old_error, str(exc)
else:
    raise AssertionError('train must reject before touching the absent model')
_require_untruncated_prompts(tokenizer, positive, negative, 384)
result = {
    'author': 'PI/gpt-6-sol', 'schema': 'bsbench-extraction-cap-repair-proof-v1',
    'pair_count': len(positive), 'persona_identity': identity,
    'tokenizer_snapshot': str(snapshot),
    'tokenizer_config_sha256': hashlib.sha256((snapshot / 'tokenizer_config.json').read_bytes()).hexdigest(),
    'old_cap': 64, 'old_error': old_error, 'training_entry_rejected_before_model': True,
    'new_cap': 384, 'new_cap_passed': True,
    'elapsed_seconds': time.monotonic() - start,
}
out = root / 'slop/verification/20260923_extraction_cap_repair_proof.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
