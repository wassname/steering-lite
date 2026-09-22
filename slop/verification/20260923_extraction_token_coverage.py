"""Reproduce the real 200 extraction pairs and measure 64-token coverage offline. — PI/gpt-6-sol"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from steering_lite.benchmark.sweep import persona_extraction_identity
from steering_lite.data.personas import load_suffixes, make_persona_pairs

root = Path(__file__).resolve().parents[2]
snapshot = Path('/home/code/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
assert snapshot.exists()
identity = persona_extraction_identity()
raw = root / 'src/steering_lite/data/branching_suffixes_filt.json'
assert hashlib.sha256(raw.read_bytes()).hexdigest() == identity['corpus_sha256']
tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
assert tok.is_fast
assert tok.truncation_side == 'right', tok.truncation_side

pos, neg = make_persona_pairs(
    tok, n_pairs=identity['requested_pairs'], thinking=identity['thinking'],
    persona_pairs=[tuple(pair) for pair in identity['pairs']],
    template=identity['template'], seed=identity['seed'],
)
assert len(pos) == len(neg) == identity['actual_pairs'] == 200
entries = load_suffixes(thinking=identity['thinking'])
rng = random.Random(identity['seed'])
sampled = rng.sample(entries, min(identity['requested_pairs'], len(entries)))
for entry in sampled:
    rng.choice(identity['pairs'])
texts = pos + neg
encoded = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
ids = encoded['input_ids']
offsets = encoded['offset_mapping']
assert len(ids) == 2 * len(pos)
rows = []
for i, entry in enumerate(sampled):
    sides = []
    for side, text, tokens, spans in (
        ('positive', pos[i], ids[i], offsets[i]),
        ('negative', neg[i], ids[i + len(pos)], offsets[i + len(pos)]),
    ):
        assert text.endswith(entry['suffix'])
        clipped = tokens[:64]
        assert clipped == tok(text, add_special_tokens=False, truncation=True, max_length=64)['input_ids']
        suffix_start = len(text) - len(entry['suffix'])
        suffix_tokens = sum(end > suffix_start for start, end in spans)
        suffix_tokens_retained = sum(end > suffix_start for start, end in spans[:64])
        clipped_text = tok.decode(clipped, skip_special_tokens=False)
        sides.append({'side': side, 'tokens': len(tokens), 'clipped_tokens': len(clipped),
            'truncated': len(tokens) > 64,
            'assistant_header_seen': '<|im_start|>assistant' in clipped_text,
            'suffix_tokens': suffix_tokens, 'suffix_tokens_retained': suffix_tokens_retained,
            'suffix_started': suffix_tokens_retained > 0,
            'suffix_fully_retained': suffix_tokens == suffix_tokens_retained,
            'persona_string_seen': identity['pairs'][0][0 if side == 'positive' else 1] in clipped_text,
            'full_last32': tok.decode(tokens[-32:], skip_special_tokens=False),
            'clipped_last32': tok.decode(clipped[-32:], skip_special_tokens=False)})
    rows.append({'index': i, 'category': entry['cat'],
        'source_suffix_sha256': hashlib.sha256(entry['suffix'].encode()).hexdigest(),
        'pair_identical_after_clip': ids[i][:64] == ids[i + len(pos)][:64],
        'pair_identical_after_skip_first16': ids[i][16:64] == ids[i + len(pos)][16:64],
        'positive': sides[0], 'negative': sides[1]})

def quantiles(values):
    ordered = sorted(values)
    return {str(q): ordered[round(q / 100 * (len(ordered) - 1))] for q in (0, 10, 25, 50, 75, 90, 95, 100)}

def counts(side):
    group = [row[side] for row in rows]
    return {'total': len(group), 'truncated': sum(row['truncated'] for row in group),
        'assistant_header_missing': sum(not row['assistant_header_seen'] for row in group),
        'suffix_not_started': sum(not row['suffix_started'] for row in group),
        'suffix_not_fully_retained': sum(not row['suffix_fully_retained'] for row in group),
        'persona_string_missing': sum(not row['persona_string_seen'] for row in group),
        'token_length_quantiles': quantiles([row['tokens'] for row in group])}

summary = {'author': 'PI/gpt-6-sol', 'model_id': 'Qwen/Qwen3.5-4B',
    'tokenizer_snapshot': str(snapshot),
    'tokenizer_config_sha256': hashlib.sha256((snapshot / 'tokenizer_config.json').read_bytes()).hexdigest(),
    'tokenizer_truncation_side': tok.truncation_side, 'persona_identity': identity,
    'max_length': 64, 'counts': {'positive': counts('positive'), 'negative': counts('negative'),
        'pair_identical_after_clip': sum(row['pair_identical_after_clip'] for row in rows),
        'pair_identical_after_skip_first16': sum(row['pair_identical_after_skip_first16'] for row in rows),
        'positive_only_truncated': sum(row['positive']['truncated'] and not row['negative']['truncated'] for row in rows),
        'negative_only_truncated': sum(row['negative']['truncated'] and not row['positive']['truncated'] for row in rows),
        'exceed_reference_384': {
            side: sum(row[side]['tokens'] > 384 for row in rows)
            for side in ('positive', 'negative')},
    },
    'categories': {cat: {'n': sum(row['category'] == cat for row in rows),
        'truncated_pairs': sum(row['category'] == cat and row['positive']['truncated'] for row in rows)}
        for cat in sorted({row['category'] for row in rows})},
    'rows': rows}
out = root / 'slop/verification/20260923_extraction_token_coverage.json'
out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, indent=2))
