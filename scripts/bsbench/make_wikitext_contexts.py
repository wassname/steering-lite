"""Write src/steering_lite/data/wikitext_contexts.json: generic text for wiki_mean_vjp.

Consecutive WikiText-2 train lines (headings skipped) are joined until each context has at least
MIN_TOKENS Qwen3.5 tokens, so the extractor can cut each one to the length of a matched persona prompt.

    uv run python scripts/bsbench/make_wikitext_contexts.py
"""

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

N_CONTEXTS = 1024
MIN_TOKENS = 400  # > walk.py --max-length 384
OUT = Path(__file__).resolve().parents[2] / "src/steering_lite/data/wikitext_contexts.json"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
contexts, current = [], []
for line in rows["text"]:
    line = line.strip()
    if not line or line.startswith("="):
        continue
    current.append(line)
    text = " ".join(current)
    if len(tokenizer(text).input_ids) >= MIN_TOKENS:
        contexts.append(text)
        current = []
        if len(contexts) == N_CONTEXTS:
            break
assert len(contexts) == N_CONTEXTS
OUT.write_text(json.dumps({
    "source": "Salesforce/wikitext wikitext-2-raw-v1 train, consecutive non-heading lines",
    "min_tokens": MIN_TOKENS, "tokenizer": "Qwen/Qwen3.5-4B", "contexts": contexts,
}, ensure_ascii=False) + "\n")
print(f"wrote {len(contexts)} contexts to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
