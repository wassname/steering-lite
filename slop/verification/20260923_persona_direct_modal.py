"""One isolated direct-prompt persona manipulation diagnostic; no vector or judge. — PI/gpt-6-sol"""
import hashlib
import json
import time
from pathlib import Path

import modal

from scripts.run_bsbench_modal import cache, image
from steering_lite.benchmark.sweep import MODAL_GPU_STAGE_TIMEOUT_SECONDS

app = modal.App('steering-lite-bsbench-persona-direct-diagnostic')
diagnostics = modal.Volume.from_name('steering-lite-bsbench-persona-diagnostics', create_if_missing=True)


def format_prompt(tokenizer, prompt: str, persona: str | None, template: str) -> str:
    prefix = '' if persona is None else template.format(persona=persona) + '\n\n'
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prefix + prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@app.function(
    gpu='A10G', image=image.env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}),
    serialized=True, volumes={'/cache': cache, '/diagnostic': diagnostics},
    timeout=MODAL_GPU_STAGE_TIMEOUT_SECONDS,
)
def compare_direct_personas(spec: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering_lite.benchmark.cache import content_key, source_hash
    from steering_lite.benchmark.generation import generate, health

    result_path = Path('/diagnostic') / f"{spec['diagnostic_id']}.json"
    diagnostics.reload()
    if result_path.exists():
        return json.loads(result_path.read_text())
    started = time.monotonic()
    if content_key({k: v for k, v in spec.items() if k != 'diagnostic_id'}) != spec['diagnostic_id']:
        raise ValueError('diagnostic payload identity differs')
    if source_hash() != spec['source_sha256']:
        raise ValueError('scientific source differs from approved diagnostic')
    if spec['generation'] != {'max_new_tokens': 128, 'do_sample': False, 'enable_thinking': False}:
        raise ValueError('generation settings differ')
    snapshot = Path('/cache/hf/hub/models--Qwen--Qwen3.5-4B/snapshots') / spec['model_revision']
    if not snapshot.is_dir():
        raise ValueError('pinned model snapshot is absent')
    for filename, digest in spec['snapshot_sha256'].items():
        if hashlib.sha256((snapshot / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'pinned snapshot differs at {filename}')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    if tokenizer.truncation_side != 'right' or tuple(model.config.layer_types) != tuple(spec['layer_types']):
        raise ValueError('model layer types or tokenizer truncation side differs')

    observations = {}
    for condition, persona in spec['conditions'].items():
        formatted = [format_prompt(tokenizer, prompt, persona, spec['persona_template']) for prompt in spec['prompts']]
        digests = [hashlib.sha256(prompt.encode()).hexdigest() for prompt in formatted]
        if digests != spec['formatted_prompt_sha256s'][condition]:
            raise ValueError(f'formatted prompts differ for {condition}')
        answers = generate(model, tokenizer, formatted, 1, spec['generation']['max_new_tokens'])
        metrics, reasons = health(tokenizer, answers)
        observations[condition] = {
            'persona': persona, 'formatted_prompt_sha256s': digests, 'answers': answers,
            'health': {'metrics': metrics, 'reasons': reasons},
        }
    result = {
        'schema': 'bsbench-persona-direct-diagnostic-result-v1',
        'diagnostic_id': spec['diagnostic_id'], 'source_sha256': spec['source_sha256'],
        'design_sha256': spec['design_sha256'], 'model_revision': spec['model_revision'],
        'snapshot_sha256': spec['snapshot_sha256'], 'generation': spec['generation'],
        'persona_template': spec['persona_template'], 'conditions': spec['conditions'],
        'prompt_ids': spec['prompt_ids'], 'prompts': spec['prompts'],
        'formatted_prompt_sha256s': spec['formatted_prompt_sha256s'], 'observations': observations,
        'cost_receipt': {'provider': 'Modal', 'status': 'pending', 'usage': {'elapsed_seconds': time.monotonic() - started}},
    }
    temporary = result_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(result_path)
    diagnostics.commit()
    return result
