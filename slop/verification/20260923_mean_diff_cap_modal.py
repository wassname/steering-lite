"""One paid, explicitly authorized 64-vs-384 mean-difference diagnostic. — PI/gpt-6-sol"""
import base64
import hashlib
import json
import time
from pathlib import Path

import modal

from scripts.run_bsbench_modal import cache, image
from steering_lite.benchmark.sweep import MODAL_GPU_STAGE_TIMEOUT_SECONDS

app = modal.App('steering-lite-bsbench-cap-diagnostic')
diagnostics = modal.Volume.from_name('steering-lite-bsbench-cap-diagnostics', create_if_missing=True)


@app.function(
    gpu='A10G', image=image, serialized=True,
    volumes={'/cache': cache, '/diagnostic': diagnostics},
    timeout=MODAL_GPU_STAGE_TIMEOUT_SECONDS,
)
def compare_mean_diff_caps(spec: dict, old_vector_bytes_b64: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import steering_lite as sl
    from steering_lite.benchmark.cache import content_key, source_hash
    from steering_lite.benchmark.dose_search import BENCHMARK_KL_SPEC
    from steering_lite.benchmark.generation import generate, generation_inputs, health
    from steering_lite.benchmark.pipeline import method_config
    from steering_lite.calibrate import measure_kl
    from steering_lite.data.personas import make_persona_pairs, persona_corpus_identity

    result_path = Path('/diagnostic') / f"{spec['diagnostic_id']}.json"
    diagnostics.reload()
    if result_path.exists():
        return json.loads(result_path.read_text())
    start = time.monotonic()
    if content_key({key: value for key, value in spec.items() if key != 'diagnostic_id'}) != spec['diagnostic_id']:
        raise ValueError('diagnostic payload identity differs')
    if spec['coefficients'] != {'+C': 0.8, '-C': -0.8} or spec['new_extraction_max_length'] != 384 or spec['method'] != 'mean_diff':
        raise ValueError('diagnostic must use only the fixed signed dose and extraction cap')
    if source_hash() != spec['source_sha256'] or BENCHMARK_KL_SPEC != spec['kl_spec']:
        raise ValueError('diagnostic source or KL specification changed')
    snapshot = Path('/cache/hf/hub/models--Qwen--Qwen3.5-4B/snapshots') / spec['model_revision']
    if not snapshot.is_dir() or snapshot.name != spec['model_revision']:
        raise ValueError('pinned model snapshot is absent from the existing Modal HF volume')
    for filename, digest in spec['snapshot_sha256'].items():
        if hashlib.sha256((snapshot / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'pinned model snapshot differs at {filename}')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    if tokenizer.truncation_side != 'right' or tuple(model.config.layer_types) != tuple(spec['layer_types']):
        raise ValueError('model layer types or tokenizer truncation side differ from the signed run')

    persona = spec['persona_identity']
    if persona_corpus_identity(thinking=persona['thinking']) != {
        'actual_pairs': persona['actual_pairs'], 'corpus_sha256': persona['corpus_sha256'], 'thinking': persona['thinking'],
    }:
        raise ValueError('persona corpus differs from signed run')
    positive, negative = make_persona_pairs(
        tokenizer, n_pairs=persona['requested_pairs'], thinking=persona['thinking'],
        persona_pairs=[tuple(pair) for pair in persona['pairs']],
        template=persona['template'], seed=persona['seed'],
    )
    if len(positive) != len(negative) or len(positive) != persona['actual_pairs']:
        raise ValueError('persona pair count differs')
    old_bytes = base64.b64decode(old_vector_bytes_b64)
    if hashlib.sha256(old_bytes).hexdigest() != spec['old_vector_sha256']:
        raise ValueError('old vector bytes differ from immutable sidecar')
    old_path = Path('/tmp/cap-old.safetensors')
    old_path.write_bytes(old_bytes)
    old = sl.Vector.load(str(old_path))
    if old.cfg.to_dict() != spec['old_vector_config']:
        raise ValueError('old vector config differs from signed run')
    layers = tuple(old.cfg.layers)
    new = sl.train(
        model, tokenizer, positive, negative,
        method_config('mean_diff', layers=layers, target_layer=spec['target_layer'], seed=persona['seed']),
        batch_size=1, max_length=384,
    )
    new_path = Path('/tmp/cap-new.safetensors')
    new.save(str(new_path))
    new_bytes = new_path.read_bytes()
    if json.loads(json.dumps(new.cfg.to_dict())) != old.cfg.to_dict():
        raise ValueError('old and new vector configs differ apart from extraction cap')

    directions = {}
    for layer in layers:
        old_v = old.stacked[layer]['v'][0].float()
        new_v = new.stacked[layer]['v'][0].float()
        directions[str(layer)] = {
            'old_norm': float(old_v.norm()), 'new_norm': float(new_v.norm()),
            'cosine': float(torch.nn.functional.cosine_similarity(old_v, new_v, dim=0)),
        }
    prompts = generation_inputs(tokenizer, [{'prompt': prompt} for prompt in spec['prompts']])
    if [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts] != spec['formatted_prompt_sha256s']:
        raise ValueError('formatted calibration prompts changed')
    prompt_ids = [tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids[0] for prompt in prompts]
    bare_before = generate(model, tokenizer, prompts, 1, spec['max_new_tokens'])
    measurements_by_cap = {}
    for label, vector in (('old64', old), ('new384', new)):
        measurements_by_cap[label] = {}
        for side, coefficient in spec['coefficients'].items():
            with vector(model, C=coefficient):
                answers = generate(model, tokenizer, prompts, 1, spec['max_new_tokens'])
            metrics, reasons = health(tokenizer, answers)
            vector.cfg.coeff = coefficient
            kl_path = Path('/tmp') / f"cap-kl-{label}-{side}.jsonl"
            kl_path.unlink(missing_ok=True)
            kl = measure_kl(
                vector, model, tokenizer, prompt_ids,
                T=spec['kl_spec']['T'], do_sample=spec['kl_spec']['do_sample'],
                seed=spec['kl_spec']['seed'], show_pbar=False, demo_log_path=kl_path,
            )
            kl_rows = [json.loads(line) for line in kl_path.read_text().splitlines()]
            if len(kl_rows) != len(prompts):
                raise ValueError('per-prompt KL trace count differs from calibration prompts')
            measurements_by_cap[label][side] = {'coefficient': coefficient, 'answers': answers,
                                 'health': {'metrics': metrics, 'reasons': reasons}, 'kl': kl, 'per_prompt_kl': kl_rows}
    bare_after = generate(model, tokenizer, prompts, 1, spec['max_new_tokens'])
    result = {
        'schema': 'bsbench-mean-diff-cap-diagnostic-result-v1',
        'diagnostic_id': spec['diagnostic_id'],
        'source_sha256': spec['source_sha256'],
        'model_revision': spec['model_revision'],
        'snapshot_sha256': spec['snapshot_sha256'],
        'persona_identity': persona,
        'prompts': spec['prompts'],
        'formatted_prompt_sha256s': spec['formatted_prompt_sha256s'],
        'old_vector_sha256': spec['old_vector_sha256'],
        'new_vector_sha256': hashlib.sha256(new_bytes).hexdigest(),
        'new_vector_bytes_b64': base64.b64encode(new_bytes).decode(),
        'direction_comparison': directions,
        'bare_before': bare_before, 'bare_after': bare_after,
        'measurements_by_cap': measurements_by_cap,
        'cost_receipt': {'provider': 'Modal', 'status': 'pending', 'usage': {'elapsed_seconds': time.monotonic() - start}},
    }
    temporary = result_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(result_path)
    diagnostics.commit()
    return result
