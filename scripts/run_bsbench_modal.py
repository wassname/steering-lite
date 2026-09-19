"""Serializable Modal callbacks for the audited BS-bench production stages.

Importing this file defines callbacks but never dispatches them. The local CLI must
supply ``--run``, the real backend selection, credentials, and a budget preflight
before calling ``run_stage.remote``.
"""
from __future__ import annotations

import time
from pathlib import Path

import modal

from steering_lite.benchmark.sweep import MODAL_GPU_STAGE_TIMEOUT_SECONDS

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "beartype==0.22.9",
        "torch==2.11.0",
        "transformers==5.12.1",
        "safetensors==0.7.0",
        "einops==0.8.2",
        "jaxtyping==0.3.9",
        "loguru==0.7.3",
        "tabulate==0.9.0",
        "tqdm==4.67.3",
    )
    .env({"PYTHONPATH": "/repo/src", "HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(Path(__file__).parents[1] / "src", "/repo/src")
)
app = modal.App("steering-lite-bsbench")
cache = modal.Volume.from_name("steering-lite-bsbench-cache", create_if_missing=True)


def canonical_prompt_texts(tokenizer, prompts: list[str], prompt_spec: dict, *, persona: str | None = None) -> list[str]:
    instruction = prompt_spec["template"]
    thinking = prompt_spec.get("enable_thinking", False)
    prefix = "" if persona is None else f"Answer as someone who is {persona}.\n\n"
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + prompt + " " + instruction}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        for prompt in prompts
    ]


def canonical_prompt_ids(tokenizer, prompts: list[str], prompt_spec: dict, *, persona: str | None = None):
    return [
        tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        for text in canonical_prompt_texts(tokenizer, prompts, prompt_spec, persona=persona)
    ]


def _layers(model) -> tuple[tuple[int, ...], int]:
    full_attention = [index for index, kind in enumerate(model.config.layer_types) if kind == "full_attention"]
    if not full_attention:
        raise ValueError("Qwen3.5 model has no full-attention source layer")
    source_layer = full_attention[0]
    target_layer = source_layer + 1
    if target_layer >= len(model.config.layer_types):
        raise ValueError("Qwen3.5 source layer has no following target layer")
    return (source_layer,), target_layer


def _candidate_policy(vector, model, tokenizer, prompts: list[str], *, limit: int, max_new_tokens: int, generate, health) -> tuple[list[float], list[dict], dict, dict]:
    """Double one dose at a time; stop on the first health failure or the policy limit."""
    coefficients: list[float] = []
    items: list[dict] = []
    health_by_coefficient: dict[str, dict] = {}
    history: list[dict] = []
    coefficient = 0.1
    for iteration in range(limit):
        with vector(model, C=coefficient):
            answers = generate(model, tokenizer, prompts, 1, max_new_tokens)
        metrics, reasons = health(tokenizer, answers)
        record = {"iteration": iteration + 1, "coefficient": coefficient, "metrics": metrics, "reasons": reasons}
        history.append(record)
        health_by_coefficient[str(float(coefficient))] = record
        coefficients.append(coefficient)
        items.extend(
            {
                "coefficient": coefficient,
                "prompt_index": index,
                "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                "response": answer,
            }
            for index, (prompt, answer) in enumerate(zip(prompts, answers, strict=True))
        )
        if reasons:
            return coefficients, items, health_by_coefficient, {"schema": "bsbench-successive-health-bracket-v1", "limit": limit, "history": history, "termination": "coherence_failure"}
        coefficient *= 2
    return coefficients, items, health_by_coefficient, {"schema": "bsbench-successive-health-bracket-v1", "limit": limit, "history": history, "termination": "search_limit"}


@app.function(gpu="A10G", image=image, volumes={"/cache": cache}, timeout=MODAL_GPU_STAGE_TIMEOUT_SECONDS)
def run_stage(*, stage: str, method: str, config: dict, prompts: list[str], model_id: str = "Qwen/Qwen3.5-4B") -> dict:
    """Run exactly one production GPU stage and return only serializable data."""
    import base64
    import hashlib

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import steering_lite as sl
    from steering_lite.benchmark.generation import generate, health
    from steering_lite.benchmark.pipeline import method_config, run_method
    from steering_lite.data import make_persona_pairs, persona_corpus_identity

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().cuda()
    if stage == "generation":
        condition = config["condition"]
        if method != condition or method not in {"bare", "prompting"}:
            raise ValueError("direct generation stage requires matching bare or prompting condition")
        rows = [{"prompt": prompt} for prompt in prompts]
        result = run_method(
            model,
            tokenizer,
            method,
            [],
            [],
            vector_dir=Path("/cache/bsbench-direct-vectors"),
            rows=rows,
            layers=_layers(model)[0],
            target_layer=_layers(model)[1],
            max_new_tokens=config["prompt_spec"]["max_new_tokens"],
        )
        metrics, reasons = health(tokenizer, result["answers"])
        prompt_ids = config["prompt_ids"]
        if not isinstance(prompt_ids, list) or len(prompt_ids) != len(result["answers"]):
            raise ValueError("direct generation requires numbered prompt identities")
        return {
            "cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": time.monotonic() - started}},
            "answers": result["answers"],
            "health_records": [
                {"question_id": prompt_id, "metrics": metrics, "reasons": reasons}
                for prompt_id in prompt_ids
            ],
        }
    if stage == "calibration-candidates":
        identity = config["persona_source"]
        observed_corpus = persona_corpus_identity(thinking=identity["thinking"])
        if observed_corpus["actual_pairs"] != identity["actual_pairs"] or observed_corpus["corpus_sha256"] != identity["corpus_sha256"]:
            raise ValueError("persona corpus differs from the cached extraction identity")
        pos_prompts, neg_prompts = make_persona_pairs(
            tokenizer,
            n_pairs=identity["requested_pairs"],
            thinking=identity["thinking"],
            persona_pairs=[tuple(pair) for pair in identity["pairs"]],
            template=identity["template"],
            seed=identity["seed"],
        )
        if len(pos_prompts) != identity["actual_pairs"] or len(neg_prompts) != identity["actual_pairs"]:
            raise ValueError("persona extraction actual pair count differs from cached identity")
        layers, target_layer = _layers(model)
        vector = sl.train(
            model,
            tokenizer,
            pos_prompts,
            neg_prompts,
            method_config(method, layers=layers, target_layer=target_layer, seed=identity["seed"]),
            batch_size=1,
            max_length=64,
        )
        vector_path = Path("/tmp") / f"bsbench-{method}.safetensors"
        vector.save(str(vector_path))
        calibration_prompts = canonical_prompt_texts(tokenizer, prompts, config["prompt_spec"])
        baseline_answers = generate(model, tokenizer, calibration_prompts, 1, config["prompt_spec"]["max_new_tokens"])
        coefficients, candidate_items, candidate_health, search_history = _candidate_policy(
            vector,
            model,
            tokenizer,
            calibration_prompts,
            limit=config["candidate_dose_upper"],
            max_new_tokens=config["prompt_spec"]["max_new_tokens"],
            generate=generate,
            health=health,
        )
        cache.commit()
        return {
            "cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": time.monotonic() - started, "persona_actual_pairs": len(pos_prompts)}},
            "vector_bytes": vector_path.read_bytes(),
            "baseline_answers": baseline_answers,
            "candidate_coefficients": coefficients,
            "candidate_items": candidate_items,
            "candidate_health": candidate_health,
            "candidate_search": search_history,
            "method_config": vector.cfg.to_dict(),
        }
    if stage == "final-generation":
        artifact = config["vector_artifact"]
        vector_bytes = base64.b64decode(artifact["vector_bytes_b64"])
        if hashlib.sha256(vector_bytes).hexdigest() != artifact["sha256"]:
            raise ValueError("remote vector bytes do not match the persisted artifact SHA256")
        vector_path = Path("/tmp") / f"{artifact['sha256']}.safetensors"
        vector_path.write_bytes(vector_bytes)
        vector = sl.Vector.load(str(vector_path))
        target = None
        transfer_predictions = None
        final_dose_plans = None
        if config.get("schema") == "bsbench-remote-vector-final-v1":
            from steering_lite.benchmark.dose_search import CALIBRATION_CASE, TRANSFER_CASES, final_dose_plan, fit_target, predict_transfer
            from steering_lite.benchmark.transfer_data import PromptRecord

            records = {
                case_id: tuple(PromptRecord(**record) for record in case_records)
                for case_id, case_records in config["transfer_prompt_records"].items()
            }
            target = fit_target(
                vector,
                model,
                tokenizer,
                canonical_prompt_ids(tokenizer, config["calibration_prompts"], config["prompt_spec"]),
                CALIBRATION_CASE,
                config["observed"],
                method=method,
                model_id=model_id,
                measure_kwargs={},
            )
            transfer_predictions = [
                predict_transfer(
                    vector,
                    model,
                    tokenizer,
                    canonical_prompt_ids(tokenizer, [record.prompt for record in records[case.case_id]], config["prompt_spec"]),
                    target,
                    case,
                    bracket=(0.01, 2.0),
                    solver_kwargs={},
                )
                for case in TRANSFER_CASES
            ]
            final_dose_plans = [final_dose_plan(prediction) for prediction in transfer_predictions]
            plans_by_case = {plan["case"]["case_id"]: plan for plan in final_dose_plans}
            plan = [
                {
                    "case_id": case.case_id,
                    "target_id": plans_by_case[case.case_id]["target_id"],
                    "coefficient": coefficient,
                    "prompt_id": record.prompt_id,
                    "prompt": record.prompt,
                    "prompt_sha256": record.content_sha256,
                }
                for case in TRANSFER_CASES
                for record in records[case.case_id]
                for coefficient in plans_by_case[case.case_id]["coefficients"]
            ]
        else:
            plan = config["executable_generation_plan"]
        unique_prompts = {item["prompt_id"]: item["prompt"] for item in plan}
        baseline_answers = dict(zip(unique_prompts, generate(model, tokenizer, canonical_prompt_texts(tokenizer, list(unique_prompts.values()), config["prompt_spec"]), 1, config["prompt_spec"]["max_new_tokens"]), strict=True))
        answers = []
        health_records = []
        for item in plan:
            with vector(model, C=item["coefficient"]):
                answer = generate(model, tokenizer, canonical_prompt_texts(tokenizer, [item["prompt"]], config["prompt_spec"]), 1, config["prompt_spec"]["max_new_tokens"])[0]
            metrics, reasons = health(tokenizer, [answer])
            answers.append(answer)
            health_records.append({"case_id": item["case_id"], "prompt_id": item["prompt_id"], "coefficient": item["coefficient"], "metrics": metrics, "reasons": reasons})
        cache.commit()
        from steering_lite.benchmark.cache import content_key
        result = {
            "cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": time.monotonic() - started}},
            "baseline_answers": baseline_answers,
            "answers": answers,
            "health_records": health_records,
            "plan_sha256": content_key({"plan": plan}),
        }
        if target is not None:
            result |= {
                "target": target,
                "transfer_predictions": transfer_predictions,
                "final_dose_plans": final_dose_plans,
                "executable_generation_plan": plan,
            }
        return result
    raise ValueError(f"unsupported Modal stage {stage!r}")


def remote_stage_call(model_id: str, *, explicit_run: bool, budget_preflight: dict):
    """Return a stage callback that checks explicit selection and budget before dispatch."""
    from steering_lite.benchmark.adapters import RunGate

    gate = RunGate(explicit_run=explicit_run, budget_preflight=budget_preflight)

    def call(*, stage, method, config, prompts):
        gate.require()
        return run_stage.remote(
            stage=stage,
            method=method,
            config=config,
            prompts=prompts,
            model_id=model_id,
        )

    return call


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen3.5-4B"):
    raise RuntimeError("Use scripts/run_bsbench_sweep.py --run --backend real after budget preflight")
