set shell := ["bash", "-cu"]
set dotenv-load

default:
	@just --list

check: test smoke

test:
	uv run --extra test --extra hf-test pytest -q

# library smoke: every method extract -> attach -> generate -> save/load on tiny models
smoke:
	uv run --extra test --extra hf-test pytest -q tests/test_pipeline.py

# bsbench walk on CPU, tiny random qwen3, 8-token answers, 2 rungs
smoke-bsbench method="mean_diff":
	uv run --extra benchmark python scripts/bsbench/walk.py {{method}} --smoke --model wassname/qwen3-5lyr-tiny-random --device cpu --dtype float32 --n-pairs 4 --batch-size 8 --extract-batch-size 2 --max-length 256 --layers 1,2 --target-layer 4 --max-rungs 2

# dose walks on Modal (cached per question), then pull outputs/bsbench back
sweep cohort="dev" methods="mean_diff,pca,vjp_delta,vjp_cache,kv_cache_gram,prompting" seeds="0" random_seeds="0,1,2,3,4":
	uv run --extra benchmark modal run scripts/bsbench/run_modal.py::main --cohort {{cohort}} --methods {{methods}} --seeds {{seeds}}
	uv run --extra benchmark modal run scripts/bsbench/run_modal.py::main --cohort {{cohort}} --methods random --seeds {{random_seeds}}
	just pull

pull:
	uv run --extra benchmark modal volume get --force steering-lite-bsbench-v3 bsbench outputs/

# judge every COMPLETE walk (OpenRouter), then render points.json, tables and plot
results cohort="dev":
	cd scripts/bsbench && uv run --extra benchmark python judge.py --cohort {{cohort}} --refresh
	cd scripts/bsbench && uv run --extra benchmark python results.py --cohort {{cohort}}
