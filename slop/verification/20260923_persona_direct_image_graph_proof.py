"""Exercise Modal 1.5.5's post-mount build validation without a provider call. — PI/gpt-6-sol"""
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from modal._utils.async_utils import synchronizer
from modal.exception import InvalidError
from modal.mount import _Mount

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CALLBACK = ROOT / 'slop/verification/20260923_persona_direct_modal.py'
LEDGER = ROOT / 'outputs/bsbench-v2/costs.jsonl'

before = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
definition = importlib.util.spec_from_file_location('bsbench_persona_direct_modal', CALLBACK)
module = importlib.util.module_from_spec(definition)
definition.loader.exec_module(module)
image = module.compare_direct_personas._spec_.image
if image is not synchronizer._translate_in(module.image):
    raise ValueError('function does not use the unchanged production image')
deps = image._deps()
if len(deps) != 2 or not isinstance(deps[1], _Mount) or deps[0] is image:
    raise ValueError('the final image must add source as a terminal runtime mount')
mount = deps[1]
old_appended_env = synchronizer._translate_in(module.image.env({'HF_HUB_OFFLINE': '1'}))
if tuple(old_appended_env._deps()) != (image,):
    raise ValueError('old image graph does not contain the final mount as build-step parent')

old_mounts = image._deferred_mounts
image._deferred_mounts = (mount,)
try:
    try:
        asyncio.run(old_appended_env._load(old_appended_env, None, None, None))
    except InvalidError as exc:
        old_rejection = str(exc).splitlines()[0]
        if 'build step after using `image.add_local_*`' not in old_rejection:
            raise
    else:
        raise AssertionError('old appended-env graph unexpectedly passed Modal validation')
finally:
    image._deferred_mounts = old_mounts

deps[0]._assert_no_mount_layers()
after = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
proof = {
    'author': 'PI/gpt-6-sol', 'modal_version': '1.5.5',
    'callback_sha256': hashlib.sha256(CALLBACK.read_bytes()).hexdigest(),
    'old_appended_env_rejection': old_rejection,
    'repaired_function_image_is_production_image': True,
    'final_image_deps': ['production build base', 'terminal source runtime Mount'],
    'production_build_base_passes_mount_validation': True,
    'post_mount_build_step_in_repaired_graph': False,
    'provider_calls': 0, 'ledger_sha256_before': before,
    'ledger_sha256_after': after, 'ledger_unchanged': before == after,
}
path = ROOT / 'slop/verification/20260923_persona_direct_image_graph_proof.json'
path.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
