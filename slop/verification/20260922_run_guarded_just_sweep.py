"""Run the authorized actual just/CLI replay with no paid dispatch. — PI/OpenAI"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

root=Path.cwd()
run=root/'outputs/bsbench-v2'
base=root/'slop/verification/20260922_actual-just-sweep'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()=='c7d23dd06fe818d4479a9539825a7fa98dd749c1'
assert not subprocess.check_output(['git','status','--porcelain','--','src','scripts/run_bsbench_sweep.py'],text=True)
assert not (root/'.local/bsbench-cli-proof/no-shell-env').exists()
for suffix in ('-armed.json','-counts.json'):
    assert not Path(str(base)+suffix).exists(), 'Do not overwrite prior guard evidence'
bootstrap=root/'.local/bsbench-cli-proof/sitecustomize.py'
compile(bootstrap.read_text(),str(bootstrap),'exec')
shutil.copyfile(bootstrap,Path(str(base)+'-sitecustomize.py'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot():
    package=root/'src/steering_lite'
    return {'source_files':{str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*.py'))},
            'entrypoint_sha256':digest(root/'scripts/run_bsbench_sweep.py'),
            'summary_sha256':digest(run/'run-summary.json'),'ledger_sha256':digest(run/'costs.jsonl'),
            'ledger_bytes':(run/'costs.jsonl').stat().st_size,
            'provider_files':{str(p):digest(p) for p in sorted((run/'provider-evidence').rglob('*')) if p.is_file()},
            'cache_files':{str(p):digest(p) for p in sorted((run/'cache').rglob('*')) if p.is_file()},
            'vector_files':{str(p):digest(p) for p in sorted((run/'artifacts/vectors').rglob('*')) if p.is_file()}}
def save(suffix,value):
    Path(str(base)+suffix).write_text(json.dumps(value,indent=2)+'\n')
before=snapshot();save('-before.json',before)
prior=json.loads((run/'run-summary.json').read_text())
command=['just','sweep','--run --backend real --judge-pricing slop/verification/20260922_v4-provider-endpoint-metadata.json','Qwen/Qwen3.5-4B','outputs/bsbench-v2','.local/bsbench-cli-proof/no-shell-env']
env=dict(os.environ)
env['PYTHONPATH']=str(bootstrap.parent)+os.pathsep+str(root/'src')
print('Running one guarded actual just sweep; full output: '+str(base)+'.command.log',flush=True)
with Path(str(base)+'.command.log').open('w') as output:
    result=subprocess.run(command,cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
after=snapshot();save('-after.json',after)
assert Path(str(base)+'-armed.json').exists(), 'CLI guard never armed'
assert Path(str(base)+'-counts.json').exists(), 'CLI guard exit evidence missing'
armed=json.loads(Path(str(base)+'-armed.json').read_text());counts=json.loads(Path(str(base)+'-counts.json').read_text())
proof={'author':'PI/OpenAI','command':command,'exit_code':result.returncode,'passed':False,'guard_armed':armed['armed'],
       'same_guard_process':armed['pid']==counts['pid'],'callback_attempts':counts['callback_attempts'],
       'unchanged':{key:before[key]==after[key] for key in before},'scientific_identity_sha256':prior['identity_sha256']}
save('-proof.json',proof)
assert result.returncode==0
assert proof['same_guard_process'] and proof['guard_armed']
assert counts['callback_attempts']=={'gpu':0,'judge':0,'reservation':0}
assert all(proof['unchanged'].values())
assert json.loads((run/'run-summary.json').read_text())['identity_sha256']==prior['identity_sha256']
proof['passed']=True;save('-proof.json',proof)
print(json.dumps(proof,indent=2))
