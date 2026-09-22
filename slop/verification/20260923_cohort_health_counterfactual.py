"""Compare saved final health units without changing score or artifacts. — PI/gpt-6-sol"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run = root / 'outputs/bsbench-v2/results'
out = root / 'slop/research/20260923_calibration_boundary'
data = json.loads((run / 'measured-points.json').read_text())
groups = defaultdict(list)
examples_by_id = {}
for point in data['points']:
    if point['method'] in ('bare', 'prompting'):
        continue
    evidence = json.loads((run / point['raw_evidence']).read_text())
    examples = evidence['examples']
    assert len(examples) == 20
    counts = {k: sum(e['health']['metrics'][k] for e in examples) for k in ('unfinished', 'role_leaks', 'repeated')}
    cohort_healthy = counts['unfinished'] < 10 and counts['role_leaks'] < 5 and counts['repeated'] < 5
    assert point['healthy'] == (point['health_flags'] == 0)
    row = point | {'cohort_healthy': cohort_healthy, 'cohort_counts': counts}
    groups[point['method'], point['random_seed'], point['side']].append(row)
    examples_by_id[point['point_id']] = examples
assert len(groups) == 20
assert all(p['generation_health']['metrics']['answers']==4 for p in data['calibration'])
for transfer in data['transfer_groups']:
    metrics = [e['health']['metrics'] for e in transfer['examples']]
    assert len(metrics) == 2
    fail = sum(e['health']['reasons'] != [] for e in transfer['examples']) > 0
    cohort_fail = sum(m['unfinished'] for m in metrics) >= 1 or sum(m['role_leaks'] for m in metrics) >= 1 or sum(m['repeated'] for m in metrics) >= 1
    assert fail == cohort_fail

def choose(group, criterion):
    eligible = [p for p in group if p[criterion]]
    assert eligible
    return max(eligible, key=lambda p: (p['dose_score'], p['directed_intended_effect'], -p['magnitude']))

def pattern(group, criterion):
    marks = ''.join('H' if p[criterion] else 'F' for p in sorted(group,key=lambda p:p['multiplier']))
    status = ('all-healthy' if marks == 'HHH' else 'all-failed' if marks == 'FFF' else
              'nonmonotonic' if 'FH' in marks else 'bracketed' if marks in ('HHF','HFF') else 'failed-below-healthy')
    return marks,status

rows=[]
for (method,seed,side),group in sorted(groups.items()):
    old,new = choose(group,'healthy'),choose(group,'cohort_healthy')
    old_pattern,old_status=pattern(group,'healthy'); new_pattern,new_status=pattern(group,'cohort_healthy')
    rows.append({'method':method,'seed':seed,'side':side,'old_health':old_pattern,'cohort_health':new_pattern,
        'old_boundary':old_status,'cohort_boundary':new_status,'old_best_multiplier':old['multiplier'],'cohort_best_multiplier':new['multiplier'],
        'old_best_score':old['dose_score'],'cohort_best_score':new['dose_score'],'delta_score':new['dose_score']-old['dose_score'],
        'old_best_id':old['point_id'],'cohort_best_id':new['point_id'],
        'old_eligible':sum(p['healthy'] for p in group),'cohort_eligible':sum(p['cohort_healthy'] for p in group)})
assert len(rows)==20
with (out/'cohort-health-only-selections.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def ranks(criterion):
    return {side: [(r['method'],r['seed']) for r in sorted((r for r in rows if r['side']==side), key=lambda r:r[criterion],reverse=True)] for side in ('+C','-C')}
old_order=ranks('old_best_score');new_order=ranks('cohort_best_score')
changes=[r for r in rows if r['old_best_id']!=r['cohort_best_id']]
newly_eligible = sorted([p for group in groups.values() for p in group if not p['healthy'] and p['cohort_healthy']], key=lambda p:(p['method'],p['random_seed'],p['side'],p['multiplier']))
assert len(newly_eligible)==8
raw=[]
for p in newly_eligible:
    flagged=[e for e in examples_by_id[p['point_id']] if e['health']['reasons']]
    first=flagged[0]
    raw.append({'method':p['method'],'seed':p['random_seed'],'side':p['side'],'multiplier':p['multiplier'],'flags':p['health_flags'],
        'unfinished':p['cohort_counts']['unfinished'],'role_leaks':p['cohort_counts']['role_leaks'],'repeated':p['cohort_counts']['repeated'],
        'question':first['question_id'],'first_reason':','.join(first['health']['reasons']),
        'first_answer_excerpt':first['steered'][:170].replace('\n','\\n'),
        'raw_evidence':f"../../../outputs/bsbench-v2/results/{p['raw_evidence']}"})
with (out/'cohort-health-only-newly-eligible.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(raw[0]),lineterminator='\n');writer.writeheader();writer.writerows(raw)
fmt=lambda x:f'{x:+.2f}'
md=['# Counterfactual: cohort-fraction health only', '',
    'PI/gpt-6-sol · Offline diagnostic. Same 20-question scores, doses and raw judgments as the [current report](index.md). No canonical eligibility or generation record changed. Reference post-boundary and off-axis conditions are deliberately **not applied**.', '',
    'Candidate calibration called `health(tokenizer, answers)` on four answers per side/magnitude ([producer](../../../scripts/run_bsbench_modal.py#L73-L91)). The target chose the largest dose whose *both sides* had no cohort reason ([selector](../../../src/steering_lite/benchmark/dose_search.py#L62-L80)). Final generation called `health(tokenizer, [answer])` for each answer ([producer](../../../scripts/run_bsbench_modal.py#L276-L283)), so the same fractional thresholds operate on denominator one. Pinned reference computes cohort health ([walk.py](../../../docs/vendor/vjp-steering/scripts/walk.py#L408-L438)).', '',
    'Reconstruction sums saved singleton `unfinished`, `role_leaks`, `repeated` counts within each 20-answer final dose; health means <10 unfinished, <5 role leaks and <5 repeated. The source [health function](../../../src/steering_lite/benchmark/generation.py#L130-L151) uses fractions >=.5/.25/.25. This tests a consistent health *unit* without repeating tokenization, changing scoring, or deciding whether the reference off-axis criterion is suitable.', '',
    f"Final dose eligibility: {sum(p['healthy'] for group in groups.values() for p in group)}/60 current, {sum(p['cohort_healthy'] for group in groups.values() for p in group)}/60 cohort. Boundary groups: current {dict(Counter(r['old_boundary'] for r in rows))}; cohort {dict(Counter(r['cohort_boundary'] for r in rows))}. Best rows change for {len(changes)}/20 method/seed/sign groups. The four two-answer transfer cases retain their old 71/80 all-healthy and 9/80 bracketed states: per-answer and cohort fraction cutoffs coincide at denominator two (asserted against all 240 saved transfer groups).", '' ,
    '| method/seed/sign | health old→cohort | best × old→cohort | score old→cohort | selected point old→cohort |', '|:--|:--:|:--:|--:|:--|']
for r in changes:
    name=f"{r['method']}/{r['seed']} {r['side']}"
    oldurl=f"../../../outputs/bsbench-v2/results/evidence/{r['old_best_id']}.html"
    newurl=f"../../../outputs/bsbench-v2/results/evidence/{r['cohort_best_id']}.html"
    md.append(f"| {name} | {r['old_health']}→{r['cohort_health']} | {r['old_best_multiplier']:g}→{r['cohort_best_multiplier']:g} | {fmt(r['old_best_score'])}→{fmt(r['cohort_best_score'])} | [old]({oldurl}) → [cohort]({newurl}) |")
md+=['', '[All 20 selections and boundary states](cohort-health-only-selections.csv). If no row appears above, scores and selected points did not change.', '',
     '## Ranking movement', '',
     'Ranking is within sign; all ten method/seed groups are listed. Prompting stays a separate one-point control (+C +0.78), bare is algebraic zero, and candidate-grid scores never enter either ordering. These selected maxima are within-sample and not a significance test.', '',
     '| sign | current order (best to worst) | cohort-health-only order |', '|:--|:--|:--|']
for side in ('+C','-C'):
    names=lambda order: ', '.join(f'{name}{seed}' if name=='random' else name for name,seed in order)
    md.append(f"| {side} | {names(old_order[side])} | {names(new_order[side])} |")
md+=['', '## Newly eligible raw failures', '',
     'The following eight points pass only the cohort-fraction health test. A passing cohort can still contain a severe individual failure; examples are not erased. Full raw answers and reasons are linked in [the CSV](cohort-health-only-newly-eligible.csv) and each point’s numbered evidence.', '',
     '| evidence | flags/20 | unfinished/20 | role leaks/20 | repeated/20 | first flagged answer |', '|:--|--:|--:|--:|--:|:--|']
for r in raw:
    name=f"{r['method']}/{r['seed']} {r['side']} ×{r['multiplier']:g}"
    md.append(f"| [{name}]({r['raw_evidence']}) | {r['flags']} | {r['unfinished']} | {r['role_leaks']} | {r['repeated']} | {r['question']} {r['first_reason']} |")
md+=['', 'Three examples that become eligible: [mean_diff +C ×1.2, BSV2-014](../../../outputs/bsbench-v2/results/evidence/b09f18e251415702.json) has an empty steered answer (`""`); [random seed3 +C ×0.8, BSV2-011](../../../outputs/bsbench-v2/results/evidence/ae367fb838ad3200.json) ends in repeated zeros (`00000000000000000000…`); [vjp_cache −C ×1.2, BSV2-013](../../../outputs/bsbench-v2/results/evidence/fa631b39c0ea779c.json) ends `the actual physical limit of the setup rather` with no sentence ending. These failures stay visible in all saved raw evidence.', '',
     'The ninth flagged evaluation dose, random seed4 −C ×1.2, has 9/20 role leaks and fails both predicates ([raw](../../../outputs/bsbench-v2/results/evidence/2f1d30b9289f4ac3.json)). `random` dose selection and rank under this counterfactual remain conditional on individual visibly broken answers being allowed in a cohort. The current selection is unchanged pending a scientific decision.', '', '— PI/gpt-6-sol', '']
(out/'cohort-health-only.md').write_text('\n'.join(md))
print(json.dumps({'current_eligible':sum(p['healthy'] for group in groups.values() for p in group),'cohort_eligible':sum(p['cohort_healthy'] for group in groups.values() for p in group),'changed_best':len(changes),'changed_groups':[(r['method'],r['seed'],r['side'],r['old_best_multiplier'],r['cohort_best_multiplier']) for r in changes], 'current_status':dict(Counter(r['old_boundary'] for r in rows)),'cohort_status':dict(Counter(r['cohort_boundary'] for r in rows)),'rankings_current':old_order,'rankings_cohort':new_order},indent=2))
