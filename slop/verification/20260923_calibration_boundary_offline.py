"""Summarize saved signed doses without model or provider calls. — PI/gpt-6-sol"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2]
run = root / 'outputs/bsbench-v2'
source = run / 'results/measured-points.json'
out = root / 'slop/research/20260923_calibration_boundary'
out.mkdir(parents=True, exist_ok=True)
data = json.loads(source.read_text())
assert data['schema'] == 'bsbench-signed-measured-points-v3'
assert hashlib.sha256((run / 'run-summary.json').read_bytes()).hexdigest() == data['source_summary_sha256']
assert len(data['points']) == 62 and len(data['solves']) == 100 and len(data['transfer_groups']) == 240
points = defaultdict(list)
for p in data['points']:
    assert p['healthy'] == (p['health_flags'] == 0)
    points[p['method'], p['random_seed'], p['side']].append(p)
solves = {(s['method'], s['random_seed'], s['case_id'], s['side']): s for s in data['solves']}
assert len(solves) == 100
transfers = defaultdict(list)
for t in data['transfer_groups']:
    assert t['health_flags'] == sum(bool(e['health']['reasons']) for e in t['examples'])
    transfers[t['method'], t['random_seed'], t['case_id'], t['side']].append(t)
assert len(transfers) == 80

def ranked(group):
    eligible = [p for p in group if p['healthy']]
    return max(eligible, key=lambda p: (p['dose_score'], p['directed_intended_effect'], -(p['magnitude'] if p['magnitude'] is not None else 0))) if eligible else None

def ref(p):
    return f"../../../outputs/bsbench-v2/results/{p['evidence']}"

best = []
for (method, seed, side), group in sorted(points.items()):
    if method == 'bare':
        continue
    selected = ranked(group)
    best.append({'method': method, 'seed': seed, 'side': side, 'eligible': sum(p['healthy'] for p in group),
                 'total': len(group), 'score': selected['dose_score'] if selected else None,
                 'intended': selected['directed_intended_effect'] if selected else None,
                 'abs_off': selected['absolute_off_axis_change'] if selected else None,
                 'coefficient': selected['magnitude'] * (1 if side == '+C' else -1) if selected and selected['magnitude'] is not None else None,
                 'multiplier': selected['multiplier'] if selected else None,
                 'point_id': selected['point_id'] if selected else None,
                 'evidence': ref(selected) if selected else None})
assert len(best) == 21
candidate = defaultdict(list)
for p in data['calibration']:
    candidate[p['method'], p['random_seed'], p['side']].append(p)
assert len(candidate) == 20
candidate_rows = []
for (method, seed, side), group in sorted(candidate.items()):
    best_candidate = ranked([{'healthy': not p['generation_health']['reasons'], 'dose_score': p['dose_score'], 'directed_intended_effect': p['directed_intended_effect'], 'magnitude': p['magnitude'], 'point_id': p['provenance'], 'evidence': p['source']} for p in group])
    candidate_rows.append({'method':method,'seed':seed,'side':side,'healthy':sum(not p['generation_health']['reasons'] for p in group),'measured':len(group),
        'best_healthy_candidate_magnitude':best_candidate['magnitude'] if best_candidate else None,
        'best_healthy_candidate_score':best_candidate['dose_score'] if best_candidate else None,
        'source':group[0]['source']})

boundary = []
for key, solve in sorted(solves.items()):
    method, seed, case, side = key
    group = points[method, seed, side] if case == 'bsbench-v2-evaluation' else transfers[key]
    assert len(group) == 3
    by_dose = {float(p['multiplier']): p for p in group}
    assert set(by_dose) == {0.8,1.,1.2}
    assert abs(by_dose[1.]['magnitude'] - solve['magnitude']) < 1e-8
    assert all(abs(p['magnitude'] / solve['magnitude'] - dose) < 1e-8 for dose, p in by_dose.items())
    healthy = {d: p['health_flags'] == 0 if case != 'bsbench-v2-evaluation' else p['healthy'] for d,p in by_dose.items()}
    good = [d for d in sorted(healthy) if healthy[d]]
    first_failed_above = min((d for d in sorted(healthy) if d > max(good) and not healthy[d]), default=None) if good else None
    nonmonotonic = any(not healthy[lo] and healthy[hi] for lo in sorted(healthy) for hi in sorted(healthy) if lo < hi)
    status = 'all-healthy' if len(good) == 3 else 'all-failed' if not good else 'nonmonotonic' if nonmonotonic else 'bracketed' if first_failed_above else 'failed-below-healthy'
    best_eval = ranked(group) if case == 'bsbench-v2-evaluation' else None
    one = by_dose[1.]
    boundary.append({'method':method,'seed':seed,'case':case,'side':side,'predicted_1x':solve['magnitude'] * (1 if side == '+C' else -1),
        'health_08':healthy[.8], 'health_1':healthy[1.], 'health_12':healthy[1.2],
        'flags_08':by_dose[.8]['health_flags'],'flags_1':one['health_flags'],'flags_12':by_dose[1.2]['health_flags'],
        'highest_healthy_coefficient':(max(good) * solve['magnitude'] * (1 if side == '+C' else -1)) if good else None,
        'first_higher_failed_coefficient':(first_failed_above * solve['magnitude'] * (1 if side == '+C' else -1)) if first_failed_above else None,
        'status':status, 'kl_target':solve['target_rms'],'kl_achieved':solve['achieved_rms'], 'kl_abs_error':solve['absolute_error'],
        'kl_relative_error':solve['relative_residual'],'kl_within_005':solve['within_absolute_005'],
        'score_1x':one['dose_score'] if best_eval and healthy[1.] else None,
        'score_best':best_eval['dose_score'] if best_eval else None,
        'score_regret':best_eval['dose_score']-one['dose_score'] if best_eval and healthy[1.] else None,
        'best_dose':best_eval['multiplier'] if best_eval else None,
        'evidence_1x':ref(one) if case == 'bsbench-v2-evaluation' else one['source'],
        'solve_source':solve['source']})
assert len(boundary) == 100

for name,rows in [('best-final',best),('calibration-candidates-4q',candidate_rows),('boundary-20q-and-transfer',boundary)]:
    with (out / f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

count_case={case:Counter(r['status'] for r in boundary if (r['case']=='bsbench-v2-evaluation')==case) for case in (True,False)}
count_kl={case:Counter(r['kl_within_005'] for r in boundary if (r['case']=='bsbench-v2-evaluation')==case) for case in (True,False)}
print(json.dumps({'final':count_case[True],'transfer':count_case[False],'kl_final':count_kl[True],'kl_transfer':count_kl[False],
    'failed_1x_final':sum(not r['health_1'] for r in boundary if r['case']=='bsbench-v2-evaluation'),
    'failed_1x_transfer':sum(not r['health_1'] for r in boundary if r['case']!='bsbench-v2-evaluation'),
    'regret_final': [(r['method'],r['seed'],r['side'],round(r['score_regret'],3) if r['score_regret'] is not None else None,r['best_dose']) for r in boundary if r['case']=='bsbench-v2-evaluation']},indent=2))

eligibility_disagreements = []
for p in data['points']:
    if p['method'] in ('bare', 'prompting'):
        continue
    examples = json.loads((run / 'results' / p['raw_evidence']).read_text())['examples']
    assert len(examples) == 20
    n = {k: sum(e['health']['metrics'][k] for e in examples) for k in ('unfinished', 'role_leaks', 'repeated')}
    cohort_failed = n['unfinished'] >= 10 or n['role_leaks'] >= 5 or n['repeated'] >= 5
    if p['healthy'] != (not cohort_failed):
        eligibility_disagreements.append({'method':p['method'],'seed':p['random_seed'],'side':p['side'],'multiplier':p['multiplier'],
            'local_flags':p['health_flags'], **n,'reference_cohort_fraction_failed':cohort_failed,
            'point_id':p['point_id'],'raw_evidence':ref(p).replace('.html','.json')})
assert len(eligibility_disagreements) == 8
with (out / 'eligibility-disagreements.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(eligibility_disagreements[0]),lineterminator="\n");w.writeheader();w.writerows(eligibility_disagreements)

fmt = lambda x: '—' if x is None else f'{x:+.2f}'
md = ['# Signed dose boundary from saved BS-bench outputs', '',
      'PI/gpt-6-sol · Offline reconstruction from the completed 20-question evaluation and four disjoint transfer cases. No new generation or judgments.', '',
      '## Best generation-healthy measured final score', '',
      'One row per method/sign and per random seed/sign. Score = directed intended − 4 × mean absolute off-axis change. Selection is within each measured three-point dose sweep; it is not a comparison adjusted for winner selection. Bare has algebraic score 0; prompting has one +C point and no magnitude. A 4-question calibration score never enters this table.', '']
for side in ('+C','-C'):
    md += [f'### {side}', '', '| evidence | score↑ | intended↑ | abs off↓ | coefficient | best × | eligible/total |', '|:--|--:|--:|--:|--:|--:|--:|']
    for r in sorted((r for r in best if r['side']==side),key=lambda r:r['score'],reverse=True):
        name=f"{r['method']}" + (f" seed{r['seed']}" if r['method']=='random' else '')
        md.append(f"| [{name}]({r['evidence']}) | {fmt(r['score'])} | {fmt(r['intended'])} | {r['abs_off']:.2f} | {fmt(r['coefficient'])} | {f'{r['multiplier']:g}' if r['multiplier'] is not None else '—'} | {r['eligible']}/{r['total']} |")
    md += ['',]
md += ['## Predicted 1× versus the three measured evaluation doses', '',
       'Health order is 0.8×/1×/1.2×; H = no answer-level reason, F = at least one reason. Highest healthy and first higher failed are observed coefficients, not extrapolated thresholds. Regret is best healthy score minus 1× score. It is absent if 1× failed. KL error is the signed 1× solve’s absolute RMS-KL error in nats.', '',
       '| evidence | health | 1× C | highest H C | first higher F C | boundary | KL err↓ | regret↓ |', '|:--|:--:|--:|--:|--:|:--|--:|--:|']
for r in sorted((r for r in boundary if r['case']=='bsbench-v2-evaluation'),key=lambda r:(r['side'],r['method'],r['seed'])):
    label=f"{r['method']}" + (f" seed{r['seed']}" if r['method']=='random' else '')
    h=''.join('H' if r['health_'+d] else 'F' for d in ('08','1','12'))
    md.append(f"| [{label} {r['side']}]({r['evidence_1x']}) | {h} | {fmt(r['predicted_1x'])} | {fmt(r['highest_healthy_coefficient'])} | {fmt(r['first_higher_failed_coefficient'])} | {r['status']} | {r['kl_abs_error']:.3f} | {fmt(r['score_regret'])} |")
md += ['', '## Four disjoint transfer cases (two questions each)', '',
       'Transfer has generation health and RMS-KL but no behavioral judgment or score. Full 80 signed case rows, including predicted coefficient, three health flags, first failed measurement and full precision KL, are in [boundary-20q-and-transfer.csv](boundary-20q-and-transfer.csv).', '',
       '| case | groups | all H | bracketed | KL within .05 | 1× failed |', '|:--|--:|--:|--:|--:|--:|']
for case in sorted({r['case'] for r in boundary if r['case']!='bsbench-v2-evaluation'}):
    group=[r for r in boundary if r['case']==case]
    md.append(f"| {case} | {len(group)} | {sum(r['status']=='all-healthy' for r in group)} | {sum(r['status']=='bracketed' for r in group)} | {sum(r['kl_within_005'] for r in group)} | {sum(not r['health_1'] for r in group)} |")
md += ['', '## Separate four-question candidate grid', '',
       'Candidate scores and healthy magnitudes are measured on only four calibration questions. The 1× prediction comes from the highest candidate magnitude clean in both signs, then one pooled signed RMS-KL target; these per-sign score optima are not final rankings. All 20 per-sign rows are in [calibration-candidates-4q.csv](calibration-candidates-4q.csv).', '',
       '| method / seed | +C H/grid | −C H/grid | highest both-sign H | target RMS-KL |', '|:--|--:|--:|--:|--:|']
for method,seed in sorted({(r['method'],r['seed']) for r in candidate_rows}):
    plus=next(r for r in candidate_rows if (r['method'],r['seed'],r['side'])==(method,seed,'+C'))
    minus=next(r for r in candidate_rows if (r['method'],r['seed'],r['side'])==(method,seed,'-C'))
    both=set(p['magnitude'] for p in candidate[method,seed,'+C'] if not p['generation_health']['reasons']) & set(p['magnitude'] for p in candidate[method,seed,'-C'] if not p['generation_health']['reasons'])
    assert both
    target=next(s['target_rms'] for s in data['solves'] if (s['method'],s['random_seed'])==(method,seed))
    md.append(f"| {method} / {seed} | {plus['healthy']}/{plus['measured']} | {minus['healthy']}/{minus['measured']} | {max(both):g} | {target:.3f} |")
md += ['', '## Eligibility and interpretation', '',
       'The [report producer](../../../scripts/run_bsbench_results.py) marks a point healthy when `flags == 0`, with `flags = sum(bool(e["health"]["reasons"]) for e in examples)` (lines 95–101). The [production final producer](../../../scripts/run_bsbench_modal.py) calls `health(tokenizer, [answer])` for each of 168 individual answers (lines 276–283). The [health implementation](../../../src/steering_lite/benchmark/generation.py) requires a sentence-ending `[.!?\")]$`, role-leak fraction <0.25, repeated fraction <0.25 and unfinished fraction <0.5 (lines 130–151). One answer means one punctuation-only unfinished flag excludes its whole 20-question dose.', '',
       'The [pinned reference admissibility producer](../../../docs/vendor/vjp-steering/scripts/export.py) uses `not health["breakdown_reasons"] and not health["post_boundary"] and steered_off_axis <= 1.5` (lines 194–198). Its [health producer](../../../docs/vendor/vjp-steering/scripts/walk.py) evaluates all answers as a cohort, calls the same regex and fraction thresholds (lines 408–438), and sets `post_boundary` only after two consecutive failed magnitudes (lines 247–260). Local final health uses per-answer reasons, so it is stricter than the reference’s cohort threshold; it also has no post-boundary or off-axis criterion. The plan intentionally removes the off-axis cutoff, so do not silently re-add it. Whether to keep per-answer exclusion, use a cohort rate, or treat a complete punctuation-only answer differently is a scientific decision, not a plotting fix.', '',
       'Eight of the nine locally flagged 20-question doses would pass the reference’s *cohort-fraction health thresholds alone* (see [eight source-linked rows](eligibility-disagreements.csv)). The remaining random seed4 −C at 1.2× has 9/20 role leaks and fails both definitions. This comparison deliberately excludes the reference’s separate off-axis and post-boundary conditions; it does not reclassify the current report. The flagged raw answers include incomplete text, an empty answer, role leaks and a repeated-zero answer. `No\\nNo` in transfer is punctuation-only flagged, not by itself proof of truncation.', '',
       'Two measured evaluation health patterns recover after a lower failed dose: random seed0 −C is FHF, random seed3 +C is FHH. Neither gives a monotone bracket. A row marked all-healthy has no observed upper boundary; an all-failed row would have no measured lower healthy bound. A bracketed row bounds only the sampled doses and predicate, not an exact maximum. The full [final and transfer table](boundary-20q-and-transfer.csv) records every group.', '',
       'At the current predicate, evaluation: 13/20 all-healthy, 5/20 bracketed, 2/20 nonmonotonic; 1/20 failed at 1×. Transfer: 71/80 all-healthy, 9/80 bracketed, 0/80 failed at 1×. RMS-KL absolute error ≤.05: evaluation 20/20, transfer 69/80. Best measured healthy score is at 0.8× for 10/20, 1× for 6/20 and 1.2× for 4/20. Thus the 1× KL target often matches KL yet does not select the best measured score. These are within-sample maximizations on just three doses, not out-of-sample efficacy.', '',
       'Most of the 84 all-healthy case/sign groups never measure their upper health boundary; calibrating RMS-KL alone cannot prove it predicts maximum coherent dose. Nine transfer groups and five evaluation groups have a measured first higher failing point. The 11 transfer KL misses include 10 all-healthy groups and one bracketed group, so KL miss and health failure are not interchangeable.', '',
       '## Evidence files', '',
       '- [All 21 final best rows](best-final.csv), [20 four-question candidate summaries](calibration-candidates-4q.csv), [100 signed evaluation/transfer boundaries](boundary-20q-and-transfer.csv), [eight eligibility disagreements](eligibility-disagreements.csv).',
       '- [Raw measured points](../../../outputs/bsbench-v2/results/measured-points.json), [20-question evidence index](../../../outputs/bsbench-v2/results/index.md), [frozen summary](../../../outputs/bsbench-v2/run-summary.json).',
       '- Calculations: [offline-only script](../../verification/20260923_calibration_boundary_offline.py), [run output](../../verification/20260923_calibration_boundary_summary.log).', '', '— PI/gpt-6-sol', '']
(out / 'index.md').write_text('\n'.join(md))
