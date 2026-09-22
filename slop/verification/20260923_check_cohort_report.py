"""Bounded checks on the frozen real report; no model/provider calls. — PI/OpenAI"""
import importlib.util
import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import polars as pl

root = Path('outputs/bsbench-v2/results')
a = json.loads((root / 'measured-points.json').read_text())
p = json.loads((root / 'source-parity.json').read_text())
assert len(a['points']) == 62
assert Counter(x['method'] for x in a['points']) == {'bare':1,'prompting':1,'random':30,'mean_diff':6,'pca':6,'kv_cache_gram':6,'vjp_delta':6,'vjp_cache':6}
assert len(a['transfer_groups']) == 240 and len(a['solves']) == 100
assert sum(not x['within_absolute_005'] for x in a['solves']) == 11
assert sum(len(x['examples']) for x in a['transfer_groups']) == 480
assert a['health_policy']['version'] == 'cohort-fraction-v1'
assert a['health_policy']['final_cohort_size'] == 20 and a['health_policy']['candidate_cohort_size'] == 4
assert a['cohort_eligible_doses'] == 59
assert [len(r['eligible_seeds']) for r in a['random_regions']] == [5,5,4]
assert [r['included'] for r in a['random_regions']] == [True,True,False]
old_parity=json.loads(Path('slop/verification/20260922_signed-report-capture/source-parity.json').read_text())
assert p['plot']['all_point_ids'] == old_parity['plot']['all_point_ids']
assert a['persona']['approvals'] == 0 and a['persona']['count'] == 12
assert p['scientific_before'] == p['scientific_after']

for point in a['points']:
    assert math.isclose(point['dose_score'],point['directed_intended_effect'] - 4*point['absolute_off_axis_change'],abs_tol=1e-12)
    expected_axis = -point['directed_intended_effect'] if point['side']=='-C' else point['directed_intended_effect']
    assert point['signed_axis_effect'] == expected_axis
    e = json.loads((root / point['raw_evidence']).read_text())
    assert len(e['examples'])==20
    assert {x['question_id'] for x in e['examples']} == {f'BSV2-{n:03d}' for n in range(1,21)}
    assert point['health_flags'] == sum(bool(x['health']['reasons']) for x in e['examples'])
    if point['method'] in ('bare','prompting'):
        continue
    assert all(x['health']['metrics']['answers']==1 for x in e['examples'])
    counts={key:sum(x['health']['metrics'][key] for x in e['examples']) for key in ('unfinished','role_leaks','repeated')}
    expected=counts['unfinished']<10 and counts['role_leaks']<5 and counts['repeated']<5
    assert point['cohort_eligible'] == expected
    assert point['zero_individual_failures'] == (point['health_flags']==0)
    tag = f"random-seed{point['random_seed']}" if point['method']=='random' else point['method'].replace('_','-')
    audit = json.loads(Path(f'slop/verification/20260922_{tag}-final-judge-audit.json').read_text())
    row = next(r for r in audit['complete_evaluation_dose_aggregates'] if r['side']==point['side'] and r['multiplier']==point['multiplier'])
    for old,new in [('dose_score','dose_score'),('directed_intended_effect','directed_intended_effect'),('off_target_effect','absolute_off_axis_change')]:
        assert math.isclose(row[old],point[new],abs_tol=1e-12), (tag,old)

spec=importlib.util.spec_from_file_location('report','scripts/run_bsbench_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def synthetic(n, field, count):
    return [{'health':{'metrics':{'answers':1,'unfinished':int(field=='unfinished' and i<count),
                                 'role_leaks':int(field=='role_leaks' and i<count),
                                 'repeated':int(field=='repeated' and i<count)}}} for i in range(n)]
for field, threshold in [('unfinished',10),('role_leaks',5),('repeated',5)]:
    assert m.cohort_eligible(synthetic(20,field,threshold-1))
    assert not m.cohort_eligible(synthetic(20,field,threshold))
for candidate in a['calibration']:
    metrics=candidate['generation_health']['metrics']
    assert metrics['answers']==4
    reasons=candidate['generation_health']['reasons']
    expected=[reason for reason, count, threshold in [('unfinished',metrics['unfinished'],2),('role_leak',metrics['role_leaks'],1),('repetition',metrics['repeated'],1)] if count>=threshold]
    assert reasons==expected
for transfer in a['transfer_groups']:
    assert transfer['cohort_eligible']==m.cohort_eligible(transfer['examples'])

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[];self.ids=set();self.point_ids=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d:self.ids.add(d['id'])
        if tag in ('a','img'):self.links.append(d['href'] if tag=='a' else d['src'])
        if tag=='tr' and 'data-point' in d:self.point_ids.append(d['data-point'])

parsers={}
for path in root.rglob('*.html'):
    parser=Links();parser.feed(path.read_text());parsers[path.resolve()]=parser
link_count=0
for path,parser in parsers.items():
    for link in parser.links:
        target,_,fragment=link.partition('#')
        resolved=(path.parent/target).resolve() if target else path
        assert resolved.exists(),(path,link)
        if fragment:assert fragment in parsers[resolved].ids,(path,link)
        link_count+=1
md=(root/'index.md').read_text()
md_tables=[]
for name,ids in sorted(p['markdown_table_ids'].items(),key=lambda kv:md.index(f'## {kv[0]}\n')):
    section=md.split(f'## {name}\n',1)[1].split('\n## ',1)[0]
    table_lines=[line for line in section.splitlines() if line.startswith('|')]
    assert all(line.count('|')==8 for line in table_lines), 'Markdown table contains an unescaped pipe'
    found=re.findall(r'\]\(evidence/([a-f0-9]{16})\.html\)',section)
    assert found==ids,(name,found,ids)
    md_tables.extend(ids)
assert parsers[(root/'index.html').resolve()].point_ids[:len(md_tables)]==md_tables

frame=pl.DataFrame(a['points']);tables,regions,frontier=m.derive_views(frame)
assert sum(x['cohort_eligible'] for x in a['points'] if x['method'] not in ('bare','prompting')) == 59
assert sum(x['zero_individual_failures'] for x in a['points'] if x['method'] not in ('bare','prompting')) == 51
for name,table in tables.items():
    if name.startswith('All'):continue
    for row in table.to_dicts():
        assert row['cohort_eligible']
        if row['method'] in ('bare','prompting'):continue
        group=[r for r in a['points'] if r['method']==row['method'] and r['random_seed']==row['random_seed'] and r['side']==row['side'] and r['cohort_eligible']]
        field='magnitude' if name.startswith('Maximum') else 'dose_score'
        assert row[field]==max(r[field] for r in group)

selected=tables['Best measured cohort-eligible score -C'].to_dicts()
selected_vjp=next(x for x in selected if x['method']=='vjp_cache')
assert (selected_vjp['point_id'],selected_vjp['multiplier'],selected_vjp['health_flags'],selected_vjp['cohort_eligible'])==('fa631b39c0ea779c',1.2,1,True)
assert selected_vjp['dose_score'] > next(x['dose_score'] for x in selected if x['method']=='vjp_delta')
mutated=[dict(r) for r in a['points']]
row=next(r for r in mutated if r['method']=='random' and r['random_seed']==4 and r['side']=='-C' and r['multiplier']==1.2)
assert not row['cohort_eligible'] and row['health_flags']==9
row['dose_score']=1000.;row['directed_intended_effect']=1001.;row['signed_axis_effect']=-1001.
changed,_,_=m.derive_views(pl.DataFrame(mutated))
assert row['point_id'] not in changed['Best measured cohort-eligible score -C']['point_id'].to_list()
assert row['point_id'] not in changed['Maximum cohort-eligible dose -C']['point_id'].to_list()
assert len(changed['All measured points -C'])==30
boundary=list(__import__('csv').DictReader((root/'dose-boundaries.csv').open()))
assert len(boundary)==100
assert Counter(x['boundary'] for x in boundary if x['case']=='bsbench-v2-evaluation') == {'all-healthy':19,'bracketed':1}
assert Counter(x['boundary'] for x in boundary if x['case']!='bsbench-v2-evaluation') == {'all-healthy':71,'bracketed':9}
assert len(list(__import__('csv').DictReader((root/'best-measured.csv').open())))==20

proof={'author':'PI/gpt-6-sol','status':'passed','points':62,'audited_activation_metric_matches':60,'all_numbered_evidence_pages':62,
       'verified_local_links':link_count,'markdown_html_disk_rank_parity':True,'cohort_threshold_boundary_4_of_20_pass_5_of_20_fail':True,'canonical_cohort_eligible':59,'zero_individual_failures':51,'vjp_cache_negative_1p2_selected_with_raw_flag':True,
       'transfer_groups':240,'solves':100,'kl_misses':11,'scientific_hashes_and_ledger_unchanged':True,
       'fresh_eyes_oracle':'blocked: parent run35bb08f6 Codex usage limit; no verdict/retry'}
Path('slop/verification/20260923_cohort-report-focused-checks.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
