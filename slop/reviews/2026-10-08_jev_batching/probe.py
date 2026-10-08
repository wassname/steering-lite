"""Measure batching costs and question isolation on six saved BS-bench pairs. PI/gpt-6.1-sol."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from statistics import mean
import sys

import httpx

ROOT = Path('/workspace/2026/lite/steering-lite')
sys.path.insert(0, str(ROOT / 'scripts/bsbench'))
from data import load_cohort
from judge import URL, blind_request, bsb_request, pair_request

OUT = ROOT / 'slop/reviews/2026-10-08_jev_batching'
OUT.mkdir(parents=True, exist_ok=True)


def isolate(request):
    request = deepcopy(request)
    flaw = request['state'].pop('nonsensical_element')
    names = ('bs_score',) if 'bs_score' in request['questions'] else tuple(request['questions'])
    for name in names:
        instructions = request['questions'][name]['instructions']
        request['questions'][name]['instructions'] = {
            'question': instructions,
            'nonsensical_element': flaw,
        }
    return request


async def main():
    cohort = load_cohort()
    pilot = [json.loads(line) for line in (ROOT / 'slop/reviews/2026-10-05_judge_v4/pilot.jsonl').open()]
    # PI/gpt-6.1-sol: a pushback pair, an acceptance pair, a damaged pair, plus their reverse/null controls.
    chosen = [next(row for row in pilot if row['condition'] == condition and row['bare'] != row['steered'])
              for condition in ('prompt -C', 'vjp +C 0.315', 'mean_diff -C 2')]
    chosen += [dict(chosen[0], bare=chosen[0]['steered'], steered=chosen[0]['bare']),
               dict(chosen[1], bare=chosen[1]['steered'], steered=chosen[1]['bare']),
               dict(chosen[0], steered=chosen[0]['bare'], condition='bare vs bare')]
    semaphore = asyncio.Semaphore(4)
    records = []
    headers = {'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}"}

    async def run(client, index, row, kind, request):
        async with semaphore:
            response = await client.post(URL, headers=headers, json=request, timeout=90)
            response.raise_for_status()
            body = response.json()
            assert set(body['answers']) == set(request['questions']), body
        record = {'index': index, 'condition': row['condition'], 'scenario': row['scenario'], 'kind': kind,
                  'request': request, 'response': body}
        with (OUT / 'live_requests.jsonl').open('a') as file:
            file.write(json.dumps(record) + '\n')
        records.append(record)
        print('LIVE_OK', index, kind, body['usage'], flush=True)

    assert not (OUT / 'live_requests.jsonl').exists(), 'Use a new output directory for a new probe.'
    async with httpx.AsyncClient() as client:
        jobs = []
        for index, row in enumerate(chosen):
            question = cohort[row['scenario']]['prompt']
            flaw = cohort[row['scenario']]['nonsensical_element']
            aware = pair_request(question, flaw, row['bare'], row['steered'])
            isolated = isolate(aware)
            blind = blind_request(question, row['bare'], row['steered'])
            assert isolated['state'] == blind['state']
            combined = deepcopy(isolated)
            combined['questions'] |= blind['questions']
            answer = bsb_request(question, flaw, row['steered'])
            requests = {'aware_original': aware, 'aware_isolated': isolated, 'blind': blind,
                        'combined': combined, 'answer_original': answer, 'answer_isolated': isolate(answer)}
            for kind, request in requests.items():
                jobs.append(run(client, index, row, kind, request))
        await asyncio.gather(*jobs)
    by = {(r['index'], r['kind']): r['response'] for r in records}
    usage = {kind: {field: sum(r['response']['usage'][field] for r in records if r['kind'] == kind)
                    for field in ('input_tokens', 'cost')} for kind in requests}
    comparisons = []
    for index in range(len(chosen)):
        def diff(left, right, questions):
            return max(abs(by[index, left]['answers'][name]['probabilities'][level] - by[index, right]['answers'][name]['probabilities'][level])
                       for name in questions for level in by[index, left]['answers'][name]['probabilities'])
        comparisons.append({'index': index,
                            'aware_batch_max_probability_delta': diff('aware_isolated', 'combined', ('premise_change', 'off_axis')),
                            'blind_batch_max_probability_delta': diff('blind', 'combined', tuple(blind['questions'])),
                            'aware_context_max_probability_delta': diff('aware_original', 'aware_isolated', ('premise_change', 'off_axis')),
                            'answer_context_max_probability_delta': diff('answer_original', 'answer_isolated', tuple(answer['questions']))})
    combined_tokens = usage['combined']['input_tokens']
    separate_tokens = usage['aware_isolated']['input_tokens'] + usage['blind']['input_tokens']
    summary = {'n_pairs': len(chosen), 'n_requests': len(records), 'usage': usage, 'probability_deltas': comparisons,
               'combined_tokens': combined_tokens, 'separate_tokens': separate_tokens,
               'saving_fraction_if_all_pairs_need_blind': 1 - combined_tokens / separate_tokens,
               'break_even_blind_fraction': (combined_tokens - usage['aware_isolated']['input_tokens']) / usage['blind']['input_tokens']}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('SUMMARY', json.dumps(summary), flush=True)


asyncio.run(main())
