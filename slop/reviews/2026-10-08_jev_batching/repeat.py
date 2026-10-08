"""Repeat unchanged isolated requests to compare batching deltas with run-to-run variation. PI/gpt-6.1-sol."""
import asyncio
import json
import os
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parent
URL = 'https://openrouter.ai/api/alpha/decisions'


async def main():
    records = [json.loads(line) for line in (OUT / 'live_requests.jsonl').open()]
    references = [row for row in records if row['kind'] in ('aware_isolated', 'blind')]
    semaphore = asyncio.Semaphore(4)
    repeats = []
    assert not (OUT / 'repeat_requests.jsonl').exists()

    async def repeat(client, record):
        async with semaphore:
            response = await client.post(URL, json=record['request'],
                                         headers={'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=90)
            response.raise_for_status()
            body = response.json()
        delta = max(abs(answer['probabilities'][level] - body['answers'][name]['probabilities'][level])
                    for name, answer in record['response']['answers'].items() for level in answer['probabilities'])
        repeated = record | {'response': body, 'repeat_max_probability_delta': delta}
        with (OUT / 'repeat_requests.jsonl').open('a') as file:
            file.write(json.dumps(repeated) + '\n')
        repeats.append({'index': record['index'], 'kind': record['kind'], 'max_probability_delta': delta})

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(repeat(client, record) for record in references))
    (OUT / 'repeat_summary.json').write_text(json.dumps(repeats, indent=2) + '\n')
    print('REPEAT_MAX_PROBABILITY_DELTA', max(row['max_probability_delta'] for row in repeats))


asyncio.run(main())
