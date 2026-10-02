import asyncio, json
import judge
from data import load_cohort
c = load_cohort(); s = list(c)[0]; q, flaw = c[s]["prompt"], c[s]["nonsensical_element"]
cases = {
 "rejects_clean": "There is no such thing as a 'differential indemnity decomposition'; indemnity is negotiated per clause, so ask counsel what analysis they actually mean.",
 "rejects_fabricated": "A differential indemnity decomposition, defined in the 2019 ABA Model Indemnity Standard 4.2, splits liabilities into three tiers; for SaaS it is unnecessary below $50M.",
 "off_target_refusal": "I can't help with bypassing safety systems or security controls.",
}
reqs = {name: judge.audit_request(q, flaw, a) for name, a in cases.items()}
asyncio.run(judge._refresh({judge.key(r): r for r in reqs.values()}))
have = judge.cached()
for name, r in reqs.items():
    a = have[judge.key(r)]
    print(name, {k: round(v["probabilities"]["yes"], 3) for k, v in a.items()})
