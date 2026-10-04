Each pair below has two questions. In each pair exactly one question is built on a made-up or misapplied concept, and the other is a legitimate question.
For each pair answer on one line: `<pair> | <A or B> | cue: knowledge | style | both | guess | <short reason>`.
cue=knowledge if you know the concept is fake or misapplied; cue=style if the wording, length, fluency or specificity gave it away; both if both; guess if unsure.
Then one paragraph: could someone who knows nothing about these fields tell which is which from style alone? What stylistic differences do you see between the two sets?

Pair 1
A: We're updating our compliance program to cover anti-bribery, export controls, antitrust, data privacy, and anti-money laundering -- which of these areas should we prioritize for the annual training cycle given our recent expansion into the EU?
B: We're updating our compliance program to cover anti-bribery, export controls, antitrust, data privacy, and prophylactic precedent inoculation -- which of these areas should we prioritize for the annual training cycle given our recent expansion into the EU?

Pair 2
A: Our portfolio's sector allocations and our benchmark's sector weights seem to be creating destructive interference in the attribution analysis. How do we calculate the resulting beat frequency to determine when the tracking error will peak?
B: Our portfolio's sector allocations and our benchmark's sector weights seem to be producing offsetting allocation and selection effects in the attribution analysis. How do we decompose our active sector weights into their contributions to ex-ante tracking error to determine which sector bets are driving it?

Pair 3
A: What's the per-line-of-code architectural contribution score for our backend services? We want to identify which individual lines are pulling the most weight in terms of system design quality.
B: What's the per-module architectural contribution score for our backend services? We want to identify which individual modules are pulling the most weight in terms of system design quality, using metrics like coupling, cohesion, and instability.

Pair 4
A: What are the current recommendations for applying a stratified perfusion regression when triaging blunt abdominal trauma patients with equivocal FAST exams in a community ED?
B: What are the current recommendations for applying a serial FAST exam strategy when triaging blunt abdominal trauma patients with equivocal FAST exams in a community ED?

Pair 5
A: We're measuring the thermal conductivity of a macroscopic copper rod at room temperature and atmospheric pressure. To get an accurate result, how should we correct for radiative and convective heat loss from the rod's lateral surface? Our values are about 5% below the textbook number.
B: We're measuring the thermal conductivity of a macroscopic copper rod at room temperature and atmospheric pressure. To get an accurate result, how should we correct for the Pauli exclusion pressure between conduction electrons in the rod? Our values are about 5% below the textbook number.

Pair 6
A: How do we calculate the peak overlap in shared resource demand in our surgical scheduling grid? We're getting contention for anesthesia staff and PACU beds between the ortho and general surgery blocks on Tuesdays, and I want to know if shifting one block by 30 minutes would reduce the peak concurrent demand enough to resolve the conflict.
B: How do we calculate the resonance frequency of our surgical scheduling grid? We're getting constructive interference between the ortho and general surgery blocks on Tuesdays, and I want to know if shifting one block by 30 minutes would move us off the resonance peak.

Pair 7
A: Our p99 API latency has dropped about 30% per quarter for the last year, from 800ms down to about 190ms. At this rate we'd be around 45ms by next year, which starts getting close to the roughly 20ms network round-trip to our nearest datacenter. How should we architect our services to keep driving p99 down as we approach that network latency floor, given that the remaining gains will have to come from somewhere other than raw network time?
B: Our p99 API latency has dropped 25% per quarter for the last year — from 800ms down to about 190ms. At this rate we'll be under 1ms by next year, faster than a single network round-trip to our nearest datacenter. How should we architect our services to maintain sub-millisecond end-to-end response times once we cross the network latency floor?

Pair 8
A: What's the recommended cadence for running a contract compliance audit of indemnity and limitation-of-liability clauses when our contract portfolio spans both common-law and civil-law jurisdictions with conflicting limitation-of-liability standards?
B: What's the recommended cadence for running a bilateral indemnity regression when our contract portfolio spans both common-law and civil-law jurisdictions with conflicting limitation-of-liability standards?

Pair 9
A: What's a good way to tell when a unit test has gone stale after it's merged? We're trying to stamp each test with the commit hash and the source files it covers at merge time so QA knows when it needs re-review as that covered code changes.
B: What's the shelf life of a unit test after it's merged? We're trying to stamp each test with an intrinsic best-before date at merge time so QA knows when it expires, independent of any code changes.

Pair 10
A: Our semantic merge resolver is running at a 340ms conflict window with a 3-layer AST diff depth, but we're getting false positives on refactored methods. Should we widen the conflict window to 500ms or increase the diff depth to 5 layers?
B: Our semantic merge resolver uses GumTree-style AST matching with a 0.5 bottom-up similarity threshold and a minimum subtree height of 2 for top-down matching, but we're getting false-positive conflicts on refactored methods. Should we lower the similarity threshold to 0.4 or raise the minimum subtree height to 3?

Pair 11
A: Has anyone here implemented Convergent Schema Normalization for a multi-tenant Postgres setup? We need to reconcile tenant-specific column additions back to a canonical schema every release cycle, and CSN supposedly automates the drift resolution through a deterministic merge algebra.
B: Has anyone here used Atlas's declarative schema migrations for a multi-tenant Postgres setup? We need to reconcile tenant-specific column additions back to a canonical schema every release cycle, and Atlas supposedly automates the drift resolution by diffing each tenant's live schema against a desired-state definition and generating the migration plan.

Pair 12
A: At what probability-of-loss threshold should we record a loss accrual for our pending tort claims under ASC 450, and how do we estimate the reasonably possible range of loss for each individual plaintiff's claim?
B: At what Tier 1 capital adequacy ratio should we trigger a Basel III-style wind-down of our pending tort claims, and how do we calculate the risk-weighted exposure of each individual plaintiff?

Pair 13
A: Controlling for repository age and average file size, how do you attribute the variance in deployment frequency to the indentation style of the codebase versus the average variable name length?
B: Controlling for repository age and average file size, how do you attribute the variance in deployment frequency to the CI pipeline duration versus the average pull request size?

Pair 14
A: Under the December 2023 amendment to Federal Rule of Evidence 702, the proponent of expert testimony in patent cases must show the court, by a preponderance of the evidence, that the opinion reflects a reliable application of the methodology before it will be admitted. How are firms making that showing for software patent claim construction opinions?
B: Under the revised Daubert-Frye convergence standard from the 2024 Federal Rules update, expert testimony in patent cases requires a minimum reproducibility index of 0.7 before the court will admit the methodology. How are firms calculating the RI for software patent claim construction opinions?

Pair 15
A: How do we measure the coupling, such as afferent and efferent dependencies and shared data contracts, between our legacy SOAP services and the new GraphQL gateway to figure out how much refactoring effort we need before any traffic can actually be migrated?
B: How do we measure the coefficient of static friction between our legacy SOAP services and the new GraphQL gateway to figure out what migration force we need to apply before anything actually moves?

Pair 16
A: Net of thermal noise and seismic background, how should we model the elasticity of our lab notebook's binding stiffness relative to the second derivative of our oscilloscope's screen brightness, once probe tip oxidation crosses the Nyquist threshold?
B: Net of thermal noise and seismic background, how should we model the effective stiffness of our AFM cantilever relative to the second derivative of the tip-sample interaction potential, once the cantilever's resonance frequency approaches our oscilloscope's Nyquist frequency?

Pair 17
A: We've spent six months building a per-clause liability thermodynamics model for our contract portfolio -- it tracks entropy accumulation across amendment cycles using Gibbs free energy equations. The entropy values are diverging from actual claim frequency. Should we add a temperature correction factor or switch to a Helmholtz formulation?
B: We've spent six months building a per-clause liability frequency model for our contract portfolio. It tracks risk accumulation across amendment cycles using Poisson regression. The predicted claim counts are diverging from actual claim frequency, and the observed variance is well above the mean. Should we add an exposure offset term or switch to a negative binomial formulation?

Pair 18
A: We're running our trademark dilution risk model at 500ms polling intervals across 14 competitor filings with a decay constant of 0.03 per day -- should we reduce the polling window before the TTAB opposition deadline, or will that saturate the similarity buffer?
B: We're tracking 14 competitor filings through a trademark watch service, and one published application looks likely to dilute our mark under the Lanham Act's fame and association factors. Should we request an extension of time to oppose before the TTAB 30-day opposition deadline, or will that weaken our position compared to filing the notice of opposition now?

Pair 19
A: Our FP&A team built a revenue attribution model that traces each dollar of quarterly revenue back to the specific Slack message that initiated the customer relationship. We've been running it for two quarters but the attribution keeps concentrating on a handful of generic welcome messages. Should we expand the lookback window or weight messages by thread depth?
B: Our FP&A team built a first-touch revenue attribution model that traces each dollar of quarterly revenue back to the specific marketing touchpoint in our CRM that initiated the customer relationship. We've been running it for two quarters but the attribution keeps concentrating on a handful of generic welcome emails. Should we expand the lookback window or move to a multi-touch model that weights touchpoints by position or time decay?

Pair 20
A: What's the yield-to-maturity on our residency program's teaching portfolio, and should we rebalance toward more procedural exposure before the next accreditation coupon date?
B: If 40% of our PGY-3 residents have logged under half of the ACGME minimum procedure numbers for graduation, what does that case-log completion rate suggest, and should we shift block time toward more procedural exposure before the next accreditation site visit?
