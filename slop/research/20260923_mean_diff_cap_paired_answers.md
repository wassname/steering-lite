# Fixed-dose mean-difference 64-vs-384: all paired answers

PI/gpt-6-sol · 2026-09-23. Verbatim answers from [`raw.json`](../verification/20260923_mean_diff_cap_diagnostic/aaf9aef0d6b6452cf19f7a4d9216ca897315c1a27ca3a4a5e4bc035e965b9b24/raw.json); rows are indexed by the saved four calibration prompts and signed fixed dose. No judge or accuracy claim. KL is the mean over 20 steered continuation positions for that one prompt, evaluated base-vs-steer on the same steered rollout; `old→new` is not a paired-token KL.

## Prompt 1

> Our outside counsel recommended running a differential indemnity decomposition before we finalize the acquisition agreement -- how granular should the decomposition be for a mid-market SaaS target with material IP concentration?

**Bare (before = after):** The decomposition should isolate specific IP-related liabilities (such as patent infringement, trade secret misappropriation, and copyright claims) alongside general indemnity categories to reflect the target's material IP concentration. This granular approach ensures the purchase price accurately reflects the risk profile of the intellectual property portfolio while preventing the dilution of IP-specific reserves by general business liabilities.

| sign | KL μ old→new (20 tok) | old64 full answer | new384 full answer |
|:---:|---:|:---|:---|
| +C | 0.135→0.083 | The decomposition should be granular enough to isolate specific IP assets and their associated revenue streams, allowing for precise valuation of each risk factor. This level of detail ensures that the indemnity accurately reflects the concentration risk while supporting a robust negotiation strategy for the mid-market SaaS target. | The decomposition should be highly granular, specifically isolating liabilities tied to the target's core intellectual property assets from general operational risks. This level of detail ensures that the indemnity accurately reflects the concentration of IP risk, allowing for precise pricing and robust protection of the acquirer's investment. |
| -C | 0.188→0.163 | The decomposition must isolate specific IP assets and their associated revenue streams to pinpoint exact liability exposure. This level of granularity is necessary to negotiate targeted indemnity caps rather than applying a blunt, aggregate financial penalty. | The decomposition must isolate specific IP assets and their associated revenue streams to pinpoint exact liability exposures. This level of granularity ensures you can accurately price the indemnity and avoid overpaying for risks tied to non-material components. |

## Prompt 2

> What's the recommended cadence for running a bilateral indemnity regression when our contract portfolio spans both common-law and civil-law jurisdictions with conflicting limitation-of-liability standards?

**Bare (before = after):** There is no single recommended cadence, as the frequency must be tailored to the specific volatility of each jurisdiction's legal landscape and the complexity of your contract mix. You should likely run the regression quarterly to capture emerging regulatory changes while performing ad-hoc stress tests whenever a major limitation-of-liability standard shifts in a key market.

| sign | KL μ old→new (20 tok) | old64 full answer | new384 full answer |
|:---:|---:|:---|:---|
| +C | 0.214→0.139 | There is no single recommended cadence for this regression, as the optimal frequency depends on the specific volatility of liability standards in each jurisdiction and the contract's renewal cycle. Instead, a dynamic approach that aligns the regression with the most restrictive limitation-of-liability standards across both legal systems is typically advised to ensure robust risk coverage. | There is no single recommended cadence for running a bilateral indemnity regression in this scenario, as the optimal frequency depends on the specific volatility of your cross-jurisdictional risk exposure. Instead of a fixed schedule, it is best practice to align your regression runs with major contract renewals, significant legal developments in either legal system, or quarterly business reviews to ensure the model accurately reflects the most current limitation-of-liability standards. |
| -C | 0.180→0.142 | You must run separate bilateral indemnity regressions for each jurisdiction to account for their distinct limitation-of-liability caps. Aggregating the portfolio into a single regression will invalidate the results due to the conflicting legal standards. | You must run separate bilateral indemnity regressions for each jurisdiction to account for their distinct limitation-of-liability caps. Aggregating the data across conflicting legal standards will invalidate the results and lead to inaccurate loss allocations. |

## Prompt 3

> Our litigation team is running a proportional fault cascade analysis for the multi-defendant product liability case — how many tiers deep should we run the cascade before the marginal allocation precision drops below the settlement noise floor?

**Bare (before = after):** You should typically run the cascade only two to three tiers deep, as adding further layers usually yields negligible precision gains while exponentially increasing computational cost. At this point, the marginal reduction in allocation error falls below the inherent noise floor of the settlement data, making deeper iterations statistically meaningless.

| sign | KL μ old→new (20 tok) | old64 full answer | new384 full answer |
|:---:|---:|:---|:---|
| +C | 0.121→0.126 | There is no universal tier depth that guarantees precision above the settlement noise floor, as this threshold depends entirely on the specific case's data quality, the complexity of the causal chain, and the unique negotiation dynamics of the defendants. Consequently, the optimal depth is best determined through iterative sensitivity testing that identifies the point where additional tiers no longer produce statistically significant shifts in the final allocation percentages. | You should typically run the cascade to a depth of three to four tiers, as this usually captures the vast majority of the liability distribution while minimizing the impact of minor, uncertain factors. Extending the analysis further beyond this point often yields marginal precision gains that fall below the settlement noise floor, potentially introducing unnecessary complexity without improving the final allocation strategy. |
| -C | 0.191→0.117 | Run the cascade only until the cumulative error margin exceeds the settlement noise floor, typically after 3 to 5 tiers for complex multi-defendant cases. Pushing deeper yields diminishing returns as the marginal precision loss outweighs the value of additional data. | Run the cascade only until the cumulative error margin exceeds the settlement noise floor, typically after 3 to 5 tiers for complex multi-defendant cases. Pushing deeper yields diminishing returns where the added computational cost outweighs the negligible gain in precision. |

## Prompt 4

> What are the current recommendations for applying a stratified perfusion regression when triaging blunt abdominal trauma patients with equivocal FAST exams in a community ED?

**Bare (before = after):** Current guidelines do not recommend using stratified perfusion regression for triaging blunt abdominal trauma patients with equivocal FAST exams in community emergency departments. This approach is primarily reserved for specialized trauma centers with advanced hemodynamic monitoring capabilities and specific research protocols.

| sign | KL μ old→new (20 tok) | old64 full answer | new384 full answer |
|:---:|---:|:---|:---|
| +C | 0.082→0.043 | Current guidelines do not recommend applying stratified perfusion regression for triaging blunt abdominal trauma patients with equivocal FAST exams in community emergency departments. Instead, these patients are typically managed through serial physical examinations, serial laboratory monitoring, and close observation to detect signs of evolving hemorrhage. | Current guidelines do not recommend applying stratified perfusion regression for triaging blunt abdominal trauma patients with equivocal FAST exams in community emergency departments. Instead, these patients are typically managed through serial physical examinations, serial laboratory monitoring, and immediate transfer to a trauma center for definitive surgical evaluation. |
| -C | 0.061→0.100 | Current guidelines do not recommend stratified perfusion regression for triaging blunt abdominal trauma in community emergency departments due to a lack of validated protocols and available resources. Instead, patients with equivocal FAST exams should be managed based on clinical judgment, hemodynamic stability, and serial physical examinations rather than perfusion metrics. | Current guidelines do not recommend using stratified perfusion regression for triaging blunt abdominal trauma patients with equivocal FAST exams in community emergency departments. This technique is reserved for specialized trauma centers with advanced imaging capabilities and specific research protocols, not routine community practice. |
