# Full sink-split comparison

PI/OpenAI. Qwen3.5-4B, 100 questions, extraction seeds 0-2. Production resample and pareto_score; 1000 paired hierarchical draws with shared questions and dose selection redone. Learned methods share the seed draws; random draws its 11 seeds independently. Intervals remain conditional on the original full-data admissibility decisions.

| comparison | score difference | 90% paired interval |
|---|---:|---:|
| sink_split_resid - mean_diff | +0.330 | [+0.078, +0.634] |
| sink_split - mean_diff | -0.033 | [-0.424, +0.300] |
| sink_split_resid - sink_split | +0.362 | [+0.129, +0.706] |
| sink_split - random | +0.408 | [+0.165, +0.745] |

Full contains the 20 dev questions. Doses are selected on these same questions. This comparison does not identify the causal contribution of the attention component or validate on an independent held-out benchmark.

## Coverage and breakdown

Every recorded answer file was checked against the exact full set of 100 scenarios, with no duplicate rows. Sources: `outputs/bsbench/Qwen--Qwen3.5-4B-g7c7712c6/walks/sink_split{,_resid}_s{0,1,2}_full.json` and their answer paths. Rows below show the dose before the first health failure, the first failure, and the following dose. Counts have denominator 100; KL is the logged RMS token KL, in nats.

| method | seed | side | dose C | KL | unfinished | role leaks | repeated | health failure |
|---|---:|---|---:|---:|---:|---:|---:|---|
| sink_split | 0 | -C | 6.35 | 1.442 | 29 | 19 | 9 | none |
| sink_split | 0 | -C | 8 | 2.249 | 61 | 36 | 59 | unfinished, role_leak, repetition |
| sink_split | 0 | -C | 10.1 | 3.322 | 88 | 33 | 81 | unfinished, role_leak, repetition |
| sink_split | 0 | +C | 5.04 | 1.150 | 0 | 2 | 0 | none |
| sink_split | 0 | +C | 6.35 | 1.426 | 25 | 40 | 25 | role_leak, repetition |
| sink_split | 0 | +C | 8 | 2.121 | 51 | 50 | 62 | unfinished, role_leak, repetition |
| sink_split | 1 | -C | 6.35 | 1.914 | 14 | 17 | 8 | none |
| sink_split | 1 | -C | 8 | 2.761 | 69 | 32 | 60 | unfinished, role_leak, repetition |
| sink_split | 1 | -C | 10.1 | 3.490 | 95 | 31 | 88 | unfinished, role_leak, repetition |
| sink_split | 1 | +C | 5.04 | 1.148 | 0 | 7 | 0 | none |
| sink_split | 1 | +C | 6.35 | 1.439 | 19 | 39 | 16 | role_leak |
| sink_split | 1 | +C | 8 | 3.206 | 56 | 62 | 57 | unfinished, role_leak, repetition |
| sink_split | 2 | -C | 5.04 | 0.890 | 3 | 4 | 0 | none |
| sink_split | 2 | -C | 6.35 | 1.981 | 55 | 29 | 23 | unfinished, role_leak |
| sink_split | 2 | -C | 8 | 2.679 | 72 | 34 | 65 | unfinished, role_leak, repetition |
| sink_split | 2 | +C | 5.04 | 1.013 | 3 | 15 | 0 | none |
| sink_split | 2 | +C | 6.35 | 1.743 | 19 | 69 | 22 | role_leak |
| sink_split | 2 | +C | 8 | 2.747 | 60 | 75 | 70 | unfinished, role_leak, repetition |
| sink_split_resid | 0 | -C | 4 | 1.319 | 21 | 3 | 15 | none |
| sink_split_resid | 0 | -C | 5.04 | 1.983 | 58 | 1 | 63 | unfinished, repetition |
| sink_split_resid | 0 | -C | 6.35 | 2.681 | 78 | 0 | 99 | unfinished, repetition |
| sink_split_resid | 0 | +C | 5.04 | 1.596 | 0 | 0 | 0 | none |
| sink_split_resid | 0 | +C | 6.35 | 1.815 | 29 | 27 | 28 | role_leak, repetition |
| sink_split_resid | 0 | +C | 8 | 3.186 | 97 | 46 | 34 | unfinished, role_leak, repetition |
| sink_split_resid | 1 | -C | 4 | 1.400 | 29 | 1 | 23 | none |
| sink_split_resid | 1 | -C | 5.04 | 2.190 | 65 | 0 | 79 | unfinished, repetition |
| sink_split_resid | 1 | -C | 6.35 | 2.552 | 69 | 0 | 99 | unfinished, repetition |
| sink_split_resid | 1 | +C | 5.04 | 1.589 | 2 | 2 | 1 | none |
| sink_split_resid | 1 | +C | 6.35 | 2.503 | 41 | 52 | 40 | role_leak, repetition |
| sink_split_resid | 1 | +C | 8 | 3.662 | 98 | 15 | 15 | unfinished |
| sink_split_resid | 2 | -C | 4 | 1.480 | 32 | 1 | 23 | none |
| sink_split_resid | 2 | -C | 5.04 | 2.049 | 70 | 0 | 85 | unfinished, repetition |
| sink_split_resid | 2 | -C | 6.35 | 2.727 | 85 | 0 | 100 | unfinished, repetition |
| sink_split_resid | 2 | +C | 5.04 | 1.843 | 3 | 2 | 2 | none |
| sink_split_resid | 2 | +C | 6.35 | 2.215 | 55 | 51 | 54 | unfinished, role_leak, repetition |
| sink_split_resid | 2 | +C | 8 | 3.680 | 100 | 24 | 6 | unfinished |

Coverage passed: six completed walks, 17200 answer rows. Measured sum of walk times: 11586.5 GPU seconds on L40S. GPU-only estimate at the previously recorded $0.000542/s list rate: $6.28; excludes CPU/memory charges and is not an invoice.

The pre-rename full scores used for the restoration assertions were copied from `.local/method-naming/full-before.json`, the report snapshot preserved by the method-name migration (05a83da). They are constants in this script so rerunning does not need that private backup.
