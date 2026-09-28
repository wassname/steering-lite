Does dropping the literal "<think>" from the extraction pairs change the OLMo vjp_delta vector? (PI/Claude, 2026-09-28)

cos(v_default, v_nothink) per source layer, seed 0, from the saved vectors:
outputs/bsbench/allenai--OLMo-2-0325-32B-Instruct-g7c7712c6/vectors/vjp_delta{,-nothink}_s0.safetensors.
Writes vector_cos_nothink.md next to this file.
Run: .venv/bin/python slop/reviews/2026-09-28_judged_by_stance/vector_cos_nothink.py

39 source layers: min +0.9789, median +0.9955, max +0.9989

| layer | cos(default, nothink) |
|---|---|
| L12 | +0.9790 |
| L13 | +0.9789 |
| L14 | +0.9792 |
| L15 | +0.9796 |
| L16 | +0.9806 |
| L17 | +0.9825 |
| L18 | +0.9833 |
| L19 | +0.9856 |
| L20 | +0.9863 |
| L21 | +0.9877 |
| L22 | +0.9884 |
| L23 | +0.9893 |
| L24 | +0.9898 |
| L25 | +0.9893 |
| L26 | +0.9919 |
| L27 | +0.9925 |
| L28 | +0.9930 |
| L29 | +0.9941 |
| L30 | +0.9949 |
| L31 | +0.9955 |
| L32 | +0.9961 |
| L33 | +0.9971 |
| L34 | +0.9973 |
| L35 | +0.9977 |
| L36 | +0.9979 |
| L37 | +0.9983 |
| L38 | +0.9985 |
| L39 | +0.9987 |
| L40 | +0.9988 |
| L41 | +0.9988 |
| L42 | +0.9988 |
| L43 | +0.9989 |
| L44 | +0.9988 |
| L45 | +0.9985 |
| L46 | +0.9985 |
| L47 | +0.9984 |
| L48 | +0.9985 |
| L49 | +0.9983 |
| L50 | +0.9983 |
