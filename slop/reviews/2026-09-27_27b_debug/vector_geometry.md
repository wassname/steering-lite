| model | method | seeds | cross-seed cos mean (min) | top-1 share | top-10 share | cos(., mean_diff) same layer |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | mean_diff | 3 | +1.000 (+0.997) | 0.017 | 0.061 | +1.000 |
| Qwen3.5-4B | vjp_delta | 3 | +1.000 (+1.000) | 0.009 | 0.061 | -0.031 |
| Qwen3.5-4B | vjp_cache | 3 | +1.000 (+1.000) | 0.058 | 0.218 | n/a |
| Qwen3.5-4B | random | 3 | -0.002 (-0.041) | 0.005 | 0.039 | -0.003 |
| Qwen3.5-27B | mean_diff | 3 | +1.000 (+0.998) | 0.017 | 0.048 | +1.000 |
| Qwen3.5-27B | vjp_delta | 3 | +0.999 (+0.986) | 0.007 | 0.042 | +0.011 |
| Qwen3.5-27B | vjp_cache | 3 | +0.994 (+0.925) | 0.028 | 0.152 | n/a |
| Qwen3.5-27B | random | 3 | -0.002 (-0.045) | 0.003 | 0.022 | -0.000 |
| OLMo-2-32B | mean_diff | 1 | 1 seed | 0.010 | 0.041 | +1.000 |
| OLMo-2-32B | vjp_delta | 1 | 1 seed | 0.027 | 0.102 | +0.014 |
| OLMo-2-32B | vjp_cache | 1 | 1 seed | 0.034 | 0.147 | n/a |
| OLMo-2-32B | random | 3 | -0.002 (-0.045) | 0.003 | 0.022 | -0.002 |
