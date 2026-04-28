"""Replot aggregate distribution-shift KL plot from cached CSV with the
new iso-KL calibration source (greedy p95=1)."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, json
csv = 'outputs/trajectory/agg_dist_shift__Qwen--Qwen3.5-0.8B__N8__T256__a0.5_a1.0_a4.0__seed0.csv'
df = pd.read_csv(csv)
iso_path = 'outputs/iso_kl/iso_kl__Qwen--Qwen3.5-0.8B__L4__greedy_kl_p95_1.0__T20__N4__seeds0__1777355924.json'
iso = json.load(open(iso_path))
methods_coeffs = {r['method']: r['calibrated_coeff'] for r in iso['summary']}
alphas = sorted(df.alpha.unique())
fig, axes = plt.subplots(1, len(alphas), figsize=(5.5*len(alphas), 4.5), squeeze=False)
colors = plt.cm.tab10.colors
mc = {m: colors[i] for i, m in enumerate(methods_coeffs)}
window = 1  # no smoothing
for j, alpha in enumerate(alphas):
    ax = axes[0, j]
    for m in methods_coeffs:
        sub = df[(df.method == m) & (df.alpha == alpha) & (df.metric == 'kl_sb')].sort_values('t')
        if len(sub) == 0:
            continue
        t = sub['t'].values
        p50 = sub['p50'].values
        p10 = sub['p10'].values
        p90 = sub['p90'].values
        lab = f'{m} (c={methods_coeffs[m]*alpha:.2g})' if j == 0 else None
        ax.plot(t, p50, label=lab, color=mc[m], lw=1.4)
        ax.fill_between(t, np.maximum(p10, 1e-3), p90, alpha=0.12, color=mc[m])
    ax.set_xlabel('token position t')
    ax.set_ylabel('KL(steer || base) [nats]')
    ax.set_title(f'\u03b1 = {alpha}\u00d7 calibrated coeff')
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e1)
    ax.axhline(1.0, color='black', lw=1.6, alpha=0.85, zorder=5)
    ax.axvline(20, color='black', lw=1.0, ls=':', alpha=0.7, zorder=5)
    ax.grid(True, alpha=0.3)
axes[0, 0].legend(loc='best', fontsize=8)
fig.suptitle(
    'KL(steered \u2225 base) per token \u00b7 \u03b1 = 0.5\u00d7, 1\u00d7, 4\u00d7 calibrated coeff \u00b7 '
    'N=8 prompts \u00b7 T=256 \u00b7 no smoothing\n'
    'Qwen/Qwen3.5-0.8B \u00b7 calibration: greedy KL_p95=1 over first 20 tokens \u00b7 '
    'solid black @ KL=1 = side of road \u00b7 dotted @ t=20 = end of calibration window',
    fontsize=9,
)
fig.tight_layout()
out = csv.replace('.csv', '__isoKL_roadlines.png')
fig.savefig(out, dpi=110, bbox_inches='tight')
print(out)
