import pandas as pd
import matplotlib.pyplot as plt
import sys
df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for m, grp in df.groupby("method"):
    steps = grp["step"].values
    axes[0].plot(steps, grp["delta_nll"].rolling(16).mean(), label=m, alpha=0.8)
    axes[1].plot(steps, grp["cumsum_delta"], label=m, alpha=0.8)
axes[0].set_ylabel("Delta NLL (rolling mean 16)")
axes[0].set_title("Steered NLL - Base NLL per token")
axes[1].set_ylabel("Cumulative Delta NLL")
axes[1].set_xlabel("Token position")
axes[0].legend()
plt.tight_layout()
plt.savefig(sys.argv[1].replace(".csv", ".png"))
print(f"saved {sys.argv[1].replace('.csv', '.png')}")
