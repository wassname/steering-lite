# Goal: Trajectory NLL Calibration and Plotting

## Context

The hardest part of calibration is calibrating cheaply on a short trajectory and generalizing to a long stable trajectory. We want to measure trajectory change directly by looking at the NLL of the steered model on the baseline model's generated trajectory (forward) and vice versa (reverse).

1. **Plot it over a longer steered trajectory**: Visualize how NLL changes over a 4096-token trajectory. Does it compound exponentially or self-correct?
2. **Try it as a calibration target**: Replace TV with target $\Delta NLL$ for calibration in `iso_tv_calibrate.py`.

## Specifications

1. **`scripts/plot_trajectory_nll.py`**:
   - Generates a long trajectory (e.g., 1024-4096 tokens) from the base model.
   - Computes per-token NLL of both base and steered models on this trajectory.
   - Plots/saves the cumulative $\Delta NLL$ (Steered NLL - Base NLL) over time.
   - Applies the token-efficient logging skill.

2. **Update `scripts/iso_tv_calibrate.py`**:
   - Replace TV-based bisection target with $\Delta NLL$ on a baseline greedy generation (e.g., 128 tokens).
   - Target $\Delta NLL \approx 0.1$ nats/token (or similar calibrated value).
   - Also compute `flip_rate` (how often argmax changes) as a reliability column.
   - Maintain the token-efficient logging output format per `SKILL.md`.

## Verification (LGTM)
- `plot_trajectory_nll.py` produces a CSV/plot showing NLL over time for 6 methods.
- `iso_tv_calibrate.py` successfully calibrates using the new NLL target.
- Smoke tests pass.
- Log outputs follow the `token-efficient-logging` principles (BLUF, cue emoji 🟢, tabulate tsv).
