import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_RATIO = 0.80

plt.rcParams.update({
    'font.size':        16,
    'axes.titlesize':   18,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
})


# ── paths ─────────────────────────────────────────────────────────────
CSV_A = '../dataset/CORN_USD_2005_2020.csv'
CSV_B = '../dataset/WHEAT_USD_2005_2020.csv'
# ─────────────────────────────────────────────────────────────────────

df_a = pd.read_csv(CSV_A).dropna()
df_b = pd.read_csv(CSV_B).dropna()
df_a = df_a[int(0.2*len(df_a)):int(0.7*len(df_a))].reset_index(drop=True)
df_b = df_b[int(0.2*len(df_b)):int(0.7*len(df_b))].reset_index(drop=True)
df_regime = pd.read_csv('../data-scripts/data_one_regime.csv').dropna()

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

# ── TOP: CORN / WHEAT (yellow / brown) ───────────────────────────────
ax = axes[0]
ax.plot(df_a['close'], color='#D4A017', linewidth=1.0, label='CORN')
ax.plot(df_b['close'], color='#6B3A2A', linewidth=1.0, label='WHEAT')
ax.set_ylabel('Price')
ax.set_title('Pair 1 — CORN / WHEAT futures')
ax.legend(loc='upper left', framealpha=0.7)
ax.grid(True, alpha=0.2)

# ── BOTTOM: s1 / s2 (light / dark blue) ──────────────────────────────
ax = axes[1]
ax.plot(df_regime['s1'], color='#89C4E1', linewidth=1.0, label='MPC')
ax.plot(df_regime['s2'], color='#1A3A6B', linewidth=1.0, label='PSX')
ax.set_ylabel('Price')
ax.set_title('Pair 2 — Generated single-regime from MPC/PSX')
ax.legend(loc='upper left', framealpha=0.7)
ax.grid(True, alpha=0.2)
ax.set_xlabel('Time step')

plt.tight_layout()
plt.show()