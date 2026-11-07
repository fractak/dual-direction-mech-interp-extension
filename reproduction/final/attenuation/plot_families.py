"""Plot the cumulative distribution of family-level interactions."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.4, 2.15))
for cohort, label, color, style in [
    ('original', 'Original families', '#2b5287', '-'),
    ('fresh', 'Fresh families', '#a34a0e', '--'),
]:
    with (ROOT / f'{cohort}_families.csv').open(newline='') as handle:
        values = np.sort([float(row['interaction_pp']) for row in csv.DictReader(handle)])
    x = np.r_[-125, values, 110]
    y = np.r_[0, np.arange(1, len(values) + 1) / len(values), 1]
    ax.step(x, y, where='post', linewidth=1.55, color=color, linestyle=style, label=label)
ax.axvline(0, color='#777777', linewidth=.7)
ax.set(xlim=(-125, 110), ylim=(0, 1.02), xlabel='Family interaction (percentage points)',
       ylabel='Fraction of families', xticks=[-100, -50, 0, 50, 100], yticks=[0, .25, .5, .75, 1])
ax.legend(loc='lower right', frameon=False, fontsize=7)
fig.tight_layout(pad=.65)
fig.savefig(ROOT.parents[1] / 'figures' / 'family_interaction_ecdf.pdf')
