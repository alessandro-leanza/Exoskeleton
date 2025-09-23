import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# opzionale per CI95:
from math import sqrt
from scipy.stats import t as t_dist

csv_path = "descriptive_stats.csv"
# conditions_order = ["EC1 - No Exo", "EC2 - Exo without Vision", "EC3 - Exo with Vision"]
questions_order = [f"Q{i}" for i in range(1, 14)]
ylim = (1, 4)
save_path = "questionnaire_means_barplot.png"

plt.rcParams.update({"font.size": 22})

conditions_order = ["EC1", "EC2", "EC3"]  # <-- abbreviate

name_map = {
    "NoExo": "EC1",
    "ExoNoVision": "EC2",
    "ExoVision": "EC3",
}

condition_colors = {
    "EC1": "#E4F9FF",
    "EC2": "#80C7F0",
    "EC3": "#3A63EA",
}
# ======= NUOVO: carico anche SD e n =======
df = pd.read_csv(csv_path)[["question", "condition", "mean", "std", "n"]].copy()

# name_map = {
#     "NoExo": "EC1 - No Exo",
#     "ExoNoVision": "EC2 - Exo without Vision",
#     "ExoVision": "EC3 - Exo with Vision",
# }
df["condition"] = df["condition"].map(name_map)

# pivot per mean, sd, n
p_mean = df.pivot(index="question", columns="condition", values="mean")
p_sd   = df.pivot(index="question", columns="condition", values="std")
p_n    = df.pivot(index="question", columns="condition", values="n")

p_mean = p_mean.reindex(index=questions_order, columns=conditions_order)
p_sd   = p_sd.reindex_like(p_mean)
p_n    = p_n.reindex_like(p_mean)

# ======= SCEGLI il tipo di errore: 'SD', 'SEM', 'CI95' =======
ERR_KIND = "SD"  # <-- cambia in 'SEM' o 'CI95' se preferisci

def compute_err(sd_vals, n_vals):
    if ERR_KIND == "SD":
        return sd_vals
    elif ERR_KIND == "SEM":
        return sd_vals / np.sqrt(n_vals)
    elif ERR_KIND == "CI95":
        # t * SEM, con gradi di libertà n-1 (gestisce NaN)
        sem = sd_vals / np.sqrt(n_vals)
        # calcola il quantile t per ogni cella (usa ~1.96 se n grande)
        tcrit = np.vectorize(lambda nn: t_dist.ppf(0.975, nn-1) if nn and nn > 1 else np.nan)(n_vals)
        return sem * tcrit
    else:
        raise ValueError("ERR_KIND must be 'SD', 'SEM', or 'CI95'")

p_err = compute_err(p_sd, p_n)

# ======= Plot =======
num_questions = len(p_mean.index)
num_conditions = len(conditions_order)
x = np.arange(num_questions)

bar_width = 0.25
offsets = (np.arange(num_conditions) - (num_conditions - 1) / 2.0) * bar_width

fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=False)

for i, cond in enumerate(conditions_order):
    heights = p_mean[cond].values
    errs    = p_err[cond].values
    mask = ~np.isnan(heights)
    xi = x[mask] + offsets[i]
    hi = heights[mask]
    ei = errs[mask] if errs is not None else None

    ax.bar(
        xi, hi, width=bar_width, label=cond,
        color=condition_colors.get(cond, None),
        edgecolor="black", linewidth=0.5,
        yerr=ei, capsize=5,
        error_kw=dict(elinewidth=2.0, capthick=2.0, alpha=0.95)
    )


ax.set_ylabel("Mean score (Likert 1–4)")
ax.set_xlabel("Question")
ax.set_title("Questionnaire means by condition")
ax.set_xticks(x)
ax.set_xticklabels(p_mean.index, rotation=0)
ax.set_ylim(*ylim)

ax.legend(title="Condition", ncol=1, loc="upper center",
          bbox_to_anchor=(0.2, 1.0), frameon=False,
          borderaxespad=0.0, handlelength=1.4, columnspacing=1.0)

ax.grid(axis="y", linestyle=":", alpha=0.8)
leg = ax.legend(
    title="Condition",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),  # centrata sotto l'asse
    ncol=3,                       # una riga
    frameon=True,
    handlelength=1.4,
    columnspacing=1.2,
    borderaxespad=0.0,
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.25)  # lascia spazio per la legenda
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {save_path}")
