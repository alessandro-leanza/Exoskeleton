import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "descriptive_stats.csv"
conditions_order = ["EC1 - No Exo", "EC2 - Exo without Vision", "EC3 - Exo with Vision"]
questions_order = [f"Q{i}" for i in range(1, 14)]
ylim = (1, 4)
save_path = "questionnaire_means_barplot.png"

plt.rcParams.update({"font.size": 22})

condition_colors = {
    "EC1 - No Exo": "#E4F9FF",
    "EC2 - Exo without Vision": "#80C7F0",
    "EC3 - Exo with Vision": "#3A63EA",
}

df = pd.read_csv(csv_path)[["question", "condition", "mean"]].copy()

# Map nomi dal CSV alle etichette paper
name_map = {
    "NoExo": "EC1 - No Exo",
    "ExoNoVision": "EC2 - Exo without Vision",
    "ExoVision": "EC3 - Exo with Vision",
}
df["condition"] = df["condition"].map(name_map)

pivot = df.pivot(index="question", columns="condition", values="mean")
pivot = pivot.reindex(index=questions_order, columns=conditions_order)

num_questions = len(pivot.index)
num_conditions = len(conditions_order)
x = np.arange(num_questions)

bar_width = 0.25
offsets = (np.arange(num_conditions) - (num_conditions - 1) / 2.0) * bar_width

fig, ax = plt.subplots(figsize=(12, 5))

for i, cond in enumerate(conditions_order):
    heights = pivot[cond].values
    mask = ~np.isnan(heights)
    xi = x[mask] + offsets[i]
    hi = heights[mask]
    ax.bar(xi, hi, width=bar_width, label=cond,
           color=condition_colors.get(cond, None),
           edgecolor="black", linewidth=0.5)

ax.set_ylabel("Mean score (Likert 1–4)")
ax.set_xlabel("Question")
ax.set_title("Questionnaire means by condition")
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=0)
ax.set_ylim(*ylim)

ax.legend(title="Condition", ncol=1, loc="upper center",
          bbox_to_anchor=(0.2, 1.0), frameon=False,
          borderaxespad=0.0, handlelength=1.4, columnspacing=1.0)

ax.grid(axis="y", linestyle=":", alpha=0.8)
plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.show()

print(f"Saved plot to: {save_path}")
