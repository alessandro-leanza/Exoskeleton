# -*- coding: utf-8 -*-
"""
Questionario HRC su scala Likert (1-4)
- Inserisci i dati sotto.
- Output: statistiche descrittive, test (Friedman, Wilcoxon), effect size, grafico PNG.

Requisiti: numpy, pandas, matplotlib, scipy
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
from itertools import combinations

# ===========================
# === 1) INSERISCI DATI  ===
# ===========================
# Ogni riga = un partecipante. Ogni riga ha 13 valori (Q1..Q13) su scala 1..4.
# Se non hai una risposta, usa None.

no_exo = [
    (2,), (2,), (3,), (2,), (2,),
    (2,), (1,), (1,), (2,), (4,),
    (2,), (3,),(1,), (1,), (2,)
]

exo_no_vision = [
    (2,2,2,3,2,2,1,1,2,2,2,2,2),
    (2,2,2,1,1,1,2,1,3,4,3,3,3),
    (3,4,3,2,2,2,3,3,3,3,3,2,2),
    (3,3,3,2,3,3,3,3,3,3,3,3,4),
    (2,2,2,2,1,2,2,2,2,1,2,2,2),
    (2,3,2,3,2,3,3,3,3,3,3,3,3),
    (2,2,2,4,4,1,4,3,3,3,4,2,4),
    (1,2,2,1,2,2,2,2,2,2,2,2,2),
    (3,2,2,2,3,1,1,2,2,3,3,3,2),
    (2,3,3,2,3,2,2,2,3,2,2,3,2),
    (2,2,2,2,1,1,2,1,2,1,2,1,1),
    (2,3,3,2,1,2,2,1,1,2,2,1,2),
    (2,2,2,1,1,2,2,2,1,1,2,1,2),
    (2,1,2,2,1,2,2,2,1,2,1,1,2),
    (2,2,1,2,2,1,2,2,1,2,1,1,2),
]


exo_vision = [
    (3,3,3,3,3,2,3,3,3,3,3,3,3),
    (3,3,2,3,3,3,2,2,3,3,3,3,3),
    (4,3,3,3,3,3,3,3,2,3,4,2,3),
    (3,3,2,2,3,2,2,2,2,3,2,2,2),
    (4,3,3,3,3,3,3,3,3,4,4,3,4),
    (3,2,3,3,3,3,3,3,3,3,3,3,3),
    (4,4,4,4,3,4,4,4,4,4,4,4,4),
    (3,3,3,4,3,4,3,4,4,3,3,3,3),
    (4,4,3,3,4,2,3,4,3,4,4,4,4),
    (4,4,4,3,3,4,4,4,4,4,4,4,4),
    (4,4,4,3,3,3,4,4,4,4,3,4,4),
    (3,3,3,2,1,2,2,2,1,2,3,2,2),
    (4,4,4,4,3,4,4,4,4,4,4,3,4),
    (4,4,4,3,4,3,4,3,4,4,4,3,4),
    (4,3,4,4,3,4,4,4,4,4,4,3,4),
]



# Etichette domande (modifica se vuoi)
QUESTIONS = [
    "The lifting task was not physically demanding.",
    "The human-robot team worked fluently together.",
    "The robot contributed to the fluency of the interaction.",
    "I didn't have to carry the weight to improve the human-robot team.",
    "The robot contributed equally to the team performance.",
    "The robot was the important team member on the team.",
    "I trusted the robot to do the right thing at the right time.",
    "The robot was intelligent.",
    "The robot was trustworthy.",
    "The robot was committed to the task.",
    "I feel comfortable with the robot.",
    "The robot and I understand each other.",
    "The robot perceives accurately what my goals are.",
]
NUM_Q = 13

# =====================================================
# === 2) NORMALIZZA INPUT E COSTRUISCI I DATAFRAME  ===
# =====================================================

def pad_row_to_13(row):
    """Accetta tuple/list di lunghezza 1 o 13 e riempie con None fino a 13."""
    row = list(row)
    if len(row) == 1:
        # solo Q1 -> completa con None
        row = row + [None] * (NUM_Q - 1)
    elif len(row) < NUM_Q:
        row = row + [None] * (NUM_Q - len(row))
    return tuple(row[:NUM_Q])

no_exo = [pad_row_to_13(r) for r in no_exo]
exo_no_vision = [pad_row_to_13(r) for r in exo_no_vision]
exo_vision = [pad_row_to_13(r) for r in exo_vision]

def to_long_df(data, condition_name):
    rows = []
    for sid, row in enumerate(data, start=1):
        for qi in range(NUM_Q):
            val = row[qi]
            if val is not None:
                rows.append({
                    "subject": sid,
                    "condition": condition_name,
                    "question": f"Q{qi+1}",
                    "question_text": QUESTIONS[qi],
                    "score": float(val),
                })
    return pd.DataFrame(rows)

df_long = pd.concat([
    to_long_df(no_exo, "NoExo"),
    to_long_df(exo_no_vision, "ExoNoVision"),
    to_long_df(exo_vision, "ExoVision"),
], ignore_index=True)

# ===================================================
# === 3) STATISTICHE DESCRITTIVE PER DOMANDA/COND ===
# ===================================================

def describe_by_question_condition(long_df):
    out = []
    for q in [f"Q{i}" for i in range(1, NUM_Q+1)]:
        for cond in ["NoExo", "ExoNoVision", "ExoVision"]:
            scores = long_df[(long_df["question"] == q) & (long_df["condition"] == cond)]["score"].dropna()
            if len(scores) == 0:
                mean = sd = vmin = vmax = np.nan
                n = 0
            else:
                mean = scores.mean()
                sd = scores.std(ddof=1) if len(scores) > 1 else 0.0
                vmin = scores.min()
                vmax = scores.max()
                n = len(scores)
            out.append({
                "question": q,
                "condition": cond,
                "n": n,
                "mean": mean,
                "std": sd,
                "min": vmin,
                "max": vmax
            })
    return pd.DataFrame(out)

desc = describe_by_question_condition(df_long)
print("\n=== Descrittive per domanda/condizione ===")
print(desc.to_string(index=False))

# ============================================
# === 4) TEST: FRIEDMAN su Q1 (3 condizioni) ==
#     + post-hoc Wilcoxon con correzione Holm
# ============================================

def align_three_conditions_for_q1(no_exo_data, exo_no_vis_data, exo_vis_data):
    """
    Restituisce tre array allineati (stessi soggetti) per Q1.
    L'allineamento è per indice: assume che le liste siano nello stesso ordine soggetti.
    Taglia alla lunghezza minima comune.
    """
    n = min(len(no_exo_data), len(exo_no_vis_data), len(exo_vis_data))
    a = [no_exo_data[i][0] for i in range(n)]
    b = [exo_no_vis_data[i][0] for i in range(n)]
    c = [exo_vis_data[i][0] for i in range(n)]
    # filtra eventuali None
    aligned = [(x, y, z) for (x, y, z) in zip(a, b, c) if x is not None and y is not None and z is not None]
    if len(aligned) == 0:
        return np.array([]), np.array([]), np.array([])
    a1, b1, c1 = zip(*aligned)
    return np.array(a1, dtype=float), np.array(b1, dtype=float), np.array(c1, dtype=float)

def holm_correction(pvals):
    """Correzione di Holm-Bonferroni. Ritorna p-corretti nello stesso ordine di input."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        adj = (m - k) * pvals[idx]
        adj = max(adj, prev)  # garantisce monotonia non decrescente
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted

def friedman_posthoc_wilcoxon(a, b, c, labels=("NoExo","ExoNoVision","ExoVision")):
    """Esegue Wilcoxon per tutte le coppie con correzione Holm."""
    pairs = [("NoExo","ExoNoVision", a, b),
             ("NoExo","ExoVision", a, c),
             ("ExoNoVision","ExoVision", b, c)]
    p_raw = []
    stats = []
    for (l1,l2,x,y) in pairs:
        # ignora differenze nulle (zero_method='wilcox' default)
        try:
            res = wilcoxon(x, y, alternative='two-sided')
            p_raw.append(res.pvalue)
            stats.append((l1, l2, res.statistic, res.pvalue))
        except ValueError:
            p_raw.append(np.nan)
            stats.append((l1, l2, np.nan, np.nan))
    p_adj = holm_correction(np.array([p if not math.isnan(p) else 1.0 for p in p_raw]))
    table = []
    for i, (l1, l2, W, p) in enumerate(stats):
        table.append({
            "pair": f"{l1} vs {l2}",
            "W": W,
            "p_raw": p,
            "p_holm": p_adj[i] if not math.isnan(p) else np.nan
        })
    return pd.DataFrame(table)

# Esegui Friedman su Q1 se possibile
q1_a, q1_b, q1_c = align_three_conditions_for_q1(no_exo, exo_no_vision, exo_vision)
if len(q1_a) >= 3:
    fr_stat, fr_p = friedmanchisquare(q1_a, q1_b, q1_c)
    N = len(q1_a); k = 3
    kendall_W = fr_stat / (N * (k - 1)) if N > 0 else np.nan  # effect size
    print("\n=== Q1: Friedman (3 condizioni) ===")
    print(f"chi2 = {fr_stat:.3f}, p = {fr_p:.5f}, Kendall's W = {kendall_W:.3f} (N={N})")

    posthoc = friedman_posthoc_wilcoxon(q1_a, q1_b, q1_c)
    print("\nPost-hoc Wilcoxon (Holm):")
    print(posthoc.to_string(index=False))
else:
    print("\n[Q1] Dati insufficienti per Friedman (servono ≥3 soggetti con tutte e 3 le condizioni).")

# ==========================================================
# === 5) TEST: WILCOXON Q2..Q13 (ExoNoVision vs ExoVision) ==
#      + effect size rank-biserial + CORREZIONE HOLM
# ==========================================================

def paired_arrays_for_two_conditions(data_A, data_B, question_index):
    """Allinea coppie A/B per domanda question_index (0-based)."""
    n = min(len(data_A), len(data_B))
    A = []
    B = []
    for i in range(n):
        a = data_A[i][question_index]
        b = data_B[i][question_index]
        if a is not None and b is not None:
            A.append(float(a))
            B.append(float(b))
    return np.array(A), np.array(B)

def rank_biserial_from_diffs(A, B):
    """
    Effect size rank-biserial per Wilcoxon: (n_pos - n_neg) / (n_pos + n_neg).
    Ignora differenze zero.
    """
    diff = B - A
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    denom = n_pos + n_neg
    if denom == 0:
        return np.nan
    return (n_pos - n_neg) / denom

print("\n=== Q2..Q13: Wilcoxon (ExoNoVision vs ExoVision) + rank-biserial + Holm ===")
rows = []
p_values = []

for qi in range(2, NUM_Q+1):
    A, B = paired_arrays_for_two_conditions(exo_no_vision, exo_vision, qi-1)
    if len(A) < 5:
        rows.append({"question": f"Q{qi}", "n": len(A), "W": np.nan, "p_raw": np.nan,
                     "p_holm": np.nan, "rank_biserial": np.nan})
        p_values.append(np.nan)
        continue
    try:
        # Nota: puoi usare method="auto" (SciPy recente) oppure lasciare default
        res = wilcoxon(A, B, alternative='two-sided')
        rb = rank_biserial_from_diffs(A, B)
        rows.append({"question": f"Q{qi}", "n": len(A), "W": res.statistic, "p_raw": res.pvalue,
                     "p_holm": np.nan, "rank_biserial": rb})
        p_values.append(res.pvalue)
    except ValueError:
        rows.append({"question": f"Q{qi}", "n": len(A), "W": np.nan, "p_raw": np.nan,
                     "p_holm": np.nan, "rank_biserial": np.nan})
        p_values.append(np.nan)

# --- Applica Holm solo ai p non-NaN, mantenendo l'ordine ---
p_array = np.array(p_values, dtype=float)
valid_mask = ~np.isnan(p_array)
if valid_mask.any():
    p_valid = p_array[valid_mask]
    # Usa la tua funzione definita sopra
    p_valid_holm = holm_correction(p_valid)
    # reinserisci nelle righe
    idx_valid = np.where(valid_mask)[0]
    for i_rel, i_abs in enumerate(idx_valid):
        rows[i_abs]["p_holm"] = p_valid_holm[i_rel]

wilc_table = pd.DataFrame(rows)
print(wilc_table.to_string(index=False))

# ============================================
# === 5b) T-TEST APPAIATI (parametrici)    ===
# ============================================
from scipy.stats import ttest_rel

def cohen_d_paired(A, B):
    """Cohen's d per campioni appaiati: mean(diff)/sd(diff)."""
    diff = np.array(B, dtype=float) - np.array(A, dtype=float)
    if diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)

# --- Q1: tre confronti appaiati (EC1, EC2, EC3) con Holm tra i 3 p-values ---
print("\n=== Q1: Paired t-tests (3 confronti) + Holm ===")
if len(q1_a) >= 3:
    pairs_q1 = [
        ("NoExo vs ExoNoVision", q1_a, q1_b),
        ("NoExo vs ExoVision",   q1_a, q1_c),
        ("ExoNoVision vs ExoVision", q1_b, q1_c),
    ]
    p_raw_q1 = []
    rows_q1 = []
    for name, X, Y in pairs_q1:
        res = ttest_rel(X, Y)
        d = cohen_d_paired(X, Y)
        rows_q1.append({"pair": name, "n": len(X), "t": res.statistic, "p_raw": res.pvalue, "p_holm": np.nan, "cohen_d": d})
        p_raw_q1.append(res.pvalue)
    p_raw_q1 = np.array(p_raw_q1, dtype=float)
    p_holm_q1 = holm_correction(p_raw_q1)
    for i in range(len(rows_q1)):
        rows_q1[i]["p_holm"] = p_holm_q1[i]
    ttest_q1_table = pd.DataFrame(rows_q1)
    print(ttest_q1_table.to_string(index=False))
    ttest_q1_table.to_csv("ttest_q1_pairwise.csv", index=False)
    print("Salvato: ttest_q1_pairwise.csv")
else:
    print("[Q1] Dati insufficienti per t-test (servono ≥3 soggetti con tutte e 3 le condizioni).")

# --- Q2..Q13: t-test appaiato EC2 vs EC3 con Holm tra le 12 domande ---
print("\n=== Q2..Q13: Paired t-test (ExoNoVision vs ExoVision) + Cohen's d + Holm ===")
rows_t = []
p_vals_t = []
ns = []
for qi in range(2, NUM_Q+1):
    A, B = paired_arrays_for_two_conditions(exo_no_vision, exo_vision, qi-1)  # riusa funzione già definita
    if len(A) < 5:
        row = {"question": f"Q{qi}", "n": len(A), "t": np.nan, "p_raw": np.nan, "p_holm": np.nan, "cohen_d": np.nan}
        rows_t.append(row)
        p_vals_t.append(np.nan)
        ns.append(len(A))
        continue
    res = ttest_rel(A, B)
    d = cohen_d_paired(A, B)
    row = {"question": f"Q{qi}", "n": len(A), "t": res.statistic, "p_raw": res.pvalue, "p_holm": np.nan, "cohen_d": d}
    rows_t.append(row)
    p_vals_t.append(res.pvalue)
    ns.append(len(A))

# Correzione di Holm sui soli p validi, preservando l'ordine delle domande
p_arr = np.array(p_vals_t, dtype=float)
mask = ~np.isnan(p_arr)
if mask.any():
    p_holm = holm_correction(p_arr[mask])
    idx_valid = np.where(mask)[0]
    for j_rel, j_abs in enumerate(idx_valid):
        rows_t[j_abs]["p_holm"] = p_holm[j_rel]

ttest_table = pd.DataFrame(rows_t)
print(ttest_table.to_string(index=False))
ttest_table.to_csv("ttest_results_q2_q13.csv", index=False)
print("Salvato: ttest_results_q2_q13.csv")

# ============================================
# === 6) GRAFICO A BARRE ORIZZONTALI (PNG)  ===
# ============================================

def plot_means(desc_df, out_path="qualitative_questionnaire.png"):
    # Preparazione matrici medie con NaN per voci mancanti
    conds = ["NoExo", "ExoNoVision", "ExoVision"]
    means = {c: [] for c in conds}
    labels = []
    for qi in range(1, NUM_Q+1):
        qlab = QUESTIONS[qi-1]
        labels.append(qlab)
        for c in conds:
            val = desc_df[(desc_df["question"] == f"Q{qi}") & (desc_df["condition"] == c)]["mean"]
            m = float(val.values[0]) if len(val) else np.nan
            means[c].append(m)

    y = np.arange(NUM_Q)
    height = 0.22

    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    ax.barh(y + height, means["NoExo"], height, label="No Exoskeleton")
    ax.barh(y,          means["ExoNoVision"], height, label="No Vision")
    ax.barh(y - height, means["ExoVision"], height, label="With Vision")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(1, 4)
    ax.set_ylim(1, 4)
    ax.set_xlabel("Likert (1–4)")
    ax.set_title("Qualitative Questionnaire")
    ax.grid(axis='x', alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nGrafico salvato: {out_path}")

plot_means(desc)

# ============================================
# === 7) ESPORTO CSV (opzionale)            ===
# ============================================
desc.to_csv("descriptive_stats.csv", index=False)
wilc_table.to_csv("wilcoxon_results_q2_q13.csv", index=False)
print("Salvati: descriptive_stats.csv, wilcoxon_results_q2_q13.csv")
