#!/usr/bin/env python3
"""
Phase 8bis — Reconstruction du bootstrap AUROC pairé (manquant dans les livrables)
Paired bootstrap AUROC (2000 resamples) + correction Benjamini-Hochberg
Comparaison : les 6 méthodes d'incertitude (Phase 5) sur les mêmes 718 patients test.

Ce script recalcule ce qui était cité dans le Tableau 11 du papier / Tableau 10 du Lot2,
mais dont aucun script ni sortie n'existait dans les livrables originaux.
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_auc_score

def benjamini_hochberg(pvals, alpha=0.05):
    """Réimplémentation pure numpy de la correction BH (équivalent statsmodels fdr_bh)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh_vals = ranked * n / (np.arange(1, n + 1))
    # Rendre la séquence monotone décroissante en repartant de la fin (cummin depuis la fin)
    bh_vals_monotone = np.minimum.accumulate(bh_vals[::-1])[::-1]
    bh_vals_monotone = np.clip(bh_vals_monotone, 0, 1)
    p_corrected = np.empty(n)
    p_corrected[order] = bh_vals_monotone
    rejected = p_corrected <= alpha
    return rejected, p_corrected

SEED = 42
N_RESAMPLES = 2000

BASE = Path("/home/claude/confidx_extract/CONFIDX")
PHASE5_DIR = BASE / "phase5_results_v2"

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

detailed = load_json(PHASE5_DIR / "phase5_fusion_detailed.json")
records = detailed["patient_records"]
n = len(records)
print(f"Patients : {n}")

y_true = np.array([r["gt_uncertain"] for r in records])
s1 = np.array([r["s1_verbalized"] for r in records])
s2 = np.array([r["s2_inconsistency"] for r in records])
s3 = np.array([r["s3_logprob_uncertainty"] for r in records])
wscore = np.array([r["fusion_weighted_score"] for r in records])
lr_proba = np.array([r["fusion_lr_proba"] for r in records])

SCORES = {
    "S1_Verbalized": s1,
    "S2_SelfConsistency": s2,
    "S3_LogProb": s3,
    "Fusion_Weighted": wscore,
    "Fusion_LogisticRegression": lr_proba,
}

def point_auroc(scores, y):
    return roc_auc_score(y, scores)

print("\nAUROC ponctuels (vérification vs phase5_summary.json) :")
for name, sc in SCORES.items():
    print(f"  {name:<28} AUROC = {point_auroc(sc, y_true):.4f}")

rng = np.random.default_rng(SEED)
idx_all = np.arange(n)

# Pré-génération des resamples (mêmes indices pour toutes les paires -> comparaison appariée correcte)
resample_indices = rng.integers(0, n, size=(N_RESAMPLES, n))

def bootstrap_delta_auroc(name_a, name_b):
    sa, sb = SCORES[name_a], SCORES[name_b]
    deltas = np.empty(N_RESAMPLES)
    for i, idx in enumerate(resample_indices):
        yb = y_true[idx]
        # AUROC non défini si toutes les classes identiques dans le resample -> on saute (rare)
        if len(np.unique(yb)) < 2:
            deltas[i] = np.nan
            continue
        auc_a = roc_auc_score(yb, sa[idx])
        auc_b = roc_auc_score(yb, sb[idx])
        deltas[i] = auc_a - auc_b
    deltas = deltas[~np.isnan(deltas)]
    point_delta = point_auroc(sa, y_true) - point_auroc(sb, y_true)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # p-value bootstrap bilatérale : proportion de resamples où le signe s'inverse par rapport au point estimate
    p_value = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    p_value = min(p_value, 1.0)
    return point_delta, ci_low, ci_high, p_value

pairs = list(combinations(SCORES.keys(), 2))
results = []
print(f"\n{'='*90}\nBOOTSTRAP AUROC PAIRÉ ({N_RESAMPLES} resamples, seed={SEED})\n{'='*90}")
for a, b in pairs:
    delta, lo, hi, p = bootstrap_delta_auroc(a, b)
    results.append({"method_a": a, "method_b": b, "delta_auroc": delta,
                     "ci_low": lo, "ci_high": hi, "p_raw": p})
    print(f"{a:<28} vs {b:<28} | ΔAUROC={delta:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.4f}")

pvals = [r["p_raw"] for r in results]
rejected, pvals_bh = benjamini_hochberg(pvals, alpha=0.05)

print(f"\n{'='*90}\nCORRECTION BENJAMINI-HOCHBERG (FDR=0.05)\n{'='*90}")
for r, p_bh, rej in zip(results, pvals_bh, rejected):
    r["p_bh"] = float(p_bh)
    r["significant"] = bool(rej)
    marker = "✅ significatif" if rej else "— n.s."
    print(f"{r['method_a']:<24} vs {r['method_b']:<24} ΔAUROC={r['delta_auroc']:+.3f} "
          f"pBH={p_bh:.4f}  {marker}")

n_sig = sum(rejected)
print(f"\n{n_sig}/{len(pvals)} comparaisons significatives après correction BH")

out = {
    "n_patients": n,
    "n_resamples": N_RESAMPLES,
    "seed": SEED,
    "point_auroc": {k: float(point_auroc(v, y_true)) for k, v in SCORES.items()},
    "pairwise_bootstrap": results,
}
out_path = Path("/mnt/user-data/outputs/phase8bis_bootstrap_auroc_RECALCULE.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n💾 Sauvegardé : {out_path}")
