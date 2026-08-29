#!/usr/bin/env python3
"""
Phase 8 — Validation statistique
McNemar (accuracy binaire appariée) + Wilcoxon signed-rank (scores continus)
+ correction Benjamini-Hochberg pour comparaisons multiples
Comparaison : les 6 méthodes d'incertitude (Phase 5) sur les mêmes 718 patients test.
"""
 
import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
 
BASE = Path("/content/CONFIDX")
PHASE5_DIR = BASE / "phase5_results_v2"
OUT_DIR = BASE / "phase8_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
 
def save_json(data, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  💾 {p.name}")
 
# ============================================================
# 1. CHARGEMENT
# ============================================================
 
summary = load_json(PHASE5_DIR / "phase5_fusion_detailed.json")
records = summary["patient_records"]
calib = load_json(PHASE5_DIR / "phase5_calibration_params.json")
 
n = len(records)
print(f"Patients : {n}")
 
y_true = np.array([r["gt_uncertain"] for r in records])
s1 = np.array([r["s1_verbalized"] for r in records])
s2 = np.array([r["s2_inconsistency"] for r in records])
s3 = np.array([r["s3_logprob_uncertainty"] for r in records])
wscore = np.array([r["fusion_weighted_score"] for r in records])
lr_proba = np.array([r["fusion_lr_proba"] for r in records])
 
# Binarisation des signaux individuels avec les seuils calibrés sur VAL
pred_s1 = (s1 >= calib["th1"]).astype(int)
pred_s2 = (s2 >= calib["th2"]).astype(int)
pred_s3 = (s3 >= calib["th3"]).astype(int)
pred_majority = np.array([r["fusion_majority_vote"] for r in records])
pred_weighted = np.array([r["fusion_weighted_pred"] for r in records])
pred_lr = np.array([r["fusion_lr_pred"] for r in records])
 
METHODS = {
    "S1_Verbalized": (pred_s1, s1),
    "S2_SelfConsistency": (pred_s2, s2),
    "S3_LogProb": (pred_s3, s3),
    "Fusion_MajorityVote": (pred_majority, pred_majority.astype(float)),  # binaire, pas de score continu propre
    "Fusion_Weighted": (pred_weighted, wscore),
    "Fusion_LogisticRegression": (pred_lr, lr_proba),
}
 
# Correction binaire : 1 si la méthode a bien classé le patient (pred == y_true)
correctness = {name: (pred == y_true).astype(int) for name, (pred, _) in METHODS.items()}
 
# ============================================================
# 2. McNEMAR — comparaisons appariées par paires (accuracy binaire)
# ============================================================
 
print("\n" + "=" * 75)
print("McNEMAR TEST — comparaisons par paires (accuracy)")
print("=" * 75)
 
mcnemar_results = []
names = list(METHODS.keys())
for a, b in combinations(names, 2):
    ca, cb = correctness[a], correctness[b]
    # Table de contingence 2x2 : [both correct, a correct & b wrong; a wrong & b correct, both wrong]
    both_correct = int(np.sum((ca == 1) & (cb == 1)))
    a_only = int(np.sum((ca == 1) & (cb == 0)))
    b_only = int(np.sum((ca == 0) & (cb == 1)))
    both_wrong = int(np.sum((ca == 0) & (cb == 0)))
    table = [[both_correct, a_only], [b_only, both_wrong]]
 
    # exact=True recommandé si a_only+b_only < 25 (cas fréquent ici)
    use_exact = (a_only + b_only) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)
 
    mcnemar_results.append({
        "method_a": a, "method_b": b,
        "acc_a": float(ca.mean()), "acc_b": float(cb.mean()),
        "a_only_correct": a_only, "b_only_correct": b_only,
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "exact_test": use_exact
    })
 
for r in mcnemar_results:
    print(f"{r['method_a']:<28} vs {r['method_b']:<28} | "
          f"acc={r['acc_a']:.3f} vs {r['acc_b']:.3f} | p={r['p_value']:.4f}")
 
# ============================================================
# 3. WILCOXON SIGNED-RANK — scores continus par paires
# ============================================================
 
print("\n" + "=" * 75)
print("WILCOXON SIGNED-RANK — comparaisons par paires (scores continus)")
print("=" * 75)
 
# Uniquement les méthodes avec un score continu significatif
CONTINUOUS = {
    "S1_Verbalized": s1,
    "S2_SelfConsistency": s2,
    "S3_LogProb": s3,
    "Fusion_Weighted": wscore,
    "Fusion_LogisticRegression": lr_proba,
}
 
wilcoxon_results = []
cnames = list(CONTINUOUS.keys())
for a, b in combinations(cnames, 2):
    xa, xb = CONTINUOUS[a], CONTINUOUS[b]
    diff = xa - xb
    if np.all(diff == 0):
        stat, p = np.nan, 1.0
    else:
        stat, p = wilcoxon(xa, xb, zero_method="wilcox")
    wilcoxon_results.append({
        "method_a": a, "method_b": b,
        "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
        "statistic": float(stat) if not np.isnan(stat) else None,
        "p_value": float(p)
    })
 
for r in wilcoxon_results:
    print(f"{r['method_a']:<28} vs {r['method_b']:<28} | "
          f"median={r['median_a']:.3f} vs {r['median_b']:.3f} | p={r['p_value']:.4f}")
 
# ============================================================
# 4. CORRECTION BENJAMINI-HOCHBERG (sur l'ensemble des tests)
# ============================================================
 
print("\n" + "=" * 75)
print("CORRECTION BENJAMINI-HOCHBERG (FDR)")
print("=" * 75)
 
all_pvals = [r["p_value"] for r in mcnemar_results] + [r["p_value"] for r in wilcoxon_results]
all_labels = (
    [f"McNemar: {r['method_a']} vs {r['method_b']}" for r in mcnemar_results] +
    [f"Wilcoxon: {r['method_a']} vs {r['method_b']}" for r in wilcoxon_results]
)
 
rejected, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
 
corrected_results = []
for label, p_raw, p_corr, rej in zip(all_labels, all_pvals, pvals_corrected, rejected):
    corrected_results.append({
        "comparison": label, "p_raw": float(p_raw),
        "p_corrected_bh": float(p_corr), "significant_at_0.05": bool(rej)
    })
    marker = "✅ significatif" if rej else "—"
    print(f"{label:<60} p_raw={p_raw:.4f}  p_BH={p_corr:.4f}  {marker}")
 
n_significant = sum(rejected)
print(f"\n{n_significant}/{len(all_pvals)} comparaisons significatives après correction BH (α=0.05)")
 
# ============================================================
# 5. SAUVEGARDE
# ============================================================
 
final = {
    "n_patients": n,
    "method_accuracies": {k: float(v.mean()) for k, v in correctness.items()},
    "mcnemar_pairwise": mcnemar_results,
    "wilcoxon_pairwise": wilcoxon_results,
    "benjamini_hochberg_corrected": corrected_results,
    "n_significant_after_correction": int(n_significant),
    "n_total_comparisons": len(all_pvals)
}
save_json(final, OUT_DIR / "phase8_statistical_validation.json")
 
print("\n" + "=" * 75)
print("PHASE 8 TERMINÉE")
print(f"Résultats dans : {OUT_DIR}")
print("=" * 75)
 
