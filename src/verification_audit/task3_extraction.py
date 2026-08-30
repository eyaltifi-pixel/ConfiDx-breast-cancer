#!/usr/bin/env python3
"""
Reconstruction — Task 3 : extraction confident/uncertain + AccuracyEU/F1EU.
Convention conservative documentée dans le rapport Lot2 (Section 4.4.1) :
un patient sans label extractable est compté "confident" par défaut.
"""

import json
import re
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

BASE = Path("/home/claude/confidx_extract/CONFIDX")
data = json.load(open(BASE / "predictions/task3_test_predictions.json"))

def extract_label(text):
    t = text.lower()
    has_uncertain = re.search(r"\buncertain\b", t) is not None
    has_confident = re.search(r"\bconfident\b", t) is not None
    if has_uncertain and not has_confident:
        return "uncertain"
    if has_confident and not has_uncertain:
        return "confident"
    return None  # ambigu ou absent

results = []
for patient in data:
    votes = [extract_label(g["text"]) for g in patient["generations"]]
    votes = [v for v in votes if v is not None]
    if votes:
        # majorité simple ; défaut confident en cas d'égalité stricte
        pred = "uncertain" if votes.count("uncertain") > votes.count("confident") else "confident"
        resolved = True
    else:
        pred = "confident"  # convention conservative
        resolved = False
    results.append({
        "patient_id": patient["patient_id"],
        "reference": patient["reference"],
        "predicted": pred,
        "resolved": resolved,
    })

n_total = len(results)
n_resolved = sum(r["resolved"] for r in results)
y_true = [r["reference"] for r in results]
y_pred = [r["predicted"] for r in results]

acc_conservative = accuracy_score(y_true, y_pred)
f1_conservative = f1_score(y_true, y_pred, pos_label="uncertain")

resolved_only = [r for r in results if r["resolved"]]
if resolved_only:
    acc_resolved = accuracy_score([r["reference"] for r in resolved_only], [r["predicted"] for r in resolved_only])
    f1_resolved = f1_score([r["reference"] for r in resolved_only], [r["predicted"] for r in resolved_only], pos_label="uncertain")
else:
    acc_resolved, f1_resolved = None, None

print("=" * 70)
print("RECONSTRUCTION TASK 3 — Uncertainty Recognition")
print("=" * 70)
print(f"Patients total                : {n_total}")
print(f"Résolus (label extractable)   : {n_resolved} ({100*n_resolved/n_total:.1f}%)")
print(f"AccuracyEU (conservative)     : {acc_conservative:.4f}")
print(f"F1EU (conservative)           : {f1_conservative:.4f}")
if resolved_only:
    print(f"AccuracyEU (résolus only)     : {acc_resolved:.4f}")
    print(f"F1EU (résolus only)           : {f1_resolved:.4f}")

print()
print("--- Comparaison avec le papier (Tableau 8 / Lot2 Tableau 9) ---")
print("Papier   : Résolus=22.8% | AccuracyEU=0.802 | F1EU=0.053")
print(f"Recalculé: Résolus={100*n_resolved/n_total:.1f}% | AccuracyEU={acc_conservative:.3f} | F1EU={f1_conservative:.3f}")

out = {
    "n_total": n_total, "n_resolved": n_resolved,
    "resolution_rate": n_resolved / n_total,
    "accuracy_eu_conservative": acc_conservative,
    "f1_eu_conservative": f1_conservative,
    "accuracy_eu_resolved_only": acc_resolved,
    "f1_eu_resolved_only": f1_resolved,
    "patient_level_results": results,
}
out_path = Path("/mnt/user-data/outputs/task3_extraction_RECALCULE.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n💾 Sauvegardé : {out_path}")
