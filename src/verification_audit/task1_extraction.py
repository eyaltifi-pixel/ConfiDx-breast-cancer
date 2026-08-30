#!/usr/bin/env python3
"""
Reconstruction — Task 1 : extraction du sous-type depuis les générations + métriques.
Implémente les mêmes règles que celles décrites dans le rapport Lot2 (Section 2.3.2) :
recherche directe des 4 sous-types, gestion du markdown, acronymes, vote majoritaire sur 5 générations.
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score

BASE = Path("/home/claude/confidx_extract/CONFIDX")
data = json.load(open(BASE / "predictions/task1_test_predictions.json"))

SUBTYPES = ["Luminal A", "Luminal B", "HER2-enriched", "Triple-negative"]

# Patterns ordonnés du plus spécifique au plus général, insensibles à la casse et au markdown (**...**)
PATTERNS = [
    r"\*{0,2}luminal\s*a\*{0,2}",
    r"\*{0,2}luminal\s*b\*{0,2}",
    r"\*{0,2}her2[\s\-]?(enriched|positive|\+)\*{0,2}",
    r"\*{0,2}triple[\s\-]?neg(ative)?\*{0,2}|\btn\b",
]
LABELS = ["Luminal A", "Luminal B", "HER2-enriched", "Triple-negative"]

def extract_label(text):
    """Retourne le sous-type détecté dans un texte généré, ou None si aucun match."""
    t = text.lower()
    for pat, label in zip(PATTERNS, LABELS):
        if re.search(pat, t):
            return label
    return None

def extract_patient_label(generations):
    """Vote majoritaire sur les 5 générations ; None si aucune génération n'est parseable."""
    votes = [extract_label(g["text"]) for g in generations]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None, 0
    counts = Counter(votes)
    top_label, top_count = counts.most_common(1)[0]
    return top_label, len(votes)

results = []
for patient in data:
    pred, n_resolved = extract_patient_label(patient["generations"])
    results.append({
        "patient_id": patient["patient_id"],
        "reference": patient["reference"],
        "predicted": pred,
        "n_resolved_generations": n_resolved,
        "correct": (pred == patient["reference"]) if pred is not None else False,
    })

n_total = len(results)
n_resolved = sum(1 for r in results if r["predicted"] is not None)
n_unresolved = n_total - n_resolved

# Accuracy conservative : non résolus comptés comme faux
acc_conservative = sum(r["correct"] for r in results) / n_total

# Accuracy résolus uniquement
resolved = [r for r in results if r["predicted"] is not None]
acc_resolved_only = sum(r["correct"] for r in resolved) / len(resolved) if resolved else 0.0

y_true_resolved = [r["reference"] for r in resolved]
y_pred_resolved = [r["predicted"] for r in resolved]
macro_f1 = f1_score(y_true_resolved, y_pred_resolved, labels=SUBTYPES, average="macro")
per_subtype_f1 = f1_score(y_true_resolved, y_pred_resolved, labels=SUBTYPES, average=None)

print("=" * 70)
print("RECONSTRUCTION TASK 1 — Diagnostic Accuracy")
print("=" * 70)
print(f"Patients total                 : {n_total}")
print(f"Résolus (≥1 génération parsée)  : {n_resolved} ({100*n_resolved/n_total:.1f}%)")
print(f"Non résolus                     : {n_unresolved} ({100*n_unresolved/n_total:.1f}%)")
print(f"Accuracy (conservative)         : {acc_conservative:.4f}")
print(f"Accuracy (résolus uniquement)   : {acc_resolved_only:.4f}")
print(f"Macro F1 (résolus)              : {macro_f1:.4f}")
print()
print("F1 par sous-type :")
for label, f1 in zip(SUBTYPES, per_subtype_f1):
    print(f"  {label:<18} {f1:.4f}")

print()
print("--- Comparaison avec le papier (Tableau 8) ---")
print(f"Papier   : Accuracy conservative = 0.727 | résolus only = 0.769 | Macro F1 = 0.691")
print(f"Recalculé: Accuracy conservative = {acc_conservative:.3f} | résolus only = {acc_resolved_only:.3f} | Macro F1 = {macro_f1:.3f}")

out = {
    "n_total": n_total, "n_resolved": n_resolved, "n_unresolved": n_unresolved,
    "accuracy_conservative": acc_conservative, "accuracy_resolved_only": acc_resolved_only,
    "macro_f1_resolved": float(macro_f1),
    "per_subtype_f1": {label: float(f1) for label, f1 in zip(SUBTYPES, per_subtype_f1)},
    "patient_level_results": results,
}
out_path = Path("/mnt/user-data/outputs/task1_extraction_RECALCULE.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n💾 Sauvegardé : {out_path}")
