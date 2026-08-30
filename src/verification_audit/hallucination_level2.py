#!/usr/bin/env python3
"""
Reconstruction — Détection hallucinations Niveau 2 (Fact-Conflicting).
Étape 1 : extraire ER/PR/HER2/Ki-67 depuis le champ narratif "input" (data/processed/test/task1_test.json)
Étape 2 : valider l'extraction en réappliquant l'Algorithme 1 (Lot1, Section 3.3.3) et en comparant
          au diagnostic de référence "output" (doit concorder à ~100%, car assigné par la même règle)
Étape 3 : comparer le diagnostic recalculé au diagnostic PRÉDIT par le modèle (task1_extraction_RECALCULE.json)
          -> taux d'hallucination Niveau 2 (proxy)
"""

import json
import re
from pathlib import Path

BASE = Path("/home/claude/repo_build/ConfiDx-breast-cancer")
test_task1 = json.load(open(BASE / "data/processed/test/task1_test.json"))
model_preds = json.load(open("/mnt/user-data/outputs/task1_extraction_RECALCULE.json"))
pred_by_id = {r["patient_id"]: r["predicted"] for r in model_preds["patient_level_results"]}

def find_status(text, keywords):
    sentences = re.split(r"(?<=[.])\s+", text)
    for sent in sentences:
        low = sent.lower()
        if any(k in low for k in keywords):
            if "equivocal" in low:
                return "Equivocal"
            if "unknown" in low:
                return "Unknown"
            if "positive" in low:
                return "Positive"
            if "negative" in low:
                return "Negative"
    return None

def find_ki67(text):
    # Cherche un pourcentage à proximité du mot Ki-67
    m = re.search(r"ki-?67[^%\d]{0,40}?(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of tumor cells were\s*(?:approximately\s*)?ki-?67", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Cas générique : "X% ... Ki-67" dans la même phrase
    sentences = re.split(r"(?<=[.])\s+", text)
    for sent in sentences:
        if "ki-67" in sent.lower() or "ki67" in sent.lower():
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", sent)
            if m:
                return float(m.group(1))
    return None

def assign_diagnosis(er, pr, her2, ki67):
    """Algorithme 1, Lot1 Section 3.3.3."""
    if her2 == "Positive":
        return "HER2-enriched"
    if her2 == "Equivocal":
        return "Luminal B"
    if er == "Positive" or pr == "Positive":
        if pr == "Positive" and ki67 is not None and ki67 < 20:
            return "Luminal A"
        else:
            return "Luminal B"
    if er == "Negative" and pr == "Negative" and her2 in ("Negative", "Unknown", None):
        return "Triple-negative"
    return None  # cas non couvert (ex: ER/PR Unknown) -> pas de recalcul possible

results = []
n_extracted_all = 0
n_matches_reference = 0
n_recomputable = 0

for ex in test_task1:
    text = ex["input"]
    er = find_status(text, ["er status", "estrogen receptor", "er immunohistochemistry", "er ihc"])
    pr = find_status(text, ["pr status", "progesterone receptor", "pr immunohistochemistry"])
    her2 = find_status(text, ["her2"])
    ki67 = find_ki67(text)

    recomputed = assign_diagnosis(er, pr, her2, ki67)
    reference = ex["output"]
    patient_id = ex["patient_id"]
    predicted_by_model = pred_by_id.get(patient_id)

    if recomputed is not None:
        n_recomputable += 1
        if recomputed == reference:
            n_matches_reference += 1

    results.append({
        "patient_id": patient_id, "ER": er, "PR": pr, "HER2": her2, "Ki67": ki67,
        "recomputed_diagnosis": recomputed, "reference_diagnosis": reference,
        "model_predicted_diagnosis": predicted_by_model,
    })

n_total = len(results)
print("=" * 70)
print("VALIDATION DE L'EXTRACTION (recalcul vs référence)")
print("=" * 70)
print(f"Patients total                          : {n_total}")
print(f"Diagnostic recalculable (ER/PR/HER2 ok)  : {n_recomputable} ({100*n_recomputable/n_total:.1f}%)")
print(f"Recalcul == référence (parmi recalculable): {n_matches_reference}/{n_recomputable} "
      f"({100*n_matches_reference/n_recomputable:.1f}%)" if n_recomputable else "N/A")

# Taux d'hallucination Niveau 2 : recalculé vs prédiction du MODÈLE (pas la référence)
comparable = [r for r in results if r["recomputed_diagnosis"] is not None and r["model_predicted_diagnosis"] is not None]
n_comparable = len(comparable)
n_mismatch = sum(1 for r in comparable if r["recomputed_diagnosis"] != r["model_predicted_diagnosis"])
rate_level2 = n_mismatch / n_comparable if n_comparable else None

print()
print("=" * 70)
print("NIVEAU 2 — FACT-CONFLICTING (proxy NCCN/ESMO)")
print("=" * 70)
print(f"Générations comparables (recalcul + prédiction dispo) : {n_comparable}/{n_total} ({100*n_comparable/n_total:.1f}%)")
print(f"Désaccords (hallucination proxy)                      : {n_mismatch}")
print(f"Taux Niveau 2 (sur comparables)                        : {rate_level2:.4f}" if rate_level2 is not None else "N/A")
print()
print("--- Comparaison avec le papier (Tableau 9, Niveau 2 = 25.5%) ---")

out = {
    "n_total": n_total, "n_recomputable": n_recomputable, "n_matches_reference": n_matches_reference,
    "n_comparable_level2": n_comparable, "n_mismatch_level2": n_mismatch,
    "rate_level2": rate_level2,
    "patient_level_results": results,
}
out_path = Path("/mnt/user-data/outputs/hallucination_level2_RECALCULE.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"💾 Sauvegardé : {out_path}")
