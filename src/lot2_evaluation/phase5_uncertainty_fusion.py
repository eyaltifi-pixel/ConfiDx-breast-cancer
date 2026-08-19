#!/usr/bin/env python3
"""
Phase 5 : Fusion d'Incertitude (3 signaux)
Objectif : combiner 3 signaux independants pour decider si un diagnostic
est "confident" ou "uncertain", au lieu de se fier uniquement au label
verbalise par le modele (Task 3), connu pour etre mal calibre.

Basé sur la Section 3.7 du document méthodologique (Phase 5).

Les 3 signaux :
  1. Verbalized confidence : label produit par Task 3 du modele (confident/uncertain)
  2. Self-consistency       : accord entre 5 generations a temperature 0.7
  3. Log-probability        : log-probabilite moyenne de la sequence generee

Regle de fusion : incertain si AU MOINS 2 signaux sur 3 l'indiquent.
Les seuils (consistency_thresh, logprob_thresh) sont calibres par grid
search sur le validation set, en maximisant l'AUROC contre le vrai label
d'incertitude (uncertainty_category present dans les metadata).

IMPORTANT - DEPENDANCE AU LOT 1 :
Les signaux 2 (self-consistency) et 3 (log-probability) necessitent le
modele fine-tune (Phase 3, Lot 1) pour generer plusieurs echantillons et
recuperer les log-probs. Tant que ce modele n'est pas disponible, ce
script fonctionne en mode DEMONSTRATION : il simule ces 2 signaux pour
valider que la logique de fusion et de calibration est correcte. Il
suffira de remplacer extract_self_consistency() et extract_logprob()
par de vrais appels au modele une fois la Phase 3 terminee.
"""

import json
import random
from pathlib import Path
from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. CHARGEMENT DES DONNEES (task3 + metadata)
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ground_truth_uncertainty(split_name):
    """
    Charge le vrai label d'incertitude (uncertainty_category) depuis
    task3_{split}.json. None = confident, "A" ou "B" = uncertain.
    """
    path = PROCESSED_DIR / split_name / f"task3_{split_name}.json"
    if not path.exists():
        print(f"  ATTENTION : {path} introuvable")
        return []

    data = load_json(path)
    result = []
    for ex in data:
        cat = ex.get("metadata", {}).get("uncertainty_category")
        verbalized = ex.get("output", "confident")
        result.append({
            "patient_id": ex.get("patient_id"),
            "true_uncertain": cat is not None,
            "verbalized_label": verbalized.strip().lower() if verbalized else "confident",
        })
    return result

# ============================================================
# 2. LES 3 SIGNAUX
# ============================================================

def signal_verbalized_confidence(verbalized_label):
    """Signal 1 : True si le modele a lui-meme dit 'uncertain'."""
    return verbalized_label == "uncertain"

def extract_self_consistency(patient_id, n_generations=5, temperature=0.7):
    """
    Signal 2 : taux d'ACCORD entre n_generations diagnostics generes a
    temperature > 0. Un taux d'accord FAIBLE indique une forte incertitude.

    !! PLACEHOLDER !! Necessite le modele fine-tune (Lot 1, Phase 3).
    """
    rng = random.Random(str(patient_id) + "_consistency")
    return round(rng.uniform(0.3, 1.0), 2)

def extract_logprob(patient_id):
    """
    Signal 3 : log-probabilite moyenne de la sequence generee.

    !! PLACEHOLDER !! Necessite le modele fine-tune (Lot 1, Phase 3).
    """
    rng = random.Random(str(patient_id) + "_logprob")
    return round(rng.uniform(-3.0, -0.2), 2)

# ============================================================
# 3. FUSION DES 3 SIGNAUX
# ============================================================

def fuse_signals(verbalized_uncertain, consistency_rate, avg_logprob,
                  consistency_thresh, logprob_thresh):
    """
    Combine les 3 signaux binaires. Retourne True (incertain) si au moins
    2 des 3 signaux l'indiquent.
    """
    sig1 = verbalized_uncertain
    sig2 = consistency_rate < consistency_thresh
    sig3 = avg_logprob < logprob_thresh

    n_votes_uncertain = sum([sig1, sig2, sig3])
    return n_votes_uncertain >= 2, {
        "signal1_verbalized": sig1,
        "signal2_low_consistency": sig2,
        "signal3_low_logprob": sig3,
        "n_votes_uncertain": n_votes_uncertain,
    }

# ============================================================
# 4. CALIBRATION DES SEUILS PAR GRID SEARCH (sur validation set)
# ============================================================

def calibrate_thresholds(val_records, consistency_grid=None, logprob_grid=None):
    """
    Cherche la combinaison de seuils qui maximise l'AUROC entre le score
    de fusion et le vrai label d'incertitude, sur le validation set.
    """
    if consistency_grid is None:
        consistency_grid = np.arange(0.3, 0.95, 0.05)
    if logprob_grid is None:
        logprob_grid = np.arange(-2.5, -0.3, 0.1)

    y_true = [r["true_uncertain"] for r in val_records]

    best_auroc = -1
    best_thresholds = None

    for c_thresh, l_thresh in product(consistency_grid, logprob_grid):
        y_score = []
        for r in val_records:
            fused, _ = fuse_signals(
                r["verbalized_uncertain"], r["consistency_rate"],
                r["avg_logprob"], c_thresh, l_thresh,
            )
            y_score.append(1.0 if fused else 0.0)

        if len(set(y_true)) < 2:
            continue

        auroc = roc_auc_score(y_true, y_score)
        if auroc > best_auroc:
            best_auroc = auroc
            best_thresholds = (round(float(c_thresh), 2), round(float(l_thresh), 2))

    return best_thresholds, best_auroc

# ============================================================
# 5. PIPELINE COMPLET SUR UN SPLIT
# ============================================================

def build_records_with_signals(split_name):
    """Charge les donnees et calcule les 3 signaux pour chaque patient."""
    gt = load_ground_truth_uncertainty(split_name)
    records = []
    for item in gt:
        pid = item["patient_id"]
        records.append({
            "patient_id": pid,
            "true_uncertain": item["true_uncertain"],
            "verbalized_uncertain": signal_verbalized_confidence(item["verbalized_label"]),
            "consistency_rate": extract_self_consistency(pid),
            "avg_logprob": extract_logprob(pid),
        })
    return records

def apply_fusion(records, consistency_thresh, logprob_thresh):
    """Applique la fusion calibree a une liste de records."""
    results = []
    for r in records:
        fused_uncertain, detail = fuse_signals(
            r["verbalized_uncertain"], r["consistency_rate"], r["avg_logprob"],
            consistency_thresh, logprob_thresh,
        )
        results.append({
            "patient_id": r["patient_id"],
            "true_uncertain": r["true_uncertain"],
            "fused_uncertain": fused_uncertain,
            "correct": fused_uncertain == r["true_uncertain"],
            **detail,
        })
    return results

# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 5 : FUSION D'INCERTITUDE (3 SIGNAUX)")
    print("=" * 60)
    print("\n!! MODE DEMONSTRATION !!")
    print("Les signaux self-consistency et log-probability sont SIMULES")
    print("en attendant le modele fine-tune (Lot 1, Phase 3).")
    print("La logique de fusion et de calibration, elle, est reelle et validee.\n")

    print("--- Chargement + calcul des signaux (VAL) ---")
    val_records = build_records_with_signals("val")
    print(f"  {len(val_records)} patients (val)")

    print("\n--- Calibration des seuils (grid search, maximise AUROC) ---")
    best_thresholds, best_auroc = calibrate_thresholds(val_records)
    if best_thresholds is None:
        print("  ERREUR : impossible de calibrer (une seule classe presente dans val ?)")
        return
    c_thresh, l_thresh = best_thresholds
    print(f"  Meilleur consistency_thresh : {c_thresh}")
    print(f"  Meilleur logprob_thresh     : {l_thresh}")
    print(f"  AUROC obtenu (val)          : {round(best_auroc, 3)}")

    print("\n--- Application sur TEST avec les seuils calibres ---")
    test_records = build_records_with_signals("test")
    test_results = apply_fusion(test_records, c_thresh, l_thresh)

    n_total = len(test_results)
    n_correct = sum(1 for r in test_results if r["correct"])
    accuracy = round(n_correct / n_total * 100, 2) if n_total else 0.0

    print(f"  {n_total} patients (test)")
    print(f"  Accuracy fusion vs vrai label : {accuracy}%")

    output_path = OUTPUT_DIR / "phase5_uncertainty_fusion_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "DEMONSTRATION (signaux 2 et 3 simules, en attente Lot1 Phase3)",
            "calibration": {
                "consistency_thresh": c_thresh,
                "logprob_thresh": l_thresh,
                "auroc_val": round(best_auroc, 3),
            },
            "test_accuracy_pct": accuracy,
            "test_details": test_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nRapport sauvegarde : {output_path}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()