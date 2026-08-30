#!/usr/bin/env python3
"""
Phase 5 : Fusion d'Incertitude (3 signaux) — MODE REEL
Utilise les vraies predictions du modele (Lot 1, Phase 3)

Signaux :
1. Verbalized confidence : output de Task 3 (confident/uncertain)
2. Self-consistency : taux d'accord entre 5 generations (temperature 0.7)
3. Log-probability : mean_logprob moyen des generations

Regle de fusion : incertain si AU MOINS 2 signaux sur 3 l'indiquent.
Seuils calibres par grid search sur le validation set (maximise AUROC).
"""

import json
import random
from pathlib import Path
from itertools import product
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
PREDICTIONS_DIR = BASE_DIR / "predictions"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_predictions(task_num, split_name="test"):
    """Charge les predictions du modele pour une tache."""
    path = PREDICTIONS_DIR / f"task{task_num}_{split_name}_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Predictions introuvables: {path}")
    data = load_json(path)
    return {p["patient_id"]: p for p in data}

def load_ground_truth(split_name="test"):
    """Charge le vrai label d'incertitude depuis task3 (with_guidelines)."""
    path = PROCESSED_DIR / split_name / f"task3_{split_name}.json"
    data = load_json(path)
    result = {}
    for ex in data:
        pid = ex.get("patient_id")
        cat = ex.get("metadata", {}).get("uncertainty_category")
        result[pid] = {
            "true_uncertain": cat is not None,  # None = confident, "A"/"B" = uncertain
            "verbalized_label": ex.get("output", "confident").strip().lower(),
        }
    return result

# ============================================================
# 2. LES 3 SIGNAUX (REELS)
# ============================================================

def signal_verbalized_confidence(verbalized_label):
    """Signal 1: True si le modele a dit 'uncertain'."""
    return verbalized_label == "uncertain"

def signal_self_consistency(patient_id, generations):
    """
    Signal 2: taux d'accord entre les N generations.
    On compare les textes (apres nettoyage) pour voir combien sont identiques.
    Un taux d'accord FAIBLE = forte incertitude.
    """
    texts = []
    for g in generations:
        text = g.get("text", "").strip().lower()
        # Nettoyage: garder juste le diagnostic principal
        text = text.split("\n")[0][:100]  # premiere ligne, 100 chars max
        texts.append(text)
    
    if len(texts) < 2:
        return 1.0  # pas assez de generations pour mesurer
    
    # Comparer chaque paire
    agreements = 0
    comparisons = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # Similarite simple: memes mots cles de diagnostic
            comparisons += 1
            if texts[i] == texts[j]:
                agreements += 1
    
    return agreements / comparisons if comparisons > 0 else 1.0

def signal_log_probability(generations):
    """
    Signal 3: mean_logprob moyen des generations.
    ATTENTION: gerer les -Infinity (generations tronquees).
    """
    logprobs = []
    for g in generations:
        lp = g.get("mean_logprob", 0)
        if lp == float('-inf') or lp == -float('inf'):
            continue  # ignorer les generations tronquees
        logprobs.append(lp)
    
    if not logprobs:
        return -3.0  # valeur par defaut si toutes tronquees
    
    return sum(logprobs) / len(logprobs)

# ============================================================
# 3. FUSION DES 3 SIGNAUX
# ============================================================

def fuse_signals(verbalized_uncertain, consistency_rate, avg_logprob, 
                  consistency_thresh, logprob_thresh):
    """
    Combine les 3 signaux binaires.
    Retourne True (incertain) si au moins 2 des 3 signaux l'indiquent.
    """
    sig1 = verbalized_uncertain
    sig2 = consistency_rate < consistency_thresh  # faible accord = incertain
    sig3 = avg_logprob < logprob_thresh  # logprob faible = incertain
    
    n_votes_uncertain = sum([sig1, sig2, sig3])
    return n_votes_uncertain >= 2, {
        "signal1_verbalized": sig1,
        "signal2_low_consistency": sig2,
        "signal3_low_logprob": sig3,
        "n_votes_uncertain": n_votes_uncertain,
        "consistency_rate": round(consistency_rate, 3),
        "avg_logprob": round(avg_logprob, 4),
    }

# ============================================================
# 4. CALIBRATION DES SEUILS (grid search sur validation set)
# ============================================================

def calibrate_thresholds(val_records):
    """
    Cherche la combinaison de seuils qui maximise l'AUROC.
    """
    consistency_grid = np.arange(0.2, 0.95, 0.05)
    logprob_grid = np.arange(-2.5, -0.2, 0.1)
    
    y_true = [r["true_uncertain"] for r in val_records]
    
    if len(set(y_true)) < 2:
        print("  ATTENTION: une seule classe dans val, calibration impossible")
        return (0.6, -1.5), None  # seuils par defaut
    
    best_auroc = -1
    best_thresholds = None
    
    for c_thresh, l_thresh in product(consistency_grid, logprob_grid):
        y_score = []
        for r in val_records:
            fused, _ = fuse_signals(
                r["verbalized_uncertain"],
                r["consistency_rate"],
                r["avg_logprob"],
                c_thresh, l_thresh
            )
            y_score.append(1.0 if fused else 0.0)
        
        try:
            auroc = roc_auc_score(y_true, y_score)
            if auroc > best_auroc:
                best_auroc = auroc
                best_thresholds = (round(float(c_thresh), 2), round(float(l_thresh), 2))
        except ValueError:
            continue  # une seule classe predite avec ces seuils
    
    return best_thresholds, best_auroc

# ============================================================
# 5. CONSTRUCTION DES RECORDS AVEC SIGNAUX REELS
# ============================================================

def build_records(task3_preds, task1_preds, ground_truth, split_name):
    """
    Construit les records avec les 3 vrais signaux pour chaque patient.
    """
    records = []
    for pid, gt in ground_truth.items():
        pred3 = task3_preds.get(pid)
        pred1 = task1_preds.get(pid)
        
        if pred3 is None or pred1 is None:
            continue
        
        generations = pred1.get("generations", [])
        
        records.append({
            "patient_id": pid,
            "true_uncertain": gt["true_uncertain"],
            "verbalized_uncertain": signal_verbalized_confidence(gt["verbalized_label"]),
            "consistency_rate": signal_self_consistency(pid, generations),
            "avg_logprob": signal_log_probability(generations),
        })
    
    return records

def apply_fusion(records, consistency_thresh, logprob_thresh):
    """Applique la fusion calibree."""
    results = []
    for r in records:
        fused_uncertain, detail = fuse_signals(
            r["verbalized_uncertain"],
            r["consistency_rate"],
            r["avg_logprob"],
            consistency_thresh, logprob_thresh
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
    print("PHASE 5 : FUSION D'INCERTITUDE (3 SIGNAUX) — MODE REEL")
    print("=" * 60)
    
    # Charger predictions Task 1 (pour generations + logprobs) et Task 3 (pour verbalized)
    print("\n--- Chargement des predictions ---")
    task1_preds = load_predictions(1, "test")
    task3_preds = load_predictions(3, "test")
    
    # Charger ground truth (val pour calibration, test pour evaluation)
    print("--- Chargement ground truth (VAL) ---")
    gt_val = load_ground_truth("val")
    val_records = build_records(task3_preds, task1_preds, gt_val, "val")
    print(f"  {len(val_records)} patients (val)")
    
    print("\n--- Calibration des seuils ---")
    best_thresholds, best_auroc = calibrate_thresholds(val_records)
    c_thresh, l_thresh = best_thresholds
    print(f"  Meilleur consistency_thresh : {c_thresh}")
    print(f"  Meilleur logprob_thresh : {l_thresh}")
    if best_auroc:
        print(f"  AUROC obtenu (val) : {round(best_auroc, 3)}")
    
    # Application sur TEST
    print("\n--- Application sur TEST ---")
    gt_test = load_ground_truth("test")
    test_records = build_records(task3_preds, task1_preds, gt_test, "test")
    test_results = apply_fusion(test_records, c_thresh, l_thresh)
    
    n_total = len(test_results)
    n_correct = sum(1 for r in test_results if r["correct"])
    accuracy = round(n_correct / n_total * 100, 2) if n_total else 0
    
    # Detail par vraie classe
    true_positives = sum(1 for r in test_results if r["true_uncertain"] and r["fused_uncertain"])
    true_negatives = sum(1 for r in test_results if not r["true_uncertain"] and not r["fused_uncertain"])
    false_positives = sum(1 for r in test_results if not r["true_uncertain"] and r["fused_uncertain"])
    false_negatives = sum(1 for r in test_results if r["true_uncertain"] and not r["fused_uncertain"])
    
    print(f"\n  {n_total} patients (test)")
    print(f"  Accuracy fusion : {accuracy}%")
    print(f"  TP (incertitude detectee correctement) : {true_positives}")
    print(f"  TN (confiance detectee correctement) : {true_negatives}")
    print(f"  FP (fausse alerte) : {false_positives}")
    print(f"  FN (incertitude manquee) : {false_negatives}")
    
    # Sauvegarde
    output_path = OUTPUT_DIR / "phase5_real_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "REEL (predictions du modele Llama-3.1-8B)",
            "calibration": {
                "consistency_thresh": c_thresh,
                "logprob_thresh": l_thresh,
                "auroc_val": round(best_auroc, 3) if best_auroc else None,
            },
            "test_results": {
                "n_total": n_total,
                "accuracy_pct": accuracy,
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
            "details": test_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nRapport sauvegarde : {output_path}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)

if __name__ == "__main__":
    main()