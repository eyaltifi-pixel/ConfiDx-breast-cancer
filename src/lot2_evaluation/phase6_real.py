#!/usr/bin/env python3
"""
Phase 6 : Metriques Automatiques — MODE REEL
Utilise les vraies predictions du modele.
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).parent.parent.parent
PREDICTIONS_DIR = BASE_DIR / "predictions"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_predictions(task_num, split_name="test"):
    path = PREDICTIONS_DIR / f"task{task_num}_{split_name}_predictions.json"
    data = load_json(path)
    return {p["patient_id"]: p for p in data}

def load_references(task_num, split_name="test"):
    path = PROCESSED_DIR / split_name / f"task{task_num}_{split_name}.json"
    data = load_json(path)
    return {ex["patient_id"]: ex for ex in data}

def extract_prediction_text(pred):
    """Extrait le texte predit (premiere generation)."""
    gens = pred.get("generations", [])
    if not gens:
        return ""
    return gens[0].get("text", "").strip()

def extract_diagnosis(text):
    """Extrait le diagnostic du texte genere."""
    import re
    match = re.search(r"(Luminal A|Luminal B|HER2-enriched|Triple-negative)", text, re.I)
    return match.group(1) if match else text.split()[0] if text else "Unknown"

# ============================================================
# 3.8.1 DIAGNOSTIC ACCURACY (Task 1)
# ============================================================

def compute_diagnostic_accuracy(predictions, references):
    y_true, y_pred = [], []
    for pid, pred in predictions.items():
        ref = references.get(pid)
        if ref is None:
            continue
        true_label = ref.get("output", "").strip()
        pred_text = extract_prediction_text(pred)
        pred_label = extract_diagnosis(pred_text)
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    return acc, y_true, y_pred

# ============================================================
# 3.8.2 UNCERTAINTY RECOGNITION (Task 3)
# ============================================================

def compute_uncertainty_metrics(predictions, references):
    y_true, y_pred = [], []
    for pid, pred in predictions.items():
        ref = references.get(pid)
        if ref is None:
            continue
        true_label = ref.get("output", "").strip().lower()
        pred_text = extract_prediction_text(pred).lower()
        pred_label = "uncertain" if "uncertain" in pred_text else "confident"
        y_true.append(1 if true_label == "uncertain" else 0)
        y_pred.append(1 if pred_label == "uncertain" else 0)
    
    true_uncertain = [i for i, v in enumerate(y_true) if v == 1]
    correctly_recognized = sum(1 for i in true_uncertain if y_pred[i] == 1)
    accuracy_eu = round(correctly_recognized / len(true_uncertain) * 100, 2) if true_uncertain else 0
    
    f1 = round(f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)
    precision = round(precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)
    recall = round(recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)
    
    return {
        "accuracy_eu": accuracy_eu,
        "f1_eu": f1,
        "precision_eu": precision,
        "recall_eu": recall,
    }

# ============================================================
# 3.8.4 EXPECTED CALIBRATION ERROR (ECE)
# ============================================================

def compute_ece(confidences, accuracies, n_bins=10):
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        n_bin = np.sum(in_bin)
        if n_bin > 0:
            acc_bin = np.mean(accuracies[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += (n_bin / total) * abs(acc_bin - conf_bin)
    
    return round(float(ece), 4)

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 6 : METRIQUES AUTOMATIQUES — MODE REEL")
    print("=" * 60)
    
    report = {"mode": "REEL", "split": "test"}
    
    # Task 1: Diagnostic Accuracy
    print("\n--- Task 1 : Diagnostic Accuracy ---")
    preds1 = load_predictions(1, "test")
    refs1 = load_references(1, "test")
    acc1, y_true1, y_pred1 = compute_diagnostic_accuracy(preds1, refs1)
    print(f"  Accuracy : {acc1}%")
    report["diagnostic_accuracy_pct"] = acc1
    
    # Confidences simulees pour ECE (pas de vrai confidence dans predictions)
    # On utilise mean_logprob normalise comme proxy
    confidences1 = []
    correct1 = []
    for pid, pred in preds1.items():
        ref = refs1.get(pid)
        if ref is None:
            continue
        pred_text = extract_prediction_text(pred)
        pred_label = extract_diagnosis(pred_text)
        true_label = ref.get("output", "").strip()
        
        # Proxy confidence: base sur mean_logprob (moins negatif = plus confiant)
        logprobs = [g.get("mean_logprob", -1) for g in pred.get("generations", []) if g.get("mean_logprob", 0) > -10]
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1
        confidence = min(0.99, max(0.5, 1 + avg_logprob))  # normaliser [-1,0] -> [0,1]
        
        confidences1.append(confidence)
        correct1.append(1 if pred_label == true_label else 0)
    
    ece1 = compute_ece(confidences1, correct1)
    print(f"  ECE (Task 1) : {ece1}")
    report["ece_task1"] = ece1
    
    # Task 3: Uncertainty Recognition
    print("\n--- Task 3 : Uncertainty Recognition ---")
    preds3 = load_predictions(3, "test")
    refs3 = load_references(3, "test")
    eu_metrics = compute_uncertainty_metrics(preds3, refs3)
    print(f"  AccuracyEU : {eu_metrics['accuracy_eu']}%")
    print(f"  F1EU       : {eu_metrics['f1_eu']}%")
    print(f"  Precision  : {eu_metrics['precision_eu']}%")
    print(f"  Recall     : {eu_metrics['recall_eu']}%")
    report["uncertainty_recognition"] = eu_metrics
    
    # ECE Task 3
    confidences3 = []
    correct3 = []
    for pid, pred in preds3.items():
        ref = refs3.get(pid)
        if ref is None:
            continue
        pred_text = extract_prediction_text(pred).lower()
        pred_label = "uncertain" if "uncertain" in pred_text else "confident"
        true_label = ref.get("output", "").strip().lower()
        
        logprobs = [g.get("mean_logprob", -1) for g in pred.get("generations", []) if g.get("mean_logprob", 0) > -10]
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1
        confidence = min(0.99, max(0.5, 1 + avg_logprob))
        
        confidences3.append(confidence)
        correct3.append(1 if pred_label == true_label else 0)
    
    ece3 = compute_ece(confidences3, correct3)
    print(f"  ECE (Task 3) : {ece3}")
    report["ece_task3"] = ece3
    
    # Sauvegarde
    output_path = OUTPUT_DIR / "phase6_real_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nRapport sauvegarde : {output_path}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)

if __name__ == "__main__":
    main()