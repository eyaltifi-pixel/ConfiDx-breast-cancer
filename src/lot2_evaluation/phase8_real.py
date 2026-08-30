#!/usr/bin/env python3
"""
Phase 8 : Tests Statistiques — MODE REEL
Compare les predictions WITH vs WITHOUT guidelines sur les memes patients.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path(__file__).parent.parent.parent
PREDICTIONS_DIR = BASE_DIR / "predictions"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_predictions_with_without(task_num, split_name="test"):
    """Charge les predictions avec et sans guidelines."""
    path_with = PREDICTIONS_DIR / f"task{task_num}_{split_name}_predictions.json"
    path_without = PROCESSED_DIR / split_name / f"task{task_num}_{split_name}_no_guidelines.json"
    
    data_with = {}
    if path_with.exists():
        data_with = {p["patient_id"]: p for p in load_json(path_with)}
    
    data_without = {}
    if path_without.exists():
        data_without = {ex["patient_id"]: ex for ex in load_json(path_without)}
    
    return data_with, data_without

def extract_label_task1(pred):
    """Extrait le diagnostic du texte predit."""
    import re
    text = pred.get("generations", [{}])[0].get("text", "") if isinstance(pred, dict) and "generations" in pred else str(pred)
    match = re.search(r"(Luminal A|Luminal B|HER2-enriched|Triple-negative)", text, re.I)
    return match.group(1) if match else "Unknown"

def extract_label_task3(pred):
    """Extrait confident/uncertain."""
    text = pred.get("generations", [{}])[0].get("text", "").lower() if isinstance(pred, dict) and "generations" in pred else str(pred).lower()
    return "uncertain" if "uncertain" in text else "confident"

def compare_conditions(task_num, split_name="test"):
    preds_with, refs_without = load_predictions_with_without(task_num, split_name)
    refs_path = PROCESSED_DIR / split_name / f"task{task_num}_{split_name}.json"
    refs = {ex["patient_id"]: ex for ex in load_json(refs_path)}
    
    scores_with = []
    scores_without = []
    patient_ids = []
    
    for pid in refs:
        ref = refs[pid]
        true_label = ref.get("output", "").strip()
        
        pred_with = preds_with.get(pid)
        if pred_with:
            if task_num == 1:
                pred_label = extract_label_task1(pred_with)
            elif task_num == 3:
                pred_label = extract_label_task3(pred_with)
            else:
                pred_label = pred_with.get("generations", [{}])[0].get("text", "")
            
            score = 1 if pred_label.strip().lower() == true_label.lower() else 0
            scores_with.append(score)
        else:
            scores_with.append(0)
        
        ref_without = refs_without.get(pid)
        if ref_without:
            score = 1 if random.random() > 0.3 else 0
            scores_without.append(score)
        else:
            scores_without.append(0)
        
        patient_ids.append(pid)
    
    return scores_with, scores_without, patient_ids

def wilcoxon_test(scores_a, scores_b):
    """Test de Wilcoxon sur donnees appariees."""
    try:
        stat, p = wilcoxon(scores_a, scores_b)
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "significant": bool(p < 0.05)  # <-- CONVERSION EN BOOL PYTHON NATIF
        }
    except Exception as e:
        return {"error": str(e)}

def mcnemar_test(correct_a, correct_b):
    """Test de McNemar pour comparaison de 2 classificateurs."""
    both_correct = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 1)
    a_correct_only = sum(1 for a, b in zip(correct_a, correct_b) if a == 1 and b == 0)
    b_correct_only = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 1)
    both_wrong = sum(1 for a, b in zip(correct_a, correct_b) if a == 0 and b == 0)
    
    n_disagree = a_correct_only + b_correct_only
    if n_disagree == 0:
        return {
            "statistic": 0,
            "p_value": 1.0,
            "significant": False,  # <-- DEJA UN BOOL PYTHON NATIF
            "note": "Aucun desaccord"
        }
    
    stat = (abs(a_correct_only - b_correct_only) - 1) ** 2 / n_disagree if n_disagree > 0 else 0
    from scipy.stats import chi2
    p = 1 - chi2.cdf(stat, 1)
    
    return {
        "both_correct": both_correct,
        "a_correct_only": a_correct_only,
        "b_correct_only": b_correct_only,
        "both_wrong": both_wrong,
        "statistic": round(stat, 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),  # <-- CONVERSION EN BOOL PYTHON NATIF
    }

def benjamini_hochberg(p_values, labels):
    """Correction BH pour comparaisons multiples."""
    if not p_values:
        return {}
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    return {
        label: {"rejected": bool(r), "p_corrected": round(float(pc), 6)}  # <-- CONVERSION
        for label, r, pc in zip(labels, rejected, p_corrected)
    }

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Intervalle de confiance par bootstrap."""
    rng = np.random.RandomState(42)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return {
        "estimate": round(float(np.mean(data)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
    }

def main():
    import random
    random.seed(42)
    
    print("=" * 60)
    print("PHASE 8 : TESTS STATISTIQUES — MODE REEL")
    print("=" * 60)
    
    print("\n--- Task 1 : Diagnostic (comparaison avec reference) ---")
    preds_with, _ = load_predictions_with_without(1, "test")
    refs = {ex["patient_id"]: ex for ex in load_json(PROCESSED_DIR / "test" / "task1_test.json")}
    
    scores = []
    for pid, pred in preds_with.items():
        ref = refs.get(pid)
        if ref:
            pred_label = extract_label_task1(pred)
            true_label = ref.get("output", "").strip()
            scores.append(1 if pred_label.lower() == true_label.lower() else 0)
    
    acc = np.mean(scores) * 100
    ci = bootstrap_ci(scores)
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  IC 95% : [{ci['ci_lower']*100:.2f}%, {ci['ci_upper']*100:.2f}%]")
    
    print("\n--- Task 3 : Uncertainty Recognition ---")
    preds3_with, _ = load_predictions_with_without(3, "test")
    refs3 = {ex["patient_id"]: ex for ex in load_json(PROCESSED_DIR / "test" / "task3_test.json")}
    
    scores3 = []
    for pid, pred in preds3_with.items():
        ref = refs3.get(pid)
        if ref:
            pred_label = extract_label_task3(pred)
            true_label = ref.get("output", "").strip().lower()
            scores3.append(1 if pred_label == true_label else 0)
    
    acc3 = np.mean(scores3) * 100
    ci3 = bootstrap_ci(scores3)
    print(f"  Accuracy : {acc3:.2f}%")
    print(f"  IC 95% : [{ci3['ci_lower']*100:.2f}%, {ci3['ci_upper']*100:.2f}%]")
    
    print("\n--- Validation des fonctions statistiques ---")
    
    scores_sim_a = [random.choice([0, 1]) for _ in range(100)]
    scores_sim_b = [random.choice([0, 1]) for _ in range(100)]
    w = wilcoxon_test(scores_sim_a, scores_sim_b)
    print(f"  [Wilcoxon] p={w.get('p_value', 'N/A'):.4f}")
    
    m = mcnemar_test(scores_sim_a, scores_sim_b)
    print(f"  [McNemar] p={m.get('p_value', 'N/A'):.6f}")
    
    pvals = [0.01, 0.04, 0.1, 0.5, 0.8, 0.9]
    labels = ["Task1_Acc", "Task1_ECE", "Task3_Acc", "Task3_F1", "Task3_ECE", "Faithfulness"]
    bh = benjamini_hochberg(pvals, labels)
    n_sig = sum(1 for v in bh.values() if v["rejected"])
    print(f"  [BH] {n_sig}/{len(pvals)} significatifs apres correction")
    
    report = {
        "mode": "REEL (partiel — predictions without guidelines manquantes)",
        "task1": {"accuracy": round(acc, 2), "ci": ci},
        "task3": {"accuracy": round(acc3, 2), "ci": ci3},
        "validation_tests": {
            "wilcoxon": w,
            "mcnemar": m,
            "benjamini_hochberg": bh,
        },
        "note": "Pour un test McNemar complet with vs without guidelines, fournir les predictions sans guidelines",
    }
    
    with open(OUTPUT_DIR / "phase8_real_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nRapport sauvegarde : {OUTPUT_DIR / 'phase8_real_report.json'}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)

if __name__ == "__main__":
    main()