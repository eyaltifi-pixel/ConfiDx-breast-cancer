#!/usr/bin/env python3
"""
Phase 6 : Metriques Automatiques d'Evaluation
Objectif : calculer les metriques quantitatives sur les 4 sous-taches.

Basé sur la Section 3.8 du document méthodologique (Phase 6).

Métriques implémentées :
  - 3.8.1 Diagnostic Accuracy (Task 1)
  - 3.8.2 Uncertainty Recognition : AccuracyEU, F1EU (Task 3)
  - 3.8.3 Explanation Faithfulness : BERTScore, SentenceBERT, METEOR (Task 2)
  - 3.8.4 Expected Calibration Error (ECE), pour Task 1 et Task 3

IMPORTANT - DEPENDANCE AU LOT 1 :
Toutes les metriques ci-dessous comparent une PREDICTION du modele a la
reference (ground truth). Tant que le modele fine-tune (Lot 1, Phase 3)
n'a pas produit de vraies predictions, ce script fonctionne en mode
DEMONSTRATION : il simule des predictions imparfaites (bruitees a partir
de la reference) pour valider que le calcul de chaque metrique est
correct. Il suffira de remplacer simulate_predictions() par le chargement
des vraies predictions exportees par le Lot 1 (Section "Inference & Export").
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

SIMULATED_ERROR_RATE = 0.15

# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_task(split_name, task_num):
    path = PROCESSED_DIR / split_name / f"task{task_num}_{split_name}.json"
    if not path.exists():
        print(f"  ATTENTION : {path} introuvable")
        return []
    return load_json(path)

# ============================================================
# 2. SIMULATION DE PREDICTIONS (MODE DEMONSTRATION UNIQUEMENT)
# ============================================================

def simulate_predictions(data, label_field="output", possible_labels=None,
                          error_rate=SIMULATED_ERROR_RATE):
    """
    !! PLACEHOLDER !! Simule des predictions imparfaites a partir de la
    reference, pour valider le calcul des metriques avant que le Lot 1
    ne fournisse les vraies predictions.
    """
    if possible_labels is None:
        possible_labels = sorted(set(ex[label_field].strip() for ex in data))

    predictions = []
    for ex in data:
        true_label = ex[label_field].strip()
        rng = random.Random(str(ex.get("patient_id")) + "_pred")
        if rng.random() < error_rate and len(possible_labels) > 1:
            other_labels = [l for l in possible_labels if l != true_label]
            pred_label = rng.choice(other_labels)
        else:
            pred_label = true_label
        confidence = rng.uniform(0.75, 0.99) if pred_label == true_label \
            else rng.uniform(0.4, 0.85)
        predictions.append({
            "patient_id": ex.get("patient_id"),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": round(confidence, 3),
        })
    return predictions

# ============================================================
# 3.8.1 DIAGNOSTIC ACCURACY (Task 1)
# ============================================================

def compute_diagnostic_accuracy(predictions):
    y_true = [p["true_label"] for p in predictions]
    y_pred = [p["pred_label"] for p in predictions]
    return round(accuracy_score(y_true, y_pred) * 100, 2)

# ============================================================
# 3.8.2 UNCERTAINTY RECOGNITION : AccuracyEU, F1EU (Task 3)
# ============================================================

def compute_uncertainty_recognition(predictions):
    y_true = [1 if p["true_label"] == "uncertain" else 0 for p in predictions]
    y_pred = [1 if p["pred_label"] == "uncertain" else 0 for p in predictions]

    true_uncertain_idx = [i for i, v in enumerate(y_true) if v == 1]
    if not true_uncertain_idx:
        return {"accuracy_eu": None, "f1_eu": None, "precision_eu": None, "recall_eu": None}

    correctly_recognized = sum(1 for i in true_uncertain_idx if y_pred[i] == 1)
    accuracy_eu = round(correctly_recognized / len(true_uncertain_idx) * 100, 2)

    f1_eu = round(f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)
    precision_eu = round(precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)
    recall_eu = round(recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100, 2)

    return {
        "accuracy_eu": accuracy_eu, "f1_eu": f1_eu,
        "precision_eu": precision_eu, "recall_eu": recall_eu,
    }

# ============================================================
# 3.8.3 EXPLANATION FAITHFULNESS : BERTScore, SentenceBERT, METEOR (Task 2)
# ============================================================

def compute_bertscore(references, hypotheses):
    """BERTScore F1 moyen. Retourne None si le package n'est pas installe."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("  [SKIP] bert-score non installe (pip install bert-score)")
        return None
    _, _, f1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    return round(float(f1.mean()) * 100, 2)

def compute_sentence_bert_similarity(references, hypotheses):
    """Similarite cosinus moyenne (SentenceBERT). None si non installe."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("  [SKIP] sentence-transformers non installe (pip install sentence-transformers)")
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_ref = model.encode(references, convert_to_tensor=True)
    emb_hyp = model.encode(hypotheses, convert_to_tensor=True)
    sims = util.cos_sim(emb_ref, emb_hyp).diagonal()
    return round(float(sims.mean()) * 100, 2)

def compute_meteor(references, hypotheses):
    """METEOR moyen. None si nltk n'est pas installe/configure."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except ImportError:
        print("  [SKIP] nltk non installe (pip install nltk)")
        return None

    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(meteor_score([ref.split()], hyp.split()))
    return round(float(np.mean(scores)) * 100, 2)

def compute_explanation_faithfulness(task2_data, task2_predictions_text=None):
    """
    Compare les explications de reference (output) aux explications
    "generees" (simulees en mode demonstration, ou reelles plus tard).
    """
    references = [ex["output"] for ex in task2_data]

    if task2_predictions_text is None:
        rng = random.Random("faithfulness_demo")
        hypotheses = []
        for ref in references:
            words = ref.split()
            if len(words) > 5 and rng.random() < 0.3:
                drop_idx = rng.randrange(len(words))
                words = words[:drop_idx] + words[drop_idx + 1:]
            hypotheses.append(" ".join(words))
    else:
        hypotheses = task2_predictions_text

    return {
        "bertscore_f1": compute_bertscore(references, hypotheses),
        "sentence_bert_similarity": compute_sentence_bert_similarity(references, hypotheses),
        "meteor": compute_meteor(references, hypotheses),
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
# 6. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 6 : METRIQUES AUTOMATIQUES D'EVALUATION")
    print("=" * 60)
    print("\n!! MODE DEMONSTRATION !!")
    print("Les predictions sont SIMULEES (bruit ajoute a la reference)")
    print("en attendant les vraies predictions du modele (Lot 1, Phase 3).\n")

    split_name = "test"
    report = {"mode": "DEMONSTRATION", "split": split_name}

    print("--- Task 1 : Diagnostic Accuracy ---")
    task1_data = load_task(split_name, 1)
    if task1_data:
        preds1 = simulate_predictions(task1_data)
        acc1 = compute_diagnostic_accuracy(preds1)
        print(f"  Accuracy : {acc1}%")
        report["diagnostic_accuracy_pct"] = acc1

        confidences1 = [p["confidence"] for p in preds1]
        correct1 = [1 if p["pred_label"] == p["true_label"] else 0 for p in preds1]
        ece1 = compute_ece(confidences1, correct1)
        print(f"  ECE (Task 1) : {ece1}")
        report["ece_task1"] = ece1

    print("\n--- Task 3 : Uncertainty Recognition ---")
    task3_data = load_task(split_name, 3)
    if task3_data:
        preds3 = simulate_predictions(task3_data, possible_labels=["confident", "uncertain"])
        eu_metrics = compute_uncertainty_recognition(preds3)
        print(f"  AccuracyEU : {eu_metrics['accuracy_eu']}%")
        print(f"  F1EU       : {eu_metrics['f1_eu']}%")
        report["uncertainty_recognition"] = eu_metrics

        confidences3 = [p["confidence"] for p in preds3]
        correct3 = [1 if p["pred_label"] == p["true_label"] else 0 for p in preds3]
        ece3 = compute_ece(confidences3, correct3)
        print(f"  ECE (Task 3) : {ece3}")
        report["ece_task3"] = ece3

    print("\n--- Task 2 : Explanation Faithfulness ---")
    task2_data = load_task(split_name, 2)
    if task2_data:
        subset = task2_data[:100]
        print(f"  (calcul sur un sous-echantillon de {len(subset)} cas pour la demo)")
        faithfulness = compute_explanation_faithfulness(subset)
        for metric_name, value in faithfulness.items():
            print(f"  {metric_name} : {value}")
        report["explanation_faithfulness"] = faithfulness

    output_path = OUTPUT_DIR / "phase6_metrics_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nRapport sauvegarde : {output_path}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()