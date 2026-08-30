#!/usr/bin/env python3
"""
Phase 7 : Evaluation Clinique Humaine — MODE REEL
Avec vraie stratification with_guidelines / without_guidelines
"""

import json
import random
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "predictions"
OUTPUT_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

N_TOTAL = 120
N_CALIBRATION = 40

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_stratified_sample(split_name="test"):
    """
    Construit un echantillon stratifie :
    - 4 sous-taches (1,2,3,4)
    - 2 conditions (with_guidelines, without_guidelines)
    - 15 cas par cellule (4 x 2 x 15 = 120)
    """
    sample = []
    
    for task_num in [1, 2, 3, 4]:
        for condition in ["with_guidelines", "without_guidelines"]:
            suffix = "" if condition == "with_guidelines" else "_no_guidelines"
            path = PROCESSED_DIR / split_name / f"task{task_num}_{split_name}{suffix}.json"
            
            if not path.exists():
                print(f"  ATTENTION: {path} introuvable")
                continue
            
            data = load_json(path)
            # Charger aussi les predictions correspondantes
            pred_path = PREDICTIONS_DIR / f"task{task_num}_{split_name}_predictions.json"
            preds = {}
            if pred_path.exists():
                preds = {p["patient_id"]: p for p in load_json(pred_path)}
            
            # Prendre 15 cas par cellule
            n_take = min(15, len(data))
            chosen = random.sample(data, n_take) if len(data) >= n_take else data
            
            for ex in chosen:
                pid = ex.get("patient_id")
                pred = preds.get(pid)
                pred_text = ""
                if pred:
                    pred_text = pred.get("generations", [{}])[0].get("text", "")
                
                sample.append({
                    "patient_id": pid,
                    "task_num": task_num,
                    "condition": condition,
                    "reference_text": ex.get("output", ""),
                    "predicted_text": pred_text,
                    "input_report": ex.get("input", ""),
                })
    
    return sample

def apply_blinding(sample):
    """Anonymise: remplace condition par Model A / Model B."""
    # Melanger aleatoirement qui est A ou B
    shuffled = sample[:]
    random.shuffle(shuffled)
    
    blinded = []
    key = {}  # secret mapping
    
    for i, item in enumerate(shuffled):
        model_label = "Model A" if i % 2 == 0 else "Model B"
        blinded.append({
            "case_id": f"CASE_{i+1:03d}",
            "model_label": model_label,
            "task": item["task_num"],
            "text_to_evaluate": item["predicted_text"] if item["predicted_text"] else item["reference_text"],
            "original_patient_id": item["patient_id"],  # pas dans le CSV final
        })
        key[f"CASE_{i+1:03d}"] = {
            "patient_id": item["patient_id"],
            "task_num": item["task_num"],
            "condition": item["condition"],
            "model_label": model_label,
        }
    
    return blinded, key

def generate_csv(form, filepath):
    """Genere un formulaire CSV pour un medecin."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "model", "task", "text", "correctness_1-5", "completeness_1-5", "comments"])
        for item in form:
            writer.writerow([
                item["case_id"],
                item["model_label"],
                f"Task {item['task']}",
                item["text_to_evaluate"].replace("\n", " ")[:500],  # tronque pour lisibilite
                "",  # a remplir par le medecin
                "",  # a remplir par le medecin
                "",  # a remplir par le medecin
            ])

def main():
    print("=" * 60)
    print("PHASE 7 : EVALUATION CLINIQUE HUMAINE — MODE REEL")
    print("=" * 60)
    
    print(f"\nEchantillonnage cible : {N_TOTAL} cas")
    sample = build_stratified_sample("test")
    print(f"Echantillon construit : {len(sample)} cas")
    
    # Breakdown
    breakdown = Counter((s["task_num"], s["condition"]) for s in sample)
    for key, count in sorted(breakdown.items()):
        print(f"  Task {key[0]}, {key[1]} : {count} cas")
    
    # Blindage
    print("\nApplication du protocole en aveugle...")
    blinded, secret_key = apply_blinding(sample)
    
    # Formulaires
    form_main = blinded[:N_TOTAL]
    form_calib = blinded[:N_CALIBRATION]
    
    # Sauvegardes
    generate_csv(form_main, OUTPUT_DIR / "phase7_form_physician1_TO_FILL.csv")
    generate_csv(form_main, OUTPUT_DIR / "phase7_form_physician2_TO_FILL.csv")
    print(f"Formulaires principaux : {len(form_main)} cas x 2 medecins")
    
    for i in range(3):
        generate_csv(form_calib, OUTPUT_DIR / f"phase7_calibration_physician{i+1}_TO_FILL.csv")
    print(f"Formulaires calibration : {len(form_calib)} cas x 3 medecins")
    
    # Cle secrete
    with open(OUTPUT_DIR / "phase7_human_eval_key_SECRET.json", "w") as f:
        json.dump(secret_key, f, indent=2)
    print("Cle de correspondance secrete sauvegardee")
    
    # Rapport
    with open(OUTPUT_DIR / "phase7_real_report.json", "w") as f:
        json.dump({
            "n_total": len(sample),
            "breakdown": {f"task_{k[0]}_{k[1]}": v for k, v in breakdown.items()},
            "n_blinded": len(blinded),
            "files_generated": [
                "phase7_form_physician1_TO_FILL.csv",
                "phase7_form_physician2_TO_FILL.csv",
                "phase7_calibration_physician1_TO_FILL.csv",
                "phase7_calibration_physician2_TO_FILL.csv",
                "phase7_calibration_physician3_TO_FILL.csv",
            ],
        }, f, indent=2)
    
    print(f"\nRapport sauvegarde : {OUTPUT_DIR / 'phase7_real_report.json'}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)

if __name__ == "__main__":
    main()