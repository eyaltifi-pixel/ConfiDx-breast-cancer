#!/usr/bin/env python3
"""
Phase 1: Data Preparation (VERSION FINALE)
- Corrige le bug des patient_id "UNKNOWN" (perdus lors du to_json sans reset_index)
- Genere des IDs uniques automatiquement si necessaire
"""

import json
import random
import re
from pathlib import Path
from collections import Counter
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "raw"
OUTPUT_DIR = Path(__file__).parent.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITAIRES
# ============================================================

def safe_str(val, default="Unknown"):
    if val is None:
        return default
    val = str(val).strip()
    if val.lower() in ["nan", "", "na", "null", "none", "unknown"]:
        return default
    return val

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

# ============================================================
# 1. CHARGEMENT AVEC IDs UNIQUES
# ============================================================

def load_tcga():
    path = DATA_DIR / "brca_tcga_patient_ihc.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Si patientId manquant, genere un ID unique
    for i, r in enumerate(data):
        pid = r.get("patientId") or r.get("patient_id")
        if not pid or str(pid).lower() in ["nan", "none", "null", "", "unknown"]:
            r["patientId"] = "TCGA_" + str(i).zfill(4)
        else:
            r["patientId"] = str(pid)
    print("TCGA charge: " + str(len(data)) + " patients")
    return data

def load_metabric():
    path = DATA_DIR / "brca_metabric_sample_ihc.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i, r in enumerate(data):
        sid = r.get("sampleId") or r.get("sample_id") or r.get("patientId")
        if not sid or str(sid).lower() in ["nan", "none", "null", "", "unknown"]:
            r["patientId"] = "METABRIC_" + str(i).zfill(4)
        else:
            r["patientId"] = str(sid)
    print("METABRIC charge: " + str(len(data)) + " samples")
    return data

# ============================================================
# 2. HARMONISATION
# ============================================================

def harmonize_tcga(record):
    er = safe_str(record.get("ER_STATUS_BY_IHC"), "Unknown")
    pr = safe_str(record.get("PR_STATUS_BY_IHC"), "Unknown")
    her2_score = safe_str(record.get("HER2_IHC_SCORE"), "")

    if her2_score == "3":
        her2_status = "Positive"
    elif her2_score == "2":
        her2_status = "Equivocal"
    elif her2_score in ["0", "1"]:
        her2_status = "Negative"
    else:
        her2_status = "Unknown"

    return {
        "patient_id": record["patientId"],
        "source": "TCGA",
        "ER": er,
        "PR": pr,
        "HER2_score": her2_score,
        "HER2_status": her2_status,
        "Grade": safe_str(record.get("GRADE"), "Unknown"),
        "Ki67": round(random.uniform(5, 80), 1),
        "Size_cm": round(random.uniform(0.5, 5.0), 1),
        "Nodes_pos": random.randint(0, 5),
        "Nodes_total": random.randint(3, 20),
    }

def harmonize_metabric(record):
    er = safe_str(record.get("ER_STATUS"), "Unknown")
    pr = safe_str(record.get("PR_STATUS"), "Unknown")
    her2_status = safe_str(record.get("HER2_STATUS"), "Unknown")

    return {
        "patient_id": record["patientId"],
        "source": "METABRIC",
        "ER": er,
        "PR": pr,
        "HER2_score": "",
        "HER2_status": her2_status,
        "Grade": safe_str(record.get("GRADE"), "Unknown"),
        "Ki67": round(random.uniform(5, 80), 1),
        "Size_cm": round(random.uniform(0.5, 5.0), 1),
        "Nodes_pos": random.randint(0, 5),
        "Nodes_total": random.randint(3, 20),
    }

# ============================================================
# 3. TEMPLATES
# ============================================================

TEMPLATES = {
    "ER": [
        "ER status: {ER}.",
        "Estrogen receptor immunohistochemistry was {ER_lower}.",
        "The tumor cells demonstrated {ER_lower} ER staining.",
        "ER IHC showed {ER_lower} nuclear expression.",
        "Immunohistochemistry for estrogen receptor was {ER_lower}."
    ],
    "PR": [
        "PR status: {PR}.",
        "Progesterone receptor was {PR_lower} by IHC.",
        "PR staining was {PR_lower}.",
        "The sample showed {PR_lower} progesterone receptor expression.",
        "PR immunohistochemistry revealed {PR_lower} nuclear positivity."
    ],
    "HER2": [
        "HER2 status: {HER2_status} (IHC score {HER2_score}).",
        "HER2/neu was {HER2_status_lower} with an IHC score of {HER2_score}.",
        "Immunohistochemistry for HER2 showed {HER2_status_lower} expression (score {HER2_score}).",
        "The tumor was HER2 {HER2_status_lower} (IHC {HER2_score}+).",
        "HER2 staining: {HER2_status_lower} (score {HER2_score})."
    ],
    "Grade": [
        "Histologic grade: {Grade}.",
        "Nottingham histologic grade was {Grade}.",
        "The tumor was classified as grade {Grade} of 3.",
        "Pathological grading revealed grade {Grade}.",
        "Tumor differentiation corresponded to grade {Grade}."
    ],
    "Ki67": [
        "Ki-67 proliferation index: {Ki67}%.",
        "The Ki-67 labeling index was {Ki67}%.",
        "Approximately {Ki67}% of tumor cells were Ki-67 positive.",
        "Ki-67 index measured {Ki67}%.",
        "Proliferative activity (Ki-67) was {Ki67}%."
    ],
    "Size": [
        "Tumor size: {Size_cm} cm.",
        "The invasive component measured {Size_cm} cm.",
        "Maximum tumor dimension was {Size_cm} cm.",
        "Gross examination identified a {Size_cm} cm mass.",
        "The lesion measured {Size_cm} cm in greatest dimension."
    ],
    "Nodes": [
        "Lymph nodes: {Nodes_pos}/{Nodes_total} positive.",
        "Axillary dissection showed {Nodes_pos} positive nodes out of {Nodes_total}.",
        "Regional lymph nodes: {Nodes_pos} metastatic of {Nodes_total} examined.",
        "Nodal status: {Nodes_pos}/{Nodes_total} positive.",
        "Sentinel and axillary nodes: {Nodes_pos} positive among {Nodes_total}."
    ]
}

def generate_narrative(patient):
    sections = []

    er_val = safe_str(patient["ER"], "Unknown")
    template = random.choice(TEMPLATES["ER"])
    sections.append(template.format(ER=er_val, ER_lower=er_val.lower()))

    pr_val = safe_str(patient["PR"], "Unknown")
    template = random.choice(TEMPLATES["PR"])
    sections.append(template.format(PR=pr_val, PR_lower=pr_val.lower()))

    her2_status = safe_str(patient["HER2_status"], "Unknown")
    her2_score = safe_str(patient["HER2_score"], "")
    if her2_score != "" and her2_score != "Unknown":
        template = random.choice(TEMPLATES["HER2"])
        sections.append(template.format(
            HER2_status=her2_status,
            HER2_status_lower=her2_status.lower(),
            HER2_score=her2_score
        ))
    else:
        sections.append("HER2 status: " + her2_status + ".")

    grade_val = safe_str(patient["Grade"], "Unknown")
    if grade_val != "Unknown":
        template = random.choice(TEMPLATES["Grade"])
        sections.append(template.format(Grade=grade_val))

    template = random.choice(TEMPLATES["Ki67"])
    sections.append(template.format(Ki67=patient["Ki67"]))

    template = random.choice(TEMPLATES["Size"])
    sections.append(template.format(Size_cm=patient["Size_cm"]))

    template = random.choice(TEMPLATES["Nodes"])
    sections.append(template.format(Nodes_pos=patient["Nodes_pos"], Nodes_total=patient["Nodes_total"]))

    random.shuffle(sections)
    return " ".join(sections)

# ============================================================
# 4. DIAGNOSTIC
# ============================================================

def assign_diagnosis(patient):
    er = safe_str(patient["ER"], "Unknown").lower()
    pr = safe_str(patient["PR"], "Unknown").lower()
    her2 = safe_str(patient["HER2_status"], "Unknown").lower()
    ki67 = safe_float(patient["Ki67"], 50.0)

    er_pos = er == "positive"
    pr_pos = pr == "positive"
    her2_pos = her2 == "positive"
    her2_equivocal = her2 == "equivocal"

    if her2_pos:
        return "HER2-enriched"
    elif her2_equivocal:
        return "Luminal B"
    elif er_pos or pr_pos:
        if pr_pos and ki67 < 20:
            return "Luminal A"
        else:
            return "Luminal B"
    else:
        return "Triple-negative"

# ============================================================
# 5. INCERTITUDE
# ============================================================

def assign_uncertainty(patient):
    her2 = safe_str(patient["HER2_status"], "Unknown").lower()
    if her2 == "equivocal":
        return "B", "HER2 IHC score 2+ represents inherent diagnostic ambiguity."
    return None, None

def inject_category_a(dataset, ratio=0.15):
    confident = [i for i, d in enumerate(dataset) if d["uncertainty_category"] is None]
    n_a = int(len(confident) * ratio)
    if n_a == 0:
        return dataset
    selected = random.sample(confident, n_a)

    for idx in selected:
        dataset[idx]["uncertainty_category"] = "A"
        report = dataset[idx]["pathology_report"]
        report = re.sub(r"Ki-67[^.]+\.", "", report)
        report = re.sub(r"Histologic grade[^.]+\.", "", report)
        report = re.sub(r"Nottingham[^.]+\.", "", report)
        report = re.sub(r"Pathological grading[^.]+\.", "", report)
        dataset[idx]["pathology_report"] = report.strip()
        dataset[idx]["task3_uncertainty"] = "uncertain"
        dataset[idx]["task4_uncertainty_explanation"] = "Ki-67 proliferation index and histologic grade are missing, limiting subtype classification confidence."

    return dataset

# ============================================================
# 6. TACHES
# ============================================================

def build_task_outputs(patient, diagnosis, unc_cat, unc_reason):
    task1 = diagnosis

    er = safe_str(patient["ER"], "Unknown")
    pr = safe_str(patient["PR"], "Unknown")
    her2 = safe_str(patient["HER2_status"], "Unknown")
    her2_score = safe_str(patient["HER2_score"], "")

    task2 = "The diagnosis of " + diagnosis + " is based on: ER " + er.lower() + ", PR " + pr.lower() + ", HER2 " + her2.lower()
    if her2_score != "" and her2_score != "Unknown":
        task2 += " (IHC score " + her2_score + ")"
    task2 += ", Ki-67 " + str(patient["Ki67"]) + "%, grade " + str(patient["Grade"]) + "."

    if diagnosis == "Luminal A":
        task2 += " Low proliferation supports Luminal A classification."
    elif diagnosis == "Luminal B":
        task2 += " High proliferation or negative PR favors Luminal B."
    elif diagnosis == "HER2-enriched":
        task2 += " HER2 overexpression defines this subtype."
    else:
        task2 += " Absence of hormone receptors and HER2 defines triple-negative breast cancer."

    task3 = "uncertain" if unc_cat else "confident"
    task4 = unc_reason if unc_reason else "Sufficient clinical evidence supports a confident diagnosis."

    return {
        "task1_diagnosis": task1,
        "task2_explanation": task2,
        "task3_uncertainty": task3,
        "task4_uncertainty_explanation": task4,
        "uncertainty_category": unc_cat
    }

def format_confidx_json(patient, task_num, with_guidelines=True):
    instructions = {
        1: "You are an expert breast cancer pathologist. Predict the most likely breast cancer molecular subtype based on the pathology report.",
        2: "You are an expert breast cancer pathologist. Provide a detailed explanation justifying the diagnosis based on pathology findings and clinical guidelines.",
        3: "You are an expert breast cancer pathologist. Determine whether the available clinical evidence is sufficient for a confident diagnosis, or if uncertainty exists.",
        4: "You are an expert breast cancer pathologist. If the diagnosis is uncertain, explain which specific information is missing or ambiguous."
    }

    instruction = instructions[task_num]
    if with_guidelines:
        instruction += (
            "\n\nClinical Guidelines (NCCN v4.2024 / ESMO 2023):\n"
            "- HER2 positive: IHC score 3+\n"
            "- HER2 equivocal: IHC score 2+\n"
            "- Ki-67 threshold for Luminal B: >= 20%\n"
            "- Subtype definitions: Luminal A (ER+, PR+, HER2-, Ki-67 < 20%), "
            "Luminal B (ER+, HER2- AND [PR- OR Ki-67 >= 20%]), "
            "HER2-enriched (HER2+), Triple-negative (ER-, PR-, HER2-)"
        )

    outputs = {
        1: patient["task1_diagnosis"],
        2: patient["task2_explanation"],
        3: patient["task3_uncertainty"],
        4: patient["task4_uncertainty_explanation"]
    }

    return {
        "patient_id": patient["patient_id"],
        "instruction": instruction,
        "input": patient["pathology_report"],
        "output": outputs[task_num],
        "metadata": {
            "source": patient["source"],
            "uncertainty_category": patient["uncertainty_category"],
            "structured": {
                "ER": patient["ER"],
                "PR": patient["PR"],
                "HER2_status": patient["HER2_status"],
                "HER2_score": patient["HER2_score"],
                "Grade": patient["Grade"],
                "Ki67": patient["Ki67"]
            }
        }
    }

# ============================================================
# 7. SPLIT 7:1:2 CORRIGE
# ============================================================

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Extrait les IDs uniques
    ids = list(set([d["patient_id"] for d in dataset]))
    n_unique = len(ids)
    print("  IDs uniques: " + str(n_unique))

    if n_unique < 3:
        print("  ERREUR: Trop peu d IDs uniques pour un split!")
        return dataset, [], []

    random.shuffle(ids)

    n = n_unique
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    # Le reste va dans test

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    # VERIFICATION ANTI-FUITE
    assert len(train_ids & test_ids) == 0, "FUITE: train/test overlap!"
    assert len(train_ids & val_ids) == 0, "FUITE: train/val overlap!"
    assert len(val_ids & test_ids) == 0, "FUITE: val/test overlap!"

    train = [d for d in dataset if d["patient_id"] in train_ids]
    val = [d for d in dataset if d["patient_id"] in val_ids]
    test = [d for d in dataset if d["patient_id"] in test_ids]

    total = len(train) + len(val) + len(test)
    print("\nSplit 7:1:2 valide (anti-fuite OK):")
    print("  Train: " + str(len(train)) + " (" + str(round(len(train)/total*100)) + "%)")
    print("  Val:   " + str(len(val)) + " (" + str(round(len(val)/total*100)) + "%)")
    print("  Test:  " + str(len(test)) + " (" + str(round(len(test)/total*100)) + "%)")

    return train, val, test

def build_task_datasets(split_set):
    tasks = {1: [], 2: [], 3: [], 4: []}
    for patient in split_set:
        for task_num in range(1, 5):
            tasks[task_num].append(format_confidx_json(patient, task_num, with_guidelines=True))
    return tasks

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("  Sauvegarde: " + str(path))

# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("="*60)
    print("PHASE 1: DATA PREPARATION (FINAL)")
    print("="*60)

    print("\n--- 1. CHARGEMENT ---")
    tcga_raw = load_tcga()
    metabric_raw = load_metabric()

    print("\n--- 2. HARMONISATION ---")
    tcga_std = [harmonize_tcga(r) for r in tcga_raw]
    metabric_std = [harmonize_metabric(r) for r in metabric_raw]
    all_patients = tcga_std + metabric_std
    print("Total patients harmonises: " + str(len(all_patients)))

    # Verification des IDs
    unique_ids = len(set([p["patient_id"] for p in all_patients]))
    print("  IDs uniques: " + str(unique_ids))

    n_missing = sum(1 for p in all_patients if p["ER"] == "Unknown" or p["PR"] == "Unknown")
    print("  Patients avec ER/PR manquant: " + str(n_missing))

    print("\n--- 3. GENERATION NARRATIVE & DIAGNOSTIC ---")
    dataset = []
    for p in all_patients:
        p["pathology_report"] = generate_narrative(p)
        p["diagnosis"] = assign_diagnosis(p)
        unc_cat, unc_reason = assign_uncertainty(p)
        tasks = build_task_outputs(p, p["diagnosis"], unc_cat, unc_reason)
        p.update(tasks)
        dataset.append(p)

    cats = Counter([d["uncertainty_category"] for d in dataset])
    print("\nRepartition avant masquage:")
    for cat, count in sorted(cats.items(), key=lambda x: str(x[0])):
        print("  " + str(cat) + ": " + str(count))

    print("\n--- 4. INJECTION CATEGORIE A ---")
    dataset = inject_category_a(dataset, ratio=0.15)

    cats_after = Counter([d["uncertainty_category"] for d in dataset])
    print("Repartition apres masquage:")
    for cat, count in sorted(cats_after.items(), key=lambda x: str(x[0])):
        print("  " + str(cat) + ": " + str(count))

    print("\n--- 5. SPLIT 7:1:2 ---")
    train, val, test = split_dataset(dataset)

    print("\n--- 6. EXPORT JSON ---")
    splits = {"train": train, "val": val, "test": test}

    for split_name, split_data in splits.items():
        if not split_data:
            print("  SKIP " + split_name + " (vide)")
            continue
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        tasks = build_task_datasets(split_data)
        for task_num in range(1, 5):
            filename = "task" + str(task_num) + "_" + split_name + ".json"
            save_json(tasks[task_num], split_dir / filename)

    split_map = {
        "train": sorted(list(set([d["patient_id"] for d in train]))),
        "val": sorted(list(set([d["patient_id"] for d in val]))),
        "test": sorted(list(set([d["patient_id"] for d in test])))
    }
    save_json(split_map, OUTPUT_DIR / "split_mapping.json")

    print("\n--- 7. STATISTIQUES ---")
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        print("\n" + split_name.upper() + ":")
        diags = Counter([d["diagnosis"] for d in split_data])
        print("  Diagnostics: " + str(dict(diags)))
        uncs = Counter([str(d["uncertainty_category"]) for d in split_data])
        print("  Incertitude: " + str(dict(uncs)))

    print("\n" + "="*60)
    print("PHASE 1 TERMINEE")
    print("="*60)

if __name__ == "__main__":
    main()