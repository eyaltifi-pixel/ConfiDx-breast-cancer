#!/usr/bin/env python3
import requests
import pandas as pd
import json
import sys
from pathlib import Path

THRESHOLD = 150
OUTPUT_DIR = Path(__file__).parent.parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ETUDES A AUDITER
STUDIES = {
    "TCGA": "brca_tcga",
    "METABRIC": "brca_metabric"
}

def fetch_and_pivot(study_id, data_type):
    url = "https://www.cbioportal.org/api/studies/" + study_id + "/clinical-data"
    r = requests.get(url, params={"clinicalDataType": data_type, "pageSize": 50000}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    idx = "patientId" if data_type == "PATIENT" else "sampleId"
    pivot = df.pivot_table(index=idx, columns="clinicalAttributeId", values="value", aggfunc="first")
    return pivot

def find_ihc_columns(pivot):
    cols = list(pivot.columns)
    her2 = [c for c in cols if "HER2" in c.upper()]
    er = [c for c in cols if c.upper() in ["ER_STATUS", "ER_IHC"]]
    pr = [c for c in cols if c.upper() in ["PR_STATUS", "PR_IHC"]]
    grade = [c for c in cols if "GRADE" in c.upper()]
    return her2, er, pr, grade

def audit_study(study_name, study_id):
    print("\n" + "="*60)
    print("AUDIT: " + study_name + " (" + study_id + ")")
    print("="*60)
    
    # Essaie d'abord SAMPLE (donnees tumorales)
    pivot = fetch_and_pivot(study_id, "SAMPLE")
    if pivot is None or len(pivot.columns) < 5:
        pivot = fetch_and_pivot(study_id, "PATIENT")
    
    if pivot is None:
        print("ERREUR: Aucune donnee recuperee")
        return None
    
    print("Patients/Samples: " + str(len(pivot)))
    print("Colonnes: " + str(len(pivot.columns)))
    
    her2_cols, er_cols, pr_cols, grade_cols = find_ihc_columns(pivot)
    print("\nColonnes IHC detectees:")
    print("  HER2: " + str(her2_cols))
    print("  ER:   " + str(er_cols))
    print("  PR:   " + str(pr_cols))
    print("  GRADE: " + str(grade_cols))
    
    # CATEGORIE B: HER2 equivocal
    n_b = 0
    if her2_cols:
        her2_col = her2_cols[0]
        her2_vals = pivot[her2_col].astype(str)
        # Cherche "2+", "Equivocal", "2"
        is_equivocal = her2_vals.str.contains(r"2\+|equivocal|^2$", case=False, na=False)
        n_b = int(is_equivocal.sum())
        print("\nDistribution HER2 (" + her2_col + "):")
        print(her2_vals.value_counts().head(10))
    
    # CATEGORIE C: Desaccord (non disponible dans la plupart des etudes)
    n_c = 0
    
    # CATEGORIE A: Epistemique (calculee plus tard par masquage)
    
    stats = {
        "total": len(pivot),
        "category_B_HER2_equivocal": n_b,
        "category_C_interobserver": n_c,
        "category_A_epistemic": len(pivot) - n_b - n_c
    }
    
    print("\nRESUME:")
    for k, v in stats.items():
        print("  " + k + ": " + str(v))
    
    # Decisions
    print("\nDECISIONS:")
    for cat, count in stats.items():
        if cat == "total": continue
        if count >= THRESHOLD:
            print("  OK " + cat + ": " + str(count) + " -> RETENUE")
        else:
            print("  WARNING " + cat + ": " + str(count) + " -> FUSION A")
    
    # Sauvegarde
    out_file = OUTPUT_DIR / (study_id + "_clinical.json")
    with open(out_file, 'w') as f:
        json.dump(pivot.reset_index().to_dict(orient='records'), f, indent=2)
    print("\nSauvegarde: " + str(out_file))
    
    return stats

# EXECUTION
print("="*60)
print("PHASE 0: FEASIBILITY AUDIT - VERSION FINALE")
print("="*60)

results = {}
for name, sid in STUDIES.items():
    results[name] = audit_study(name, sid)

# Rapport final
report = {
    "threshold": THRESHOLD,
    "studies": results,
    "recommendations": {
        "TCGA": "Utiliser brca_tcga (Firehose Legacy) - donnees IHC disponibles",
        "METABRIC": "Utiliser brca_metabric - donnees IHC disponibles"
    }
}

with open(OUTPUT_DIR / "audit_report_final.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("AUDIT TERMINE")
print("="*60)
