import requests
import pandas as pd
import json
from pathlib import Path

THRESHOLD = 150
OUTPUT_DIR = Path(__file__).parent.parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_pivot(study_id, data_type):
    url = "https://www.cbioportal.org/api/studies/" + study_id + "/clinical-data"
    r = requests.get(url, params={"clinicalDataType": data_type, "pageSize": 50000}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    idx = "patientId" if data_type == "PATIENT" else "sampleId"
    return df.pivot_table(index=idx, columns="clinicalAttributeId", values="value", aggfunc="first")

def audit_tcga():
    print("\n" + "="*60)
    print("AUDIT TCGA (brca_tcga) - DONNEES PATIENT")
    print("="*60)
    
    pivot = fetch_pivot("brca_tcga", "PATIENT")
    print("Patients: " + str(len(pivot)))
    print("Colonnes: " + str(len(pivot.columns)))
    
    # Noms exacts des colonnes IHC dans TCGA
    her2_col = "HER2_IHC_SCORE"
    er_col = "ER_STATUS_BY_IHC"
    pr_col = "PR_STATUS_BY_IHC"
    grade_col = "GRADE"
    
    print("\nColonnes IHC:")
    for c in [her2_col, er_col, pr_col, grade_col]:
        status = "OK" if c in pivot.columns else "MANQUANT"
        print("  " + c + ": " + status)
    
    # CATEGORIE B: HER2 equivocal (score 2)
    n_b = 0
    if her2_col in pivot.columns:
        her2_vals = pivot[her2_col].astype(str)
        print("\nDistribution HER2_IHC_SCORE:")
        print(her2_vals.value_counts())
        is_equivocal = her2_vals.str.contains(r"^2$|2\+|equivocal", case=False, na=False)
        n_b = int(is_equivocal.sum())
        print("HER2 equivocal (2+): " + str(n_b))
    
    # CATEGORIE A: Epistemique (masquage artificiel)
    n_a = len(pivot) - n_b
    
    print("\nRESUME TCGA:")
    print("  Total: " + str(len(pivot)))
    print("  B (HER2 equivocal): " + str(n_b))
    print("  A (Epistemique): " + str(n_a))
    
    # Sauvegarde
    pivot.to_json(OUTPUT_DIR / "brca_tcga_patient_ihc.json", orient='records', indent=2)
    return {"total": len(pivot), "B": n_b, "A": n_a}

def audit_metabric():
    print("\n" + "="*60)
    print("AUDIT METABRIC (brca_metabric) - DONNEES SAMPLE")
    print("="*60)
    
    pivot = fetch_pivot("brca_metabric", "SAMPLE")
    print("Samples: " + str(len(pivot)))
    print("Colonnes: " + str(len(pivot.columns)))
    
    her2_col = "HER2_STATUS"
    er_col = "ER_STATUS"
    pr_col = "PR_STATUS"
    grade_col = "GRADE"
    
    print("\nColonnes IHC:")
    for c in [her2_col, er_col, pr_col, grade_col]:
        status = "OK" if c in pivot.columns else "MANQUANT"
        print("  " + c + ": " + status)
    
    # CATEGORIE B: HER2 equivocal
    n_b = 0
    if her2_col in pivot.columns:
        her2_vals = pivot[her2_col].astype(str)
        print("\nDistribution HER2_STATUS:")
        print(her2_vals.value_counts())
        # Cherche "Equivocal" ou "2+" ou "2"
        is_equivocal = her2_vals.str.contains(r"equivocal|2\+|^2$", case=False, na=False)
        n_b = int(is_equivocal.sum())
        print("HER2 equivocal: " + str(n_b))
    
    n_a = len(pivot) - n_b
    
    print("\nRESUME METABRIC:")
    print("  Total: " + str(len(pivot)))
    print("  B (HER2 equivocal): " + str(n_b))
    print("  A (Epistemique): " + str(n_a))
    
    pivot.to_json(OUTPUT_DIR / "brca_metabric_sample_ihc.json", orient='records', indent=2)
    return {"total": len(pivot), "B": n_b, "A": n_a}

print("="*60)
print("PHASE 0: AUDIT CORRIGE")
print("="*60)

tcga_stats = audit_tcga()
meta_stats = audit_metabric()

print("\n" + "="*60)
print("DECISIONS FINALES")
print("="*60)
print("TCGA:")
for k, v in tcga_stats.items():
    print("  " + k + ": " + str(v))
print("METABRIC:")
for k, v in meta_stats.items():
    print("  " + k + ": " + str(v))

# Decision
print("\nCategorie B (HER2 equivocal):")
total_b = tcga_stats["B"] + meta_stats["B"]
print("  TCGA: " + str(tcga_stats["B"]))
print("  METABRIC: " + str(meta_stats["B"]))
print("  TOTAL: " + str(total_b))

if total_b >= THRESHOLD:
    print("  -> RETENUE (>= " + str(THRESHOLD) + ")")
else:
    print("  -> FUSION dans A (< " + str(THRESHOLD) + ")")

report = {
    "TCGA": tcga_stats,
    "METABRIC": meta_stats,
    "total_B": total_b,
    "decision_B": "RETENIR" if total_b >= THRESHOLD else "FUSION_A"
}

with open(OUTPUT_DIR / "audit_report_corrected.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\nAUDIT CORRIGE TERMINE")
