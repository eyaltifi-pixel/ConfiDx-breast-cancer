import requests
import pandas as pd

STUDY = "brca_tcga_pan_can_atlas_2018"

def explore(data_type):
    url = "https://www.cbioportal.org/api/studies/" + STUDY + "/clinical-data"
    r = requests.get(url, params={"clinicalDataType": data_type}, timeout=30)
    data = r.json()
    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index="patientId" if data_type == "PATIENT" else "sampleId",
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first"
    )
    cols = sorted(pivot.columns)
    print("\n=== TYPE: " + data_type + " | Entrees: " + str(len(data)) + " ===")
    print("Total colonnes: " + str(len(cols)))
    keywords = ["HER2", "ER", "PR", "KI67", "GRADE", "STAGE", "REVIEW", "SECOND"]
    print("\n--- COLONNES INTERESSANTES ---")
    for kw in keywords:
        matches = [c for c in cols if kw in c.upper()]
        if matches:
            print("  [" + kw + "] -> " + str(matches))
    print("\n--- TOUTES LES COLONNES (50 premieres) ---")
    for i, c in enumerate(cols[:50]):
        print("  " + str(i+1).rjust(3) + ". " + c)
    return pivot

print("EXPLORATION CBIOPORTAL - TCGA-BRCA")
patient = explore("PATIENT")
sample = explore("SAMPLE")
