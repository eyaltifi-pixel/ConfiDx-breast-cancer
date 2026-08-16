import requests

STUDY = "brca_tcga"
url = "https://www.cbioportal.org/api/studies/" + STUDY + "/clinical-attributes"
r = requests.get(url, timeout=30)
attrs = r.json()

print("=== ATTRIBUTS pour " + STUDY + " ===")
print("Total: " + str(len(attrs)))

keywords = ["HER2", "ER", "ESTROGEN", "PR", "PROGESTERONE", "KI67", "GRADE", "IHC"]
for a in attrs:
    aid = a.get("clinicalAttributeId", "").upper()
    aname = a.get("displayName", "").upper()
    if any(k in aid or k in aname for k in keywords):
        print("  -> " + a["clinicalAttributeId"] + ": " + a.get("displayName", "N/A"))

# Test aussi les donnees SAMPLE
url2 = "https://www.cbioportal.org/api/studies/" + STUDY + "/clinical-data"
r2 = requests.get(url2, params={"clinicalDataType": "SAMPLE", "pageSize": 100}, timeout=30)
data = r2.json()
if data:
    import pandas as pd
    df = pd.DataFrame(data)
    pivot = df.pivot_table(index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first")
    print("\nColonnes SAMPLE: " + str(sorted(list(pivot.columns))))
    for col in ["ER_STATUS", "PR_STATUS", "HER2_STATUS", "GRADE"]:
        if col in pivot.columns:
            print("\nDistribution " + col + ":")
            print(pivot[col].value_counts().head(10))
