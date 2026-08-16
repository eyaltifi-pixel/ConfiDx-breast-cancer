import requests
import pandas as pd

def list_studies():
    url = "https://www.cbioportal.org/api/studies"
    r = requests.get(url, params={"projection": "SUMMARY", "pageSize": 1000}, timeout=30)
    studies = r.json()
    brca = [s for s in studies if "brca" in s["studyId"].lower()]
    print("=== ETUDES BRCA ===")
    for s in brca:
        print("  - " + s["studyId"] + ": " + s.get("name", "N/A"))
    return [s["studyId"] for s in brca]

def list_attrs(study_id):
    url = "https://www.cbioportal.org/api/studies/" + study_id + "/clinical-attributes"
    r = requests.get(url, timeout=30)
    attrs = r.json()
    keywords = ["HER2", "ER", "ESTROGEN", "PR", "PROGESTERONE", "KI67", "KI-67", "GRADE", "IHC"]
    print("\n=== ATTRIBUTS: " + study_id + " ===")
    found = []
    for a in attrs:
        aid = a.get("clinicalAttributeId", "").upper()
        aname = a.get("displayName", "").upper()
        if any(k in aid or k in aname for k in keywords):
            found.append(a)
            print("  -> " + a["clinicalAttributeId"] + ": " + a.get("displayName", "N/A"))
    if not found:
        print("  (Aucun attribut IHC)")
    return found

print("RECHERCHE DONNEES IHC DANS CBIOPORTAL")
studies = list_studies()
for study in studies[:5]:
    list_attrs(study)
