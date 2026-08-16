#!/usr/bin/env python3
"""
Phase 0: Feasibility Audit for TCGA-BRCA
Vérifie que les 3 catégories d'incertitude ont ≥150 cas.
"""

import requests
import pandas as pd
import json
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
THRESHOLD = 150  # Seuil minimum par catégorie
OUTPUT_DIR = Path(__file__).parent.parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IDs des études cBioPortal
STUDY_TCGA = "brca_tcga_pan_can_atlas_2018"  # TCGA-BRCA Pan-Cancer Atlas


def fetch_clinical_data(study_id):
    """
    Récupère les données cliniques depuis l'API cBioPortal.
    """
    base_url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data"
    
    print(f"🔍 Connexion à cBioPortal pour l'étude: {study_id}")
    print(f"   URL: {base_url}")
    
    try:
        r = requests.get(base_url, params={"clinicalDataType": "PATIENT"}, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion: {e}")
        print("💡 Astuce: Vérifie ta connexion internet ou essaie plus tard.")
        sys.exit(1)
    
    data = r.json()
    print(f"✅ Données récupérées: {len(data)} entrées brutes")
    
    # Conversion en DataFrame pivoté (1 ligne = 1 patient)
    df = pd.DataFrame(data)
    
    if df.empty:
        print("❌ Aucune donnée reçue!")
        sys.exit(1)
    
    # Pivot: patientId en index, clinicalAttributeId en colonnes
    pivot = df.pivot_table(
        index="patientId",
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first"
    )
    
    print(f"📊 Patients uniques: {len(pivot)}")
    print(f"📊 Attributs cliniques disponibles: {list(pivot.columns)[:10]}...")  # Affiche les 10 premiers
    
    return pivot


def count_category_b_her2(pivot):
    """
    Catégorie B: HER2 IHC 2+ (équivoque)
    Cherche les colonnes contenant 'HER2' et compte les valeurs '2+' ou 'equivocal'.
    """
    # Cherche toutes les colonnes contenant HER2 (insensible à la casse)
    her2_cols = [c for c in pivot.columns if "HER2" in c.upper()]
    print(f"\n🔬 Colonnes HER2 trouvées: {her2_cols}")
    
    if not her2_cols:
        print("⚠️ WARNING: Aucune colonne HER2 trouvée!")
        return 0, []
    
    # Utilise la première colonne HER2 (généralement la principale)
    her2_col = her2_cols[0]
    her2_vals = pivot[her2_col].astype(str)
    
    # Compte les cas équivoques (2+ ou 'equivocal')
    is_equivocal = her2_vals.str.contains(r"2\+|equivocal", case=False, na=False)
    n_equivocal = is_equivocal.sum()
    
    # Affiche la distribution
    print(f"\n📊 Distribution HER2 ({her2_col}):")
    print(her2_vals.value_counts().head(10))
    
    return int(n_equivocal), her2_cols


def count_category_c_disagreement(pivot):
    """
    Catégorie C: Désaccord inter-observateur
    Cherche des colonnes de relecture et compare les paires.
    """
    # Cherche les colonnes suggérant une 2e relecture
    review_keywords = ["review", "second", "pathology", "pathologist"]
    review_cols = []
    
    for col in pivot.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in review_keywords):
            review_cols.append(col)
    
    print(f"\n🔍 Colonnes de relecture candidates: {review_cols}")
    
    if len(review_cols) < 2:
        print(f"⚠️ WARNING: Seulement {len(review_cols)} colonne(s) de relecture trouvée(s).")
        print("   Catégorie C ne peut pas être calculée.")
        return 0, review_cols
    
    # Compare les 2 premières colonnes de relecture
    col1, col2 = review_cols[0], review_cols[1]
    first = pivot[col1].astype(str)
    second = pivot[col2].astype(str)
    
    # Exclusion des valeurs manquantes
    valid = (
        first.notna() & second.notna() &
        (first != "nan") & (second != "nan") &
        (first != "NA") & (second != "NA") &
        (first != "") & (second != "")
    )
    
    n_disagreement = ((first != second) & valid).sum()
    n_comparable = valid.sum()
    
    print(f"\n📊 Comparaison {col1} vs {col2}:")
    print(f"   Cas comparables: {n_comparable}")
    print(f"   Désaccords: {n_disagreement}")
    print(f"   Taux de désaccord: {n_disagreement/n_comparable*100:.1f}%" if n_comparable > 0 else "   N/A")
    
    return int(n_disagreement), review_cols


def apply_thresholds(stats, threshold=THRESHOLD):
    """
    Applique le seuil et décide quelles catégories garder.
    """
    print(f"\n{'='*60}")
    print("📋 DÉCISIONS DE LA PHASE 0")
    print(f"{'='*60}")
    print(f"Seuil minimum: {threshold} cas\n")
    
    decisions = {}
    
    for cat, count in stats.items():
        if cat == "total":
            continue
        
        if count >= threshold:
            decisions[cat] = "✅ RETENIR"
            print(f"✅ {cat}: {count} cas → RETENUE")
        else:
            decisions[cat] = "⚠️ FUSIONNER dans A"
            print(f"⚠️ {cat}: {count} cas (<{threshold}) → FUSION dans Catégorie A (épistémique)")
    
    return decisions


def save_raw_data(pivot, study_id):
    """
    Sauvegarde les données brutes pour les phases suivantes.
    """
    output_file = OUTPUT_DIR / f"{study_id}_clinical.json"
    
    # Conversion en dict pour sauvegarde
    data_dict = pivot.reset_index().to_dict(orient='records')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"\n💾 Données brutes sauvegardées: {output_file}")
    print(f"   Taille: {len(data_dict)} patients")


def main():
    print("="*60)
    print("PHASE 0: FEASIBILITY AUDIT - TCGA-BRCA")
    print("="*60)
    
    # 1. Récupération des données
    pivot = fetch_clinical_data(STUDY_TCGA)
    n_total = len(pivot)
    
    # 2. Comptage Catégorie B
    n_equivocal, her2_cols = count_category_b_her2(pivot)
    
    # 3. Comptage Catégorie C
    n_disagreement, review_cols = count_category_c_disagreement(pivot)
    
    # 4. Résumé
    stats = {
        "total": n_total,
        "category_B_HER2_equivocal": n_equivocal,
        "category_C_interobserver_disagreement": n_disagreement,
        "category_A_epistemic": n_total - n_equivocal - n_disagreement  # Approximation
    }
    
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DES COMPTAGES")
    print(f"{'='*60}")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    # 5. Décisions
    decisions = apply_thresholds(stats, threshold=THRESHOLD)
    
    # 6. Sauvegarde
    save_raw_data(pivot, STUDY_TCGA)
    
    # 7. Export du rapport d'audit
    audit_report = {
        "study_id": STUDY_TCGA,
        "threshold": THRESHOLD,
        "statistics": stats,
        "decisions": decisions,
        "her2_columns_found": her2_cols,
        "review_columns_found": review_cols,
        "notes": {
            "category_A": "Créée artificiellement par masquage si B/C insuffisantes",
            "category_B": "HER2 IHC 2+ (équivoque réel)",
            "category_C": "Désaccord inter-observateur (2e lecture)"
        }
    }
    
    report_file = OUTPUT_DIR / "audit_report_tcga.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2)
    
    print(f"\n📄 Rapport d'audit sauvegardé: {report_file}")
    print(f"\n{'='*60}")
    print("✅ PHASE 0 TERMINÉE")
    print(f"{'='*60}")
    print("\nProchaines étapes:")
    print("   1. Vérifier les décisions dans audit_report_tcga.json")
    print("   2. Si B/C < 150: préparer le masquage artificiel (Catégorie A)")
    print("   3. Passer à la Phase 1: Génération des narratives")


if __name__ == "__main__":
    main()

