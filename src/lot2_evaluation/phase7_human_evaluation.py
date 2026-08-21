#!/usr/bin/env python3
"""
Phase 7 : Evaluation Clinique Humaine
Objectif : preparer et dimensionner l'evaluation par des medecins, avec
protocole en aveugle, et fournir les outils de calcul d'accord
inter-annotateurs une fois les notations recueillies.

Basé sur la Section 3.9 du document méthodologique (Phase 7).

Contenu :
  1. Echantillonnage dimensionne (n=120), reparti proportionnellement
     sur les 4 sous-taches et les 2 conditions (avec/sans guidelines).
  2. Protocole en aveugle : chaque cas recoit un label neutre
     "Model A" / "Model B" (le mapping reel reste secret, stocke a part).
  3. Generation d'un formulaire de notation (CSV) pret a remplir par les
     medecins : correctness (1-5) et completeness (1-5).
  4. Calcul de l'accord inter-annotateurs UNE FOIS les formulaires remplis :
     - Cohen's kappa (quadratic weighted) pour deux medecins
     - Fleiss' kappa pour trois medecins sur le sous-echantillon de calibration (n=40)

IMPORTANT - CE QUI DEPEND DE TIERS EXTERNES :
  - Le contenu reel des colonnes "correctness"/"completeness" ne peut etre
    rempli QUE par de vrais medecins (etape humaine, hors de portee du code).
  - Ce script prepare tout le necessaire (echantillon, formulaires vides,
    fonctions de calcul), pret a etre utilise des que les notations arrivent.
  - Comme observe en Phase 4/5, le dataset actuel ne contient que des cas
    "avec guidelines" (voir note transmise au Lot 1) : la stratification
    par condition sera incomplete tant que cela n'est pas corrige.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

try:
    from statsmodels.stats.inter_rater import fleiss_kappa
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

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

N_TOTAL_SAMPLE = 120          # Section 3.9 : n >= 120
N_CALIBRATION_SAMPLE = 40     # sous-echantillon note par les 3 medecins (Fleiss)

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
# 2. ECHANTILLONNAGE PROPORTIONNEL (n=120, 4 sous-taches x 2 conditions)
# ============================================================

def build_human_eval_sample(split_name="test", n_total=N_TOTAL_SAMPLE):
    """
    Tire n_total cas repartis proportionnellement sur les 4 sous-taches
    et les 2 conditions (avec/sans guidelines), conformement a la
    Section 3.9 ("Cases are drawn proportionally from the four subtasks
    and the two conditions").
    """
    strata = []

    for task_num in [1, 2, 3, 4]:
        data = load_task(split_name, task_num)
        for ex in data:
            has_guidelines = "guideline" in ex.get("instruction", "").lower()
            condition = "with_guidelines" if has_guidelines else "without_guidelines"
            strata.append((task_num, condition, ex))

    groups = {}
    for task_num, condition, ex in strata:
        key = (task_num, condition)
        groups.setdefault(key, []).append(ex)

    n_strata = len(groups)
    if n_strata == 0:
        print("  ERREUR : aucune donnee disponible pour l'echantillonnage")
        return []

    n_per_stratum = max(1, n_total // n_strata)

    sample = []
    for (task_num, condition), pool in sorted(groups.items()):
        n_take = min(n_per_stratum, len(pool))
        if n_take < n_per_stratum:
            print(f"  ATTENTION : task{task_num}/{condition} n'a que {len(pool)} cas "
                  f"(< {n_per_stratum} demandes)")
        chosen = random.sample(pool, n_take)
        for ex in chosen:
            sample.append({
                "case_id": f"T{task_num}_{condition}_{ex.get('patient_id')}",
                "patient_id": ex.get("patient_id"),
                "task_num": task_num,
                "condition": condition,
                "input": ex.get("input", ""),
                "output": ex.get("output", ""),
            })

    return sample

# ============================================================
# 3. PROTOCOLE EN AVEUGLE (Model A / Model B)
# ============================================================

def blind_sample(sample):
    """
    Attribue un label neutre "Model A"/"Model B" a chaque cas, de facon
    aleatoire, et conserve le mapping reel dans un fichier separe (secret),
    conformement au "Blinded protocol" de la Section 3.9.
    """
    blinded = []
    key_mapping = []

    for item in sample:
        rng = random.Random(item["case_id"] + "_blind")
        label = rng.choice(["Model A", "Model B"])

        blinded.append({
            "case_id": item["case_id"],
            "blinded_label": label,
            "task_num": item["task_num"],
            "input": item["input"],
            "output_shown": item["output"],
        })
        key_mapping.append({
            "case_id": item["case_id"],
            "blinded_label": label,
            "true_condition": item["condition"],
            "patient_id": item["patient_id"],
        })

    return blinded, key_mapping

# ============================================================
# 4. GENERATION DU FORMULAIRE DE NOTATION (CSV)
# ============================================================

def generate_rating_form(blinded_sample, output_path):
    """
    Genere un CSV vide, pret a etre rempli par un medecin :
    correctness (1-5) et completeness (1-5), conformement a la Section 3.9.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "blinded_label", "task_num",
            "pathology_report", "generated_output",
            "correctness_1_5", "completeness_1_5", "comments",
        ])
        for item in blinded_sample:
            writer.writerow([
                item["case_id"], item["blinded_label"], item["task_num"],
                item["input"], item["output_shown"],
                "", "", "",
            ])
    print(f"  Formulaire genere : {output_path}")

# ============================================================
# 5. CALCUL D'ACCORD INTER-ANNOTATEURS (une fois les formulaires remplis)
# ============================================================

def compute_pairwise_cohens_kappa(ratings_physician1, ratings_physician2):
    """
    Cohen's kappa pondere quadratique, pour deux medecins notant les
    memes cas sur une echelle 1-5.
    """
    return round(cohen_kappa_score(
        ratings_physician1, ratings_physician2, weights="quadratic"
    ), 4)

def compute_fleiss_kappa_from_ratings(ratings_matrix_3_raters, n_categories=5):
    """
    Fleiss' kappa pour 3 medecins sur le meme sous-echantillon (n=40).
    """
    if not HAS_STATSMODELS:
        print("  [SKIP] statsmodels non installe (pip install statsmodels)")
        return None

    n_cases = len(ratings_matrix_3_raters)
    counts = np.zeros((n_cases, n_categories), dtype=int)
    for i, case_ratings in enumerate(ratings_matrix_3_raters):
        for r in case_ratings:
            counts[i, int(r) - 1] += 1

    return round(float(fleiss_kappa(counts)), 4)

def load_completed_ratings(csv_path):
    """Charge un formulaire CSV rempli (correctness/completeness non vides)."""
    if not Path(csv_path).exists():
        return None
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["correctness_1_5"].strip():
                rows.append(row)
    return rows

# ============================================================
# 5bis. CALCUL DES VRAIS RESULTATS (A LANCER APRES NOTATION MEDICALE)
# ============================================================

def compute_real_agreement_report(form_path_physician1, form_path_physician2,
                                   calibration_form_path=None):
    """
    FONCTION DE PRODUCTION - a executer une fois que les medecins ont
    reellement rempli les formulaires CSV generes par ce script.

    Usage reel (exemple, a executer plus tard) :
        report = compute_real_agreement_report(
            "phase7_form_physician1_completed.csv",
            "phase7_form_physician2_completed.csv",
            calibration_form_path=[
                "phase7_calib_physician1.csv",
                "phase7_calib_physician2.csv",
                "phase7_calib_physician3.csv",
            ]
        )
    """
    rows1 = load_completed_ratings(form_path_physician1)
    rows2 = load_completed_ratings(form_path_physician2)

    if rows1 is None or rows2 is None:
        raise FileNotFoundError(
            "Un ou plusieurs formulaires sont introuvables. Ce calcul ne "
            "peut etre lance qu'apres reception des formulaires remplis "
            "par les medecins."
        )
    if len(rows1) == 0 or len(rows2) == 0:
        raise ValueError(
            "Les formulaires existent mais aucune ligne n'est remplie "
            "(colonne correctness_1_5 vide partout). Rien a calculer."
        )

    ids1 = {r["case_id"]: r for r in rows1}
    ids2 = {r["case_id"]: r for r in rows2}
    common_ids = sorted(set(ids1) & set(ids2))
    if len(common_ids) < len(rows1):
        print(f"  ATTENTION : seulement {len(common_ids)}/{len(rows1)} case_id "
              f"communs entre les deux formulaires")

    correctness1 = [int(ids1[cid]["correctness_1_5"]) for cid in common_ids]
    correctness2 = [int(ids2[cid]["correctness_1_5"]) for cid in common_ids]
    completeness1 = [int(ids1[cid]["completeness_1_5"]) for cid in common_ids]
    completeness2 = [int(ids2[cid]["completeness_1_5"]) for cid in common_ids]

    result = {
        "n_cases_compared": len(common_ids),
        "cohens_kappa_correctness": compute_pairwise_cohens_kappa(correctness1, correctness2),
        "cohens_kappa_completeness": compute_pairwise_cohens_kappa(completeness1, completeness2),
    }

    if calibration_form_path is not None:
        calib_rows = [load_completed_ratings(p) for p in calibration_form_path]
        if all(r is not None and len(r) > 0 for r in calib_rows):
            calib_ids = [set(r["case_id"] for r in rows) for rows in calib_rows]
            common_calib_ids = sorted(set.intersection(*calib_ids))
            ratings_matrix = []
            for cid in common_calib_ids:
                case_ratings = []
                for rows in calib_rows:
                    row = next(r for r in rows if r["case_id"] == cid)
                    case_ratings.append(int(row["correctness_1_5"]))
                ratings_matrix.append(case_ratings)
            result["n_calibration_cases"] = len(common_calib_ids)
            result["fleiss_kappa_correctness"] = compute_fleiss_kappa_from_ratings(ratings_matrix)
        else:
            print("  ATTENTION : formulaires de calibration incomplets, Fleiss kappa non calcule")

    return result

# ============================================================
# 6. MAIN - GENERATION DU MATERIEL D'EVALUATION (ETAPE REELLE)
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 7 : EVALUATION CLINIQUE HUMAINE")
    print("=" * 60)
    print("\nCe script produit le materiel REEL d'evaluation (echantillon,")
    print("protocole en aveugle, formulaires). Le calcul des accords")
    print("inter-annotateurs (Cohen/Fleiss kappa) ne peut se faire qu'une")
    print("fois les formulaires ci-dessous remplis par de vrais medecins")
    print("-> utiliser ensuite compute_real_agreement_report() (voir plus bas).\n")

    print(f"Echantillonnage cible : {N_TOTAL_SAMPLE} cas "
          f"(4 sous-taches x 2 conditions)")
    sample = build_human_eval_sample(split_name="test", n_total=N_TOTAL_SAMPLE)
    print(f"Echantillon reellement construit : {len(sample)} cas")
    if len(sample) < N_TOTAL_SAMPLE:
        print(f"  ATTENTION : cible non atteinte ({len(sample)}/{N_TOTAL_SAMPLE}). "
              f"Voir la note sur la condition 'without_guidelines' (Lot 1).")

    print("\nApplication du protocole en aveugle (Model A / Model B)...")
    blinded_sample, key_mapping = blind_sample(sample)

    form_physician1_path = OUTPUT_DIR / "phase7_form_physician1_TO_FILL.csv"
    form_physician2_path = OUTPUT_DIR / "phase7_form_physician2_TO_FILL.csv"
    generate_rating_form(blinded_sample, form_physician1_path)
    generate_rating_form(blinded_sample, form_physician2_path)

    key_path = OUTPUT_DIR / "phase7_human_eval_key_SECRET.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key_mapping, f, indent=2, ensure_ascii=False)
    print(f"  Cle de correspondance (SECRETE, a ne pas partager avec les "
          f"medecins ni pousser publiquement) : {key_path}")

    print(f"\nSous-echantillon de calibration (3 medecins) : "
          f"{N_CALIBRATION_SAMPLE} cas")
    calibration_sample = random.sample(
        blinded_sample, min(N_CALIBRATION_SAMPLE, len(blinded_sample))
    )
    calib_paths = []
    for i in [1, 2, 3]:
        p = OUTPUT_DIR / f"phase7_calibration_physician{i}_TO_FILL.csv"
        generate_rating_form(calibration_sample, p)
        calib_paths.append(p)

    report = {
        "status": "MATERIEL PRET - en attente de notation par de vrais medecins",
        "n_sample_built": len(sample),
        "n_sample_target": N_TOTAL_SAMPLE,
        "n_calibration_sample": len(calibration_sample),
        "forms_to_fill": {
            "physician1": str(form_physician1_path),
            "physician2": str(form_physician2_path),
            "calibration_physician1": str(calib_paths[0]),
            "calibration_physician2": str(calib_paths[1]),
            "calibration_physician3": str(calib_paths[2]),
        },
        "next_step": (
            "Une fois les 5 CSV remplis (colonnes correctness_1_5 et "
            "completeness_1_5), appeler compute_real_agreement_report() "
            "pour obtenir les vrais Cohen's kappa et Fleiss' kappa."
        ),
    }
    report_path = OUTPUT_DIR / "phase7_human_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nRapport sauvegarde : {report_path}")
    print("\n5 formulaires CSV generes, prets a etre envoyes aux medecins :")
    for name, path in report["forms_to_fill"].items():
        print(f"  - {name} : {path}")
    print("\nUne fois remplis, relancer avec compute_real_agreement_report()")
    print("pour obtenir les VRAIS resultats (voir docstring de la fonction).")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()