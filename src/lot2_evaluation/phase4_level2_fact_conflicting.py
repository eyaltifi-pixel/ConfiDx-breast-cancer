#!/usr/bin/env python3
"""
Phase 4 - Niveau 2 : Détection d'hallucinations Fact-Conflicting
Objectif : comparer le diagnostic (task1) et l'explication (task2) générés
aux règles cliniques officielles NCCN/ESMO définies dans GUIDELINE_FACTS.json.

Contrairement au Niveau 1 (qui compare le texte généré au rapport source),
le Niveau 2 vérifie que le RAISONNEMENT respecte les seuils cliniques
officiels (ex: HER2 IHC 3+ = positif, Ki-67 >= 20% = haute prolifération).

Basé sur la Section 3.6.2 du document méthodologique (Phase 4, Niveau 2).

IMPORTANT : ce niveau est un "proxy rapide", pas une mesure de fidélité
clinique définitive (il réutilise les mêmes faits injectés à l'entraînement).
Le Niveau 3 (LLM-as-judge) reste la vérification obligatoire finale.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
GUIDELINES_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR = BASE_DIR / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# On réutilise les mêmes extracteurs que le Niveau 1 pour rester cohérent
sys.path.insert(0, str(Path(__file__).parent))
from phase4_level1_input_conflicting import (
    extract_er_status, extract_pr_status, extract_her2_status,
    extract_her2_score, extract_ki67, extract_grade,
)

# ============================================================
# 1. CHARGEMENT DES GUIDELINES OFFICIELLES
# ============================================================

def load_guideline_facts():
    """Charge le fichier GUIDELINE_FACTS.json produit par le Lot 1 (Phase 2)."""
    path = GUIDELINES_DIR / "GUIDELINE_FACTS.json"
    if not path.exists():
        raise FileNotFoundError(
            f"GUIDELINE_FACTS.json introuvable a : {path}\n"
            "Ce fichier vient du Lot 1 (Phase 2). Vérifie qu'il a bien été"
            " récupéré depuis le repo GitHub."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 2. VERIFICATION DE COHERENCE HER2 (score IHC <-> statut)
# ============================================================

def check_her2_consistency(her2_score_str, her2_status_str, facts):
    """
    Vérifie que le statut HER2 annoncé correspond bien au score IHC
    selon les seuils officiels NCCN/ESMO.
    """
    if her2_score_str is None or her2_status_str is None:
        return False, "not_verifiable"

    try:
        score = int(her2_score_str)
    except (TypeError, ValueError):
        return False, "not_verifiable"

    her2_facts = facts["her2"]
    status_lower = her2_status_str.strip().lower()

    if score == her2_facts["positive_ihc_score"]:
        expected = "positive"
    elif score == her2_facts["equivocal_ihc_score"]:
        expected = "equivocal"
    elif score in her2_facts["negative_ihc_scores"]:
        expected = "negative"
    else:
        return False, "not_verifiable"

    if status_lower != expected:
        return True, (
            f"HER2 IHC score {score} devrait donner statut '{expected}' "
            f"(NCCN/ESMO) mais le texte généré indique '{status_lower}'"
        )
    return False, "match"

# ============================================================
# 3. VERIFICATION DE COHERENCE Ki-67 (seuil Luminal A/B)
# ============================================================

def check_ki67_threshold_consistency(ki67_value, diagnosis, facts):
    """
    Vérifie que la classification Luminal A / Luminal B respecte
    le seuil officiel de Ki-67 (par défaut 20%).
    """
    if ki67_value is None or diagnosis is None:
        return False, "not_verifiable"

    threshold = facts["ki67"]["luminal_b_threshold"]
    diag_lower = diagnosis.strip().lower()

    if diag_lower == "luminal a" and ki67_value >= threshold:
        return True, (
            f"Diagnostic 'Luminal A' incompatible avec Ki-67={ki67_value}% "
            f"(seuil NCCN/ESMO = {threshold}%, devrait être Luminal B)"
        )
    return False, "match"

# ============================================================
# 4. VERIFICATION DE LA DEFINITION DU SOUS-TYPE (subtypes)
# ============================================================

def check_subtype_definition(er, pr, her2_status, diagnosis, facts):
    """
    Vérifie la cohérence globale du sous-type annoncé avec les règles
    HER2/ER/PR définies dans GUIDELINE_FACTS['subtypes'].
    """
    if diagnosis is None or her2_status is None:
        return False, "not_verifiable"

    diag_lower = diagnosis.strip().lower()
    her2_lower = her2_status.strip().lower()
    er_lower = (er or "").strip().lower()
    pr_lower = (pr or "").strip().lower()

    if her2_lower == "positive" and diag_lower != "her2-enriched":
        return True, (
            f"HER2 positif mais diagnostic annoncé = '{diagnosis}' "
            f"(devrait être HER2-enriched selon GUIDELINE_FACTS)"
        )

    if (her2_lower == "negative" and er_lower == "negative"
            and pr_lower == "negative" and diag_lower != "triple-negative"):
        return True, (
            f"ER-/PR-/HER2- mais diagnostic annoncé = '{diagnosis}' "
            f"(devrait être Triple-negative selon GUIDELINE_FACTS)"
        )

    return False, "match"

# ============================================================
# 5. TRAITEMENT COMBINE (task1 = diagnostic, task2 = explication)
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_by_patient(data):
    """Transforme une liste d'exemples en dict indexé par patient_id."""
    return {ex["patient_id"]: ex for ex in data if "patient_id" in ex}

def run_level2_check(split_name, facts):
    """
    Exécute la vérification Niveau 2 pour un split donné (train/val/test)
    en croisant task1 (diagnostic) et task2 (explication).
    """
    task1_path = PROCESSED_DIR / split_name / f"task1_{split_name}.json"
    task2_path = PROCESSED_DIR / split_name / f"task2_{split_name}.json"

    if not task1_path.exists() or not task2_path.exists():
        print(f"  SKIP {split_name} : fichiers task1/task2 introuvables")
        return []

    task1_data = index_by_patient(load_json(task1_path))
    task2_data = index_by_patient(load_json(task2_path))

    results = []
    for patient_id, ex2 in task2_data.items():
        ex1 = task1_data.get(patient_id)
        if ex1 is None:
            continue

        diagnosis = ex1.get("output")
        explanation = ex2.get("output", "")

        er = extract_er_status(explanation)
        pr = extract_pr_status(explanation)
        her2_status = extract_her2_status(explanation)
        her2_score = extract_her2_score(explanation)
        ki67 = extract_ki67(explanation)

        conflicts = []

        c1, d1 = check_her2_consistency(her2_score, her2_status, facts)
        if c1:
            conflicts.append({"check": "her2_score_vs_status", "detail": d1})

        c2, d2 = check_ki67_threshold_consistency(ki67, diagnosis, facts)
        if c2:
            conflicts.append({"check": "ki67_threshold", "detail": d2})

        c3, d3 = check_subtype_definition(er, pr, her2_status, diagnosis, facts)
        if c3:
            conflicts.append({"check": "subtype_definition", "detail": d3})

        results.append({
            "patient_id": patient_id,
            "split": split_name,
            "diagnosis": diagnosis,
            "has_fact_conflict": len(conflicts) > 0,
            "conflicts": conflicts,
        })

    return results

# ============================================================
# 6. RAPPORT GLOBAL
# ============================================================

def summarize(results):
    total = len(results)
    n_conflict = sum(1 for r in results if r["has_fact_conflict"])
    check_types = Counter()
    for r in results:
        for c in r["conflicts"]:
            check_types[c["check"]] += 1

    rate = (n_conflict / total * 100) if total > 0 else 0.0

    return {
        "total_examples": total,
        "n_fact_conflicting": n_conflict,
        "fact_conflict_rate_level2_pct": round(rate, 2),
        "conflicts_by_check": dict(check_types),
    }

# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 4 - NIVEAU 2 : DETECTION FACT-CONFLICTING")
    print("=" * 60)

    print("\nChargement de GUIDELINE_FACTS.json...")
    facts = load_guideline_facts()
    print(f"  Version guidelines : {facts.get('version')}")

    all_results = []
    for split_name in ["test", "val", "train"]:
        print(f"\nAnalyse du split : {split_name}")
        split_results = run_level2_check(split_name, facts)
        print(f"  {len(split_results)} patients analysés")
        all_results.extend(split_results)

    summary = summarize(all_results)

    print("\n--- RESULTATS GLOBAUX (tous splits) ---")
    print(f"Exemples analysés          : {summary['total_examples']}")
    print(f"Conflits fact-conflicting  : {summary['n_fact_conflicting']}")
    print(f"Taux (Niveau 2)            : {summary['fact_conflict_rate_level2_pct']}%")
    print(f"Détail par type de check   : {summary['conflicts_by_check']}")

    output_path = OUTPUT_DIR / "phase4_level2_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "guideline_version": facts.get("version"),
            "summary": summary,
            "details": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nRapport sauvegardé : {output_path}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()