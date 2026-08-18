#!/usr/bin/env python3
"""
Phase 4 - Niveau 1 : Détection d'hallucinations Input-Conflicting
Objectif : extraire les affirmations cliniques (ER, PR, HER2, Ki-67, Grade)
depuis le rapport source (input) ET depuis l'explication générée (output),
puis comparer les deux pour détecter des incohérences (hallucinations).

Basé sur la Section 3.6.1 du document méthodologique (Phase 4, Niveau 1).
"""

import json
import re
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. EXTRACTION DES AFFIRMATIONS CLINIQUES (REGEX)
# ============================================================

def extract_er_status(text):
    """Extrait le statut ER (Positive/Negative/Equivocal/Unknown) d'un texte."""
    text_lower = text.lower()
    match = re.search(r"er\s*(?:status|ihc|immunohistochemistry)?[:\s]*(?:was\s+)?(positive|negative|equivocal)", text_lower)
    if match:
        return match.group(1).capitalize()
    if re.search(r"estrogen receptor.*?(positive|negative)", text_lower):
        m = re.search(r"estrogen receptor.*?(positive|negative)", text_lower)
        return m.group(1).capitalize()
    return None

def extract_pr_status(text):
    """Extrait le statut PR d'un texte."""
    text_lower = text.lower()
    match = re.search(r"pr\s*(?:status|ihc)?[:\s]*(?:was\s+)?(positive|negative|equivocal)", text_lower)
    if match:
        return match.group(1).capitalize()
    if re.search(r"progesterone receptor.*?(positive|negative)", text_lower):
        m = re.search(r"progesterone receptor.*?(positive|negative)", text_lower)
        return m.group(1).capitalize()
    return None

def extract_her2_status(text):
    """Extrait le statut HER2 (Positive/Negative/Equivocal) d'un texte."""
    text_lower = text.lower()
    match = re.search(r"her2[/\s]*(?:neu)?\s*(?:status)?[:\s]*(?:was\s+)?(positive|negative|equivocal)", text_lower)
    if match:
        return match.group(1).capitalize()
    return None

def extract_her2_score(text):
    """Extrait le score IHC HER2 (0, 1, 2, 3) d'un texte."""
    match = re.search(r"her2.*?(?:ihc|score)[^\d]{0,15}(\d)\s*\+?", text.lower())
    if match:
        return match.group(1)
    match = re.search(r"(?:ihc|score)[^\d]{0,15}(\d)\s*\+.*?her2", text.lower())
    if match:
        return match.group(1)
    return None

def extract_ki67(text):
    """Extrait la valeur numérique du Ki-67 (%) d'un texte."""
    match = re.search(r"ki-?67[^\d]{0,25}(\d+(?:\.\d+)?)\s*%", text.lower())
    if match:
        return float(match.group(1))
    return None

def extract_grade(text):
    """Extrait le grade histologique (1, 2 ou 3) d'un texte."""
    match = re.search(r"grade[^\d]{0,15}(\d)", text.lower())
    if match:
        return match.group(1)
    return None

EXTRACTORS = {
    "ER": extract_er_status,
    "PR": extract_pr_status,
    "HER2_status": extract_her2_status,
    "HER2_score": extract_her2_score,
    "Ki67": extract_ki67,
    "Grade": extract_grade,
}

# ============================================================
# 2. COMPARAISON RAPPORT SOURCE vs EXPLICATION GENEREE
# ============================================================

def compare_field(field_name, source_val, generated_val, tolerance=2.0):
    """
    Compare une valeur extraite du rapport source à celle extraite
    de l'explication générée. Retourne True si conflit détecté.
    """
    if source_val is None or generated_val is None:
        return False, "not_verifiable"

    if field_name == "Ki67":
        try:
            diff = abs(float(source_val) - float(generated_val))
            if diff > tolerance:
                return True, f"écart Ki-67: source={source_val}%, généré={generated_val}% (diff={diff:.1f})"
            return False, "match"
        except (TypeError, ValueError):
            return False, "not_verifiable"

    if str(source_val).strip().lower() != str(generated_val).strip().lower():
        return True, f"conflit {field_name}: source='{source_val}', généré='{generated_val}'"
    return False, "match"

def detect_input_conflicting_hallucination(source_report, generated_explanation):
    """
    Compare le rapport source et l'explication générée sur les 6 champs cliniques.
    """
    result = {
        "has_hallucination": False,
        "conflicts": [],
        "fields_checked": 0,
        "fields_verifiable": 0,
    }

    for field_name, extractor in EXTRACTORS.items():
        source_val = extractor(source_report)
        generated_val = extractor(generated_explanation)

        result["fields_checked"] += 1
        is_conflict, detail = compare_field(field_name, source_val, generated_val)

        if detail != "not_verifiable":
            result["fields_verifiable"] += 1

        if is_conflict:
            result["has_hallucination"] = True
            result["conflicts"].append({
                "field": field_name,
                "source_value": source_val,
                "generated_value": generated_val,
                "detail": detail,
            })

    return result

# ============================================================
# 3. TRAITEMENT D'UN FICHIER JSON COMPLET (task2 = explication)
# ============================================================

def process_task2_file(filepath, output_field="output", input_field="input"):
    """Parcourt un fichier task2_*.json et applique la détection Niveau 1."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for example in data:
        source_report = example.get(input_field, "")
        generated_explanation = example.get(output_field, "")

        detection = detect_input_conflicting_hallucination(source_report, generated_explanation)
        detection["patient_id"] = example.get("patient_id")
        results.append(detection)

    return results

# ============================================================
# 4. RAPPORT GLOBAL
# ============================================================

def summarize_results(results):
    """Calcule le taux d'hallucination Niveau 1 sur un ensemble de résultats."""
    total = len(results)
    n_hallucinated = sum(1 for r in results if r["has_hallucination"])
    conflict_types = Counter()
    for r in results:
        for c in r["conflicts"]:
            conflict_types[c["field"]] += 1

    rate = (n_hallucinated / total * 100) if total > 0 else 0.0

    return {
        "total_examples": total,
        "n_hallucinated": n_hallucinated,
        "hallucination_rate_level1_pct": round(rate, 2),
        "conflicts_by_field": dict(conflict_types),
    }

# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 4 - NIVEAU 1 : DETECTION INPUT-CONFLICTING")
    print("=" * 60)

    task2_test_path = PROCESSED_DIR / "test" / "task2_test.json"

    if not task2_test_path.exists():
        print(f"\nERREUR : fichier introuvable -> {task2_test_path}")
        print("Vérifie que data/processed/test/task2_test.json existe bien.")
        return

    print(f"\nLecture de : {task2_test_path}")
    results = process_task2_file(task2_test_path)

    summary = summarize_results(results)

    print("\n--- RESULTATS ---")
    print(f"Exemples analysés     : {summary['total_examples']}")
    print(f"Hallucinations (N1)   : {summary['n_hallucinated']}")
    print(f"Taux d'hallucination  : {summary['hallucination_rate_level1_pct']}%")
    print(f"Conflits par champ    : {summary['conflicts_by_field']}")

    output_path = OUTPUT_DIR / "phase4_level1_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "details": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nRapport sauvegardé : {output_path}")
    print("=" * 60)
    print("TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()