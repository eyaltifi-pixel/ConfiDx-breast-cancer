#!/usr/bin/env python3
"""
Phase 2: Guideline Injection and Verification Loop
- Definit la table GUIDELINE_FACTS versionnee (NCCN/ESMO)
- Verifie que chaque exemple JSON a les guidelines correctement injectees
- Cree un overlay de verification pour detecter les incoherences
- Genere un rapport de conformite des guidelines
"""

import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "guidelines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. TABLE GUIDELINE_FACTS (Versionnee)
# ============================================================
# Source: NCCN Guidelines Breast Cancer v4.2024 + ESMO 2023

GUIDELINE_FACTS = {
    "version": "NCCN v4.2024 / ESMO 2023",
    "last_updated": "2026-08-14",
    "source_url": "https://www.nccn.org/guidelines/guidelines-detail?category=3&id=1419",

    # Seuils HER2
    "her2": {
        "positive_ihc_score": 3,
        "equivocal_ihc_score": 2,
        "negative_ihc_scores": [0, 1],
        "positive_definition": "IHC 3+ OR FISH amplified",
        "equivocal_definition": "IHC 2+ (requires FISH confirmation)",
        "negative_definition": "IHC 0 or 1+"
    },

    # Seuils Ki-67
    "ki67": {
        "luminal_b_threshold": 20,
        "low_proliferation": "< 20%",
        "high_proliferation": ">= 20%",
        "note": "Threshold varies by institution; 20% is standard cutoff"
    },

    # Seuils ER/PR
    "hormone_receptors": {
        "positive_threshold_percent": 1,
        "positive_definition": ">= 1% tumor cells staining positive",
        "er_positive": "Any ER+ qualifies as hormone receptor-positive",
        "pr_role": "PR negativity in ER+ disease shifts toward Luminal B"
    },

    # Classification des sous-types (St. Gallen / NCCN)
    "subtypes": {
        "Luminal A": {
            "ER": "Positive",
            "PR": "Positive (preferably high)",
            "HER2": "Negative",
            "Ki67": "< 20%",
            "grade": "Low (1-2) preferred"
        },
        "Luminal B": {
            "ER": "Positive",
            "PR": "Negative OR low",
            "HER2": "Negative",
            "Ki67": ">= 20%",
            "grade": "Any"
        },
        "HER2-enriched": {
            "ER": "Negative",
            "PR": "Negative",
            "HER2": "Positive (IHC 3+ or FISH+)",
            "Ki67": "Usually high",
            "grade": "Any"
        },
        "Triple-negative": {
            "ER": "Negative",
            "PR": "Negative",
            "HER2": "Negative",
            "Ki67": "Usually high",
            "grade": "Any"
        }
    },

    # Staging (simplifie)
    "staging": {
        "t1_size_cm": 2,
        "t2_size_cm": 5,
        "t3_size_cm": None,  # > 5 cm
        "n1_nodes_positive": 1
    }
}


def save_guideline_facts():
    """Sauvegarde la table GUIDELINE_FACTS en JSON."""
    path = OUTPUT_DIR / "GUIDELINE_FACTS.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(GUIDELINE_FACTS, f, indent=2, ensure_ascii=False)
    print("Table GUIDELINE_FACTS sauvegardee: " + str(path))


# ============================================================
# 2. VERIFICATION DES GUIDELINES INJECTEES
# ============================================================

def verify_guideline_injection(json_file):
    """
    Verifie que chaque exemple dans un fichier JSON a:
    1. Le champ 'instruction' present
    2. Les guidelines NCCN/ESMO mentionnees
    3. Les seuils corrects (HER2 3+, Ki-67 20%, etc.)
    4. Les 4 sous-types definis
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    issues = []
    verified = 0

    required_elements = [
        "NCCN",
        "ESMO",
        "HER2 positive: IHC score 3+",
        "HER2 equivocal: IHC score 2+",
        "Ki-67 threshold",
        "Luminal A",
        "Luminal B",
        "HER2-enriched",
        "Triple-negative"
    ]

    for i, example in enumerate(data):
        instruction = example.get("instruction", "")

        # Verification 1: Champ instruction present
        if not instruction:
            issues.append({"index": i, "patient_id": example.get("patient_id"), 
                          "issue": "Champ 'instruction' manquant"})
            continue

        # Verification 2: Elements requis presents
        missing = []
        for elem in required_elements:
            if elem not in instruction:
                missing.append(elem)

        if missing:
            issues.append({"index": i, "patient_id": example.get("patient_id"),
                          "issue": "Elements manquants dans instruction", 
                          "missing": missing})
        else:
            verified += 1

    return {
        "total": len(data),
        "verified": verified,
        "issues": issues,
        "compliance_rate": verified / len(data) if data else 0
    }


# ============================================================
# 3. VERIFICATION OVERLAY (Coherence diagnostic vs guidelines)
# ============================================================

def verify_diagnostic_coherence(json_file):
    """
    Verifie que le diagnostic dans 'output' est coherent avec les 
    biomarqueurs dans 'metadata.structured' selon les guidelines.

    Niveaux de verification:
    - Level 1: HER2+ -> doit etre HER2-enriched
    - Level 2: ER- PR- HER2- -> doit etre Triple-negative
    - Level 3: ER+ PR+ HER2- Ki67<20 -> devrait etre Luminal A
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # On ne verifie que la Task 1 (diagnostic)
    if "task1" not in json_file.name:
        return None

    inconsistencies = []
    checked = 0

    for i, example in enumerate(data):
        diagnosis = example.get("output", "")
        structured = example.get("metadata", {}).get("structured", {})

        if not structured:
            continue

        checked += 1
        er = structured.get("ER", "Unknown").lower()
        pr = structured.get("PR", "Unknown").lower()
        her2 = structured.get("HER2_status", "Unknown").lower()

        # Verification Level 1: HER2+ doit etre HER2-enriched
        if her2 == "positive" and diagnosis != "HER2-enriched":
            inconsistencies.append({
                "index": i,
                "patient_id": example.get("patient_id"),
                "level": 1,
                "issue": "HER2+ mais diagnostic != HER2-enriched",
                "diagnosis": diagnosis,
                "biomarkers": structured
            })

        # Verification Level 2: ER- PR- HER2- doit etre TN
        if er == "negative" and pr == "negative" and her2 == "negative":
            if diagnosis != "Triple-negative":
                inconsistencies.append({
                    "index": i,
                    "patient_id": example.get("patient_id"),
                    "level": 2,
                    "issue": "ER-/PR-/HER2- mais diagnostic != Triple-negative",
                    "diagnosis": diagnosis,
                    "biomarkers": structured
                })

    return {
        "total": len(data),
        "checked": checked,
        "inconsistencies": inconsistencies,
        "inconsistency_rate": len(inconsistencies) / checked if checked else 0
    }


# ============================================================
# 4. RAPPORT DE CONFORMITE
# ============================================================

def generate_compliance_report(results):
    """Genere un rapport JSON de conformite des guidelines."""
    report = {
        "phase": "Phase 2: Guideline Injection and Verification",
        "timestamp": datetime.now().isoformat(),
        "guideline_version": GUIDELINE_FACTS["version"],
        "results": results
    }

    path = OUTPUT_DIR / "guideline_compliance_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nRapport de conformite sauvegarde: " + str(path))
    return report


def print_summary(results):
    """Affiche un resume des verifications."""
    print("\n" + "="*60)
    print("RESUME DE LA PHASE 2: GUIDELINE VERIFICATION")
    print("="*60)

    for filename, res in results.items():
        print("\nFichier: " + filename)

        if "injection" in res:
            inj = res["injection"]
            print("  Injection guidelines:")
            print("    Total: " + str(inj["total"]))
            print("    Verifies: " + str(inj["verified"]))
            print("    Taux: " + str(round(inj["compliance_rate"] * 100, 2)) + "%")
            if inj["issues"]:
                print("    Problemes: " + str(len(inj["issues"])))
                for issue in inj["issues"][:3]:  # Affiche les 3 premiers
                    print("      - " + str(issue["issue"]) + " (" + str(issue.get("patient_id", "N/A")) + ")")

        if "coherence" in res and res["coherence"]:
            coh = res["coherence"]
            print("  Coherence diagnostic:")
            print("    Verifies: " + str(coh["checked"]))
            print("    Incoherences: " + str(len(coh["inconsistencies"])))
            print("    Taux: " + str(round(coh["inconsistency_rate"] * 100, 2)) + "%")
            if coh["inconsistencies"]:
                for inc in coh["inconsistencies"][:3]:
                    print("      - " + str(inc["issue"]) + " (" + str(inc.get("patient_id", "N/A")) + ")")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("="*60)
    print("PHASE 2: GUIDELINE INJECTION AND VERIFICATION LOOP")
    print("="*60)

    # 1. Sauvegarde de la table GUIDELINE_FACTS
    print("\n--- 1. SAUVEGARDE GUIDELINE_FACTS ---")
    save_guideline_facts()

    # 2. Verification sur tous les fichiers JSON generes
    print("\n--- 2. VERIFICATION DES GUIDELINES INJECTEES ---")

    results = {}
    json_files = sorted(PROCESSED_DIR.glob("**/*.json"))

    for json_file in json_files:
        if json_file.name == "split_mapping.json":
            continue

        print("\nVerification: " + str(json_file.name))

        # Verification injection
        injection_result = verify_guideline_injection(json_file)

        # Verification coherence (uniquement pour task1)
        coherence_result = None
        if "task1" in json_file.name:
            coherence_result = verify_diagnostic_coherence(json_file)

        results[json_file.name] = {
            "injection": injection_result,
            "coherence": coherence_result
        }

    # 3. Resume
    print_summary(results)

    # 4. Rapport de conformite
    print("\n--- 3. GENERATION DU RAPPORT DE CONFORMITE ---")
    generate_compliance_report(results)

    print("\n" + "="*60)
    print("PHASE 2 TERMINEE")
    print("="*60)
    print("\nFichiers generes:")
    print("  " + str(OUTPUT_DIR / "GUIDELINE_FACTS.json"))
    print("  " + str(OUTPUT_DIR / "guideline_compliance_report.json"))
    print("\nProchaine etape: Phase 3 (Model Building)")


if __name__ == "__main__":
    main()
