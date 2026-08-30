# ConfiDx — Quantification de l'incertitude pour la prédiction du sous-type moléculaire du cancer du sein

Ce dépôt contient l'intégralité du travail expérimental du projet ConfiDx : un LLM (Llama-3.1-8B-Instruct) évalué en zero-shot avec injection de guidelines cliniques (NCCN v4.2024 / ESMO 2023) pour prédire le sous-type moléculaire du cancer du sein (Luminal A, Luminal B, HER2-enrichi, Triple-négatif) à partir de rapports de pathologie, avec une couche de **quantification et de fusion de l'incertitude**, une **détection d'hallucinations à trois niveaux**, et une **validation statistique rigoureuse**.

## Structure du dépôt

```
ConfiDx-breast-cancer/
├── data/
│   ├── phase0_audit/            # Scripts d'audit de faisabilité (cBioPortal, Phase 0)
│   └── processed/                # Données prétraitées TCGA-BRCA + METABRIC (Phase 1)
│       ├── split_mapping.json
│       └── train/ val/ test/     # task{1-4}_{split}.json + variantes _no_guidelines
│                                  # (donnees pretes, mais AUCUNE inference n'a ete faite
│                                  #  sur les variantes _no_guidelines - voir Limitations)
├── guidelines/                   # Sorties Phase 2 (guidelines) et Phase 4 (hallucinations)
│   ├── GUIDELINE_FACTS.json / guideline_compliance_report.json
│   ├── phase4_level{1,2,3}_report.json          # Detection hallucinations - REEL
│   ├── phase5_uncertainty_fusion_report.json    # Prototype MODE DEMONSTRATION (signaux simules)
│   ├── phase6_metrics_report.json               # Prototype MODE DEMONSTRATION (signaux simules)
│   ├── phase7_*_TO_FILL.csv                     # Formulaires vierges (evaluation humaine planifiee,
│                                                 #   NON executee - voir rapport Lot2 Section 5.1)
│   └── phase8_statistical_tests_report.json
├── notebooks/
│   └── ConfiDx_full_pipeline.ipynb   # Notebook Colab complet : inference (Task 1-4),
│                                     # extraction corrigee, Phase 4/5/6/8 REELLES.
│                                     # Source de verite pour les resultats publies dans le papier.
├── results/
│   ├── predictions/                          # Generations brutes du modele (val/test)
│   └── phase5_uncertainty_fusion/            # Resultats REELS utilises dans le papier
│                                              # (accuracy 72.7%, AUROC fusion 0.619, etc.)
└── src/
    ├── phase2_guideline_verification.py
    ├── phase8_statistics.py                  # McNemar / Wilcoxon (version autonome)
    ├── phase3_training/
    │   └── phase3_model_building.py          # Pipeline LoRA/QLoRA (smoke test 30 steps ;
    │                                          #   entrainement complet NON execute, quota GPU)
    ├── lot2_evaluation/
    │   ├── phase4_level{1,2,3}_*.py           # Detection hallucinations - REEL
    │   ├── phase5_uncertainty_fusion.py       # Prototype MODE DEMONSTRATION (signaux simules)
    │   ├── phase6_metrics.py                  # Prototype MODE DEMONSTRATION (signaux simules)
    │   ├── phase7_human_evaluation.py
    │   └── phase8_statistical_tests.py
    └── verification_audit/                    # Scripts de verification independante (30/08/2026)
        ├── task1_extraction.py / task3_extraction.py
        ├── phase8_bootstrap_auroc.py
        └── hallucination_level2.py
        # Reproduisent independamment les resultats publies (accuracy, AUROC, hallucinations)
        # a partir des predictions brutes, en complement du notebook pour la tracabilite.
```

## Pipeline expérimental (phases)

0. **Audit de faisabilité** (`data/phase0_audit/`) : validation statistique des catégories d'incertitude sur TCGA-BRCA + METABRIC via l'API cBioPortal.
1. **Préparation des données** : harmonisation, génération narrative anti-overfitting, split patient-level 7:1:2.
2. **Injection des guidelines** (`src/phase2_guideline_verification.py`) : table `GUIDELINE_FACTS`, vérification de conformité (100% sur 3 590 exemples).
3. **Model Building** (`src/phase3_training/`) : configuration LoRA (r=8, α=16) + QLoRA 4-bit. **Validé sur smoke test uniquement** (30 steps) ; entraînement complet non exécuté faute de quota GPU.
4. **Inférence** (`notebooks/ConfiDx_full_pipeline.ipynb`) : génération des prédictions zero-shot (Task 1 : 5 générations/32 tokens ; Tasks 2-4 : 3 générations/tokens réduits, contrainte de temps documentée).
5. **Détection d'hallucinations** (3 niveaux : regex, proxy guidelines, LLM-as-Judge calibré) — voir `guidelines/phase4_level*`.
6. **Fusion de l'incertitude** (3 signaux, calibration par grid search) — voir `results/phase5_uncertainty_fusion/`.
7. **Métriques automatisées** (accuracy, F1, BERTScore, ECE, bootstrap CI).
8. **Validation statistique** (McNemar, Wilcoxon, bootstrap AUROC, correction Benjamini-Hochberg).

## Notes importantes sur la traçabilité

- **`guidelines/phase5_uncertainty_fusion_report.json` et `phase6_metrics_report.json`** sont des **prototypes en mode démonstration** (signaux simulés par graine aléatoire, explicitement étiquetés `"mode": "DEMONSTRATION"` dans leur JSON de sortie). **Ne pas confondre avec les résultats réels**, qui se trouvent dans `results/phase5_uncertainty_fusion/` et proviennent du notebook `notebooks/ConfiDx_full_pipeline.ipynb`.
- **Modèle et checkpoints LoRA** : non inclus (trop volumineux ; voir `.gitignore`). Le pipeline d'entraînement est entièrement documenté et reproductible (`src/phase3_training/`), mais n'a été validé qu'en smoke test.
- **Baseline sans guidelines** : les données `_no_guidelines` existent (`data/processed/`), mais **aucune inférence n'a été réalisée dessus** — la comparaison guideline-grounded vs zero-shot pur reste un travail futur (limitation explicitement documentée dans le papier).
- **Historique Git** : certains commits antérieurs contenant des résultats "mode réel" buggés (fuite de données en Phase 5, extraction incomplète en Phase 6) ont été corrigés via `git revert` documenté plutôt que supprimés, pour préserver la traçabilité complète de l'audit.

## Reproduire les résultats

```bash
pip install numpy scipy scikit-learn
python src/verification_audit/task1_extraction.py       # Accuracy Task 1
python src/verification_audit/phase8_bootstrap_auroc.py # Bootstrap AUROC (Tableau 11 du papier)
python src/verification_audit/hallucination_level2.py   # Proxy NCCN/ESMO (Niveau 2)
```

Pour le pipeline complet (inférence + Phases 4-8), voir `notebooks/ConfiDx_full_pipeline.ipynb` (nécessite un environnement Google Colab avec accès GPU et au Drive du projet).
