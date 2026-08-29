# ConfiDx — Quantification de l'incertitude pour la prédiction du sous-type moléculaire du cancer du sein

Ce dépôt contient le travail expérimental du projet ConfiDx : un LLM fine-tuné (LoRA) est utilisé pour prédire le sous-type moléculaire du cancer du sein (Luminal A, Luminal B, HER2-enrichi, Triple-négatif) à partir de rapports de pathologie, avec une couche de **quantification et de fusion de l'incertitude** (signaux verbalisés, cohérence interne, log-probabilités) puis une **validation statistique** des résultats.

## Structure du dépôt

```
ConfiDx-breast-cancer/
├── data/
│   └── processed/              # Jeux de données prétraités (train/val/test) issus de TCGA & METABRIC
│       ├── split_mapping.json  # Mapping patient_id -> split (train/val/test)
│       ├── train/               # task{1..4}_train.json (+ variantes _no_guidelines)
│       ├── val/                 # task{1..4}_val.json (+ variantes _no_guidelines)
│       └── test/                # task{1..4}_test.json (+ variantes _no_guidelines)
├── results/
│   ├── predictions/             # Générations du modèle (LoRA) par tâche, sur val/test
│   └── phase5_uncertainty_fusion/
│       ├── phase5_summary.json            # Métriques (AUROC, AUPRC, Brier, F1...) par méthode d'incertitude
│       ├── phase5_calibration_params.json # Seuils et poids calibrés sur le split val
│       └── phase5_fusion_detailed.json    # Détail patient par patient de la fusion des signaux
└── src/
    └── phase8_statistics.py     # Tests statistiques (McNemar, Wilcoxon, correction Benjamini-Hochberg)
```

## Description des tâches

Chaque tâche (task1–task4) correspond à une reformulation ou une variante de l'instruction de prédiction du sous-type moléculaire (avec/sans guidelines cliniques NCCN/ESMO incluses dans le prompt). Les patients proviennent des cohortes publiques **TCGA** et **METABRIC**.

## Pipeline expérimental (phases)

1. **Fine-tuning LoRA** du modèle sur les données d'entraînement (`data/processed/train`).
2. **Génération de prédictions** sur val/test avec plusieurs échantillons par patient et leurs log-probabilités (`results/predictions/`).
3. **Phase 5 — Fusion de l'incertitude** : combinaison de 3 signaux (incertitude verbalisée S1, incohérence inter-génération S2, incertitude par log-probabilité S3) via vote majoritaire, moyenne pondérée et régression logistique, avec calibration des seuils sur le split val (`results/phase5_uncertainty_fusion/`).
4. **Phase 8 — Validation statistique** (`src/phase8_statistics.py`) : test de McNemar (accuracy binaire appariée) et test de Wilcoxon (scores continus) entre les 6 méthodes d'incertitude, avec correction de Benjamini-Hochberg pour comparaisons multiples, sur les 718 patients du set de test.

## Notes importantes

- **Modèle et checkpoints LoRA** : les poids du modèle fine-tuné (`models/phase3_lora/`) ne sont **pas inclus** dans ce dépôt (trop volumineux pour Git). Pensez à les héberger séparément (Hugging Face Hub, Git LFS, ou lien Drive) et à référencer le lien ici.
- **Logs d'entraînement** (`logs/phase3/`) : idem, non inclus — à ajouter séparément si besoin.
- Certains fichiers de prédictions contiennent des valeurs `-Infinity` (log-probabilité), qui ne sont pas du JSON strict — à garder en tête si vous les rechargez avec un parseur strict.

## Reproduire les statistiques (Phase 8)

```bash
pip install numpy scipy statsmodels
python src/phase8_statistics.py
```

> Le script attend les données sous `/content/CONFIDX` (chemin Google Colab) — adaptez la variable `BASE` en haut du fichier si vous l'exécutez en local.
