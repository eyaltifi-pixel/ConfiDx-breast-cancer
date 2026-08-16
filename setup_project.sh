#!/bin/bash
# Script de création de l'arborescence du projet ConfiDx

cd ~/ConfiDx

# Créer data/ avec ses sous-dossiers
mkdir -p data/phase0_audit
mkdir -p data/raw
mkdir -p data/processed/train
mkdir -p data/processed/val
mkdir -p data/processed/test
mkdir -p data/guidelines

# Créer configs/
mkdir -p configs

# Créer src/
mkdir -p src

echo "✅ Arborescence créée avec succès !"

# Afficher la structure
echo ""
echo "📁 Structure du projet :"
tree data/ configs/ src/ 2>/dev/null || ls -R data/ configs/ src/
