# Prédiction de Prix Immobiliers

## Objectif du projet
Ce projet vise à prédire le prix au mètre carré de biens immobiliers en France à partir de données publiques de transactions foncières. L'objectif est de construire un pipeline complet de data science, du nettoyage des données à l'entraînement d'un modèle de machine learning.

## Pipeline de traitement
Le projet s'articule autour de trois scripts principaux :

### 1. Chargement des données brutes (`upload_data.py`)
- Charge le fichier brut `ValeursFoncieres-2024.txt` (données publiques, format texte, séparateur `|`).
- Affiche la liste des colonnes pour repérage et exploration initiale.

### 2. Nettoyage des données (`clean_data.py`)
- Sélectionne les colonnes pertinentes pour la modélisation (prix, surface, localisation, etc.).
- Nettoie la colonne "Valeur fonciere" (remplacement des virgules, suppression des espaces, conversion en numérique).
- Supprime les lignes incomplètes et les biens de moins de 10 m².
- Sauvegarde un fichier nettoyé `cleaned_data.txt` pour la suite du pipeline.

### 3. Entraînement et évaluation du modèle (`train_model.py`)
- Charge les données nettoyées.
- Prépare les variables explicatives (features) et la cible (prix au m²).
- Transforme la date en nombre de jours depuis la première transaction.
- Limite le nombre de communes à 20 (les plus fréquentes) pour éviter un encodage trop volumineux.
- Encode les variables catégorielles (`Commune`, `Type local`) en variables numériques (one-hot encoding).
- Sépare les données en train/test (80/20).
- Entraîne un modèle de forêt aléatoire (`RandomForestRegressor`).
- Évalue le modèle (MAE, RMSE, R²) et affiche les résultats.
- Sauvegarde le modèle entraîné dans `models/linear_regression_model.joblib`.

## Fichiers d'entrée et de sortie
- **Entrée principale** : `valeursfoncieres-2024.txt/ValeursFoncieres-2024.txt` (données brutes, ~440 Mo)
- **Données nettoyées** : `valeursfoncieres-2024.txt/cleaned_data.txt` (généré par `clean_data.py`)
- **Modèle entraîné** : `models/linear_regression_model.joblib` (généré par `train_model.py`)

## Structure des dossiers
```
price_prediction/
├── upload_data.py
├── clean_data.py
├── train_model.py
├── models/
│   └── linear_regression_model.joblib
├── valeursfoncieres-2024.txt/
│   ├── ValeursFoncieres-2024.txt
│   ├── cleaned_data.txt
│   └── ...
```

## Technologies utilisées
- Python 3
- pandas
- numpy
- scikit-learn
- joblib

## Instructions d'exécution
1. **Charger et explorer les données**
   ```bash
   python upload_data.py
   ```
2. **Nettoyer les données**
   ```bash
   python clean_data.py
   ```
3. **Entraîner le modèle**
   ```bash
   python train_model.py
   ```

## Exemple d'utilisation
Après exécution de `train_model.py`, le script affiche les métriques d'évaluation du modèle et sauvegarde le modèle entraîné dans le dossier `models/`.

## Auteurs et crédits
Projet réalisé dans le cadre du BOOTCAMP Master IA.

---

Pour toute question ou amélioration, n'hésitez pas à ouvrir une issue ou à proposer une pull request ! 