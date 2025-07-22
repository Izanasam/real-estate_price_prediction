# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # Pour sauvegarder le modèle

# 1️⃣ Chargement des données nettoyées
df = pd.read_csv(r"C:\Users\imhac\Desktop\BOOTCAMP Master IA\Data Science exo\price_prediction\valeursfoncieres-2024.txt\cleaned_data.txt")

# 2️⃣ Sélection des variables utiles
features = ["Date mutation", "Code postal", "Commune", "Type local", "Surface reelle bati", "Nombre pieces principales", "Surface terrain"]
X = df[features].copy()

# Conversion de la date
X["Date mutation"] = pd.to_datetime(X["Date mutation"], errors="coerce")
min_date = X["Date mutation"].min(skipna=True)
X["days_since"] = (X["Date mutation"] - min_date).dt.days
X.drop(columns=["Date mutation"], inplace=True)

df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]

# Suppression des lignes avec NaN dans X
X = X.dropna()
y = df.loc[X.index, "prix_m2"]

# Limiter le nombre de modalités pour 'Commune'
top_communes = X['Commune'].value_counts().nlargest(20).index
X['Commune'] = X['Commune'].where(X['Commune'].isin(top_communes), 'Autre')

# Encodage des variables catégorielles
X = pd.get_dummies(X, columns=["Commune", "Type local"])

# 3️⃣ Séparation du jeu de données en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Création et entraînement du modèle de régression linéaire
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Prédictions sur les données de test
y_pred = model.predict(X_test)

# 6️⃣ Évaluation du modèle
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 7️⃣ Affichage des résultats
print("Évaluation du modèle LinearRegression :")
print(f"- MAE (erreur absolue moyenne)      : {mae:.2f} €")
print(f"- RMSE (racine de l'erreur quadratique moyenne) : {rmse:.2f} €")
print(f"- R² (coefficient de détermination) : {r2:.4f}")

# 8️⃣ Sauvegarde du modèle entraîné
joblib.dump(model, "models/linear_regression_model.joblib")
print("✅ Modèle sauvegardé dans 'models/linear_regression_model.joblib'")
