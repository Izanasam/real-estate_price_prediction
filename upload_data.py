import pandas as pd

# 1. Charger les données
fichier = r"C:\Users\imhac\Desktop\BOOTCAMP Master IA\Data Science exo\price_prediction\valeursfoncieres-2024.txt\ValeursFoncieres-2024.txt"
df = pd.read_csv(fichier, sep='|', low_memory=False)

# 2. Afficher les colonnes pour repérage
print(df.columns.tolist())