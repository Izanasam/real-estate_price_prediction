import pandas as pd

fichier = r"C:\Users\imhac\Desktop\BOOTCAMP Master IA\Data Science exo\price_prediction\valeursfoncieres-2024.txt\ValeursFoncieres-2024.txt"

df=pd.read_csv(fichier, sep='|', low_memory=False)

# 2. Afficher les colonnes pour repérage
colonnes_utiles = [   'Valeur fonciere',
    'Code postal',
    'Commune',
    'Type local',
    'Surface reelle bati',
    'Nombre pieces principales',
    'Surface terrain',
    'Date mutation',
    'Nombre de lots',
    'Type local'
    ]

df_selection = df[colonnes_utiles]

df= df_selection.dropna()

df['Valeur fonciere'] = df['Valeur fonciere'].astype(str) \
                                            .str.replace(',', '.') \
                                            .str.replace(' ', '') \
                                            .str.strip()
df['Valeur fonciere'] = pd.to_numeric(df['Valeur fonciere'], errors='coerce')

df = df[df['Surface reelle bati'] > 10]

df.to_csv(r"C:\Users\imhac\Desktop\BOOTCAMP Master IA\Data Science exo\price_prediction\valeursfoncieres-2024.txt\cleaned_data.txt", index=False)

        
print("Colonnes disponibles :", df.columns.tolist())
        
print("✅ Données nettoyées sauvegardées dans data/cleaned_data.csv")
