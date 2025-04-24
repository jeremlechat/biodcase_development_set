from datetime import datetime
import pandas as pd
from PIL import Image
import os

# === CONFIGURATION ===
T = 300  # Durée en secondes
input_csv = "./train/annotations/elephantisland2013.csv"
image_folder = "./train/spectres/elephantisland2013"
output_folder = "./output/elephantisland2013"

os.makedirs(output_folder, exist_ok=True)

# === LECTURE DU CSV ===
df = pd.read_csv(input_csv)  # on garde ','

# Convertir les colonnes de temps en datetime
df['start_datetime'] = pd.to_datetime(df['start_datetime'])
df['end_datetime'] = pd.to_datetime(df['end_datetime'])

# Grouper les annotations par fichier (pour calculer un point de référence temporel par image)
grouped = df.groupby('filename')

for filename, group in grouped:
    ref_time = group['start_datetime'].min()  # référence = début de la première annotation
    for i, row in group.iterrows():
        try:
            start_seconds = (row['start_datetime'] - ref_time).total_seconds()
            end_seconds = (row['end_datetime'] - ref_time).total_seconds()

            basename = os.path.splitext(filename)[0]  # enlève .wav
            image_path = os.path.join(image_folder, f"{basename}.png") 

            if not os.path.exists(image_path):
                print(f"Image non trouvée : {image_path}")
                continue

            image = Image.open(image_path)
            width, height = image.size

            seconds_per_pixel = T / width
            start_px = int(start_seconds / seconds_per_pixel)
            end_px = int(end_seconds / seconds_per_pixel)

            start_px = max(0, min(start_px, width))
            end_px = max(0, min(end_px, width))

            if end_px <= start_px:
                print(f"Durée invalide pour {filename}, ligne {i}")
                continue

            cropped = image.crop((start_px, 0, end_px, height))

            annotation_label = str(row['annotation']).replace(" ", "_")  # pour éviter les espaces
            output_path = os.path.join(output_folder, f"{annotation_label}_{i}.jpg")


            cropped = image.crop((start_px, 0, end_px, height)).convert("RGB")

            cropped.save(output_path)
            print(f"Image sauvegardée : {output_path}")

        except Exception as e:
            print(f"Erreur à la ligne {i} : {e}")
