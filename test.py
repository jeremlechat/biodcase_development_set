import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Exemple de fichier d'annotation (ligne)
ligne = {
    'filename': '2015-01-15T17-00-00_000.png',  # Exemple de filename
    'low_frequency': 21.9,
    'high_frequency': 28.4,
    'start_datetime': '2015-02-04T03:27:32.053000+00:00',
    'end_datetime': '2015-02-04T03:27:43.709000+00:00',
}

# Charger l'image PNG du spectrogramme
pathfile = r"biodcase_development_set\train\spectrogrammes\ballenyislands2015\2015-01-15T17-00-00_000.png"
spectrogram_image = np.array(Image.open(pathfile))  # Charge ton spectrogramme PNG ici
print(spectrogram_image.shape[:2])
# Fonction pour extraire les indices de temps et découper l'image
def crop_spectrogram_image(spectrogram_image, ligne, t, f):
    # Extraire le nom du fichier (filename)
    filename = ligne['filename']
    
    # Convertir le nom de fichier en heure (timestep en secondes)
    file_time_str = filename.split('_')[0]  # On garde uniquement la partie avant '_000'
    file_time = datetime.strptime(file_time_str, "%Y-%m-%dT%H-%M-%S")  # Ignorer les fractions de seconde
    
    base_time = datetime.strptime('2015-01-15T17:00:00', "%Y-%m-%dT%H:%M:%S")
    
    # Calculer la différence en secondes entre l'heure du fichier et l'heure de base
    time_diff = (file_time - base_time).total_seconds()
    print(f"Time difference: {time_diff} seconds")

    # Calculer la proportion du temps dans l'image (entre 0 et 1)
    time_range = t[-1] - t[0]
    x_start = int((time_diff / time_range) * spectrogram_image.shape[1])
    x_end = int(((time_diff + (pd.to_datetime(ligne['end_datetime']) - pd.to_datetime(ligne['start_datetime'])).total_seconds()) / time_range) * spectrogram_image.shape[1])
    print(f"x_start: {x_start}, x_end: {x_end}")

    # Calculer la proportion des fréquences dans l'image (entre 0 et 1)
    freq_range = f[-1] - f[0]
    y_end = int(((ligne['high_frequency'] - f[0]) / freq_range) * spectrogram_image.shape[0])
    y_start = int(((ligne['low_frequency'] - f[0]) / freq_range) * spectrogram_image.shape[0])
    print(f"y_start: {y_start}, y_end: {y_end}")

    # Vérification des indices
    if x_start >= x_end or y_start >= y_end:
        print(f"Erreur de découpe : x_start={x_start}, x_end={x_end}, y_start={y_start}, y_end={y_end}")
        return None

    # Découper l'image entre `x_start` et `x_end`, et entre `y_start` et `y_end`
    imagette = spectrogram_image[y_start:y_end, x_start:x_end]
    
    if imagette.size == 0:
        print("L'imagette est vide.")
        return None

    return imagette

# Exemple d'axe du temps `t` et des fréquences `f`
t = np.linspace(0, spectrogram_image.shape[1], spectrogram_image.shape[1])  # Exemple d'axe du temps
f = np.linspace(0, spectrogram_image.shape[0], spectrogram_image.shape[0])  # Exemple d'axe des fréquences

# Utiliser la fonction pour découper l'image
imagette = crop_spectrogram_image(spectrogram_image, ligne, t, f)

# Si l'imagette a été découpée, l'afficher
if imagette is not None:
    # Affichage de l'imagette avec matplotlib
    plt.imshow(imagette, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')  # Optionnel: cacher les axes pour une meilleure image propre
    plt.show()  # Afficher l'image
else:
    print("imagette is none")
