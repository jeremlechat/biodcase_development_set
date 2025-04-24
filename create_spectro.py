import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def convert_wav_to_float(data: np.ndarray) -> np.ndarray:
    """Normalize WAV integer data to float in [-1, 1]."""
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128) / 128.0
    elif data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    else:
        return data.astype(np.float32)

def process_all_wavs(input_dir: str, output_dir: str):
    """
    Parcourt récursivement input_dir, calcule le spectrogramme de chaque .wav
    et enregistre une image PNG en niveaux de gris dans la même structure
    de répertoires sous output_dir.
    """
    for root, _, files in os.walk(input_dir):
        # Crée le répertoire de sortie correspondant
        rel_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue

            in_path  = os.path.join(root, fname)
            out_name = os.path.splitext(fname)[0] + '.png'
            out_path = os.path.join(target_dir, out_name)

            if os.path.exists(out_path):
                # on skip si déjà traité
                continue

            # Lecture et conversion en float
            fs, data = wavfile.read(in_path)
            data = convert_wav_to_float(data)

            # Calcul du spectrogramme
            f, t, Sxx = spectrogram(data, fs)

            # Passage en dB (avec un petit epsilon pour éviter log(0))
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)

            # Enregistrement en niveaux de gris
            # origin='lower' pour que l'axe fréquentiel aille du bas (0 Hz) vers le haut
            plt.imsave(out_path, Sxx_dB, cmap='gray', origin='lower')

if __name__ == "__main__":
    dir_path  = r"biodcase_development_set/train/audio"
    save_path = r"biodcase_development_set/train/spectrogrammes"
    process_all_wavs(dir_path, save_path)
