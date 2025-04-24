import os
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime

# ——— CONFIG ———
SPEC_DIR       = r"biodcase_development_set\train\spectrogrammes\ballenyislands2015"
ANN_DIR        = r"biodcase_development_set\train\annotations"
SAVE_DIR       = r"biodcase_development_set\train\imagettesballenyislands2015"
CSV_NAME       = "ballenyislands2015.csv"   # ou le nom de votre CSV
SPECTRO_DURATION = 3600.0   # chaque spectrogramme couvre 1 heure = 3600 s
FREQ_MAX         = 96.0     # par ex. 96 kHz selon vos données
# —————————————————

os.makedirs(SAVE_DIR, exist_ok=True)

# 1) Chargement du CSV
df = pd.read_csv(os.path.join(ANN_DIR, CSV_NAME))
df['start'] = pd.to_datetime(df['start_datetime'])
df['end']   = pd.to_datetime(df['end_datetime'])

# 2) Fonction d’interprétation du nom de fichier
def parse_spectro_start(fname_png):
    base   = os.path.splitext(fname_png)[0]       # "2015-02-04T03-00-00_000"
    dt_str = base.split('_')[0]                   # "2015-02-04T03-00-00"
    # renvoie un pandas.Timestamp avec tzinfo=UTC
    return pd.to_datetime(dt_str, format="%Y-%m-%dT%H-%M-%S", utc=True)

# 3) Boucle sur chaque annotation
for idx, row in df.iterrows():
    wav_name = row['filename']                      # "2015-02-04T03-00-00_000.wav"
    png_name = wav_name.replace(".wav", ".png")
    spec_path = os.path.join(SPEC_DIR, png_name)
    if not os.path.exists(spec_path):
        print(f"[!] Spectrogramme manquant : {spec_path}")
        continue

    # 3a) lecture de l’image
    img = Image.open(spec_path)
    W, H = img.size     # W = ~2048 px, H = ~65 px

    # 3b) calcul pixel-time
    spec_start = parse_spectro_start(png_name)
    t0 = row['start']
    t1 = row['end']
    off0 = (t0 - spec_start).total_seconds()
    off1 = (t1 - spec_start).total_seconds()
    x0 = int((off0 / SPECTRO_DURATION) * W)
    x1 = int((off1 / SPECTRO_DURATION) * W)

    # 3c) calcul pixel-frequency (origine y=0 en haut)
    f_low  = row['low_frequency']
    f_high = row['high_frequency']
    # on suppose que 0 Hz = bas de l’image, FREQ_MAX = haut
    y_low_px  = H - int((f_low  / FREQ_MAX) * H)
    y_high_px = H - int((f_high / FREQ_MAX) * H)

    # 3d) bornes dans l’image
    x0, x1 = max(0, x0), min(W, x1)
    y_high_px, y_low_px = max(0, y_high_px), min(H, y_low_px)
    if x1 <= x0 or y_low_px <= y_high_px:
        print(f"[!] zone vide pour idx={idx} ({x0}-{x1}, {y_high_px}-{y_low_px})")
        continue

    # 3e) extraction et sauvegarde
    crop = img.crop((x0, y_high_px, x1, y_low_px))
    out_name = png_name.replace(".png", f"_crop_{idx}.png")
    out_path = os.path.join(SAVE_DIR, out_name)
    crop.save(out_path)
    print(f"→ enregistré : {out_path}")
