import os
import time
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# === Paramètres ===
DATA_DIR = "base_train"
TARGET_SIZE = (128, 128)
IMG_MODE = "L"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# === Chargement des données ===
def load_flat_images(data_dir):
    X_list, y_list = [], []

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]

    for i, fname in enumerate(files):
        if "_" not in fname:
            print(f"❌ Fichier ignoré (pas de `_`) : {fname}")
            continue

        label = fname.split("_")[0]
        path = os.path.join(data_dir, fname)

        try:
            img = Image.open(path).convert(IMG_MODE).resize(TARGET_SIZE)
            arr = np.array(img, dtype=np.float32).ravel()
            X_list.append(arr)
            y_list.append(label)

            if i % 200 == 0:
                print(f"📦 {i}/{len(files)} images traitées...")

        except Exception as e:
            print(f"Erreur sur {fname} : {e}")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

print("📥 Chargement des images depuis :", DATA_DIR)
X, y = load_flat_images(DATA_DIR)
print(f"✅ {len(X)} images chargées avec {len(set(y))} classes.")

# === Encodage des labels ===
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("🏷️ Labels encodés :", list(le.classes_))

# === Modèle XGBoost ===
clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    verbosity=2  # affiche l'avancement de l'entraînement
)

# === Entraînement avec suivi de temps ===
start = time.time()
clf.fit(X, y_enc)
print(f"⏱️ Entraînement terminé en {time.time() - start:.2f} secondes.")

# === Validation croisée (5 folds) ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("🔄 Lancement de la validation croisée...")
scores = cross_val_score(clf, X, y_enc, cv=kf, scoring="accuracy", n_jobs=-1)
print(f"📈 Accuracy moyenne : {scores.mean():.3f} ± {scores.std():.3f}")

