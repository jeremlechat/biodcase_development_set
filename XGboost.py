import os
import time
import numpy as np
from PIL import Image
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# === PARAMÈTRES ===
TRAIN_DIR = "base_train"
TARGET_SIZE = (128, 128)
MODEL_PATH = "modele_xgboost_final.pkl"
IMAGE_EXT = (".jpg", ".jpeg", ".png")
IMG_MODE = "L"

# === FONCTION DE CHARGEMENT DES IMAGETTES AVEC LABELS ===
def load_images_with_labels(directory):
    X_list, y_list = [], []
    for fname in os.listdir(directory):
        if not fname.lower().endswith(IMAGE_EXT) or "_" not in fname:
            continue
        label = fname.split("_")[0]
        path = os.path.join(directory, fname)
        try:
            img = Image.open(path).convert(IMG_MODE).resize(TARGET_SIZE)
            arr = np.array(img, dtype=np.float32).ravel()
            X_list.append(arr)
            y_list.append(label)
        except:
            continue
    return np.vstack(X_list), np.array(y_list)

# === 1. Entraînement du modèle final ===
print(" Chargement des données d'entraînement...")
X, y = load_images_with_labels(TRAIN_DIR)
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"c {len(X)} images chargées — classes : {list(le.classes_)}")

print(" Entraînement du modèle final (optimisé)...")
clf = XGBClassifier(
    eval_metric="mlogloss",
    n_estimators=150,
    learning_rate=0.2,
    max_depth=4,
    random_state=42,
    n_jobs=os.cpu_count(),
    verbosity=1
)
start = time.time()
clf.fit(X, y_enc)
print(f" Entraînement terminé en {time.time() - start:.2f} s")

# === 2. Validaction croisée ===
print(" Validation croisée (2 folds)...")
kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores = []

for i, (train_idx, test_idx) in enumerate(kf.split(X, y_enc)):
    clf.fit(X[train_idx], y_enc[train_idx])
    preds = clf.predict(X[test_idx])
    acc = accuracy_score(y_enc[test_idx], preds)
    scores.append(acc)
    print(f"  Fold {i+1} — Accuracy : {acc:.3f}")

print(f" Moyenne : {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# === 3. Sauvegarde du modèle ===
joblib.dump(clf, MODEL_PATH)
print(f" Modèle sauvegardé sous '{MODEL_PATH}'")

# ===  pour les prédictions test  ===

# # === 4. Chargement et prédiction sur les images de test ===
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd

# def load_test_images(directory):
#     X_list, filenames = [], []
#     for fname in os.listdir(directory):
#         if not fname.lower().endswith(IMAGE_EXT):
#             continue
#         path = os.path.join(directory, fname)
#         try:
#             img = Image.open(path).convert(IMG_MODE).resize(TARGET_SIZE)
#             arr = np.array(img, dtype=np.float32).ravel()
#             X_list.append(arr)
#             filenames.append(fname)
#         except:
#             continue
#     return np.vstack(X_list), filenames

# TEST_DIR = "base_test"
# X_test, test_filenames = load_test_images(TEST_DIR)
# clf_loaded = joblib.load(MODEL_PATH)
# y_pred_enc = clf_loaded.predict(X_test)
# y_pred = le.inverse_transform(y_pred_enc)

# df = pd.DataFrame({
#     "filename": test_filenames,
#     "predicted_label": y_pred
# })
# df.to_csv("predictions_test.csv", index=False)
# print(" Prédictions sauvegardées dans 'predictions_test.csv'")


