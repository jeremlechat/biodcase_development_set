import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


# === PARAMÈTRES ===
TEST_DIR = "base_train"
TARGET_SIZE = (128, 128)
IMG_MODE = "L"
MODEL_PATH = "modele_xgboost_final.pkl"

# === 1. Chargement des imagettes de test ===
def load_test_images_and_labels(directory):
    X_list, y_list, filenames = [], [], []
    for fname in os.listdir(directory):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if "_" not in fname:
            continue
        label = fname.split("_")[0]  # on récupère la vraie étiquette
        path = os.path.join(directory, fname)
        try:
            img = Image.open(path).convert(IMG_MODE).resize(TARGET_SIZE)
            arr = np.array(img, dtype=np.float32).ravel()
            X_list.append(arr)
            y_list.append(label)
            filenames.append(fname)
        except:
            continue
    return np.vstack(X_list), np.array(y_list), filenames

print(" Chargement des données de test...")
X_test, y_true, test_filenames = load_test_images_and_labels(TEST_DIR)

# === 2. Chargement du modèle
print(" Chargement du modèle sauvegardé...")
model = joblib.load(MODEL_PATH)

# === 3. Prédictions
print(" Prédiction en cours...")
y_pred_enc = model.predict(X_test)

# === 4. Recode les labels pour interpréter les résultats
le = LabelEncoder()
le.fit(y_true)  # basé sur le vrai jeu de test
y_true_enc = le.transform(y_true)
y_pred = le.inverse_transform(y_pred_enc)

# === 5. Matrice de confusion
cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.title(" Matrice de confusion")
plt.tight_layout()
plt.show()
