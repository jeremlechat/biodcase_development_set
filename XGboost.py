import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1) Chargement des images + labels
DATA_DIR = "biodcase_development_set/train/imagettes/elephantisland2013/"
X_list, y_list = [], []

for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(".jpg"):
        continue
    # on suppose "classe_idx.png"
    classe = fname.split("_")[0]
    img = Image.open(os.path.join(DATA_DIR, fname)).convert("L")
    arr = np.array(img, dtype=np.float32).ravel()
    X_list.append(arr)
    y_list.append(classe)

X = np.vstack(X_list)      # shape = (n_samples, n_pixels)
y = np.array(y_list)

# 2) Encodage des labels
le     = LabelEncoder()
y_enc  = le.fit_transform(y)

# 3) Définition du classifieur
clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# — Option A : entraînement direct sur tout le train
clf.fit(X, y_enc)
print("Modèle entraîné sur tout le train.")

# — Option B : cross-validation sur le train pour avoir un score moyen
kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y_enc, cv=kf, scoring="accuracy", n_jobs=-1)
print(f"CV accuracy : {scores.mean():.3f} ± {scores.std():.3f}")
