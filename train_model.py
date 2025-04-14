import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

print("🚀 Inizio training modello finale...")

df = pd.read_csv("train_updated.csv")
features_A1 = np.load("features_A1.npy", allow_pickle=True)
features_A2 = np.load("features_A2.npy", allow_pickle=True)

feature_keys_struct = ["DoC", "CC", "D", "M", "Q", "E"]
feature_keys_stat = ["DoC", "CC", "D", "M", "Q"]

utterance_features = []
for i in range(len(features_A1)):
    struct = features_A1[i]
    stat = features_A2[i]
    vector = []

    for key in feature_keys_struct:
        vector.append(struct.get(key, 0))
    for key in feature_keys_stat:
        vector.append(stat.get(key, 0))

    utterance_features.append({
        "speaker": df.loc[i, "speaker_id"],
        "emotion": df.loc[i, "emotion"],
        "features": np.array(vector)
    })

occurrences = Counter((u["speaker"], u["emotion"]) for u in utterance_features)
utterance_features = [
    u for u in utterance_features if occurrences[(u["speaker"], u["emotion"])] >= 3
]

X = np.vstack([u["features"] for u in utterance_features])
y_labels = [u["emotion"] for u in utterance_features]

le = LabelEncoder()
y = le.fit_transform(y_labels)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

clf = RandomForestClassifier(n_estimators=500, random_state=70)
clf.fit(X_res, y_res)

print("✅ Training completato.")

# Salva modello
joblib.dump(clf, 'modello_finale.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Modello salvato come modello_finale.pkl")
print("✅ Encoder salvato come label_encoder.pkl")

print("\n🧾 Classification Report:")
print(classification_report(y, clf.predict(X), target_names=le.classes_))

print("\n🧩 Matrice di confusione:")
print(confusion_matrix(y, clf.predict(X)))

uar = recall_score(y, clf.predict(X), average='macro')
print(f"\n🎯 UAR (Unweighted Average Recall): {uar:.4f}")
