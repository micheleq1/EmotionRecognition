import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

import joblib


FEATURE_KEYS_STRUCT = ["DoC", "CC", "D", "M", "Q", "E"]
FEATURE_KEYS_STAT   = ["DoC", "CC", "D", "M", "Q"]

def load_feature_dataset(csv_path="train_updated.csv",
                         a1_path="features_A1.npy",
                         a2_path="features_A2.npy"):

    df = pd.read_csv(csv_path)
    features_A1 = np.load(a1_path, allow_pickle=True)
    features_A2 = np.load(a2_path, allow_pickle=True)


    utterance_list = []
    for i in range(len(features_A1)):
        struct = features_A1[i]
        stat   = features_A2[i]
        vec = [struct.get(k, 0.0) for k in FEATURE_KEYS_STRUCT] + \
              [stat.get(k, 0.0)   for k in FEATURE_KEYS_STAT]
        utterance_list.append({
            "speaker": df.loc[i, "speaker_id"],
            "emotion": df.loc[i, "emotion"],
            "features": np.array(vec, dtype=float)
        })


    occur = Counter((u["speaker"], u["emotion"]) for u in utterance_list)
    filtered = [u for u in utterance_list if occur[(u["speaker"], u["emotion"])] >= 3]


    X = np.vstack([u["features"] for u in filtered])
    y_labels = [u["emotion"] for u in filtered]


    le = LabelEncoder()
    y_int = le.fit_transform(y_labels)


    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y_int)

    return X_res, y_res, le


def build_cnn1d_model(input_length, num_classes):

    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_length, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("🚀 Inizio training modello CNN...")


    X_res, y_res, le = load_feature_dataset(
        csv_path="train_updated.csv",
        a1_path="features_A1.npy",
        a2_path="features_A2.npy"
    )


    X_res = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))


    y_res_cat = to_categorical(y_res)


    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res_cat, test_size=0.20, random_state=42
    )


    input_length = X_train.shape[1]
    num_classes  = y_train.shape[1]

    model = build_cnn1d_model(input_length, num_classes)


    print("🧠 Training in corso sulla CNN 1D...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.10,
        verbose=2
    )
    print("✅ Training completato.")


    model.save("cnn_emotion_model.h5")
    joblib.dump(le, "label_encoder.pkl")
    print("✅ Modello salvato come cnn_emotion_model.h5")
    print("✅ Encoder salvato come label_encoder.pkl")


    y_pred_prob = model.predict(X_test)
    y_true      = np.argmax(y_test, axis=1)
    y_pred_lbl  = np.argmax(y_pred_prob, axis=1)

    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred_lbl, target_names=le.classes_))

    print("\n🧩 Matrice di confusione:")
    print(confusion_matrix(y_true, y_pred_lbl))

    uar = recall_score(y_true, y_pred_lbl, average='macro')
    print(f"\n🎯 UAR (Unweighted Average Recall): {uar:.4f}")
