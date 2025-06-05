import numpy as np
import librosa
import joblib
import cv2
import warnings
import tensorflow as tf

from functions import sliding_window_std, sliding_window_statistical, visibility_graph, extract_graph_features

warnings.filterwarnings("ignore", category=RuntimeWarning)

def predict_emotion_from_file_cnn(audio_path):

    print(f"📥 File ricevuto: {audio_path}")


    try:
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"✅ Audio caricato: {len(audio)} campioni, sample rate: {sr}")
    except Exception as e:
        print(f"❌ Errore nel caricamento audio: {e}")
        return "errore_audio"

    try:

        model = tf.keras.models.load_model("cnn_emotion_model.h5")
        le    = joblib.load("label_encoder.pkl")


        signal_scaled = ((audio - np.min(audio)) / (np.max(audio) - np.min(audio)) * 255).astype(np.uint8)
        image = np.tile(signal_scaled, (5, 1))
        edge_positions = np.where(cv2.Canny(image, 100, 200)[0, :] > 0)[0]
        segments = np.split(audio, edge_positions) if len(edge_positions) > 0 else [audio]
        segments = [seg for seg in segments if len(seg) > 500]

        if len(segments) == 0:
            print("❌ Nessun segmento valido trovato.")
            return "errore_segmentazione"


        segment_mean_values = [np.mean(seg) for seg in segments]


        segment_len   = len(segment_mean_values)
        window_struct = 400
        step_struct   = 200
        window_stat   = 8000
        step_stat     = 6000

        adjusted_window = max(2, segment_len // 2) if segment_len < window_stat else window_stat
        adjusted_step   = max(1, adjusted_window // 2) if segment_len < window_stat else step_stat


        std_series = sliding_window_std(segment_mean_values, window_struct, step_struct)
        VG, A1     = visibility_graph(std_series)
        A2         = sliding_window_statistical(segment_mean_values, adjusted_window, adjusted_step)


        feature_keys_struct = ["DoC", "CC", "D", "M", "Q", "E"]
        feature_keys_stat   = ["DoC", "CC", "D", "M", "Q"]

        f1 = extract_graph_features(A1, is_structural=True)
        f2 = extract_graph_features(A2, is_structural=False)

        vector = [f1.get(k, 0.0) for k in feature_keys_struct] + \
                 [f2.get(k, 0.0) for k in feature_keys_stat]

        if len(vector) == 0:
            print("❌ Vettore delle feature vuoto.")
            return "errore_feature"


        X = np.array(vector, dtype=float).reshape(1, -1, 1)
        pred_prob = model.predict(X)           # (1, num_classes)
        idx       = np.argmax(pred_prob, axis=1)[0]
        emotion   = le.inverse_transform([idx])[0]

        print(f"🎯 Emozione predetta: {emotion}")
        return emotion

    except Exception as e:
        print(f"❌ Errore durante la predizione: {e}")
        return "errore_predizione"


