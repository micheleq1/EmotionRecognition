import numpy as np
import librosa
import joblib
from functions import sliding_window_std, sliding_window_statistical, visibility_graph, extract_graph_features
import warnings
import cv2

warnings.filterwarnings("ignore", category=RuntimeWarning)

def predict_emotion_from_file(audio_path):
    print(f"📥 File ricevuto: {audio_path}")

    # Caricamento audio
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration_sec = len(audio) / sr
        print(f"✅ Audio caricato: {len(audio)} campioni, sample rate: {sr}, durata: {duration_sec:.2f} secondi")
    except Exception as e:
        print(f"❌ Errore nel caricamento audio con librosa: {e}")
        return "errore_audio"

    try:
        # Carica modello e encoder
        clf = joblib.load('modello_finale.pkl')
        le = joblib.load('label_encoder.pkl')

        # Segmentazione
        signal_scaled = ((audio - np.min(audio)) / (np.max(audio) - np.min(audio)) * 255).astype(np.uint8)
        image = np.tile(signal_scaled, (5, 1))
        edge_positions = np.where(cv2.Canny(image, 100, 200)[0, :] > 0)[0]
        segments = np.split(audio, edge_positions) if len(edge_positions) > 0 else [audio]
        segments = [seg for seg in segments if len(seg) > 500]

        print(f"✅ Segmenti trovati: {len(segments)}")
        if len(segments) == 0:
            print("❌ Nessun segmento valido trovato.")
            return "errore_segmentazione"

        segment_mean_values = [np.mean(seg) for seg in segments]

        # Sliding window
        window_struct = 400
        step_struct = 200
        window_stat = 8000
        step_stat = 6000

        segment_len = len(segment_mean_values)
        adjusted_window = max(2, segment_len // 2) if segment_len < window_stat else window_stat
        adjusted_step = max(1, adjusted_window // 2) if segment_len < window_stat else step_stat

        std_series = sliding_window_std(segment_mean_values, window_struct, step_struct)

        # Grafi
        VG, A1 = visibility_graph(std_series)
        A2 = sliding_window_statistical(segment_mean_values, adjusted_window, adjusted_step)

        # Feature extraction
        feature_keys_struct = ["DoC", "CC", "D", "M", "Q", "E"]
        feature_keys_stat = ["DoC", "CC", "D", "M", "Q"]

        f1 = extract_graph_features(A1, is_structural=True)
        f2 = extract_graph_features(A2, is_structural=False)

        vector = []
        for key in feature_keys_struct:
            vector.append(f1.get(key, 0))
        for key in feature_keys_stat:
            vector.append(f2.get(key, 0))

        print(f"📊 Vettore feature: {vector}")
        if len(vector) == 0:
            print("❌ Vettore delle feature vuoto.")
            return "errore_feature"

        vector = np.array(vector).reshape(1, -1)
        print(f"📐 Shape vettore: {vector.shape}")

        pred = clf.predict(vector)
        emotion = le.inverse_transform(pred)[0]
        print(f"🎯 Emozione predetta: {emotion}")

        return emotion

    except Exception as e:
        print(f"❌ Errore durante la predizione: {e}")
        return "errore_predizione"
