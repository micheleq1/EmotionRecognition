import numpy as np
import librosa
import joblib
from functions import sliding_window_std, sliding_window_statistical, visibility_graph, extract_graph_features
import warnings
import cv2

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Carica modello e encoder
clf = joblib.load('modello_finale.pkl')
le = joblib.load('label_encoder.pkl')

# Parametri sliding window fissi
window_struct = 400
step_struct = 200
window_stat = 8000
step_stat = 6000

def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)

    # Segmentazione (come sempre)
    signal_scaled = ((audio - np.min(audio)) / (np.max(audio) - np.min(audio)) * 255).astype(np.uint8)
    image = np.tile(signal_scaled, (5, 1))
    edge_positions = np.where(cv2.Canny(image, 100, 200)[0, :] > 0)[0]
    segments = np.split(audio, edge_positions) if len(edge_positions) > 0 else [audio]
    segments = [seg for seg in segments if len(seg) > 500]
    segment_mean_values = [np.mean(seg) for seg in segments]

    segment_len = len(segment_mean_values)
    adjusted_window = max(2, segment_len // 2) if segment_len < window_stat else window_stat
    adjusted_step = max(1, adjusted_window // 2) if segment_len < window_stat else step_stat

    std_series = sliding_window_std(segment_mean_values, window_struct, step_struct)

    # Grafi A1 e A2
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

    vector = np.array(vector).reshape(1, -1)

    pred = clf.predict(vector)
    emotion = le.inverse_transform(pred)[0]

    return emotion

# Esempio di utilizzo:
#print(predict_emotion("C:/Users/miche/Desktop/1616505554877.wav"))
