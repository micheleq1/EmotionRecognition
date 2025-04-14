import pandas as pd
import librosa
import numpy as np
import cv2
import warnings
from functions import sliding_window_std, ottimizza_parametri_sliding

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("🚀 Inizio segmentazione...")

df = pd.read_csv("train_updated.csv")
audio_list = []
segment_means = []
segment_std_series = []

best_struct, best_stat = ottimizza_parametri_sliding(df)

for index, row in df.iterrows():
    file_path = row["file_path"]
    audio, sr = librosa.load(file_path, sr=None)

    if len(audio) < 1000:
        print(f"Audio {file_path} troppo corto. Skippato.")
        continue

    signal_scaled = ((audio - np.min(audio)) / (np.max(audio) - np.min(audio)) * 255).astype(np.uint8)
    image = np.tile(signal_scaled, (5, 1))

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges_canny = cv2.Canny(blurred_image, 100, 200)
    combined_edges = edges_canny

    edge_positions = np.where(combined_edges[0, :] > 0)[0]
    segments = np.split(audio, edge_positions) if len(edge_positions) > 0 else [audio]
    segments = [seg for seg in segments if len(seg) > 500]

    segment_mean_values = [np.mean(seg) for seg in segments]
    segment_means.append(segment_mean_values)

    segment_len = len(segment_mean_values)
    adjusted_window = max(2, segment_len // 2) if segment_len < best_struct[0] else best_struct[0]
    adjusted_step = max(1, adjusted_window // 2) if segment_len < best_struct[0] else best_struct[1]

    std_series = sliding_window_std(segment_mean_values, adjusted_window, adjusted_step)
    segment_std_series.append(std_series)

# Salva segmentazione correttamente
np.save("segment_means.npy", np.array(segment_means, dtype=object))
np.save("segment_std_series.npy", np.array(segment_std_series, dtype=object))

print("✅ Segmentazione completata e salvata.")
