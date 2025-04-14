import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from functions import visibility_graph, sliding_window_statistical
import warnings
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("🚀 Inizio estrazione feature finale...")

df = pd.read_csv("train_updated.csv")

# Carico segmentazioni già fatte
segment_means = np.load("segment_means.npy", allow_pickle=True)
segment_std_series = np.load("segment_std_series.npy", allow_pickle=True)

# Parametri sliding window FISSI (paper)
window_struct = 400
step_struct = 200
window_stat = 8000
step_stat = 6000

# Costruzione A1
adjacency_matrices_A1 = []
for series in segment_std_series:
    if len(series) >= 3:
        VG, adj = visibility_graph(series)
        adjacency_matrices_A1.append(adj)
    else:
        adjacency_matrices_A1.append(None)

# Costruzione A2
adjacency_matrices_A2 = []
for segment_mean_values in segment_means:
    segment_len = len(segment_mean_values)
    adjusted_window = max(2, segment_len // 2) if segment_len < window_stat else window_stat
    adjusted_step = max(1, adjusted_window // 2) if segment_len < window_stat else step_stat
    A2 = sliding_window_statistical(segment_mean_values, adjusted_window, adjusted_step)
    adjacency_matrices_A2.append(A2)

# Estrazione feature grafiche
def estrai_feature(adj, is_structural):
    if adj is None:
        return {}

    G = nx.from_numpy_array(adj)
    features = {}

    degrees = [d for _, d in G.degree()]
    features['DoC'] = np.mean(degrees)
    features['CC'] = nx.average_clustering(G)
    features['D'] = nx.density(G)
    features['M'] = np.mean(adj)

    try:
        partition = louvain_communities(G)
        features['Q'] = modularity(G, partition)
    except:
        features['Q'] = 0

    if is_structural:
        try:
            eigenvalues = np.linalg.eigvals(adj)
            features['E'] = np.sum(np.square(eigenvalues))
        except:
            features['E'] = 0

    return features

# Parallelizzazione limitata
features_A1 = Parallel(n_jobs=4, verbose=10)(
    delayed(estrai_feature)(adj, True) for adj in adjacency_matrices_A1
)

features_A2 = Parallel(n_jobs=4, verbose=10)(
    delayed(estrai_feature)(adj, False) for adj in adjacency_matrices_A2
)

np.save("features_A1.npy", features_A1)
np.save("features_A2.npy", features_A2)

print("✅ Estrazione feature completata e salvata.")
