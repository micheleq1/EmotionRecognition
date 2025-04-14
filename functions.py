import librosa
from tqdm import tqdm

# Funzione per calcolare la deviazione standard con finestra scorrevole
def sliding_window_std(time_series, window_size, step):
    std_values = []
    if len(time_series) < window_size:
        return np.array([])  # oppure np.std(time_series) se vuoi almeno 1 valore
    if window_size == 1:
        return np.zeros(len(time_series))
    for i in range(0, len(time_series) - window_size + 1, step):
        window = time_series[i:i + window_size]
        std_values.append(np.std(window))
    return np.array(std_values)



# Funzione per creare il Visibility Graph
def visibility_graph(time_series):
    n = len(time_series)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, value=time_series[i])

    for i in range(n - 1):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                x1, n1 = time_series[i], i
                x2, n2 = time_series[j], j
                x3, n3 = time_series[k], k
                if x3 >= x1 + (x2 - x1) * (n3 - n1) / (n2 - n1):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)

    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    return G, adjacency_matrix

#Suddivide un segnale in segmenti sovrapposti e calcola direttamente la matrice di adiacenza A₂ basata sulla correlazione di Pearson.
def sliding_window_statistical(signal, window_size, step):
    """
    :param signal: Array NumPy con il segnale
    :param window_size: Numero di campioni per ogni segmento
    :param step: sovrapposizione tra le finestre
    :return: Matrice di correlazione assoluta (A₂) o None se i segmenti non sono validi
    """
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segment = signal[start:start + window_size]
        segments.append(segment)

    if len(segments) < 2:
        #print("⚠️ Warning: Meno di 2 segmenti disponibili. Skipping A₂.")
        return None  # Evita il calcolo se ci sono meno di 2 segmenti

    segments = np.array(segments)

    # Controlla se ci sono segmenti con varianza zero
    std_values = np.std(segments, axis=1)
    if np.any(std_values == 0):
        #print("⚠️ Warning: Segmenti costanti rilevati. Skipping A₂.")
        return None  # Evita il calcolo della matrice di correlazione

    # Calcola la matrice di correlazione di Pearson
    correlation_matrix = np.corrcoef(segments)

    # Sostituisci NaN con 0 (se si verificano problemi numerici)
    correlation_matrix = np.nan_to_num(correlation_matrix)

    return np.abs(correlation_matrix)  # Restituisce solo valori positivi


import numpy as np
import networkx as nx
import community as community_louvain  # per modularity Louvain
#Estrae le feature da una matrice di adiacenza (A1 o A2).
def extract_graph_features(adj_matrix, is_structural=True):
    """
    :param adj_matrix: Matrice di adiacenza (numpy array)
    :param is_structural: True per A1 (Visibility Graph), False per A2 (Statistical Graph)
    :return: Dizionario con le feature estratte
    """

    G = nx.from_numpy_array(adj_matrix)

    # Se il grafo è vuoto o troppo piccolo metto feature a 0
    if len(G) == 0 or G.number_of_edges() == 0:
        features = {key: 0 for key in ['DoC', 'CC', 'D', 'M', 'Q']}
        if is_structural:
            features['E'] = 0
        return features

    features = {}

    # Degree of Connectivity
    degrees = [d for _, d in G.degree()]
    features['DoC'] = np.mean(degrees)

    # Clustering Coefficient
    features['CC'] = nx.average_clustering(G)

    # Density
    features['D'] = nx.density(G)

    # Media valori adiacenza
    features['M'] = np.mean(adj_matrix)

    # Modularity (con Louvain)
    try:
        partition = community_louvain.best_partition(G)
        features['Q'] = community_louvain.modularity(partition, G)
    except:
        features['Q'] = 0

    # Energy Measure solo per A1
    if is_structural:
        try:
            eigenvalues = np.linalg.eigvals(adj_matrix)
            features['E'] = np.sum(np.square(eigenvalues))
        except:
            features['E'] = 0

    return features




#Trova la combinazione ottimale di window e step per approccio strutturale e statistico.
def ottimizza_parametri_sliding(df, sr=16000):
    """
    Restituisce una tupla: (best_structural, best_statistical)
    """

    # --- PARAMETRI DA TESTARE ---
    parametri_statistici = [
        (2000, 1000),
        (4000, 2000),
        (6000, 4000),
        (8000, 6000),
        (12000, 6000),
        (16000, 8000)
    ]

    parametri_strutturali = [
        (50, 25),
        (100, 50),
        (150, 75),
        (200, 100),
        (300, 150),
        (400, 200)
    ]

    risultati_stat = []
    risultati_struct = []

    print("🔍 Analisi parametri per l'approccio statistico...")
    for w, s in parametri_statistici:
        conteggi = []
        for path in tqdm(df["file_path"]):
            try:
                audio, _ = librosa.load(path, sr=sr)
                n_seg = 1 + max(0, (len(audio) - w) // s)
                conteggi.append(n_seg)
            except Exception:
                pass
        media = np.mean(conteggi)
        std = np.std(conteggi)
        risultati_stat.append({"tipo": "statistico", "window": w, "step": s, "media": media, "std": std})

    print("\n🔍 Analisi parametri per l'approccio strutturale...")
    for w, s in parametri_strutturali:
        conteggi = []
        for path in tqdm(df["file_path"]):
            try:
                audio, _ = librosa.load(path, sr=sr)
                n_seg = 1 + max(0, (len(audio) - w) // s)
                conteggi.append(n_seg)
            except Exception:
                pass
        media = np.mean(conteggi)
        std = np.std(conteggi)
        risultati_struct.append({"tipo": "strutturale", "window": w, "step": s, "media": media, "std": std})

    # --- SELEZIONE MIGLIORE ---
    def seleziona_migliore(risultati):
        candidati = [r for r in risultati if r["media"] >= 2]
        if not candidati:
            return max(risultati, key=lambda r: r["media"])  # fallback
        candidati.sort(key=lambda r: (abs(r["media"] - 10), r["std"]))  # preferisci media ~10 e std bassa
        return candidati[0]

    best_stat = seleziona_migliore(risultati_stat)
    best_struct = seleziona_migliore(risultati_struct)

    print("\n✅ Parametri selezionati:")
    print(f" - Statistico  (A₂): window={best_stat['window']}, step={best_stat['step']} (media segmenti = {best_stat['media']:.2f})")
    print(f" - Strutturale (A₁): window={best_struct['window']}, step={best_struct['step']} (media segmenti = {best_struct['media']:.2f})")
    print("\n📊 Risultati strutturali:")
    for r in risultati_struct:
        print(f"window={r['window']:>5}, step={r['step']:>5}, media={r['media']:.2f}, std={r['std']:.2f}")

    print("\n📊 Risultati statistici:")
    for r in risultati_stat:
        print(f"window={r['window']:>5}, step={r['step']:>5}, media={r['media']:.2f}, std={r['std']:.2f}")

    return (best_struct["window"], best_struct["step"]), (best_stat["window"], best_stat["step"])
