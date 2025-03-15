import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.sparse import hstack

def compute_cosine_similarity(data_file, output_file, tfidf_file):
    """
        Compute and save a weighted cosine similarity matrix for a given dataset.
        :param data_file: Path to the dataset CSV file
        :param output_file: Path to save the cosine similarity matrix (.npy)
        :param tfidf_file: Path to the precomputed TF-IDF vector file (.pkl)
        :return: Cosine similarity matrix as a NumPy array
    """

    print(f"🔄 Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    print(f"🔄 Loading precomputed TF-IDF vectors from: {tfidf_file}")
    with open(tfidf_file, "rb") as f:
        tfidf_mixed_matrix = pickle.load(f)
        # Pre-weighted TF-IDF: 50% mechanics, 30% categories, 20% tags

    # Ensure numeric columns are handled properly / type conversion as needed
    df['maxplayers'] = pd.to_numeric(df['maxplayers'], errors='coerce').fillna(0)
    df['playingtime'] = pd.to_numeric(df['playingtime'], errors='coerce').fillna(0)

    # Use precomputed scaled values from the CSV
    max_players_scaled = np.expand_dims(df['maxplayers_scaled'].values, axis=1)
    playtime_scaled = np.expand_dims(df['playingtime_scaled'].values, axis=1)

    # Convert to numpy arrays for compatibility
    max_players_scaled = np.expand_dims(df['maxplayers_scaled'].values, axis=1)
    playtime_scaled = np.expand_dims(df['playingtime_scaled'].values, axis=1)

    # Apply model weighting to incorporate max player count & max playtime as factors
    WEIGHT_TFIDF = 0.70
    WEIGHT_MAXPLAYERS = 0.10
    WEIGHT_PLAYTIME = 0.20

    # Construct full feature matrix with appropriate weights
    print("🔢 Combining weighted features for similarity computation...")
    feature_matrix = hstack([
        tfidf_mixed_matrix * WEIGHT_TFIDF,
        max_players_scaled * WEIGHT_MAXPLAYERS,
        playtime_scaled * WEIGHT_PLAYTIME
    ])
    # Compute cosine similarity
    print("⚡ Computing cosine similarity...")
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

    # Save as numpy array for faster lookup
    np.save(output_file, cosine_sim)
    print(f"✅ Cosine similarity matrix saved: {output_file}")
    print(f"🟢 Matrix Shape: {cosine_sim.shape}")

    return cosine_sim