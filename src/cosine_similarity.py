import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.sparse import hstack, vstack


#helper function to handle the pkl dictionary files
def load_tfidf_matrix(file_path, df):
    """
    Load a TF-IDF object from file_path. Supports new dictionary format with keys:
    - 'matrix': TF-IDF sparse matrix
    - 'game_ids': list of game IDs matching matrix rows
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "matrix" in obj and "game_ids" in obj:
        matrix = obj["matrix"]
        id_to_index = {game_id: idx for idx, game_id in enumerate(obj["game_ids"])}
        indices = [id_to_index[game_id] for game_id in df["id"]]
        return matrix[indices]

    return obj

def get_weighted_feature_matrix(df, tfidf_tags_matrix, tfidf_categories_matrix, tfidf_mechanics_matrix,
                                max_players_scaled, playtime_scaled, avg_weight_scaled, weights):
    return hstack([
        tfidf_tags_matrix * weights["tags"],
        tfidf_categories_matrix * weights["categories"],
        tfidf_mechanics_matrix * weights["mechanics"],
        max_players_scaled * weights["maxplayers"],
        playtime_scaled * weights["playtime"],
        avg_weight_scaled * weights["avgweight"]
    ])

def compute_cosine_similarity(data_file, output_file, tfidf_files):
    """
    Compute and save a weighted cosine similarity matrix for a given dataset.

    Parameters:
    - data_file (str): Path to the dataset CSV file
    - output_file (str): Path to save the cosine similarity matrix (.npy)
    - tfidf_files (dict): Dictionary of TF-IDF .pkl paths:
        {
            'tags': path,
            'categories': path,
            'mechanics': path
        }

    Returns:
    - cosine_sim (np.ndarray): Cosine similarity matrix
    """

    #load all necessary files and put data in usable form for this function
    print(f"🔄 Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    print("🔄 Loading precomputed TF-IDF vectors...")
    tfidf_tags_matrix = load_tfidf_matrix(tfidf_files['tags'], df)
    tfidf_categories_matrix = load_tfidf_matrix(tfidf_files['categories'], df)
    tfidf_mechanics_matrix = load_tfidf_matrix(tfidf_files['mechanics'], df)

    expected_rows = df.shape[0]
    assert tfidf_tags_matrix.shape[0] == expected_rows, "Mismatch in TF-IDF tags row count"
    assert tfidf_categories_matrix.shape[0] == expected_rows, "Mismatch in TF-IDF categories row count"
    assert tfidf_mechanics_matrix.shape[0] == expected_rows, "Mismatch in TF-IDF mechanics row count"

    # Ensure scaled numeric columns exist and are numeric
    df['maxplayers_scaled'] = pd.to_numeric(df['maxplayers_scaled'], errors='coerce').fillna(0)
    df['playingtime_scaled'] = pd.to_numeric(df['playingtime_scaled'], errors='coerce').fillna(0)
    df['averageweight_scaled'] = pd.to_numeric(df['averageweight_scaled'], errors='coerce').fillna(0)

    # Expand numeric vectors to match TF-IDF sparse matrix format
    max_players_scaled = np.expand_dims(df['maxplayers_scaled'].values, axis=1)
    playtime_scaled = np.expand_dims(df['playingtime_scaled'].values, axis=1)
    avg_weight_scaled = np.expand_dims(df['averageweight_scaled'].values, axis=1)

    RECIPES = {
        "mech_heavy": {
            "tags": 0.05, "categories": 0.20, "mechanics": 0.45,
            "maxplayers": 0.05, "playtime": 0.05, "avgweight": 0.20
        },
        "cat_tag_heavy": {
            "tags": 0.15, "categories": 0.35, "mechanics": 0.25,
            "maxplayers": 0.05, "playtime": 0.05, "avgweight": 0.15
        },
        "mixed": {
            "tags": 0.10, "categories": 0.20, "mechanics": 0.30,
            "maxplayers": 0.05, "playtime": 0.10, "avgweight": 0.25
        }
    }

    for label, weights in RECIPES.items():
        print(f"🧪 Running recipe: {label}")
        feature_matrix = get_weighted_feature_matrix(
            df, tfidf_tags_matrix, tfidf_categories_matrix, tfidf_mechanics_matrix,
            max_players_scaled, playtime_scaled, avg_weight_scaled, weights
        )
        cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
        output_path = output_file.replace(".npy", f"_{label}.npy")
        np.save(output_path, cosine_sim)
        print(f"✅ Saved matrix for '{label}' at {output_path}")
        print(f"🟢 Shape: {cosine_sim.shape}")

#to run this function:
if __name__ == "__main__":

    data_file = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
    base_output_path = "/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity.npy"
    tfidf_files = {
        "tags": "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_tags.pkl",
        "categories": "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_categories.pkl",
        "mechanics": "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_mechanics.pkl"
    }

    compute_cosine_similarity(data_file, base_output_path, tfidf_files)
