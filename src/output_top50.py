#This script extracts the top 50 matching games based on each of the available CS matrix files
#and stores those games in a dataframe as .parquet file
#for use in the Streamlit app (which cannot use the giant .npy CS matrix files)

import pyarrow
import pandas as pd
import numpy as np
import os

def extract_top_50(sim_matrix, game_ids):
    """
    Compute and save a set of the top-50 similar games for any given game id

    Parameters:
    - sim_matrix (pandas.DataFrame): Similarity matrix
    - game_ids (list): List of game ids

    Returns:
    - dataframe of top 50 matches labeled by game id for use as static lookup table in app
    """
    top_matches = []
    for idx, row in enumerate(sim_matrix):
        # Get top 50 indices and scores (excluding self-match)
        sim_scores = list(enumerate(row))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [entry for entry in sim_scores if entry[0] != idx][:50]
        for sim_idx, score in sim_scores:
            top_matches.append({
                "base_game_id": game_ids[idx],
                "similar_game_id": game_ids[sim_idx],
                "similarity_score": score
            })
    return pd.DataFrame(top_matches)


# Load base data and game IDs
gamedata_path = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
df = pd.read_csv(gamedata_path)
# Also save the main game data as a Parquet file for faster loading in the app
gamedata_parquet_path = gamedata_path.replace(".csv", ".parquet")
df.to_parquet(gamedata_parquet_path, index=False, engine="pyarrow")
print(f"Saved: {gamedata_parquet_path} ({df.shape[0]} rows)")

game_ids = df["id"].tolist()

matrix_names = {
    "mech_heavy": "cosine_similarity_mech_heavy.npy",
    "cat_tag_heavy": "cosine_similarity_cat_heavy.npy",
    "mixed": "cosine_similarity_mixed.npy"
}

for label, fname in matrix_names.items():
    print(f"Processing {label}...")
    path = os.path.join("/Users/loriramey/PycharmProjects/BGapp/data", fname)
    sim_matrix = np.load(path)
    top_df = extract_top_50(sim_matrix, game_ids)
    parquet_path = os.path.join("/Users/loriramey/PycharmProjects/BGapp/data", f"top_50_matches_{label}.parquet")
    top_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"Saved: {parquet_path} ({top_df.shape[0]} rows)")