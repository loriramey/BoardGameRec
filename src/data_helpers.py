import numpy as np
import pandas as pd

# Define file paths
GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted_cleaned.csv"
COSINE_SIM_FILE = "/data/cosine_similarity_origrecipe.npy"

# Load gamedata.csv once and assume its order matches the CS matrix
df = pd.read_csv(GAMEDATA_FILE)

# Create an ID-to-index mapping assuming the CS matrix rows correspond to the order of the CSV
id_to_index = {game_id: idx for idx, game_id in enumerate(df['id'])}

# Load the precomputed cosine similarity matrix
cs_matrix = np.load(COSINE_SIM_FILE)

def get_similarity(game_id1: int, game_id2: int) -> float:
    """
    Return the cosine similarity between two games based on their IDs
    using the precomputed CS matrix.
    """
    if game_id1 not in id_to_index or game_id2 not in id_to_index:
        raise ValueError("One or both game IDs not found in the dataset.")
    idx1 = id_to_index[game_id1]
    idx2 = id_to_index[game_id2]
    return cs_matrix[idx1, idx2]

def get_game_data(game_id: int) -> pd.Series:
    """
    Return the game data row from gamedata.csv for the given game ID.
    """
    if game_id not in id_to_index:
        raise ValueError("Game ID not found in the dataset.")
    # This assumes that 'id' uniquely identifies a game in df.
    return df.loc[df['id'] == game_id].iloc[0]


