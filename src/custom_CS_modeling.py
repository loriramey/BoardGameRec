import numpy as np
import pandas as pd
import pickle
from scipy.sparse import vstack, hstack
from sklearn.metrics.pairwise import cosine_similarity

# File paths
GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv"
TFIDF_TAGS_FILE = "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_tags.pkl"
TFIDF_CATEGORIES_FILE = "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_categories.pkl"
TFIDF_MECHANICS_FILE = "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_mechanics.pkl"

# Load master data from gamedata.csv
df = pd.read_csv(GAMEDATA_FILE)


def load_tfidf_matrix(file_path, df):
    """
    Load a TF-IDF object from file_path. If it is a dictionary,
    rebuild a sparse matrix by ordering the vectors based on the 'id' column in df.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    # If the loaded object is a dict, convert it into a sparse matrix.
    if isinstance(obj, dict):
        # Rebuild a list of sparse vectors in the order of df['id']
        tfidf_list = [obj[game_id] for game_id in df['id']]
        return vstack(tfidf_list)
    else:
        return obj


def compute_custom_cs(weights):
    """
    Compute a cosine similarity matrix on the fly using the provided weights.
    'weights' is a dict with keys:
      'tags', 'categories', 'mechanics', 'maxplayers', 'playingtime', 'averageweight'
    This function assumes that:
      - The TF-IDF pickle files have been generated and are stored as dictionaries.
      - The numeric columns in df are normalized and available as:
            'maxplayers_scaled', 'playingtime_scaled', 'averageweight_scaled'
    """
    # Load the TF-IDF matrices from pickle files
    tfidf_tags_matrix = load_tfidf_matrix(TFIDF_TAGS_FILE, df)
    tfidf_categories_matrix = load_tfidf_matrix(TFIDF_CATEGORIES_FILE, df)
    tfidf_mechanics_matrix = load_tfidf_matrix(TFIDF_MECHANICS_FILE, df)

    # Expand numeric features from df (these should be precomputed normalized columns)
    max_players_scaled = np.expand_dims(df['maxplayers_scaled'].values, axis=1)
    playtime_scaled = np.expand_dims(df['playingtime_scaled'].values, axis=1)
    avg_weight_scaled = np.expand_dims(df['averageweight_scaled'].values, axis=1)

    # Multiply each feature set by its respective weight
    weighted_tags = tfidf_tags_matrix * weights['tags']
    weighted_categories = tfidf_categories_matrix * weights['categories']
    weighted_mechanics = tfidf_mechanics_matrix * weights['mechanics']
    weighted_maxplayers = max_players_scaled * weights['maxplayers']
    weighted_playtime = playtime_scaled * weights['playingtime']
    weighted_avgweight = avg_weight_scaled * weights['averageweight']

    # Combine all weighted features horizontally
    feature_matrix = hstack([weighted_tags, weighted_categories, weighted_mechanics,
                             weighted_maxplayers, weighted_playtime, weighted_avgweight])

    # Compute cosine similarity on the combined feature matrix
    custom_cs = cosine_similarity(feature_matrix, feature_matrix)
    return custom_cs


if __name__ == "__main__":
    # Define a new weight recipe; adjust these values as desired.
    new_weights = {
        'tags': 0.15,
        'categories': 0.10,
        'mechanics': 0.35,
        'maxplayers': 0.10,
        'playingtime': 0.15,
        'averageweight': 0.15
    }
    # Compute the custom cosine similarity matrix using the new recipe
    custom_cs_matrix = compute_custom_cs(new_weights)
    print("Custom cosine similarity matrix shape:", custom_cs_matrix.shape)