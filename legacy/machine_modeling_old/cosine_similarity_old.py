import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

#load game data file and the pre-computed TF-IDF tags/categories/mechanics vectors
GAMEDATA_FILE = "/data/gamedata_final.csv"
TFIDF_MIXED_FILE = "/models_old/tfidf_mixed.pkl"

df = pd.read_csv(GAMEDATA_FILE)
with open(TFIDF_MIXED_FILE, "rb") as f:
    tfidf_mixed_matrix = pickle.load(f)  #40% tags, 30% categories, 30% mechanics weights

#extract the scaled min/max vectors for maxplayers, playingtime
max_players_scaled = np.expand_dims(df['maxplayers_scaled'].values, axis = 1)
playtime_scaled = np.expand_dims(df['playingtime_scaled'].values, axis = 1)

#overall weighting in the model for similarity matches
WEIGHT_TFIDF = 0.70
WEIGHT_MAXPLAYERS = 0.10
WEIGHT_PLAYTIME = 0.20

#create a single dataframe with all factors
feature_matrix = hstack([
    tfidf_mixed_matrix * WEIGHT_TFIDF,
    max_players_scaled * WEIGHT_MAXPLAYERS,
    playtime_scaled * WEIGHT_PLAYTIME
])

#compute cosine similarity on these combined features
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

#convert to dataframe for faster lookups, save as numpy for faster retrieval
COSINE_SIM_FILE = "/data/cosine_similarity_origrecipe.npy"
np.save(COSINE_SIM_FILE, cosine_sim)

print("Cosine Similarity Matrix Shape:", cosine_sim.shape)
