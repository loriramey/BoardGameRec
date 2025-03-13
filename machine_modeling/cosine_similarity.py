import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

file_path = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv"
df = pd.read_csv(file_path)

#IF-IDF vectorize Tags, Cats, Mechs stored TF-IDF data
tfidf = TfidfVectorizer(stop_words='english')
#combine all 3 vectors for processing
df['combo_features'] = df['tags_str'] + " " + df['categories_str'] + " " + df['mechanics_str']
tfidf_matrix = tfidf.fit_transform(df['combo_features'])
#pull in normalized data like min & max players, playtime
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['minplayers_scaled', 'maxplayers_scaled', 'playingtime_scaled']])
#create a single dataframe with all factors
tfidf_dense = tfidf_matrix.toarray()
feature_matrix = np.hstack((tfidf_dense, scaled_features))

#compute cosine similarity on these combined features
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

#convert to dataframe for fast lookups, save and inspect shape
cosine_sim_df = pd.DataFrame(cosine_sim, index=df['name'], columns=df['name'])
cosine_sim_df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_matrix.csv")
print("Cosine Similarity Matrix Shape:", cosine_sim_df.shape)
