#This process should be run on clean game data in .csv format
#This script will analyze the lists of tags, mechs, and categories and generate TF IDF tokens, stores .pkl files

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# === Load game data ===
df = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv")

from clean_prep.correct_TFIDF_tokens import clean_token_string

# Ensure base tag columns are converted to token strings if not already
df["tags_str"] = df["tags"].apply(clean_token_string)
df["mechanics_str"] = df["boardgamemechanic"].apply(clean_token_string)
df["categories_str"] = df["boardgamecategory"].apply(clean_token_string)

# === Ensure columns exist and are cleaned ===
df["tags_str"] = df["tags_str"].fillna("")
df["categories_str"] = df["categories_str"].fillna("")
df["mechanics_str"] = df["mechanics_str"].fillna("")

# === TF-IDF for Tags (12 unique tags) ===
vectorizer_tags = TfidfVectorizer()
X_tags = vectorizer_tags.fit_transform(df["tags_str"])

output_tags = {
    "matrix": X_tags,
    "vectorizer": vectorizer_tags,
    "game_ids": df["id"].tolist(),
}
with open("../models_old/tfidf_tags.pkl", "wb") as f:
    pickle.dump(output_tags, f)

# === TF-IDF for Mechanics (~90 unique) ===
vectorizer_mech = TfidfVectorizer(max_features=90)
X_mech = vectorizer_mech.fit_transform(df["mechanics_str"])

output_mech = {
    "matrix": X_mech,
    "vectorizer": vectorizer_mech,
    "game_ids": df["id"].tolist(),
}
with open("../models_old/tfidf_mechanics.pkl", "wb") as f:
    pickle.dump(output_mech, f)

# === TF-IDF for Categories (~190 unique) ===
vectorizer_cat = TfidfVectorizer(max_features=200)
X_cat = vectorizer_cat.fit_transform(df["categories_str"])

output_cat = {
    "matrix": X_cat,
    "vectorizer": vectorizer_cat,
    "game_ids": df["id"].tolist(),
}
with open("../models_old/tfidf_categories.pkl", "wb") as f:
    pickle.dump(output_cat, f)

print("TF-IDF vectorization complete. Files saved to /models_old.")