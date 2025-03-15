import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# 📌 Paths
TRAIN_DATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_train.csv"
TFIDF_TRAIN_FILE = "/Users/loriramey/PycharmProjects/BGapp/models/tfidf_train.pkl"

# 🔄 Load training dataset
print(f"🔄 Loading training data from: {TRAIN_DATA_FILE}")
df_train = pd.read_csv(TRAIN_DATA_FILE)

# Ensure text columns have no NaN values
df_train["tags_str"] = df_train["tags_str"].fillna("")
df_train["categories_str"] = df_train["categories_str"].fillna("")
df_train["mechanics_str"] = df_train["mechanics_str"].fillna("")

# 🔢 Initialize TF-IDF vectorizers
tfidf_tags = TfidfVectorizer(ngram_range=(1, 2))
tfidf_categories = TfidfVectorizer(ngram_range=(1, 2))
tfidf_mechanics = TfidfVectorizer(ngram_range=(1, 2))

# 🎯 Fit and transform each column separately (only on TRAINING data)
tfidf_tags_matrix = tfidf_tags.fit_transform(df_train["tags_str"])
tfidf_categories_matrix = tfidf_categories.fit_transform(df_train["categories_str"])
tfidf_mechanics_matrix = tfidf_mechanics.fit_transform(df_train["mechanics_str"])

# 🏗️ Apply weights to each matrix
WEIGHT_TAGS = 0.50
WEIGHT_CATEGORIES = 0.30
WEIGHT_MECHANICS = 0.20

tfidf_tags_matrix *= WEIGHT_TAGS
tfidf_categories_matrix *= WEIGHT_CATEGORIES
tfidf_mechanics_matrix *= WEIGHT_MECHANICS

# 🛠️ Stack into a single TF-IDF matrix
tfidf_train_matrix = hstack([tfidf_tags_matrix, tfidf_categories_matrix, tfidf_mechanics_matrix])

# 💾 Save the new TF-IDF training set
with open(TFIDF_TRAIN_FILE, "wb") as f:
    pickle.dump(tfidf_train_matrix, f)

print(f"✅ TF-IDF for training data saved: {TFIDF_TRAIN_FILE}")
print(f"🔢 Matrix shape: {tfidf_train_matrix.shape}")