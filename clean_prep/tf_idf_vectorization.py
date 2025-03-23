#updated March 22, 2025 to handle average weight too
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle  # To save TF-IDF matrices for reuse
from scipy.sparse import hstack

# Load cleaned dataset
datafile = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted.csv"
df = pd.read_csv(datafile)

# Ensure there are no NaN values in text columns
df["tags_str"] = df["tags_str"].fillna("")
df["categories_str"] = df["categories_str"].fillna("")
df["mechanics_str"] = df["mechanics_str"].fillna("")

# Ensure game id column "id" and "averageweight_scaled" is numeric
df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
df["averageweight_scaled"] = pd.to_numeric(df["averageweight_scaled"], errors="coerce").fillna(0)

# Initialize TF-IDF vectorizers (using 1-word and 2-word phrases)
tfidf_tags = TfidfVectorizer(ngram_range=(1, 2))
tfidf_categories = TfidfVectorizer(ngram_range=(1, 2))
tfidf_mechanics = TfidfVectorizer(ngram_range=(1, 2))

# Fit and transform each column separately
tfidf_tags_matrix = tfidf_tags.fit_transform(df["tags_str"])
tfidf_categories_matrix = tfidf_categories.fit_transform(df["categories_str"])
tfidf_mechanics_matrix = tfidf_mechanics.fit_transform(df["mechanics_str"])

# Print shape to verify correct processing
print("TF-IDF Tags Shape:", tfidf_tags_matrix.shape)
print("TF-IDF Categories Shape:", tfidf_categories_matrix.shape)
print("TF-IDF Mechanics Shape:", tfidf_mechanics_matrix.shape)

# Convert sparse matricies to dictionaries {id: vector row}
tfidf_tags_dict = {game_id: tfidf_tags_matrix[idx] for idx, game_id in enumerate(df["id"])}
tfidf_categories_dict = {game_id: tfidf_categories_matrix[idx] for idx, game_id in enumerate(df["id"])}
tfidf_mechanics_dict = {game_id: tfidf_mechanics_matrix[idx] for idx, game_id in enumerate(df["id"])}

# Save the TF-IDF matrices as dictionaries using .pkl
with open("../models/tfidf_tags.pkl", "wb") as f:
    pickle.dump(tfidf_tags_dict, f)

with open("../models/tfidf_categories.pkl", "wb") as f:
    pickle.dump(tfidf_categories_dict, f)

with open("../models/tfidf_mechanics.pkl", "wb") as f:
    pickle.dump(tfidf_mechanics_dict, f)

# Save the vectorizers too (for decoding later)
with open("../models/tfidf_tags_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_tags, f)

with open("../models/tfidf_categories_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_categories, f)

with open("../models/tfidf_mechanics_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_mechanics, f)

averageweight_dict = {game_id: weight for game_id, weight in zip(df["id"], df["averageweight_scaled"])}
with open("../models/norm_avg_weight.pkl", "wb") as f:
    pickle.dump(averageweight_dict, f)

# Apply weights
tfidf_tags_matrix *= 0.20
tfidf_categories_matrix *= 0.30
tfidf_mechanics_matrix *= 0.50

# Stack into a final mixed matrix
final_tfidf_matrix = hstack([tfidf_tags_matrix, tfidf_categories_matrix, tfidf_mechanics_matrix])

# Save it for quick lookups
with open("../models/tfidf_mixed.pkl", "wb") as f:
    pickle.dump(final_tfidf_matrix, f)
    print("Saved mixed vectorization")

print("🎉 All .pkl files updated with Game IDs for better data alignment!")