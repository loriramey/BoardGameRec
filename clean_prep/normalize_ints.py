import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load cleaned dataset with TD-IDF info already inserted
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf.csv"
df = pd.read_csv(file_path)

# Columns to normalize
columns_to_normalize = ['minplayers', 'maxplayers', 'playingtime']

# Apply Min-Max Scaling
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Save normalized dataset
df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf_normalized.csv", index=False)

# Check results
print(df[columns_to_normalize].describe())  # Should now be in range [0, 1]

# Load both datasets
original_file = "/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf.csv"
normalized_file = "/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf_normalized.csv"

df_original = pd.read_csv(original_file)
df_normalized = pd.read_csv(normalized_file)

# Select only the original columns for min/max players & playtime
original_columns = df_original[['minplayers', 'maxplayers', 'playingtime']]

# Merge the original values back into the normalized dataset
df_final = pd.concat([df_normalized, original_columns], axis=1)

# Save the final dataset
df_final.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv", index=False)

# Verify it worked
print(df_final.head())
