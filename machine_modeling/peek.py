import pandas as pd

# Update with your actual path
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf.csv"

# Load dataset
# Attempt to load using a different encoding
df = pd.read_csv(file_path, encoding="ISO-8859-1")  # or use "latin-1"

print(df.head())  # Check if data loads correctly

# Calculate min/max values
min_max_values = df[['minplayers', 'maxplayers', 'playingtime']].agg(['min', 'max'])

# Print results
print(min_max_values)
df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf_cleaned_utf8.csv", index=False, encoding="utf-8")