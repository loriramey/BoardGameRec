
import pandas as pd

# Load the dataset
DATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
df = pd.read_csv(DATA_FILE)

# Ensure 'averageweight' is numeric
df['averageweight'] = pd.to_numeric(df['averageweight'], errors='coerce').fillna(0)

# Normalize "average weight" (scaled between 0 and 1)
df['averageweight_scaled'] = (df['averageweight'] - df['averageweight'].min()) / (
    df['averageweight'].max() - df['averageweight'].min()
)

# Save the updated dataset
UPDATED_DATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
df.to_csv(UPDATED_DATA_FILE, index=False)

print(f"Updated dataset saved with 'averageweight_scaled' at: {UPDATED_DATA_FILE}")

import pickle
import numpy as np

# Load the updated dataset
UPDATED_DATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf.csv"
df = pd.read_csv(UPDATED_DATA_FILE)

# Ensure 'averageweight_scaled' exists and is numeric
if 'averageweight_scaled' not in df.columns:
    raise ValueError("❌ Column 'averageweight_scaled' not found. Make sure dataset was updated.")

# Convert to NumPy array
averageweight_scaled = np.expand_dims(df['averageweight_scaled'].values, axis=1)

# Save as a pickle file
PICKLE_FILE = "/models/norm_avg_weight.pkl"
with open(PICKLE_FILE, "wb") as f:
    pickle.dump(averageweight_scaled, f)

print(f"✅ 'norm_avg_weight.pkl' saved at: {PICKLE_FILE}")