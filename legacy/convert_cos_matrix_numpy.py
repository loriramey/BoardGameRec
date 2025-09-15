import pandas as pd
import numpy as np

# Load the CSV-based similarity matrix
cosine_sim_df = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_matrix.csv", index_col=0)

# Convert to NumPy array
cosine_sim_array = cosine_sim_df.to_numpy()

# Save as `.npy`
np.save("/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_matrix.npy", cosine_sim_array)

