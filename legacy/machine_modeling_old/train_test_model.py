import pandas as pd
import numpy as np
import pickle
from src.cosine_similarity import compute_cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------TRAINING on 80% of data-----------------------
# Define file paths
TRAIN_DATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_train.csv"
TRAIN_COSINE_SIM_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_train.npy"
TFIDF_TRAIN_FILE = "/legacy/models_old/tfidf_train.pkl"

# Compute cosine similarity for training set using helper function
cosine_sim_train = compute_cosine_similarity(  # Capture returned matrix here
    data_file=TRAIN_DATA_FILE,
    output_file=TRAIN_COSINE_SIM_FILE,
    tfidf_file=TFIDF_TRAIN_FILE
)

# Save the trained similarity matrix
np.save(TRAIN_COSINE_SIM_FILE, cosine_sim_train)
print(f"✅ Cosine similarity matrix saved: {TRAIN_COSINE_SIM_FILE}")

# -----------------------TESTING with 20% of data-----------------------
# Load datasets - test file, set up data frames with train & test data & CS matrix
TEST_DATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_test.csv"
test_df = pd.read_csv(TEST_DATA_FILE)
train_df = pd.read_csv(TRAIN_DATA_FILE)
cosine_sim_train = np.load(TRAIN_COSINE_SIM_FILE)

# Function to evaluate recommendations
def evaluate_recommendations(test_df, train_df, cosine_sim_matrix, top_n=10):
    """
    Evaluate the model by checking if the recommended games appear in the test dataset.

    :param test_df: The test dataset containing actual game names.
    :param train_df: The training dataset used to build the similarity matrix.
    :param cosine_sim_matrix: The precomputed cosine similarity matrix.
    :param top_n: Number of recommendations to consider for each game.
    """
    total_games = len(test_df)
    evaluated_games = 0  # Counter for how many games were actually evaluated
    success_count = 0

    #convert stored game id data to integers
    train_df['id'] = pd.to_numeric(train_df['id'], errors='coerce')
    train_df = train_df.dropna(subset=['id'])  # Drop rows where 'id' is NaN
    train_df['id'] = train_df['id'].astype(int)  # Ensure IDs are integers

    test_df['id'] = pd.to_numeric(test_df['id'], errors='coerce')
    test_df = test_df.dropna(subset=['id'])  # Drop rows where 'id' is NaN
    test_df['id'] = test_df['id'].astype(int)  # Ensure IDs are integers

    #DEBUG issue of no games showing!
    print(f"🔍 First 10 train game IDs: {train_df['id'].head(10).tolist()}")
    print(f"🔍 First 10 test game IDs: {test_df['id'].head(10).tolist()}")
    print(f"🧐 Train ID Type: {train_df['id'].dtype}")
    print(f"🧐 Test ID Type: {test_df['id'].dtype}")
    overlap_count = sum(test_df['id'].isin(train_df['id']))
    print(f"🎯 Games in test set that exist in training: {overlap_count} out of {len(test_df)}")

    # Build a quick lookup from game name to index (training set)
    train_game_index = {game_id: idx for idx, game_id in enumerate(train_df['id'])}

    for _, row in test_df.iterrows():  #map game ids to indexes in the training set
        game_id = row['id']
        if game_id not in train_game_index:
            continue  # Skip if game ID isn't in training set
        evaluated_games += 1

        # Get the index of the game in the training set
        game_idx = train_game_index[game_id]

        # generate top 10 recommendations from similarity matrix & grab names from training set
        sim_scores = list(enumerate(cosine_sim_matrix[game_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        recommended_ids = [train_df.iloc[i[0]]['id'] for i in sim_scores if i[0] < len(train_df)]

        # Check if at least one recommendation exists in test set
        if any(game_id in test_df['id'].values for game_id in recommended_ids):
            success_count += 1

    #avoid div by zero
    accuracy = success_count / evaluated_games if evaluated_games > 0 else 0

    print(f"🔍 Total test games: {total_games}")
    print(f"✅ Games evaluated: {evaluated_games}")  # New counter output
    print(f"🎯 Success count: {success_count}")  # Adjust if using a better success metric
    print(f"🤩 Model Accuracy: {accuracy:.2%}")

# Evaluate on test set
evaluate_recommendations(test_df, train_df, cosine_sim_train)