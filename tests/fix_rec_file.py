import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv"
COSINE_SIM_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_weighted.npy"

# Load master data and cosine similarity matrix
df = pd.read_csv(GAMEDATA_FILE)
cosine_sim = np.load(COSINE_SIM_FILE)  # Precomputed cosine similarity matrix
game_index = {name: idx for idx, name in enumerate(df['name'])}  # For fuzzy matching

# Ensure 'yearpublished' is numeric, drop NaNs, and convert to int
df['yearpublished'] = pd.to_numeric(df['yearpublished'], errors='coerce')
df = df.dropna(subset=['yearpublished'])
df['yearpublished'] = df['yearpublished'].astype(int)

# FUNCTION: Use fuzzy matching on game name to handle user input
def find_closest_name(user_input, auto_select=False):
    matches = process.extract(user_input, game_index.keys(), scorer=fuzz.WRatio, limit=10)
    if auto_select:
        return matches[0][0]  # Automatically choose the highest-confidence match
    print("\nDid you mean:")
    for idx, (match, score, _) in enumerate(matches):
        print(f"{idx+1}. {match} ({round(score, 1)}%)")
    choice = input("Enter number to select or press Enter to choose best match: ")
    if choice.isdigit() and 1 <= int(choice) <= len(matches):
        return matches[int(choice) - 1][0]
    return matches[0][0]

# FUNCTION: Extract root game title to avoid duplicate series entries
def get_root_title(title):
    return title.split(':')[0].split('(')[0].strip().lower()

def get_rec_by_name_debug(game_name, auto_select=False):
    """
    Debug version: runs the recommendation pipeline up to duplicate filtering and similarity mapping,
    then prints the resulting DataFrame (without applying user filters).
    """
    # Fuzzy match the game name
    game_name = find_closest_name(game_name, auto_select=auto_select)
    print(f"Matched game: {game_name}")
    if not game_name:
        return "Game not found. Try entering the name again."

    # Get similarity scores for the matched game
    game_idx = game_index[game_name]
    sim_scores = list(enumerate(cosine_sim[game_idx]))
    print(f"Raw similarity scores for {game_name}: {sim_scores[:10]}")
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]  # Top 20 matches (exclude self)
    print(f"Sorted similarity scores: {sim_scores[:5]}")
    similarity_dict = {i[0]: i[1] for i in sim_scores}

    valid_indices = [i[0] for i in sim_scores if i[0] in df.index]
    if not valid_indices:
        return "No valid recommendations found."

    sim_game_names = df.loc[valid_indices, 'name'].str.strip().str.lower().tolist()
    print(f"Top matches for {game_name}: {sim_game_names[:10]}")
    print(f"🧐 Checking sim_game_names: {sim_game_names}")

    # Prepare DataFrame for recommendations using lowercase names for matching
    df['name_lower'] = df['name'].str.lower()
    sim_game_names_lower = [name.lower() for name in sim_game_names]
    recommended_games = df[df['name_lower'].isin(sim_game_names_lower)].copy()
    print(f"🔍 Before filtering: {len(recommended_games)} games found")

    # Preserve original indices by explicitly adding a column
    recommended_games['original_index'] = recommended_games.index

    # Remove duplicate series entries using drop_duplicates
    recommended_games = recommended_games.drop_duplicates(subset='name_lower', keep='first')
    print(f"🎯 After filtering clones: {len(recommended_games)} games remaining")

    # Map similarity scores using the preserved DataFrame index
    print("Similarity dict keys:", list(similarity_dict.keys()))
    print("Recommended games index:", recommended_games.index.tolist())
    recommended_games["similarity"] = recommended_games.index.map(similarity_dict)
    print("Mapped similarity column values:", recommended_games["similarity"].tolist())

    print("DEBUG: DataFrame with similarity scores (before user filters):")
    print(recommended_games[['name', 'playingtime', 'similarity']])
    return recommended_games


def get_rec_by_name_debug_filtered(game_name, auto_select=False, max_time=None):
    # Run the same initial steps as in get_rec_by_name_debug...
    game_name = find_closest_name(game_name, auto_select=True)
    print(f"Matched game: {game_name}")
    if not game_name:
        return "Game not found. Try entering the name again."

    game_idx = game_index[game_name]
    sim_scores = list(enumerate(cosine_sim[game_idx]))
    print(f"Raw similarity scores for {game_name}: {sim_scores[:10]}")
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]
    print(f"Sorted similarity scores: {sim_scores[:5]}")
    similarity_dict = {i[0]: i[1] for i in sim_scores}

    valid_indices = [i[0] for i in sim_scores if i[0] in df.index]
    if not valid_indices:
        return "No valid recommendations found."

    sim_game_names = df.loc[valid_indices, 'name'].str.strip().str.lower().tolist()
    print(f"Top matches for {game_name}: {sim_game_names[:10]}")

    df['name_lower'] = df['name'].str.lower()
    sim_game_names_lower = [name.lower() for name in sim_game_names]
    recommended_games = df[df['name_lower'].isin(sim_game_names_lower)].copy()
    print(f"🔍 Before filtering: {len(recommended_games)} games found")

    # Remove duplicates using drop_duplicates (preserves index)
    recommended_games = recommended_games.drop_duplicates(subset='name_lower', keep='first')
    print(f"🎯 After filtering clones: {len(recommended_games)} games remaining")

    # Apply the max_time filter (for example, 60 minutes)
    if max_time:
        recommended_games = recommended_games[recommended_games['playingtime'] <= max_time]
    print(f"🎯 After applying max_time filter: {len(recommended_games)} games remaining")

    # Map similarity scores using the DataFrame's index
    recommended_games = recommended_games[recommended_games.index.isin(similarity_dict.keys())]
    recommended_games["similarity"] = recommended_games.index.map(similarity_dict)
    print("Mapped similarity column values (after filtering):", recommended_games["similarity"].tolist())

    recommended_games = recommended_games.sort_values(by="similarity", ascending=False)
    print("Final debug DataFrame (after filtering):")
    print(recommended_games[['name', 'playingtime', 'similarity']])
    return recommended_games

'''
# For quick testing:
if __name__ == "__main__":
    # Try with "Arboretum" and a max_time filter of 60 minutes
    debug_filtered = get_rec_by_name_debug_filtered("Arboretum", max_time=60)
'''