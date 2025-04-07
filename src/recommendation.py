import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
COSINE_SIM_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_cat_heavy.npy"

# Load master data and cosine similarity matrix
df = pd.read_csv(GAMEDATA_FILE)
cosine_sim = np.load(COSINE_SIM_FILE)  # Precomputed cosine similarity matrix
game_index = {name: idx for idx, name in enumerate(df['name'])}  # For fuzzy matching

# Ensure 'yearpublished' is numeric, drop NaNs, and convert to int
df['yearpublished'] = pd.to_numeric(df['yearpublished'], errors='coerce')
df = df.dropna(subset=['yearpublished'])
df['yearpublished'] = df['yearpublished'].astype(int)


#FUNCTION to print a quick summary of recommended info
def print_recommendations_summary(recs_df):
    """
    Print a compact summary for each recommended game:
      name / min players / max players / playingtime / average / average weight / tags / cats / mechs
    """
    for _, row in recs_df.iterrows():
        print(f"{row['name']} / Players: {row['minplayers']}-{row['maxplayers']} / "
              f"Playtime: {row['playingtime']} min / Rating: {row['average']:.2f}⭐ / "
              f"Avg Weight: {row['averageweight']} / Tags: {row['tags']} / "
              f"Category: {row['categories_str']} /  Mechanics: {row['mechanics_str']}"
        )


#FUNCTION for pulling and printing a single game's information
def print_game_info(game_name):
    df_sorted = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted.csv")

    # Filter the DataFrame for the game name (case insensitive)
    game_row = df_sorted[df_sorted['name'].str.lower() == game_name.lower()]
    if game_row.empty:
        print(f"Game '{game_name}' not found in gamedata_sorted.csv.")
        return
    # Get the first match (if there are duplicates)
    row = game_row.iloc[0]

    # Format and print the desired information:
    print(f"{row['id']}. {row['name']} ({row['yearpublished']}) - {row['average']:.2f}⭐")
    print(f"   Players: {row['minplayers']} - {row['maxplayers']}, Playtime: {row['playingtime']} min")
    print(f"   Tags: {row['tags']}")
    print(f"   Categories: {row['category_list']}")
    print(f"   Mechanics: {row['mech_list']}")
    print(f"   Average Weight: {row['averageweight']}")
    # Use description_clean if it exists; otherwise fallback to description:
    desc = row.get('description_clean', row.get('description', 'No description available'))
    print(f"   Description: {desc[:250]}...\n")


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

# FUNCTION: Find "similar" games given a user input game, applying various filters
def get_rec_by_name(game_name, min_players=None, max_players=None,
                    max_time=None, min_rating=None, min_year=None, auto_select=False):

    game_name = find_closest_name(game_name, auto_select=auto_select)
    print(f"Matched game: {game_name}")
    print_game_info(game_name)  # Assumes print_game_info is available in the current scope
    if not game_name:
        return "Game not found. Try entering the name again."

    # Get index and compute similarity scores
    game_idx = game_index[game_name]
    sim_scores = list(enumerate(cosine_sim[game_idx]))
    #print(f"Raw similarity scores for {game_name}: {sim_scores[:10]}")
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:26]  # Top 20 matches (exclude self)
    #print(f"Sorted similarity scores: {sim_scores[:5]}")

    # Build a dictionary for mapping similarity scores
    similarity_dict = {i[0]: i[1] for i in sim_scores}
    valid_indices = [i[0] for i in sim_scores if i[0] in df.index]
    if not valid_indices:
        return "No valid recommendations found."

    sim_game_names = df.loc[valid_indices, 'name'].str.strip().str.lower().tolist()
    #print(f"Top matches for {game_name}: {sim_game_names[:10]}")
    #print(f"🧐 Checking sim_game_names: {sim_game_names}")

    # Prepare DataFrame for filtering: work with lowercase names for matching
    df['name_lower'] = df['name'].str.lower()
    sim_game_names_lower = [name.lower() for name in sim_game_names]
    recommended_games = df[df['name_lower'].isin(sim_game_names_lower)].copy()
    print(f"🔍 Before filtering: {len(recommended_games)} games found")

    # Remove duplicate series using drop_duplicates while preserving original index
    recommended_games = recommended_games.drop_duplicates(subset='name_lower', keep='first')
    print(f"🎯 After dropping clones: {len(recommended_games)} games remaining")

    # Apply user filters
    if min_players:
        recommended_games = recommended_games[recommended_games['minplayers'] >= min_players]
    if max_players:
        recommended_games = recommended_games[recommended_games['maxplayers'] <= max_players]
    if max_time:
        recommended_games = recommended_games[recommended_games['playingtime'] <= max_time]
    if min_rating:
        recommended_games = recommended_games[recommended_games['average'] >= min_rating]
    if min_year:
        recommended_games = recommended_games[recommended_games['yearpublished'] >= min_year]
    print(f"🎯 After filtering: {len(recommended_games)} games remaining")

    # IMPORTANT: Restrict the recommended_games to only those rows whose index is in similarity_dict keys
    recommended_games = recommended_games[recommended_games.index.isin(similarity_dict.keys())]
    #print("Indices after restricting:", recommended_games.index.tolist())

    # Map similarity scores using the DataFrame's index
    recommended_games["similarity"] = recommended_games.index.map(similarity_dict)
    recommended_games = recommended_games.sort_values(by="similarity", ascending=False)

    return recommended_games[['name', 'description', 'thumbnail', 'yearpublished',
                'category_list', 'mech_list', 'tags', 'Board Game Rank', 'tags_str',
                'categories_str', 'mechanics_str', 'minplayers', 'maxplayers', 'playingtime',
                'average', 'averageweight', 'similarity']]

# For debugging purposes: a separate function to test filtering and mapping
def get_rec_by_name_debug_filtered(game_name, auto_select=False, max_time=None):
    recs = get_rec_by_name(game_name, auto_select=auto_select, max_time=max_time)
    print("Final debug DataFrame:")
    print(recs[['name', 'playingtime', 'similarity']])
    return recs

if __name__ == "__main__":
    # Quick test with debug function
    #print_game_info("Star Wars: Rebellion")
    debug_recs = get_rec_by_name_debug_filtered("Power Grid", auto_select=True, max_time=130)
    print("\nReturned recommendations:")
    print(debug_recs)