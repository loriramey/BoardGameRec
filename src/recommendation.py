import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv"
COSINE_SIM_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_weighted.npy"

df = pd.read_csv(GAMEDATA_FILE)
cosine_sim = np.load(COSINE_SIM_FILE) #numpy will give faster reads on this large matrix
game_index = {name: idx for idx, name in enumerate(df['name'])}  #faster index for searching
# Ensure 'yearpublished' is a numeric column in the dataset for later filtering
df['yearpublished'] = pd.to_numeric(df['yearpublished'], errors='coerce')
# Drop any rows where year is still NaN after conversion
df = df.dropna(subset=['yearpublished'])
# Convert column again to ensure it's an integer
df['yearpublished'] = df['yearpublished'].astype(int)

#FUNCTION: use Fuzzy Match on game name to handle user input
def find_closest_name(user_input):
    matches = process.extract(user_input, game_index.keys(), scorer=fuzz.WRatio, limit=10)  # WRatio helps with misspellings

    print("\n Did you mean:")
    for idx, (match, score, _) in enumerate(matches):
        print(f"{idx+1}. {match} ({round(score, 1)}%)")

    choice = input("Enter number to select or press Enter to choose best match: ")
    if choice.isdigit() and 1 <= int(choice) <= len(matches):
        return matches[int(choice) - 1][0]
    return matches[0][0]  # Default to best match if user skips selection

#FUNCTION: find "similar" games to user input game
def get_rec_by_name(game_name, min_players = None, max_players = None,
                    max_time = None, min_rating = None, min_year = None):

    game_name = find_closest_name(game_name)
    print(f"Matched game: {game_name}")
    #handle unidentifiable user input after fast fuzzy search
    if not game_name:
        return f"Game not found. Try entering the name again."

    #index for similarity search
    game_idx = game_index[game_name]
    #find similar games - fetch top 25
    sim_scores = list(enumerate(cosine_sim[game_idx]))

    # ✅ Debug: Print raw similarity scores
    print(f"Raw similarity scores for {game_name}: {sim_scores[:10]}")  # Top 10 scores

    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)[1:26]     #return top 20 game matches
    # ✅ Debug: Check if similarity scores are too low
    print(f"Sorted similarity scores: {sim_scores[:5]}")  # Print top 5

    #create dict to hold sim scores so we can add ranked column for returning results
    similarity_dict = {i[0]: i[1] for i in sim_scores}
    #convert indices back to rows of game data
    valid_indices = [i[0] for i in sim_scores if i[0] in df.index]  # Ensure indices exist
    if not valid_indices:
        return "No valid recommendations found."

    sim_game_names = df.loc[valid_indices, 'name'].str.strip().str.lower().tolist()

    # ✅ Debug: Print the matched games
    print(f"Top matches for {game_name}: {sim_game_names[:10]}")
    print(f"🧐 Checking sim_game_names: {sim_game_names}")
    print(f"🧐 Checking df['name'] sample: {df['name'].head(10).tolist()}")

    #grab full game data from source data, handle capitalization for sake of filtering
    sim_game_names_lower = [name.lower() for name in sim_game_names]  # Convert to lowercase
    df['name_lower'] = df['name'].str.lower()  # Create a lowercase version of names in df
    recommended_games = df[df['name_lower'].isin(sim_game_names_lower)].copy()
    print(f"🔍 Before filtering: {len(recommended_games)} games found")

    #apply user filters
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

    #sort the recommended games so they are returned "in order" of closest match
    recommended_games["similarity"] = recommended_games.index.map(similarity_dict)
    recommended_games = recommended_games.sort_values(by=['similarity'], ascending=False)

    return recommended_games[['name', 'description', 'thumbnail', 'yearpublished',
                'category_list', 'mech_list', 'tags', 'Board Game Rank',
                'minplayers', 'maxplayers', 'playingtime', 'average']]