import numpy as np
import pandas as pd
from rapidfuzz import process

GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv"
COSINE_SIM_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/cosine_similarity_matrix.npy"

df = pd.read_csv(GAMEDATA_FILE)
cosine_sim = np.load(COSINE_SIM_FILE) #numpy will give faster reads on this large matrix
game_index = {name: idx for idx, name in enumerate(df['name'])}  #faster index for searching

#FUNCTION: use Fuzzy Match on game name to handle user input
#search for name that's at least 80% match to what they typed
def find_closest_name(user_input):
    matches = process.extract(user_input, game_index.keys(), limit=5)
    print("\n Did you mean:")
    for idx, (match, score, _) in enumerate(matches):
        print(f"{idx+1}. {match} ({score}%)")

    choice = input("Enter number to select or press Enter to choose best match: ")
    if choice.isdigit() and 1<= int(choice) <= len(matches):
        return matches[int(choice)-1][0]
    return matches[0][0] #default to best match on skip

#FUNCTION: find "similar" games to user input game
def get_rec_by_name(game_name, min_players = None, max_players = None,
                    max_time = None, min_rating = None):
    game_name = find_closest_name(game_name)

    # ✅ Debugging: Print fuzzy-matched game
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

    #convert indices back to rows of game data
    sim_game_names = df.iloc[[i[0] for i in sim_scores]]['name'].str.strip().str.lower().tolist()
    df['name'] = df['name'].str.strip().str.lower()

    # ✅ Debug: Print the matched games
    print(f"Top matches for {game_name}: {sim_game_names[:10]}")

    print(f"🧐 Checking sim_game_names: {sim_game_names}")
    print(f"🧐 Checking df['name'] sample: {df['name'].head(10).tolist()}")

    #grab full game data from source data
    recommended_games = df[df['name'].isin(sim_game_names)].copy()

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

    print(f"🎯 After filtering: {len(recommended_games)} games remaining")

    return recommended_games[['name', 'description', 'thumbnail', 'yearpublished',
                'category_list', 'mech_list', 'tags', 'Board Game Rank',
                'minplayers', 'maxplayers', 'playingtime', 'average']]