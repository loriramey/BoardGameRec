# tests/test_cosine_similarity.py
import pytest
import numpy as np
import pandas as pd
from src.data_helpers import get_similarity, df
from src.recommendation import get_rec_by_name, print_recommendations_summary

def test_recommendations_for_random_games():
    """
    For 20 random games from the master DataFrame, retrieve recommendations
    with auto_select enabled. For each game, print a compact summary of the top 5
    recommendations along with the top 5 average similarity and a random baseline average similarity.
    Finally, output a summary table listing each game name, its top 5 average similarity, and random average similarity.
    """
    summary_list = []  # to collect final summary for each game

    # Randomly sample 20 games from the master DataFrame
    random_games = df.sample(n=20, random_state=685)
    for _, game in random_games.iterrows():
        game_name = game['name']
        print(f"\n=== Recommendations for: {game_name} ===")

        # Get recommendations for this game with auto_select enabled (no extra filters)
        recs = get_rec_by_name(game_name, auto_select=True)
        if isinstance(recs, str):
            print(f"Error: {recs}")
            continue

        # Sort recommendations by similarity and take the top 5
        recs_sorted = recs.sort_values(by="similarity", ascending=False)
        top5 = recs_sorted.head(5)
        print("Top 5 Recommendations Summary:")
        print_recommendations_summary(top5)
        top5_avg = top5["similarity"].mean()

        # Sample 5 random games (excluding the input game) for baseline similarity
        random_ids = df[df['name'].str.lower() != game_name.lower()]['id'].sample(5, random_state=42).tolist()
        random_avg = np.mean([get_similarity(game['id'], rid) for rid in random_ids])

        print(f"Top 5 average similarity: {top5_avg}")
        print(f"Random sample average similarity: {random_avg}\n")

        summary_list.append({
            "Game Name": game_name,
            "Top 5 Avg Similarity": top5_avg,
            "Random Avg Similarity": random_avg
        })

    # Create a summary DataFrame and print it
    pd.set_option('display.max_columns', None)
    summary_df = pd.DataFrame(summary_list)
    print("\nFinal Summary Table:")
    print(summary_df.to_string(index=False))
