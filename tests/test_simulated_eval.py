# tests/test_bulk_recommendation_short.py
import pytest
import numpy as np
import pandas as pd
from src.recommendation import get_rec_by_name

def test_bulk_recommendation_short():
    """
    For a few anchor games, measure the average similarity of top recommendations
    compared to random selections from a representative sample of gamedata.
    """
    # Use a representative sample from gamedata_sorted.csv
    df_all = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted.csv")
    # Optionally, sample 100 games to work with a smaller, more representative set
    df_sample = df_all.sample(100, random_state=42)

    # Define a few anchor games. Make sure these titles are in your sampled data.
    anchor_games = ["Parade", "Pandemic", "Ticket to Ride", "CATAN", "Carcassonne", "Castle Panic","Netrunner", "Star Realms", "Samurai Spirit", "Azul" ]
    results = []

    for game in anchor_games:
        # Get recommendations from your engine; adjust max_results if needed
        recs = get_rec_by_name(game, max_results=5)
        # Here we assume each recommended row has a "similarity" column added in get_rec_by_name
        avg_sim = np.mean(recs["similarity"].values)

        # For a random baseline, sample 5 random games from the sample that aren't the anchor game
        random_df = df_sample[df_sample["name"].str.lower() != game.lower()].sample(5, random_state=42)
        # Assume these random rows also have a "similarity" field computed, or use a placeholder
        random_avg_sim = np.mean([r.get("similarity", 0) for r in random_df.to_dict("records")])

        results.append({
            "game": game,
            "avg_rec_similarity": avg_sim,
            "avg_random_similarity": random_avg_sim
        })

    # Print out the results for each anchor game
    for r in results:
        print(f"{r['game']} -> recommended avg sim: {r['avg_rec_similarity']:.3f}, "
              f"random avg sim: {r['avg_random_similarity']:.3f}")
        # Assert that recommendations have a higher average similarity than random
        assert r["avg_rec_similarity"] > r["avg_random_similarity"], (
            f"Recommendations for {r['game']} are not beating random baseline!"
        )