import pytest
import numpy as np
import pandas as pd
from src.recommendation import get_rec_by_name, print_recommendations_summary
from src.data_helpers import get_similarity, df  # ensure df is your master DataFrame


def test_bulk_recommendation_short():
    """
    For a few anchor games, measure the average similarity of top recommendations
    compared to random selections from a representative sample of gamedata.
    """
    # Load a representative sample from gamedata_sorted_cleaned.csv
    df_all = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted_cleaned.csv")
    # Optionally, sample 100 games for a more representative set
    df_sample = df_all.sample(100, random_state=67)

    # Define a few anchor games (ensure these titles exist in the sample)
    anchor_games = [
        "Parade", "Pandemic", "Ticket to Ride", "CATAN", "Carcassonne",
        "Castle Panic", "Netrunner", "Star Realms", "Samurai Spirit", "Azul"
    ]
    results = []

    for game in anchor_games:
        print(f"\n=== Testing recommendations for: {game} ===")

        # Get recommendations from your engine using the standard CS matrix
        recs = get_rec_by_name(game, auto_select=True)
        if isinstance(recs, str):
            print(f"Error retrieving recommendations for {game}: {recs}")
            continue

        # Calculate average similarity of the recommendations (top 5)
        recs_sorted = recs.sort_values(by="similarity", ascending=False)
        top5 = recs_sorted.head(5)
        avg_rec_sim = top5["similarity"].mean()

        print("Top 5 Recommendations Summary:")
        print_recommendations_summary(top5)

        # For a random baseline: sample 5 random game IDs (from df_sample)
        # excluding the anchor game, and use get_similarity to compute similarity.
        anchor_row = df_all[df_all["name"].str.lower() == game.lower()].iloc[0]
        anchor_id = anchor_row["id"]
        random_ids = df_sample[df_sample["name"].str.lower() != game.lower()]["id"].sample(5, random_state=42).tolist()
        random_avg_sim = np.mean([get_similarity(anchor_id, rid) for rid in random_ids])

        print(
            f"Anchor '{game}': Top 5 avg similarity = {avg_rec_sim:.3f}, Random avg similarity = {random_avg_sim:.5f}")

        results.append({
            "Game": game,
            "Top 5 Avg Similarity": avg_rec_sim,
            "Random Avg Similarity": random_avg_sim
        })

    # Print final summary table
    summary_df = pd.DataFrame(results)
    print("\nFinal Summary Table:")
    print(summary_df.to_string(index=False))

    # Assert that for each game, the top 5 avg similarity is greater than the random baseline
    for r in results:
        assert r["Top 5 Avg Similarity"] > r["Random Avg Similarity"], (
            f"Recommendations for {r['Game']} are not beating random baseline!"
        )