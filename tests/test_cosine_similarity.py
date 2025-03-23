# tests/test_similarity.py
import pytest
import numpy as np
from src.data_helpers import get_similarity, df

def test_known_similar_games():
    """
    Test that two games known to be similar (by domain knowledge) have a high similarity score.
    Replace GAME_ID_A and GAME_ID_B with actual IDs from your CSV.
    """
    GAME_ID_A = 12345  # replace with an actual game id from your CSV
    GAME_ID_B = 12346  # replace with an actual similar game id
    sim = get_similarity(GAME_ID_A, GAME_ID_B)
    assert sim > 0.7, f"Expected similarity > 0.7 but got {sim}"


def test_similarity_vs_random():
    """
    For a given game, check that its recommended similarity is higher than random comparisons.
    """
    # Choose a game id from your CSV
    game_id = df['id'].iloc[0]

    # Get similarity for the top 5 similar games (excluding self)
    idx = [i for i in range(len(df)) if df['id'].iloc[i] != game_id]
    similarities = [get_similarity(game_id, df['id'].iloc[i]) for i in idx]
    top5_sim = np.mean(sorted(similarities, reverse=True)[:5])

    # Compare with the average similarity for 5 random games
    random_ids = df['id'].sample(5, random_state=42).tolist()
    random_sim = np.mean([get_similarity(game_id, rid) for rid in random_ids])

    assert top5_sim > random_sim, (
        f"Top 5 average similarity {top5_sim} not higher than random average {random_sim}"
    )