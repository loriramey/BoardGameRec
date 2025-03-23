import pytest
import time
from src.recommendation import get_rec_by_name

def test_performance_recommendations():
    """
    Time how long it takes to get recommendations for multiple queries.
    This doesn't have a strict pass/fail unless you set a threshold.
    """
    games_to_test = ["Arboretum", "Cascadia", "Terraforming Mars", "Gloomhaven", "Star Wars: Rebellion"]

    start_time = time.time()
    for game in games_to_test:
        _ = get_rec_by_name(game, max_results=10)
    total_time = time.time() - start_time

    avg_time_per_query = total_time / len(games_to_test)
    print(f"Average time per query: {avg_time_per_query:.3f} seconds.")

    # Optionally, you can assert it must be below a threshold:
    assert avg_time_per_query < 2.0, "Queries took too long on average!"