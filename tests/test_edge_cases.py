import pytest
from src.recommendation import get_rec_by_name

# List of edge-case game inputs
edge_case_games = [
    "Blang!",  # Non-existent game (should return a message)
    "Grind",  # Low BGG rank game
    "Monopoly",
    "Android Netrunner",  # Possibly not an exact match
    "Subtext",  # Low review count/popularity edge
    "Sultan",
    "On the Rocks",
    "Railway Rivals",  # Older game
    "IQ 2000",  # Check for variations like "I.Q. 2000"
    "Agamemnon",
    "Down in Flames",
    "Fragged Empire"
]


@pytest.mark.parametrize("game_input", edge_case_games)
def test_edge_case_recommendations(game_input):
    """
    For each edge-case game input, run get_rec_by_name with auto_select enabled,
    then check for the presence of a 'similarity' column. If present, sort and print
    the first 5 recommendations; if not, fail the test.
    """
    print(f"\n--- Testing input: '{game_input}' ---")
    recs = get_rec_by_name(game_input, auto_select=True)

    # Check if the function returned an error message
    if isinstance(recs, str):
        print(f"Returned message: {recs}")
        # If we expect a non-existent game to return a message, assert that
        assert "not found" in recs.lower() or recs == "", (
            f"Unexpected message for input '{game_input}': {recs}"
        )
    else:
        # Verify the 'similarity' column exists
        if "similarity" in recs.columns:
            recs_sorted = recs.sort_values(by="similarity", ascending=False)
            top5 = recs_sorted.head(5)
            print(top5[['name', 'playingtime', 'similarity']])
            assert len(top5) > 0, f"Expected some recommendations for input '{game_input}'"
        else:
            print("No 'similarity' column found in recommendations:")
            print(recs)
            pytest.fail(f"'similarity' column missing for input '{game_input}'")