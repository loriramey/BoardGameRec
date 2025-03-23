import pytest
import pandas as pd
import numpy as np
from src.recommendation import get_rec_by_name


def test_random_games_recommendation_filters_max_time():
    """
    For 5 randomly selected game titles from gamedata_sorted.csv,
    verify that get_rec_by_name returns recommendations that meet the max playtime filter.
    """
    # Load the full dataset
    df_all = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted.csv")

    # Randomly sample 5 game titles from the dataset
    random_games = df_all['name'].sample(5, random_state=None).tolist()

    # Apply a filter (e.g., maxtime = 60 minutes) for each sampled game
    print("\nTESTING FILTER: MAX PLAYTIME")
    max_time_filter = 62
    for game_title in random_games:
        print(f"\nTesting recommendations for: {game_title}")

        # Get recommendations for the game, using auto_select=True to avoid interactive prompts
        recs = get_rec_by_name(game_title, auto_select=True, max_time=max_time_filter)

        # Print and check the top recommendations
        print(f"Recommendations meeting the max playtime (<= {max_time_filter} minutes) filter:")
        for _, row in recs.iterrows():
            game_name = row['name']
            variable_to_test = row['playingtime']
            print(f"Game: {game_name}, Playingtime: {variable_to_test} minutes")
            assert variable_to_test <= max_time_filter, (
                f"Game {game_name} exceeds max playtime filter: {variable_to_test} minutes"
            )

    print("\n\nTESTING FILTER: MINIMUM PLAYER COUNT")
    min_players_filter = 2
    for game_title in random_games:
        print(f"\nTesting recommendations for: {game_title}")

        # Get recommendations for the game, using auto_select=True to avoid interactive prompts
        recs = get_rec_by_name(game_title, auto_select=True, min_players=min_players_filter)

        # Print and check the top recommendations
        print(f"Recommendations meeting the minimum players (>= {min_players_filter}) filter:")
        for _, row in recs.iterrows():
            game_name = row['name']
            variable_to_test = row['minplayers']
            print(f"Game: {game_name}, Minimum Players: {variable_to_test}")
            assert variable_to_test >= min_players_filter, (
                f"Game {game_name} fails min player count filter: {variable_to_test} players"
            )

    print("\n\nTESTING FILTER: MAX PLAYER COUNT")
    max_players_filter = 3
    for game_title in random_games:
        print(f"\nTesting recommendations for: {game_title}")

        # Get recommendations for the game, using auto_select=True to avoid interactive prompts
        recs = get_rec_by_name(game_title, auto_select=True, max_players=max_players_filter)

        # Print and check the top recommendations
        print(f"Recommendations meeting the maximum players (<= {max_players_filter}) filter:")
        for _, row in recs.iterrows():
            game_name = row['name']
            variable_to_test = row['maxplayers']
            print(f"Game: {game_name}, Max Players: {variable_to_test}")
            assert variable_to_test <= max_players_filter, (
                f"Game {game_name} fails max player count filter: {variable_to_test} players"
            )

    print("\n\nTESTING FILTER: MINIMUM PUBLICATION YEAR")
    pub_year_filter = 1987
    for game_title in random_games:
        print(f"\nTesting recommendations for: {game_title}")

        # Get recommendations for the game, using auto_select=True to avoid interactive prompts
        recs = get_rec_by_name(game_title, auto_select=True, min_year=pub_year_filter)

        # Print and check the top recommendations
        print(f"Recommendations meeting the minimum pub year (>= {pub_year_filter}) filter:")
        for _, row in recs.iterrows():
            game_name = row['name']
            variable_to_test = row['yearpublished']
            print(f"Game: {game_name}, Year Published (Minimum): {variable_to_test}")
            assert variable_to_test >= pub_year_filter, (
                f"Game {game_name} fails min pub year filter: {variable_to_test}"
            )

    print("\n\nTESTING FILTER: AVERAGE PLAYER RATING")
    min_rating_filter = 6.50
    for game_title in random_games:
        print(f"\nTesting recommendations for: {game_title}")

        # Get recommendations for the game, using auto_select=True to avoid interactive prompts
        recs = get_rec_by_name(game_title, auto_select=True, min_rating=min_rating_filter)

        # Print and check the top recommendations
        print(f"Recommendations meeting the minimum avg rating (>= {min_rating_filter}) filter:")
        for _, row in recs.iterrows():
            game_name = row['name']
            variable_to_test = row['average']
            print(f"Game: {game_name}, Avg Rating (Minimum): {variable_to_test}")
            assert variable_to_test >= min_rating_filter, (
                f"Game {game_name} fails min pub year filter: {variable_to_test}"
            )
