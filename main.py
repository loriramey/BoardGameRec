from src.recommendation import get_rec_by_name

# Example: User input
game_name = input("Enter a game name: ")
min_players = int(input("Minimum players? (or 0 to skip): ") or 0)
max_players = int(input("Maximum players? (or 0 to skip): ") or 0)
max_time = int(input("Max playtime in minutes? (or 0 to skip): ") or 0)
min_rating = float(input("Minimum rating? (or 0 to skip): ") or 0)

# Get Recommendations
results = get_rec_by_name(game_name, min_players, max_players, max_time, min_rating)

# Display results
print(results)
