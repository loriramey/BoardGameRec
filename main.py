from src.recommendation import get_rec_by_name, print_game_info

# Example: User input
game_name = input("Enter a game name: ")
min_players = int(input("Minimum players? (or 0 to skip): ") or 0)
max_players = int(input("Maximum players? (or 0 to skip): ") or 0)
max_time = int(input("Max playtime in minutes? (or 0 to skip): ") or 0)
min_rating = float(input("Minimum rating? (or 0 to skip): ") or 0)
min_year = int(input("Newer than [4 digit year]? (or 0 to skip): ") or 0)

# Get Recommendations
results = get_rec_by_name(game_name, min_players, max_players, max_time, min_rating, min_year)

# Display results
if isinstance(results, str):
    print(results)  # If error message (e.g., game not found)
else:
    print("\n🎲 **Top Recommended Games** 🎲")
    for idx, row in results.iterrows():
        print(f"{idx+1}. {row['name']} ({row['yearpublished']}) - Average Rating: {row['average']:.2f}⭐")
        print(f"   Players: {row['minplayers']} - {row['maxplayers']}, Playtime: {row['playingtime']} min")
        print(f"   Tags: {row['tags']} | Categories: {row['category_list']}")
        print(f"   Mechanics: : {row['mech_list']} | Avg Weight: {row['averageweight']:.2f}")
        print(f"   Description: {row['description'][:500]}...\n")  # Show preview
