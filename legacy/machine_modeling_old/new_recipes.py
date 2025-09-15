import numpy as np
import pandas as pd
import sys
from legacy.machine_modeling_old.custom_CS_modeling import compute_custom_cs
from src.recommendation import get_rec_by_name, print_recommendations_summary, cosine_sim
from src.data_helpers import df, get_similarity

# --- Step 1: Prompt user for weight values ---
print("Enter 6 weight values (as percentages) for the following features:")
try:
    tags_weight = float(input("Tags (%): "))
    categories_weight = float(input("Categories (%): "))
    mechanics_weight = float(input("Mechanics (%): "))
    maxplayers_weight = float(input("Max Players (%): "))
    playingtime_weight = float(input("Playing Time (%): "))
    averageweight_weight = float(input("Average Weight (%): "))
except ValueError:
    print("Please enter valid numbers.")
    sys.exit(1)

total = tags_weight + categories_weight + mechanics_weight + maxplayers_weight + playingtime_weight + averageweight_weight
if abs(total - 100) > 1e-6:
    print(f"Weights must add up to 100 (you entered {total}). Exiting.")
    sys.exit(1)

# Convert percentages to fractions
new_weights = {
    'tags': tags_weight / 100.0,
    'categories': categories_weight / 100.0,
    'mechanics': mechanics_weight / 100.0,
    'maxplayers': maxplayers_weight / 100.0,
    'playingtime': playingtime_weight / 100.0,
    'averageweight': averageweight_weight / 100.0
}
print("\nUsing new weight recipe:", new_weights)

# --- Step 2: Compute custom CS matrix using new_weights ---
custom_cs = compute_custom_cs(new_weights)
print("Custom cosine similarity matrix computed. Shape:", custom_cs.shape)

# --- Step 3: Save original global cosine_sim and prepare summary collection ---
original_cs = cosine_sim.copy()  # copy original global CS matrix

summary_list = []

# --- Step 4: Randomly sample 10 games from the master DataFrame ---
random_games = df.sample(n=10, random_state=91)
for _, game in random_games.iterrows():
    game_name = game['name']
    print(f"\n=== Testing recommendations for: {game_name} ===")

    # (A) Get recommendations using the standard (original) CS matrix
    recs_standard = get_rec_by_name(game_name, auto_select=True)
    if isinstance(recs_standard, str):
        print(f"Error with standard CS: {recs_standard}")
        continue
    recs_standard_sorted = recs_standard.sort_values(by="similarity", ascending=False)
    top5_standard = recs_standard_sorted.head(5)
    std_avg = top5_standard["similarity"].mean()

    print("\nStandard CS Recommendations (Top 5):")
    print_recommendations_summary(top5_standard)

    # (B) Override global CS with custom_cs and get custom recommendations
    cosine_sim[:] = custom_cs  # override global CS matrix
    recs_custom = get_rec_by_name(game_name, auto_select=True)
    if isinstance(recs_custom, str):
        print(f"Error with custom CS: {recs_custom}")
        # Restore original and continue
        cosine_sim[:] = original_cs
        continue
    recs_custom_sorted = recs_custom.sort_values(by="similarity", ascending=False)
    top5_custom = recs_custom_sorted.head(5)
    cust_avg = top5_custom["similarity"].mean()

    print("\nCustom CS Recommendations (Top 5):")
    print_recommendations_summary(top5_custom)

    # Restore the original CS matrix for baseline calculation
    cosine_sim[:] = original_cs

    # (C) Calculate a random baseline average similarity for the input game
    # Using original CS matrix (i.e., get_similarity)
    random_ids = df[df['id'] != game['id']]['id'].sample(5, random_state=42).tolist()
    random_avg = np.mean([get_similarity(game['id'], rid) for rid in random_ids])

    print(
        f"\nFor '{game_name}': Standard Top 5 Avg Similarity = {std_avg:.4f}, Custom Top 5 Avg Similarity = {cust_avg:.4f}, Random Baseline = {random_avg:.4f}")

    summary_list.append({
        "Game Name": game_name,
        "Standard Top 5 Avg Similarity": std_avg,
        "Custom Top 5 Avg Similarity": cust_avg,
        "Random Baseline Similarity": random_avg
    })

# --- Step 5: Print final summary table ---
summary_df = pd.DataFrame(summary_list)
print("\nFinal Summary Table:")
print(summary_df.to_string(index=False))

# (Optional) Restore the original CS matrix if needed
cosine_sim[:] = original_cs