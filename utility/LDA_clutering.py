'''mport pandas as pd
import ast
from collections import Counter

# Load dataset
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/final_cleaned_dataset.csv"
df = pd.read_csv(file_path)

# Convert mechanics & categories columns from string lists to actual lists
df['boardgamemechanic'] = df['boardgamemechanic'].apply(lambda x: ast.literal_eval(x)
   if isinstance(x, str) and x.startswith("[") else [])
df['boardgamecategory'] = df['boardgamecategory'].apply(lambda x: ast.literal_eval(x)
   if isinstance(x, str) and x.startswith("[") else [])

# Combine mechanics & categories into a single text column
df['game_tags'] = df['boardgamemechanic'] + df['boardgamecategory']

# Remove rare mechanics/categories (appear in < 5 games)
tag_counts = Counter([tag for sublist in df['game_tags'] for tag in sublist])
common_tags = {tag for tag, count in tag_counts.items() if count >= 5}

# Keep only common tags
df['game_tags'] = df['game_tags'].apply(lambda tags: [tag for tag in tags if tag in common_tags])

# Convert lists to space-separated strings (for LDA input)
df['game_tags'] = df['game_tags'].apply(lambda tags: " ".join(tags))

# Save preprocessed data for LDA
df[['id', 'name', 'game_tags']].to_csv("/Users/loriramey/PycharmProjects/BGapp/data/preprocessed_for_lda.csv", index=False)

print(f"Preprocessing complete! Saved cleaned LDA input to preprocessed_for_lda.csv")
'''

import pandas as pd
import ast
from collections import Counter

# Load dataset
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/final_cleaned_dataset.csv"
df = pd.read_csv(file_path)

# Convert mechanics & categories columns from string lists to actual lists
df['boardgamemechanic'] = df['boardgamemechanic'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
)
df['boardgamecategory'] = df['boardgamecategory'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
)

# Combine mechanics & categories into a single text column
df['game_tags'] = df['boardgamemechanic'] + df['boardgamecategory']

# Remove rare mechanics/categories (appear in < 5 games)
tag_counts = Counter([tag for sublist in df['game_tags'] for tag in sublist])
common_tags = {tag for tag, count in tag_counts.items() if count >= 5}

# Keep only common tags
df['game_tags'] = df['game_tags'].apply(lambda tags: [tag for tag in tags if tag in common_tags])

# Convert lists to space-separated strings (for LDA input)
df['game_tags'] = df['game_tags'].apply(lambda tags: " ".join(tags))

# Save preprocessed data for LDA
output_path = "/Users/loriramey/PycharmProjects/BGapp/data/preprocessed_for_lda.csv"
df[['id', 'name', 'game_tags']].to_csv(output_path, index=False)

print(f"Preprocessing complete! Saved cleaned LDA input to {output_path}")



