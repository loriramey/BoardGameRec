import pandas as pd
import ast  # To convert string lists to actual lists

# Load dataset
file_path = "/data/final_cleaned_dataset.csv"
df = pd.read_csv(file_path)

# Convert mechanics and category columns from string to actual lists
df['boardgamemechanic'] = df['boardgamemechanic'].apply(lambda x: ast.literal_eval(x)
    if isinstance(x, str) and x.startswith("[") else [])

df['boardgamecategory'] = df['boardgamecategory'].apply(lambda x: ast.literal_eval(x)
    if isinstance(x, str) and x.startswith("[") else [])

# Flatten mechanics and categories into lists
all_mechanics = [mechanic for sublist in df['boardgamemechanic'] for mechanic in sublist]
all_categories = [category for sublist in df['boardgamecategory'] for category in sublist]

# Get unique mechanics and categories
unique_mechanics = pd.Series(all_mechanics).value_counts()
unique_categories = pd.Series(all_categories).value_counts()

# Save mechanics and categories lists to CSV files for manual inspection
unique_mechanics.to_csv("mechanics_list.csv", index=True)
unique_categories.to_csv("categories_list.csv", index=True)

print(f"Extracted {len(unique_mechanics)} unique game mechanics. Saved to mechanics_list.csv.")
print(f"Extracted {len(unique_categories)} unique game categories. Saved to categories_list.csv.")