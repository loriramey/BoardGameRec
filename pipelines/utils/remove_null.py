import pandas as pd
import ast  # Converts string lists to Python lists

# Load cleaned dataset
file_path = "/data/cleaned_dataset.csv"
df = pd.read_csv(file_path)

# --- Fix `yearpublished` display ---
df['yearpublished'] = df['yearpublished'].apply(lambda x: "Unknown" if x == 0 else f"{abs(x)} B.C." if x < 0 else x)

# --- Drop games with no mechanics ---
df = df.dropna(subset=['boardgamemechanic'])

# --- Convert string lists to actual Python lists ---
def convert_to_list(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.startswith("[") else []
    except (SyntaxError, ValueError):
        return []  # If parsing fails, return an empty list

df['boardgamecategory'] = df['boardgamecategory'].apply(convert_to_list)
df['boardgamemechanic'] = df['boardgamemechanic'].apply(convert_to_list)

# --- Save cleaned dataset ---
cleaned_file_path = "/data/final_cleaned_dataset.csv"
df.to_csv(cleaned_file_path, index=False)

# --- Verify changes ---
print("\nUpdated Dataset Summary:")
print(df[['yearpublished', 'boardgamecategory', 'boardgamemechanic']].head())

print(f"\nFinal cleaned dataset saved at: {cleaned_file_path}")