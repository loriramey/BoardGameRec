import pandas as pd
import re

# Load dataset
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/game_info_select.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Remove "&#10;" and other similar HTML artifacts in 'description' field
df['description'] = df['description'].astype(str).apply(lambda x: re.sub(r'&#\d+;', ' ', x))

# Fill missing `boardgamecategory` with "Unknown" (temporary solution)
df['boardgamecategory'].fillna("Unknown", inplace=True)

# Save cleaned dataset (optional)
df['boardgamecategory'] = df['boardgamecategory'].fillna("Unknown")

# Verify changes
print("\nUpdated Missing Values:")
print(df.isnull().sum())

print("\nSample Cleaned Description:")
print(df['description'].iloc[0])  # Check a sample description