import pandas as pd

# Load the dataset
file_path = "data/game_info_select.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
drop_columns = ["field1", "suggested_num_players", "suggested_playerage",
                "suggested_language_dependence", "Strategy Game Rank", "Family Game Rank"]
df = df.drop(columns=drop_columns, errors='ignore')

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data types overview
print("\nData Types:")
print(df.dtypes)

# Summary statistics for numeric columns
print("\nSummary Statistics:")
print(df.describe())

# Save cleaned dataset to inspect manually (optional)
df.to_csv("cleaned_dataset.csv", index=False)