import pandas as pd
import ftfy
import html

def clean_text(text):
    """
    Clean text by fixing encoding issues and decoding HTML entities.
    """
    if pd.isna(text):
        return text  # Skip cleaning for NaN values
    # Fix text encoding issues
    fixed_text = ftfy.fix_text(text)
    # Convert HTML/XML entities to their corresponding characters
    clean_fixed_text = html.unescape(fixed_text)
    return clean_fixed_text

FILE_TO_CLEAN = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted.csv"
# Load your CSV file (adjust the encoding parameter if necessary)
df = pd.read_csv(FILE_TO_CLEAN, encoding='utf-8')

# Apply cleaning functions to the relevant columns
df['name'] = df['name'].apply(clean_text)
df['description_clean'] = df['description_clean'].apply(clean_text)

# Optionally, verify the changes on a sample of the data
print(df[['name', 'description_clean']].head())

# Save the cleaned dataframe to a new CSV file
df.to_csv("gamedata_sorted_cleaned.csv", index=False)