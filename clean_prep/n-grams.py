import pandas as pd
import ast  # To convert string lists back to Python lists

# Load dataset
datafile = "/Users/loriramey/PycharmProjects/BGapp/data/final_tagged_data.csv"
df = pd.read_csv(datafile)

# Function to replace spaces with underscores in a list of phrases
def format_phrases(value):
    if isinstance(value, str):
        try:
            items = ast.literal_eval(value)  # Convert string representation of a list to an actual list
            return " ".join(item.replace(" ", "_") for item in items)  # Replace spaces with underscores
        except (SyntaxError, ValueError):
            return value.replace(" ", "_")  # In case the value is a single string
    return " ".join(value).replace(" ", "_") if isinstance(value, list) else value

# Apply function to relevant columns
df["tags_str"] = df["tags"].apply(format_phrases)
df["categories_str"] = df["category_list"].apply(format_phrases)
df["mechanics_str"] = df["mech_list"].apply(format_phrases)

# Print to verify
print(df[["tags_str", "categories_str", "mechanics_str"]].head())

import re

# Function to clean double underscores and extra spaces
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"\s+", "_", text)  # Replace any whitespace with single underscore
        text = re.sub(r"_+", "_", text)  # Replace multiple underscores with one
        return text.strip("_")  # Remove leading/trailing underscores if present
    return text

# Apply fix to relevant columns
df["tags_str"] = df["tags_str"].apply(clean_text)
df["categories_str"] = df["categories_str"].apply(clean_text)
df["mechanics_str"] = df["mechanics_str"].apply(clean_text)

# Save cleaned data
df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/games_tfidf_cleaned.csv", index=False)