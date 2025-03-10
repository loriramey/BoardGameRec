import pandas as pd
import ast
from collections import Counter


datafile = "/Users/loriramey/PycharmProjects/BGapp/data/final_tagged_data.csv"
tagfile = "/Users/loriramey/PycharmProjects/BGapp/data/tags.csv"

# Load dataset where each game is tagged
df = pd.read_csv(datafile)
tags_df = pd.read_csv(tagfile)

# Ensure tags column exists and is properly formatted
if "tags" not in df.columns:
    raise ValueError("The 'tags' column is missing from final_tagged_data.csv!")

# Convert string lists to actual lists if needed
df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Flatten and count occurrences of each tag
all_tags = [tag for sublist in df["tags"] for tag in sublist]
tag_frequencies = Counter(all_tags)

# Convert to DataFrame
tag_freq_df = pd.DataFrame(tag_frequencies.items(), columns=["tag", "frequency"])

# Merge with existing tags data
tags_df = tags_df.merge(tag_freq_df, on="tag", how="left").fillna(0)

# Display results
print("Updated Tags Data:\n", tags_df.head())