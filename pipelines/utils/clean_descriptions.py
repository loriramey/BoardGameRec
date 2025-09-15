import pandas as pd
import html
import re

def clean_description(desc):
    # Unescape HTML entities (e.g., &amp; -> &)
    cleaned = html.unescape(desc)
    # Remove any stray HTML tags (if any)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Trim whitespace
    return cleaned.strip()

# Load the sorted CSV
GAMEDATA = "/Users/loriramey/PycharmProjects/BGapp/data/BGGtop300.csv"
df = pd.read_csv(GAMEDATA)

# Apply cleaning to the 'description' column
df["description_clean"] = df["description"].apply(clean_description)

# Save the cleaned CSV
output_file = GAMEDATA
df.to_csv(output_file, index=False)

print(f"Cleaned CSV saved as {output_file}")