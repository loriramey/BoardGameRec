#this function reorders the gamedate.csv file by the "id" column
#which should be unique - and verfieis that uniqueness
#so the TF IDF matrixes can be aligned to the same game order for faster processing.

import pandas as pd

GAMEDATA_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata.csv"
OUTPUT_FILE = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_sorted.csv"


# 1. Load the CSV
df = pd.read_csv(GAMEDATA_FILE)

# 2. Inspect the "id" column for duplicates or nulls
duplicates = df[df['id'].duplicated(keep=False)]
nulls = df[df['id'].isnull()]

if not duplicates.empty:
    print("Duplicate IDs found:")
    print(duplicates)
else:
    print("No duplicate IDs found.")

if not nulls.empty:
    print("Null IDs found:")
    print(nulls)
else:
    print("No null IDs found.")

# Optional: if duplicates or nulls are found, decide how to handle them
# For example, you could drop duplicates:
# df = df.drop_duplicates(subset=['id'])

# 3. Sort the DataFrame by the "id" column
df_sorted = df.sort_values(by='id')

# 4. Save the sorted DataFrame to a new CSV file
df_sorted.to_csv(OUTPUT_FILE, index=False)

print(f"CSV has been sorted by 'id' and saved as {OUTPUT_FILE}.")