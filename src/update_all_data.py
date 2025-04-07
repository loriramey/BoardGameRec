'''
This file should be used 1-2x a year when the app owner needs to update the game data with new games published this year.

UPDATE PROCESS:
A. DATA CLEANING BEFORE YOU BEGIN
    1. Obtain updated game data for this year's games from BGG.com's API.  Drop games lacking key info and non-games.
    2. Drop columns that do not match the gamedata .csv architecture. Assign tags. Check for NaN / missing values that can be replaced.
    3. Run project files (stored elsewhere) to update TF-IDF vectors and min-max scale (normalize) numeric data.
    4. Append Python-friendly lists of tags, mechanics, and similarities to the file - observe column order.
    5. Run Python ftfy script in base project to fix weird characters produced by conversion from HTML/XML in descriptions and game titles.
    Store cleaned descriptions in their own column. Overwrite game titles.
* Optional: Store backups of gamedata.csv, CS matrices, .pkl files, and df of top 50 game matches.

B. Run the script below to append new game data to the base gamedata.csv.  Output as .parquet file storing a df.

C. Save new .pkl TF-IDF vector files for use in Cosine Similarity calculations.

D. Run the script cosine_similarity.py in this package to output an updated CS matrix.
Use commented code to update the cat_heavy and original_mix "recipes." The file as written outputs mech_heavy similarity scores.

E. Run the script output_similarity_df in this package to output the updated Top 50 similar games for each game id.
Store in new .parquet file for fast loading.

F. Check all file names and file paths against streamlit_app.py and Home / About / DataViz page code
CONGRATS! YOUR APP HAS BEEN UPDATED WITH THE LATEST GAME INFO AND SIMILAR GAMES FOR RECOMMENDATIONS!
'''

import pandas as pd
from pathlib import Path

# Paths to your data files
OLD_PARQUET_PATH = Path("gamedata.parquet")
NEW_GAMES_CSV_PATH = Path("new_games.csv")
OUTPUT_PARQUET_PATH = Path("gamedata.parquet")  # overwrite for simplicity

# 1. Load existing .parquet file
if OLD_PARQUET_PATH.exists():
    df_current = pd.read_parquet(OLD_PARQUET_PATH)
    print(f"Loaded existing dataset with {len(df_current)} games.")
else:
    df_current = pd.DataFrame()
    print("No existing .parquet file found. Starting fresh.")

# 2. Load new data
df_new = pd.read_csv(NEW_GAMES_CSV_PATH)
print(f"Loaded {len(df_new)} new game entries.")

# 3. Sanity check: Remove duplicates (based on game ID)
df_combined = pd.concat([df_current, df_new])
df_combined = df_combined.drop_duplicates(subset="id", keep="last").reset_index(drop=True)

print(f"Combined dataset now has {len(df_combined)} unique games.")

# 4. Save updated .parquet
df_combined.to_parquet(OUTPUT_PARQUET_PATH, index=False)
print(f"Saved updated game data to {OUTPUT_PARQUET_PATH}")

'''
From here, the user needs to re-output the CS matrix to include the new games. 
* Run cosine_similarity.py on the new gamedata.parquet file

Next, update the stored top 50 matching games based on the new CS matrix. 
* Run output_similarity.df.py

'''