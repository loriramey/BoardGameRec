import pandas as pd
import ast
tags_file = "/Users/loriramey/PycharmProjects/BGapp/data/tags.csv"
data_file = "/Users/loriramey/PycharmProjects/BGapp/data/final_tagged_data.csv"

#initial dictionary = each tag maps to multiple categories
df_tags = pd.read_csv(tags_file, header=None)
tag_dict = {}
for row in df_tags.itertuples(index=False):
    tag_name = row[0]
    categories_for_tag = row[1:]
    # Convert to a cleaned-up list of strings, ignoring empty or NaN
    cat_list = []
    for cat in categories_for_tag:
        if pd.isna(cat):
            continue
        cat_str = str(cat).strip()
        if cat_str:
            cat_list.append(cat_str)
    tag_dict[tag_name] = cat_list

#invert dictionary for faster lookup
cat_to_tag = {}
for tag, cat_list in tag_dict.items():
    for c in cat_list:
        cat_to_tag[c] = tag

#assign tags based on boardgamecategory labels in the dataset
df_games = pd.read_csv(data_file, header=0)
def parse_list(cell_value):
    if pd.isna(cell_value):
        return []
    try:
        return ast.literal_eval(cell_value)
    except: # If there's a parsing issue, just return an empty list
        return []

#convert the boardgamecategory info into a readable Python list
df_games["category_list"] = df_games["boardgamecategory"].apply(parse_list)
def split_categories(cat_str):
    if pd.isna(cat_str):
        return []
    return [x.strip() for x in cat_str.split('|') if x.strip()]

#tag everything
tags_column = []
for categories in df_games["category_list"]:
    game_tags = []
    for category in categories:
        if category in cat_to_tag: # Use cat_to_tag if category is recognized
            mapped_tag = cat_to_tag[category]
            game_tags.append(mapped_tag)
        else:             # Category doesn't match any known tag
            game_tags.append("Unknown")
    tags_column.append(game_tags)

df_games["tags"] = tags_column
df_games["tags"] = df_games["tags"].apply(lambda tlist: sorted(set(tlist)))

df_games.to_csv(data_file, index=False)
