import pandas as pd
datafile = "/Users/loriramey/PycharmProjects/BGapp/data/final_tagged_data.csv"
categoryfile = "/Users/loriramey/PycharmProjects/BGapp/data/categories_list.csv"
mechanicsfile = "/Users/loriramey/PycharmProjects/BGapp/data/mechanics_list.csv"

df = pd.read_csv(datafile)
cat_df = pd.read_csv(categoryfile)
mech_df = pd.read_csv(mechanicsfile)
tags_df = pd.read_csv("/Users/loriramey/PycharmProjects/BGapp/data/tags.csv")

# Display first few rows for verification
print("Raw Data:\n", df.head(), "\n")
print("Categories Data:\n", cat_df.head(), "\n")
print("Mechanics Data:\n", mech_df.head(), "\n")
print("Tags Data:\n", tags_df.head(), "\n")
