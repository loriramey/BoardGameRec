'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load final cleaned dataset
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/final_cleaned_dataset.csv"
df = pd.read_csv(file_path)

# --- Compare `bayesaverage` vs. `average` ---
correlation = df[['average', 'bayesaverage']].corr()

# --- Distribution of `average` and `bayesaverage` ---
plt.figure(figsize=(10, 5))
sns.histplot(df['average'], bins=50, kde=True, label="Average Rating", color="blue", alpha=0.6)
sns.histplot(df['bayesaverage'], bins=50, kde=True, label="Bayesian Average", color="red", alpha=0.6)
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Average vs. Bayesian Average Ratings")
plt.show()

# --- Scatterplot of `usersrated` vs. `bayesaverage` ---
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['usersrated'], y=df['bayesaverage'], alpha=0.5)
plt.xlabel("Number of Users Rated")
plt.ylabel("Bayesian Average Rating")
plt.title("Bayesian Average vs. Number of Ratings")
plt.xscale("log")  # Log scale to handle large variations
plt.show()

# Display correlation results
print("\nCorrelation between `average` and `bayesaverage`:")
print(correlation)


import pandas as pd

# Load the dataset
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/final_cleaned_dataset.csv"
df = pd.read_csv(file_path)

# Count games with fewer than 50 and 25 ratings
low_rating_count = df[df['usersrated'] < 50].shape[0]
very_low_rating_count = df[df['usersrated'] < 25].shape[0]

# Percentage of dataset affected
total_games = df.shape[0]
low_rating_percentage = (low_rating_count / total_games) * 100
very_low_rating_percentage = (very_low_rating_count / total_games) * 100

# Print results
print(f"Total Games in Dataset: {total_games}")
print(f"Games with < 50 ratings: {low_rating_count} ({low_rating_percentage:.2f}%)")
print(f"Games with < 25 ratings: {very_low_rating_count} ({very_low_rating_percentage:.2f}%)")
'''

import pandas as pd

# Load the dataset
file_path = "/Users/loriramey/PycharmProjects/BGapp/data/final_cleaned_dataset.csv"
df = pd.read_csv(file_path)

# Filter games with at least 25 ratings
filtered_df = df[df['usersrated'] >= 25]

# Rank games by Bayesian Average (descending)
top_10_games = filtered_df.sort_values(by='bayesaverage', ascending=False).head(10)

# Select relevant columns for readability
top_10_games = top_10_games[['name', 'bayesaverage', 'usersrated', 'Board Game Rank']]

# Print results
print("\nTop 10 Games by Bayesian Average (at least 25 ratings):")
print(top_10_games)