import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Load preprocessed data (without modifying it)
lda_file = "/Users/loriramey/PycharmProjects/BGapp/data/preprocessed_for_lda.csv"
df = pd.read_csv(lda_file)

# Convert text data into a document-term matrix
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Limit to most relevant words
dtm = vectorizer.fit_transform(df['game_tags'])

# Train LDA model with 40 topics
lda = LatentDirichletAllocation(n_components=40, random_state=42, learning_method="batch")
lda.fit(dtm)

# Get topic probability distribution for each game
lda_probs = lda.transform(dtm)

# Assign top 5 topics per game
top_n = 5  # Adjust this if needed
top_topics_per_game = np.argsort(-lda_probs, axis=1)[:, :top_n]  # Get indices of top topics per game

# Convert topic indices to human-readable topic names
topic_labels = [f"Topic {i+1}" for i in range(40)]
top_topic_names = [[topic_labels[idx] for idx in row] for row in top_topics_per_game]

# Add topic assignments to the dataset
df['Top_Topics'] = top_topic_names

# Save to a new CSV file (without modifying preprocessed_for_lda.csv)
output_file = "/Users/loriramey/PycharmProjects/BGapp/data/lda_game_topics.csv"
df[['id', 'name', 'Top_Topics']].to_csv(output_file, index=False)

print(f"\nLDA Topic Probability Assignment Complete! Results saved to {output_file}")
print(df[['name', 'Top_Topics']].head())
