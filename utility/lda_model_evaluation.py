import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Load labeled topic names
topics_file = "/Users/loriramey/PycharmProjects/BGapp/data/lda_topics_40.csv"
topics_df = pd.read_csv(topics_file)

# Load final cleaned dataset
data_file = "/Users/loriramey/PycharmProjects/BGapp/data/final_cleaned_dataset.csv"
df = pd.read_csv(data_file)

# Load preprocessed game tags (used for LDA modeling)
lda_file = "/Users/loriramey/PycharmProjects/BGapp/data/preprocessed_for_lda.csv"
lda_df = pd.read_csv(lda_file)

# Ensure alignment of IDs
df = df.merge(lda_df[['id', 'game_tags']], on='id', how='left')

# Split into training (80%) and testing (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Vectorize game tags for LDA input
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
dtm_train = vectorizer.fit_transform(train_df['game_tags'])

# Train LDA Model
n_topics = 40  # Matching the labeled topics count
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
lda.fit(dtm_train)

# Evaluate on test set
dtm_test = vectorizer.transform(test_df['game_tags'])
test_topic_probs = lda.transform(dtm_test)

# Assign the top 3 topics per game
top_n = 3
top_topics_per_game = np.argsort(-test_topic_probs, axis=1)[:, :top_n]

# Convert topic indices to labeled names
topic_labels = topics_df.columns.tolist()
test_df['Top_Tags'] = [[topic_labels[idx] for idx in row] for row in top_topics_per_game]

# Save results to a CSV file
output_file = "/Users/loriramey/PycharmProjects/BGapp/data/lda_test_set_results.csv"
test_df[['id', 'name', 'Top_Tags']].to_csv(output_file, index=False)

# Print sample output
print("\nLDA Model Evaluation Complete! Test set results saved to lda_test_set_results.csv")
print(test_df[['name', 'Top_Tags']].head(20))