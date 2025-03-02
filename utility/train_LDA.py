import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load preprocessed data
lda_file = "/Users/loriramey/PycharmProjects/BGapp/data/preprocessed_for_lda.csv"
df = pd.read_csv(lda_file)

# Convert text data into a document-term matrix
vectorizer = CountVectorizer(stop_words='english', max_features=5000)  # Limit to most relevant words
dtm = vectorizer.fit_transform(df['game_tags'])

# Train LDA model with 40 topics
lda = LatentDirichletAllocation(n_components=40, random_state=42, learning_method="batch")
lda.fit(dtm)

# Extract topic keywords
def get_top_words(model, feature_names, n_words=10):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
    return topics

# Get top words for each topic
feature_names = vectorizer.get_feature_names_out()
lda_topics = get_top_words(lda, feature_names, n_words=10)

# Convert topics to DataFrame
lda_df = pd.DataFrame(lda_topics)

# Save LDA topics for review
lda_df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/lda_topics_40.csv", index=False)

# Print sample output
print("\nLDA Training Complete! Top topic keywords saved to lda_topics_40.csv")
print(lda_df.head())
