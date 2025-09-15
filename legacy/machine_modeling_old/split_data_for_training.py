import pandas as pd
from sklearn.model_selection import train_test_split

#load full data file of ~25,700 games & split 80/20% for testing
GAMEDATA_FILE = "/data/gamedata_final.csv"
df = pd.read_csv(GAMEDATA_FILE)

# Split into 80% training, 20% testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save these for later use
train_df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_train.csv", index=False)
test_df.to_csv("/Users/loriramey/PycharmProjects/BGapp/data/gamedata_test.csv", index=False)

print(f"Training set: {len(train_df)} games")
print(f"Testing set: {len(test_df)} games")