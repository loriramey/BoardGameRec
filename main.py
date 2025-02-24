import kagglehub

# Download latest version
path = kagglehub.dataset_download("bwandowando/boardgamegeek-board-games-reviews-jan-2025")

print("Path to dataset files:", path)