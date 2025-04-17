# viz_data.py
import pandas as pd
import matplotlib.pyplot as plt
import re

GAME_DATA = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
# --- Functions for Data Loading & Preprocessing ---
def load_data(filepath):
    """
    Load the CSV file into a pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    data['yearpublished'] = pd.to_numeric(data['yearpublished'], errors='coerce')
    data['bayesaverage'] = pd.to_numeric(data['bayesaverage'], errors='coerce')
    data['averageweight'] = pd.to_numeric(data['averageweight'], errors='coerce')
def clean_year(x):
    """
    Convert a year value (possibly as a string with 'B.C.' or 'BC') to a numeric value.
    If the value represents a BC year, return a negative number.
    """
    try:
        if pd.isna(x):
            return None
        if isinstance(x, str):
            # Check if it represents a BC year
            if "B.C." in x or "BC" in x:
                num_str = re.findall(r'\d+', x)
                if num_str:
                    return -1 * float(num_str[0])
                else:
                    return None
            else:
                num_str = re.findall(r'-?\d+', x)
                if num_str:
                    return float(num_str[0])
                else:
                    return None
        # Otherwise, assume it's already numeric
        return x
    except Exception as e:
        return None
# --- Descriptive Statistics & Basic Reports ---
def print_basic_stats(data, cols_to_print=None):
    """
    Print basic descriptive statistics about the dataset.

    Parameters:
        data (DataFrame): The full dataset.
        cols_to_print (list, optional): List of column names to include in the descriptive summary.
                                         If None, the summary for all columns is printed.
    """
    print("\n--- Basic Statistics ---")
    print(f"Total games in dataset: {len(data)}")

    # Check for expected columns and print their stats if available.
    if 'bayesaverage' in data.columns:
        print(f"Average rating [Bayes] (1-10): {data['bayesaverage'].mean():.2f}")
    if 'averageweight' in data.columns:
        print(f"Average complexity weight (1-5): {data['averageweight'].mean():.2f}")
    if 'yearpublished' in data.columns:
        print(f"Year Published range: {data['yearpublished'].min()} - {data['yearpublished'].max()}")

    # Print descriptive summary for specified columns (or all columns if not specified)
    if cols_to_print is not None:
        summary = data[cols_to_print].describe(include='all')
    else:
        summary = data.describe(include='all')

    print("\nDescriptive Summary:")
    print(summary)

# --- Static Visualizations using Matplotlib ---
def plot_bayes_rating_distribution(data):
    """
    Generate and save a histogram of average game ratings.
    """
    df = data.copy()
    df['bayesaverage'] = pd.to_numeric(df['bayesaverage'], errors='coerce')
    df_clean = df.dropna(subset=['bayesaverage'])

    plt.figure(figsize=(8, 6))
    plt.hist(df_clean['bayesaverage'], bins=20, edgecolor='black', range=(1, 10))
    plt.title("Distribution of Bayesian Average User Ratings")
    plt.xlabel("Bayesian Rating (scale 1-10)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    output_file = "average_bayesaverage_per_year.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Histogram saved to {output_file}")


def plot_games_per_year(data):
    """
    Generate and save visualizations of the number of games published per year
    and the average Bayesian rating per year, filtering the data to only include games from 1970 until now.
    (Note: Negative values representing BC years are automatically excluded.)
    """
    if 'yearpublished' not in data.columns:
        print("Column 'yearpublished' not found in data.")
        return

    # Clean the 'yearpublished' column to ensure it's numeric
    data['yearpublished'] = data['yearpublished'].apply(clean_year)

    # Filter the data for games from 1970 onward (this will exclude negative values)
    filtered_data = data[data['yearpublished'] >= 1970].copy()

    # Count games per year and sort by year
    year_counts = filtered_data['yearpublished'].value_counts().sort_index()

    # Plot and save the games-per-year bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(year_counts.index, year_counts.values)
    plt.title("Games Published per Year (1970-Present)")
    plt.xlabel("Year Published")
    plt.ylabel("Number of Games")
    plt.tight_layout()
    plt.savefig("games_per_year.png")
    plt.close()
    print("Games per year plot saved as games_per_year.png.")

    # Use the Bayesian average for computing yearly averages
    if 'bayesaverage' in filtered_data.columns:
        # Calculate average Bayesian rating per year
        avg_rating_per_year = filtered_data.groupby('yearpublished')['bayesaverage'].mean().reset_index()
        avg_rating_per_year.rename(columns={'bayesaverage': 'average_bayesaverage'}, inplace=True)

        # Save the average Bayesian rating data as a CSV
        avg_rating_per_year.to_csv("average_bayesaverage_per_year.csv", index=False)
        print("Average Bayesian rating per year exported to average_bayesaverage_per_year.csv.")

        # Plot and save the average Bayesian rating per year line chart
        plt.figure(figsize=(10, 6))
        plt.plot(avg_rating_per_year['yearpublished'], avg_rating_per_year['average_bayesaverage'], marker='o')
        plt.title("Average Bayesian Game Rating per Year (1970-Present)")
        plt.xlabel("Year Published")
        plt.ylabel("Average Bayesian Rating")
        plt.tight_layout()
        plt.savefig("average_bayesaverage_per_year.png")
        plt.close()
        print("Average Bayesian rating per year plot saved as average_bayesaverage_per_year.png.")
    else:
        print("Column 'bayesaverage' not found in data; skipping average rating computation.")

# --- Top/Bottom Listings & Exporting Data ---
def export_top_bottom_games(data):
    """
    Create CSVs for the top 25 highest-rated games, oldest 10 games, and newest 10 games.
    """
    if 'bayesaverage' in data.columns:
        top25 = data.sort_values(by='bayesaverage', ascending=False).head(25)
        top25.to_csv("top25_games.csv", index=False)
        print("Top 25 games exported to top25_games.csv.")
    else:
        print("Column 'bayesaverage' not found; skipping top 25 export.")

    if 'yearpublished' in data.columns:
        oldest10 = data.sort_values(by='yearpublished').head(10)
        newest10 = data.sort_values(by='yearpublished', ascending=False).head(10)
        oldest10.to_csv("oldest10_games.csv", index=False)
        newest10.to_csv("newest10_games.csv", index=False)
        print("Oldest 10 and Newest 10 games exported to oldest10_games.csv and newest10_games.csv respectively.")
    else:
        print("Column 'yearpublished' not found; skipping oldest/newest exports.")

# --- Main Execution ---

if __name__ == "__main__":
    # Adjust the filepath to where your CSV is stored
    filepath = "/Users/loriramey/PycharmProjects/BGapp/data/gamedata_final.csv"
    data = load_data(filepath)

    if data is not None:
        cols = ['average', 'yearpublished', 'bayesaverage', 'playingtime']
        print_basic_stats(data, cols_to_print=cols)
        plot_bayes_rating_distribution(data)
        plot_games_per_year(data)
        #export_top_bottom_games(data)
