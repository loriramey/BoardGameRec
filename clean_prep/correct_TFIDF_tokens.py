import pandas as pd

def clean_token_string(entry):
    """
    Convert list of tags/mechanics/categories into a TF-IDF-friendly string.
    Replaces spaces with underscores within items and joins them with spaces.
    """
    try:
        # Handle both stringified lists and actual lists
        items = eval(entry) if isinstance(entry, str) else entry
        if not isinstance(items, list):
            return ""
        return " ".join(
            item.strip().replace(" ", "_").replace("__", "_") for item in items
        )
    except:
        return ""

def fix_token_columns(input_csv, output_csv):
    """
    Cleans and fixes token columns from a board game dataset:
    - tags_str from tags
    - mechanics_str from mech_list
    - categories_str from category_list
    """
    df = pd.read_csv(input_csv)

    # Apply the string cleaning to the proper source columns
    df["tags_str"] = df["tags"].apply(clean_token_string)
    df["mechanics_str"] = df["mech_list"].apply(clean_token_string)
    df["categories_str"] = df["category_list"].apply(clean_token_string)

    # Save to new output CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved cleaned data to {output_csv}")


if __name__ == "__main__":
    # Example usage — update filenames as needed
    input_path = "gamedata_final.csv"
    output_path = "gamedata_with_fixed_tokens_full.csv"
    fix_token_columns(input_path, output_path)