import pandas as pd
import numpy as np
import re
import os

def clean_dataset(input_file="labeled_shopee_dataset.csv", output_file="cleaned_shopee_dataset.csv"):
    if not os.path.exists(input_file):
        print(f"[Error] Input file '{input_file}' does not exist. Please run weak_labeling.py first.")
        return

    print(f"[Data Cleaner] Loading dataset from '{input_file}'...")
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")

    # 1. Drop exact duplicate rows (where all columns are identical, typically crawling artifacts)
    df.drop_duplicates(inplace=True)
    post_dedup_rows = len(df)
    exact_duplicates_removed = initial_rows - post_dedup_rows
    print(f"Removed {exact_duplicates_removed} exact row duplicates.")

    # 2. Handle missing values in 'comment' column
    # Drop rows where comment is NaN or null
    df.dropna(subset=['comment'], inplace=True)
    post_null_drop_rows = len(df)
    null_comments_removed = post_dedup_rows - post_null_drop_rows
    print(f"Removed {null_comments_removed} reviews with missing/null comments.")

    # 3. Text cleaning and formatting normalization
    def normalize_text(text):
        if not isinstance(text, str):
            return ""
        # Replace raw newlines (\n, \r) with spaces to keep each CSV row on a single line
        text = re.sub(r'[\r\n]+', ' ', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    df['comment'] = df['comment'].apply(normalize_text)

    # Filter out reviews where comment became empty after stripping
    df = df[df['comment'] != ""]
    post_empty_strip_rows = len(df)
    empty_strips_removed = post_null_drop_rows - post_empty_strip_rows
    print(f"Removed {empty_strips_removed} reviews that had only empty/whitespace comments.")

    # 4. Fill missing values in other columns
    if 'image_count' in df.columns:
        df['image_count'] = df['image_count'].fillna(0).astype(int)
    if 'rating' in df.columns:
        df['rating'] = df['rating'].fillna(5).astype(int)
    if 'is_purchased' in df.columns:
        df['is_purchased'] = df['is_purchased'].fillna(True)

    # 5. Feature Engineering helper columns
    print("[Data Cleaner] Generating helper features (character length and word count)...")
    df['char_length'] = df['comment'].apply(len)
    df['word_count'] = df['comment'].apply(lambda x: len(x.split()))

    # Ensure correct data types for heuristics and label columns
    flag_cols = ['h1_content', 'h2_duplicate', 'h3_burst', 'h4_semantic', 'is_suspicious']
    for col in flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # 6. Save final cleaned dataset
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n[Success] Cleaned dataset saved to '{output_file}'.")
    print(f"Final clean row count: {len(df)}")
    
    # Summary stats
    if 'is_suspicious' in df.columns:
        suspicious_count = df['is_suspicious'].sum()
        pct = (suspicious_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"Cleaned Suspicious Reviews: {suspicious_count} ({pct:.2f}%)")

if __name__ == "__main__":
    clean_dataset()
