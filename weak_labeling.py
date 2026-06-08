import pandas as pd
import numpy as np
import re
import os
import argparse
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """Basic text cleaning before NLP processing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove special characters, keep only words, numbers, and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_duplicate_reviews(df):
    """Compute Cosine Similarity between reviews in the dataset to flag duplicates > 90%."""
    if len(df) < 2:
        return pd.Series(0, index=df.index)
    
    texts_tokenized = []
    indices = df.index
    
    for text in df['comment']:
        cleaned = clean_text(text)
        if not cleaned:
            texts_tokenized.append("")
        else:
            # Word tokenize using underthesea for Vietnamese compound words
            tokens = word_tokenize(cleaned, format="text")
            texts_tokenized.append(tokens)
            
    if all(t == "" for t in texts_tokenized):
        return pd.Series(0, index=df.index)
        
    try:
        # Fit TF-IDF on tokenized review text
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform(texts_tokenized)
        
        # Calculate Cosine Similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Set diagonal elements to 0 to ignore self-similarity
        np.fill_diagonal(cosine_sim, 0)
        
        # Flag if the max similarity with any other review is > 0.90
        is_dup = (cosine_sim.max(axis=1) > 0.90).astype(int)
        return pd.Series(is_dup, index=indices)
    except Exception as e:
        print(f"[Warning] TF-IDF duplicate detection failed: {e}")
        return pd.Series(0, index=df.index)

def apply_weak_labeling(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"[Error] Input file '{input_csv}' does not exist. Please run shopee_crawler.py first.")
        return
        
    print(f"[Weak Labeling] Loading dataset from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    
    if len(df) == 0:
        print("[Warning] Dataset is empty.")
        return
        
    # Copy dataframe for labeling
    df_labeled = df.copy()
    
    # ----------------------------------------------------
    # Heuristic 1: Content-based (Rating 5 but text length < 10 words)
    # ----------------------------------------------------
    print("[Weak Labeling] Applying Heuristic 1: Short 5-star reviews...")
    word_counts = []
    for idx, row in df_labeled.iterrows():
        comment = str(row['comment']) if pd.notna(row['comment']) else ""
        if row['rating'] == 5:
            tokens = word_tokenize(comment)
            word_counts.append(len(tokens))
        else:
            word_counts.append(999) # Ignored if not 5 stars
            
    df_labeled['h1_content'] = ((df_labeled['rating'] == 5) & (pd.Series(word_counts, index=df_labeled.index) < 10)).astype(int)
    
    # ----------------------------------------------------
    # Heuristic 2: Duplicate Detection (Cosine Similarity > 90%)
    # ----------------------------------------------------
    print("[Weak Labeling] Applying Heuristic 2: Duplicate detection (Cosine Similarity > 90%)...")
    df_labeled['h2_duplicate'] = check_duplicate_reviews(df_labeled)
    df_labeled['h2_duplicate'] = df_labeled['h2_duplicate'].fillna(0).astype(int)
    
    # ----------------------------------------------------
    # Heuristic 3: Burst Detection (> 10 reviews/hour)
    # ----------------------------------------------------
    print("[Weak Labeling] Applying Heuristic 3: Burst detection (> 10 reviews/hour)...")
    df_labeled['timestamp'] = pd.to_datetime(df_labeled['timestamp'])
    
    # Round timestamp to nearest hour to bin them
    df_labeled['hour_bin'] = df_labeled['timestamp'].dt.floor('h')
    
    # Calculate group count in each hour bin
    burst_counts = df_labeled.groupby('hour_bin')['user_id'].transform('count')
    df_labeled['h3_burst'] = (burst_counts > 10).astype(int)
    
    # Drop temporary bin column
    df_labeled = df_labeled.drop(columns=['hour_bin'])
    
    # ----------------------------------------------------
    # Heuristic 4: Semantic Mismatch / Spam Keywords
    # ----------------------------------------------------
    print("[Weak Labeling] Applying Heuristic 4: Semantic mismatch & Spam keywords...")
    # Spam words related to Shopee coins, song lyrics, recipes, or character repetitions
    keywords_xu = r'(farm xu|nhận xu|kiếm xu|lấy xu|tích xu|shopee xu|100 xu|hình ảnh mang tính chất|nhan xu|farmxu)'
    keywords_lyrics = r'(lời bài hát|loi bai hat|người lạ ơi|chàng trai viết lên cây|em ơi xích lại gần đây)'
    keywords_recipes = r'(công thức làm bánh|công thức nấu|công thức phở|hấp cách thủy|ninh xương)'
    pattern_gibberish = r'([a-z])\1{4,}|(ahsdg|qwert|zxcvb|hjkl)'
    
    combined_regex = f"{keywords_xu}|{keywords_lyrics}|{keywords_recipes}|{pattern_gibberish}"
    
    # Ensure empty reviews are handled
    df_labeled['comment_filled'] = df_labeled['comment'].fillna("")
    df_labeled['h4_semantic'] = df_labeled['comment_filled'].str.lower().apply(lambda x: 1 if re.search(combined_regex, x) else 0)
    df_labeled = df_labeled.drop(columns=['comment_filled'])
    
    # ----------------------------------------------------
    # Combine Heuristics (Logical OR)
    # ----------------------------------------------------
    df_labeled['is_suspicious'] = (
        (df_labeled['h1_content'] == 1) |
        (df_labeled['h2_duplicate'] == 1) |
        (df_labeled['h3_burst'] == 1) |
        (df_labeled['h4_semantic'] == 1)
    ).astype(int)
    
    # Save the output CSV file
    df_labeled.to_csv(output_csv, index=False)
    
    # Print metrics
    total = len(df_labeled)
    suspicious = df_labeled['is_suspicious'].sum()
    print("\n--- Weak Labeling Results ---")
    print(f"Total reviews processed: {total}")
    print(f"Suspicious reviews flagged: {suspicious} ({suspicious/total*100:.2f}%)")
    print(f"  - Heuristic 1 (Short 5-star): {df_labeled['h1_content'].sum()}")
    print(f"  - Heuristic 2 (Duplicate): {df_labeled['h2_duplicate'].sum()}")
    print(f"  - Heuristic 3 (Burst): {df_labeled['h3_burst'].sum()}")
    print(f"  - Heuristic 4 (Semantic/Spam): {df_labeled['h4_semantic'].sum()}")
    print(f"Saved labeled dataset to: '{output_csv}'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weak Labeling Engine for Shopee Dataset")
    parser.add_argument("--input", type=str, default="raw_shopee_dataset.csv", help="Input raw CSV file")
    parser.add_argument("--output", type=str, default="labeled_shopee_dataset.csv", help="Output labeled CSV file")
    
    args = parser.parse_args()
    apply_weak_labeling(args.input, args.output)
