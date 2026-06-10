"""
Unsupervised fake-review detection on Amazon Reviews dataset.
Mirrors the same pipeline as run_phase4.py (IF + AE + Fusion).
No ground-truth labels — outputs anomaly scores only.

Expected CSV columns:
  Reviewer Name, Profile Link, Country, Review Count, Review Date,
  Rating, Review Title, Review Text, Date of Experience
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor

try:
    import torch
    import torch.nn as nn
    import torch.utils.data as data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
if HAS_TORCH:
    torch.manual_seed(RANDOM_STATE)

os.makedirs("p4_outputs_amazon", exist_ok=True)

CSV_PATH = r"D:\Term 7\MDS301\Amazon_Reviews.csv"


# ── helpers ────────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\\n', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def parse_rating(rating_str):
    """Extract numeric rating from 'Rated X out of 5 stars'."""
    if not isinstance(rating_str, str):
        return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)', rating_str)
    return float(m.group(1)) if m else np.nan


def parse_review_count(rc_str):
    """Extract integer from '9 reviews' or '1 review'."""
    if not isinstance(rc_str, str):
        return np.nan
    m = re.search(r'(\d+)', rc_str)
    return int(m.group(1)) if m else np.nan


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 – Load & feature engineering
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 1: LOAD & FEATURE ENGINEERING")
print("="*55)

df = pd.read_csv(CSV_PATH, engine='python', on_bad_lines='skip')
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# Drop rows with missing review text
df = df.dropna(subset=['Review Text']).reset_index(drop=True)
print(f"After dropping missing Review Text: {len(df)} rows")

# ── Text features ─────────────────────────────────────────────────────────
df['text_clean'] = (df['Review Title'].fillna('') + ' ' + df['Review Text'].fillna('')).apply(clean_text)

df['word_count']        = df['text_clean'].apply(lambda x: len(x.split()))
df['char_length']       = df['text_clean'].apply(len)
df['avg_word_len']      = df['char_length'] / (df['word_count'] + 1)
df['exclamation_ratio'] = df['text_clean'].apply(lambda x: x.count('!')) / (df['char_length'] + 1)
df['newline_count']     = df['Review Text'].apply(
    lambda x: x.count('\n') + x.count('\\n') if isinstance(x, str) else 0
)

# ── Behavioral features ───────────────────────────────────────────────────
df['rating']        = df['Rating'].apply(parse_rating)
df['review_count']  = df['Review Count'].apply(parse_review_count).fillna(1)

# Extreme rating flag (1 or 5 stars tend to be more polarized)
df['is_extreme_rating'] = df['rating'].apply(lambda r: 1 if r in [1.0, 5.0] else 0)
# Single-review account flag
df['is_single_reviewer'] = (df['review_count'] == 1).astype(int)

# Parse review date for temporal features
df['review_date'] = pd.to_datetime(df['Review Date'], errors='coerce')
df['hour_of_day'] = df['review_date'].dt.hour.fillna(0)
df['day_of_week'] = df['review_date'].dt.dayofweek.fillna(0)

# Burst: reviews from same profile link in same hour
df['timestamp_hour'] = df['review_date'].dt.floor('h')
df['user_burst_1h'] = df.groupby(['Profile Link', 'timestamp_hour'])['Profile Link'].transform('count')

# user_review_count from Review Count column (behavioral)
df['user_review_count'] = df['review_count']

# account_age_days: days between earliest and current review per profile
min_date_per_user = df.groupby('Profile Link')['review_date'].transform('min')
df['account_age_days'] = (df['review_date'] - min_date_per_user) / pd.Timedelta('1D')
df['account_age_days'] = df['account_age_days'].fillna(0)

# Weak signal placeholders (not available in this dataset)
for col in ['h1_content', 'h2_duplicate', 'h3_burst', 'h4_semantic']:
    df[col] = 0

FEATURES = [
    # Group A – text
    "word_count", "char_length", "avg_word_len",
    "exclamation_ratio", "newline_count",
    # Group B – behavioral
    "user_review_count", "user_burst_1h", "account_age_days",
    "hour_of_day", "day_of_week",
    # Metadata signals
    "is_extreme_rating", "is_single_reviewer",
    # Placeholders
    "h1_content", "h2_duplicate", "h3_burst", "h4_semantic",
]

feature_df = df[['Profile Link', 'text_clean', 'review_date', 'rating'] + FEATURES].copy()
feature_df.to_csv("p4_outputs_amazon/feature_dataset.csv", index=False, encoding='utf-8-sig')
print(f"\nFeature dataset saved. Shape: {feature_df.shape}")
print("\nDescriptive statistics:")
print(feature_df[FEATURES].describe().T.to_string())


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 – Isolation Forest (unsupervised)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 2: ISOLATION FOREST")
print("="*55)

X = feature_df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# contamination=0.1 as a reasonable prior for fake review rate
IF_CONFIG = {
    "n_estimators":  100,
    "contamination": 0.1,
    "max_features":  1.0,
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

clf_if = IsolationForest(**IF_CONFIG)
clf_if.fit(X_scaled)

raw_if = clf_if.score_samples(X_scaled)
if_scores = (raw_if.max() - raw_if) / (raw_if.max() - raw_if.min() + 1e-9)

print(f"IF score stats — mean: {if_scores.mean():.4f}, std: {if_scores.std():.4f}")
print(f"Flagged as anomaly (score >= 0.50): {(if_scores >= 0.5).sum()} / {len(if_scores)}")
print(f"Flagged as anomaly (score >= 0.65): {(if_scores >= 0.65).sum()} / {len(if_scores)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 – Autoencoder
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 3: AUTOENCODER (TF-IDF + SVD embeddings)")
print("="*55)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), analyzer="word", min_df=3)
tfidf_matrix = tfidf.fit_transform(feature_df['text_clean'])
svd = TruncatedSVD(n_components=128, random_state=RANDOM_STATE)
embeddings = svd.fit_transform(tfidf_matrix)
np.save("p4_outputs_amazon/tfidf_embeddings.npy", embeddings)
print(f"TF-IDF SVD embeddings shape: {embeddings.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

# Train on 80% of data (random split, no stratify since no labels)
train_size = int(0.8 * len(embeddings))
perm = np.random.RandomState(RANDOM_STATE).permutation(len(embeddings))
train_idx = perm[:train_size]

if HAS_TORCH:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=64):
            super().__init__()
            mid = max(input_dim // 4, 128)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, mid), nn.ReLU(),
                nn.BatchNorm1d(mid), nn.Dropout(0.1),
                nn.Linear(mid, latent_dim), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, mid), nn.ReLU(),
                nn.BatchNorm1d(mid),
                nn.Linear(mid, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    class EmbeddingDS(data.Dataset):
        def __init__(self, embs):
            self.embs = torch.tensor(embs, dtype=torch.float32)
        def __len__(self): return len(self.embs)
        def __getitem__(self, i): return self.embs[i]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_CONFIG = {"epochs": 50, "lr": 1e-3, "batch_size": 64,
                 "latent_dim": 64, "patience": 8}

    val_idx = perm[train_size:]
    train_loader = data.DataLoader(EmbeddingDS(embeddings[train_idx]),
                                   batch_size=AE_CONFIG["batch_size"], shuffle=True)
    val_loader   = data.DataLoader(EmbeddingDS(embeddings[val_idx]),
                                   batch_size=AE_CONFIG["batch_size"], shuffle=False)

    model_ae = Autoencoder(embeddings.shape[1], AE_CONFIG["latent_dim"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ae.parameters(), lr=AE_CONFIG["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val, best_state, patience_ctr = float('inf'), None, 0

    for epoch in range(1, AE_CONFIG["epochs"] + 1):
        model_ae.train()
        tl = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model_ae(batch), batch)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(batch)
        tl /= len(train_idx)

        model_ae.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                vl += criterion(model_ae(batch), batch).item() * len(batch)
        vl /= len(val_idx)
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model_ae.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}: train={tl:.6f}  val={vl:.6f}")
        if patience_ctr >= AE_CONFIG["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    model_ae.load_state_dict(best_state)
    model_ae.eval()
    with torch.no_grad():
        inp_tr  = torch.tensor(embeddings[train_idx], dtype=torch.float32).to(device)
        errors_train = torch.mean((inp_tr - model_ae(inp_tr)) ** 2, dim=1).cpu().numpy()

        inp_all = torch.tensor(embeddings, dtype=torch.float32).to(device)
        errors_all = torch.mean((inp_all - model_ae(inp_all)) ** 2, dim=1).cpu().numpy()
else:
    print("[WARNING] PyTorch not available. Using MLPRegressor fallback.")
    mid = max(embeddings.shape[1] // 4, 128)
    mlp_ae = MLPRegressor(
        hidden_layer_sizes=(mid, 64, mid), activation='relu', solver='adam',
        alpha=1e-5, batch_size=64, learning_rate_init=1e-3, max_iter=50,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=8,
        random_state=RANDOM_STATE
    )
    mlp_ae.fit(embeddings[train_idx], embeddings[train_idx])
    errors_train = np.mean((embeddings[train_idx] - mlp_ae.predict(embeddings[train_idx])) ** 2, axis=1)
    errors_all   = np.mean((embeddings - mlp_ae.predict(embeddings)) ** 2, axis=1)

mean_err = errors_train.mean()
std_err  = errors_train.std()
threshold_ae = mean_err + 2.0 * std_err
print(f"AE threshold (mean+2*std): {threshold_ae:.6f}")

min_e, max_e = errors_all.min(), errors_all.max()
ae_scores = np.zeros_like(errors_all)
below = errors_all < threshold_ae
if threshold_ae > min_e:
    ae_scores[below]  = 0.5 * (errors_all[below] - min_e) / (threshold_ae - min_e)
if max_e > threshold_ae:
    ae_scores[~below] = 0.5 + 0.5 * (errors_all[~below] - threshold_ae) / (max_e - threshold_ae)
else:
    ae_scores[~below] = 0.5

print(f"AE score stats — mean: {ae_scores.mean():.4f}, std: {ae_scores.std():.4f}")
print(f"Flagged as anomaly (score >= 0.50): {(ae_scores >= 0.5).sum()} / {len(ae_scores)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 – Fusion + export
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 4: SCORE FUSION + EXPORT")
print("="*55)

final_score = 0.4 * if_scores + 0.6 * ae_scores

print(f"Fusion score stats — mean: {final_score.mean():.4f}, std: {final_score.std():.4f}")
print(f"Flagged @ 0.35: {(final_score >= 0.35).sum()}")
print(f"Flagged @ 0.50: {(final_score >= 0.50).sum()}")
print(f"Flagged @ 0.65: {(final_score >= 0.65).sum()}")

# ── Summary table by threshold ─────────────────────────────────────────
summary_rows = []
for thr in [0.35, 0.50, 0.65]:
    flagged = (final_score >= thr).sum()
    summary_rows.append({
        "Threshold": thr,
        "Flagged":   int(flagged),
        "Flagged_%": round(flagged / len(final_score) * 100, 2),
        "Not_Flagged": int(len(final_score) - flagged),
    })
summary_df = pd.DataFrame(summary_rows)
print("\nDetection Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv("p4_outputs_amazon/detection_summary.csv", index=False, encoding='utf-8-sig')

# ── Full predictions ───────────────────────────────────────────────────
out_df = df[['Reviewer Name', 'Profile Link', 'Country', 'Review Count',
             'Review Date', 'Rating', 'Review Title', 'Review Text']].copy()
out_df['if_score']      = np.round(if_scores, 6)
out_df['ae_score']      = np.round(ae_scores, 6)
out_df['final_score']   = np.round(final_score, 6)
out_df['pred_fake_35']  = (final_score >= 0.35).astype(int)
out_df['pred_fake_50']  = (final_score >= 0.50).astype(int)
out_df['pred_fake_65']  = (final_score >= 0.65).astype(int)

pred_path = "p4_outputs_amazon/final_predictions.csv"
out_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
print(f"\nFull predictions saved to: {pred_path}")

# ── Top 20 most suspicious reviews ────────────────────────────────────
top20 = out_df.nlargest(20, 'final_score')[
    ['Reviewer Name', 'Country', 'Rating', 'Review Title', 'final_score', 'if_score', 'ae_score']
]
print("\nTop 20 most suspicious reviews:")
print(top20.to_string(index=False))
top20.to_csv("p4_outputs_amazon/top20_suspicious.csv", index=False, encoding='utf-8-sig')

print("\nAll outputs saved to: p4_outputs_amazon/")
print("Done.")
