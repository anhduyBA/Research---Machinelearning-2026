"""
Baseline evaluation on Ott et al. Deceptive Opinion Spam dataset.
Mirrors the same pipeline as run_phase4.py (IF + AE + Fusion) so results
are directly comparable.

Expected CSV columns: deceptive, hotel, polarity, source, text
  deceptive: "deceptive" | "truthful"  ->  label 1 / 0
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
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

os.makedirs("p4_outputs_deceptive", exist_ok=True)

CSV_PATH = r"D:\Term 7\MDS301\deceptive-opinion.csv"


# ── helpers ────────────────────────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\\n', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def evaluate_model(y_true, y_pred, scores):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, scores),
        "PR-AUC":    average_precision_score(y_true, scores),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred),
    }


def print_evaluation(name, m):
    print(f"\n--- {name} ---")
    print(f"Precision : {m['Precision']:.4f}")
    print(f"Recall    : {m['Recall']:.4f}")
    print(f"F1 Score  : {m['F1']:.4f}")
    print(f"AUC-ROC   : {m['AUC-ROC']:.4f}")
    print(f"PR-AUC    : {m['PR-AUC']:.4f}")
    print("Confusion Matrix:")
    print(m['Confusion_Matrix'])


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 – Load & feature engineering
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 1: LOAD & FEATURE ENGINEERING")
print("="*55)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# Label: deceptive=1, truthful=0
df['label'] = (df['deceptive'] == 'deceptive').astype(int)
print(f"Label distribution:\n{df['label'].value_counts()}")

df['text_clean'] = df['text'].apply(clean_text)

# Text-derived features (mirror Group A from run_phase4.py)
df['word_count']        = df['text_clean'].apply(lambda x: len(x.split()))
df['char_length']       = df['text_clean'].apply(len)
df['avg_word_len']      = df['char_length'] / (df['word_count'] + 1)
df['exclamation_ratio'] = df['text_clean'].apply(lambda x: x.count('!')) / (df['char_length'] + 1)
df['newline_count']     = df['text'].apply(
    lambda x: x.count('\n') + x.count('\\n') if isinstance(x, str) else 0
)

# Metadata features available in this dataset
df['is_positive']   = (df['polarity'] == 'positive').astype(int)
df['is_tripadvisor'] = (df['source'] == 'TripAdvisor').astype(int)

# Behavioral features not available → set to 0
for col in ['user_review_count', 'user_burst_1h', 'image_count',
            'account_age_days', 'h1_content', 'h2_duplicate',
            'h3_burst', 'h4_semantic']:
    df[col] = 0

FEATURES = [
    # Group A – text
    "word_count", "char_length", "avg_word_len",
    "exclamation_ratio", "newline_count",
    # Metadata (replaces behavioral / weak signals)
    "is_positive", "is_tripadvisor",
    # Placeholder columns kept so IF config matches phase4 shape
    "user_review_count", "user_burst_1h", "image_count",
    "account_age_days", "h1_content", "h2_duplicate",
    "h3_burst", "h4_semantic",
]

feature_df = df[['text_clean', 'label'] + FEATURES].copy()
feature_df.to_csv("p4_outputs_deceptive/feature_dataset.csv", index=False, encoding='utf-8-sig')
print(f"\nFeature dataset saved. Shape: {feature_df.shape}")
print("\nCorrelation with label (sorted):")
corr = feature_df[FEATURES + ['label']].corr()['label'].sort_values(ascending=False)
print(corr)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 – Isolation Forest
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 2: ISOLATION FOREST")
print("="*55)

X = feature_df[FEATURES].values
y = feature_df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

contamination = y.mean()  # use true proportion
IF_CONFIG = {
    "n_estimators":  100,
    "contamination": contamination,
    "max_features":  1.0,
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}
print(f"Contamination set to true label proportion: {contamination:.4f}")

clf_if = IsolationForest(**IF_CONFIG)
clf_if.fit(X_scaled)

raw_if = clf_if.score_samples(X_scaled)
if_scores = (raw_if.max() - raw_if) / (raw_if.max() - raw_if.min() + 1e-9)

y_pred_if_50 = (if_scores >= 0.50).astype(int)
m_if_50 = evaluate_model(y, y_pred_if_50, if_scores)
print_evaluation("Isolation Forest (threshold=0.50)", m_if_50)

y_pred_if_65 = (if_scores >= 0.65).astype(int)
m_if_65 = evaluate_model(y, y_pred_if_65, if_scores)
print_evaluation("Isolation Forest (threshold=0.65)", m_if_65)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1s, cv_rocs = [], []
for fold, (tr, val) in enumerate(skf.split(X_scaled, y), 1):
    fs = StandardScaler()
    Xtr = fs.fit_transform(X_scaled[tr])
    Xval = fs.transform(X_scaled[val])
    clf_cv = IsolationForest(**IF_CONFIG)
    clf_cv.fit(Xtr)
    tr_raw = clf_cv.score_samples(Xtr)
    val_raw = clf_cv.score_samples(Xval)
    val_s = np.clip((tr_raw.max() - val_raw) / (tr_raw.max() - tr_raw.min() + 1e-9), 0, 1)
    cv_f1s.append(f1_score(y[val], (val_s >= 0.5).astype(int), zero_division=0))
    cv_rocs.append(roc_auc_score(y[val], val_s))

print(f"\n5-Fold CV (threshold=0.50):")
print(f"  F1      : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
print(f"  AUC-ROC : {np.mean(cv_rocs):.4f} ± {np.std(cv_rocs):.4f}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 – Autoencoder (TF-IDF + SVD fallback; PhoBERT N/A for English)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 3: AUTOENCODER (TF-IDF + SVD embeddings)")
print("="*55)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), analyzer="word", min_df=2)
tfidf_matrix = tfidf.fit_transform(feature_df['text_clean'])
svd = TruncatedSVD(n_components=128, random_state=RANDOM_STATE)
embeddings = svd.fit_transform(tfidf_matrix)
np.save("p4_outputs_deceptive/tfidf_embeddings.npy", embeddings)
print(f"TF-IDF SVD embeddings shape: {embeddings.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

train_idx, val_idx = train_test_split(
    np.arange(len(embeddings)), test_size=0.2,
    random_state=RANDOM_STATE, stratify=y
)

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
    AE_CONFIG = {"epochs": 50, "lr": 1e-3, "batch_size": 32,
                 "latent_dim": 64, "patience": 8, "std_mult": 2.0}

    train_loader = data.DataLoader(EmbeddingDS(embeddings[train_idx]),
                                   batch_size=AE_CONFIG["batch_size"], shuffle=True)
    model_ae = Autoencoder(embeddings.shape[1], AE_CONFIG["latent_dim"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ae.parameters(), lr=AE_CONFIG["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val, best_state, patience_ctr = float('inf'), None, 0
    val_loader = data.DataLoader(EmbeddingDS(embeddings[val_idx]),
                                  batch_size=AE_CONFIG["batch_size"], shuffle=False)
    for epoch in range(1, AE_CONFIG["epochs"] + 1):
        model_ae.train()
        tl = sum(
            (lambda b: (model_ae.zero_grad(),
                        (loss := criterion(model_ae(b := b.to(device)), b)),
                        loss.backward(), optimizer.step(), loss.item() * len(b))[-1])(batch)
            for batch in train_loader
        ) / len(train_idx)

        model_ae.eval()
        with torch.no_grad():
            vl = sum(
                criterion(model_ae(b.to(device)), b.to(device)).item() * len(b)
                for b in val_loader
            ) / len(val_idx)
        scheduler.step(vl)
        if vl < best_val:
            best_val, best_state, patience_ctr = vl, {k: v.clone() for k, v in model_ae.state_dict().items()}, 0
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
        e_tr = torch.mean(
            (t := torch.tensor(embeddings[train_idx], dtype=torch.float32).to(device)) -
            model_ae(t), dim=1).pow(2).cpu().numpy()
        # recompute properly
        inp_tr = torch.tensor(embeddings[train_idx], dtype=torch.float32).to(device)
        out_tr = model_ae(inp_tr)
        errors_train = torch.mean((inp_tr - out_tr) ** 2, dim=1).cpu().numpy()

        inp_all = torch.tensor(embeddings, dtype=torch.float32).to(device)
        out_all = model_ae(inp_all)
        errors_all = torch.mean((inp_all - out_all) ** 2, dim=1).cpu().numpy()
else:
    print("[WARNING] PyTorch not available. Using MLPRegressor fallback.")
    mid = max(embeddings.shape[1] // 4, 128)
    mlp_ae = MLPRegressor(
        hidden_layer_sizes=(mid, 64, mid), activation='relu', solver='adam',
        alpha=1e-5, batch_size=32, learning_rate_init=1e-3, max_iter=50,
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

y_pred_ae = (ae_scores >= 0.5).astype(int)
m_ae = evaluate_model(y, y_pred_ae, ae_scores)
print_evaluation("Autoencoder (threshold=0.50)", m_ae)


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 – Fusion + benchmark
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("STEP 4: SCORE FUSION + BENCHMARK")
print("="*55)

final_score = 0.4 * if_scores + 0.6 * ae_scores

m_fuse_50 = evaluate_model(y, (final_score >= 0.50).astype(int), final_score)
m_fuse_65 = evaluate_model(y, (final_score >= 0.65).astype(int), final_score)
m_fuse_35 = evaluate_model(y, (final_score >= 0.35).astype(int), final_score)

print_evaluation("Fusion (threshold=0.50)", m_fuse_50)
print_evaluation("Fusion (threshold=0.65)", m_fuse_65)
print_evaluation("Fusion (threshold=0.35)", m_fuse_35)

benchmark = [
    {"Model": "IF @ 0.50",        **{k: m_if_50[k]   for k in ["Precision","Recall","F1","AUC-ROC","PR-AUC"]}},
    {"Model": "IF @ 0.65",        **{k: m_if_65[k]   for k in ["Precision","Recall","F1","AUC-ROC","PR-AUC"]}},
    {"Model": "AE @ 0.50",        **{k: m_ae[k]      for k in ["Precision","Recall","F1","AUC-ROC","PR-AUC"]}},
    {"Model": "Fusion @ 0.50",    **{k: m_fuse_50[k] for k in ["Precision","Recall","F1","AUC-ROC","PR-AUC"]}},
    {"Model": "Fusion @ 0.65",    **{k: m_fuse_65[k] for k in ["Precision","Recall","F1","AUC-ROC","PR-AUC"]}},
    {"Model": "Fusion @ 0.35",    **{k: m_fuse_35[k] for k in ["Precision","Recall","F1","AUC-ROC","PR-AUC"]}},
]

bench_df = pd.DataFrame(benchmark)
bench_df.to_csv("p4_outputs_deceptive/benchmark_results.csv", index=False, encoding='utf-8-sig')

print("\n" + "="*55)
print("BENCHMARK TABLE (Deceptive Opinion Spam Dataset)")
print("="*55)
print(bench_df.to_string(index=False))

# Save predictions
pd.DataFrame({
    "text":         feature_df["text_clean"],
    "label":        y,
    "if_score":     if_scores,
    "ae_score":     ae_scores,
    "final_score":  final_score,
    "pred_fuse_50": (final_score >= 0.50).astype(int),
    "pred_fuse_65": (final_score >= 0.65).astype(int),
    "pred_fuse_35": (final_score >= 0.35).astype(int),
}).to_csv("p4_outputs_deceptive/final_predictions.csv", index=False, encoding='utf-8-sig')

print("\nAll outputs saved to: p4_outputs_deceptive/")
print("Done.")
