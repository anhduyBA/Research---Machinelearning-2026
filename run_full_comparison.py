# -*- coding: utf-8 -*-
"""
Unified benchmark on the Shopee dataset under ONE shared protocol, so every
number is directly comparable (apple-to-apple).

Shared protocol (paper Novoa-Paradela et al. 2024, EAAI 133:108065):
  * 10-fold CV over NORMAL reviews. Each fold: train detector on 9/10 normal
    reviews; test on the held-out normal fold + ALL anomalous reviews.
  * Anomalous class = positive. Report mean F1 +/- std (paper Tables 2-3 style).

Compared along two axes:
  Feature sets
    - Text     : MPNet sentence embeddings        (paper, group A)
    - Behavioral: 13 hand-crafted features          (your pipeline, group B)
    - Hybrid   : MPNet + behavioral concatenated    (group C)
  Detectors
    - Autoencoder (DAEF-style, IQR threshold)
    - One-Class SVM
    - Isolation Forest
    - Local Outlier Factor
    - Fusion (0.4*IF + 0.6*AE)  <- your run_phase4 method
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RANDOM_STATE = 42
N_FOLDS = 10
np.random.seed(RANDOM_STATE)
if HAS_TORCH:
    torch.manual_seed(RANDOM_STATE)

OUT_DIR = "p4_outputs_paper"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = "labeled_shopee_dataset.csv"
EMB_CACHE = os.path.join(OUT_DIR, "mpnet_embeddings.npy")
MPNET_CHECKPOINT = "paraphrase-multilingual-mpnet-base-v2"


def clean_comment(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\\n', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def count_newlines(raw):
    if not isinstance(raw, str):
        return 0
    return raw.count('\n') + raw.count('\\n')


# ── 1. Data + behavioral feature engineering (mirrors run_phase4.py) ──────────
df = pd.read_csv(DATA_PATH)
df['comment'] = df['comment'].fillna("")
df['comment_clean'] = df['comment'].apply(clean_comment)
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['word_count'] = df['comment_clean'].apply(lambda x: len(x.split()))
df['char_length'] = df['comment_clean'].apply(len)
df['avg_word_len'] = df['char_length'] / (df['word_count'] + 1)
df['exclamation_ratio'] = df['comment_clean'].apply(lambda x: x.count('!')) / (df['char_length'] + 1)
df['newline_count'] = df['comment'].apply(count_newlines)

df['user_review_count'] = df['user_id'].map(df['user_id'].value_counts())
df['timestamp_hour'] = df['timestamp'].dt.floor('h')
df['user_burst_1h'] = df.groupby(['user_id', 'timestamp_hour'])['user_id'].transform('count')
min_ts = df.groupby('user_id')['timestamp'].transform('min')
df['account_age_days'] = (df['timestamp'] - min_ts) / pd.Timedelta('1D')

for col in ['h1_content', 'h2_duplicate', 'h3_burst', 'h4_semantic', 'image_count']:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        df[col] = 0

# NOTE: is_suspicious = h1 OR h2 OR h3 OR h4 (see weak_labeling.py). The h1-h4
# columns therefore LEAK the label. We evaluate behavioral features both WITH
# them (upper-bound / sanity check) and WITHOUT them (honest comparison).
BEHAV_BASE = [
    "word_count", "char_length", "avg_word_len", "exclamation_ratio", "newline_count",
    "user_review_count", "user_burst_1h", "image_count", "account_age_days",
]
WEAK_SIGNALS = ["h1_content", "h2_duplicate", "h3_burst", "h4_semantic"]
BEHAV_FEATURES = BEHAV_BASE + WEAK_SIGNALS
behav = df[BEHAV_FEATURES].astype(float).values
behav_noleak = df[BEHAV_BASE].astype(float).values
y_all = df['is_suspicious'].fillna(0).astype(int).values
print(f"Reviews: {len(df)} | normal={int((y_all==0).sum())} | anomalous={int((y_all==1).sum())}")
print(f"Behavioral features: {len(BEHAV_FEATURES)}")


# ── 2. MPNet text embeddings (cache) ─────────────────────────────────────────
text_emb = None
encoder_name = f"MPNet ({MPNET_CHECKPOINT})"
if os.path.exists(EMB_CACHE):
    cached = np.load(EMB_CACHE)
    if len(cached) == len(df):
        text_emb = cached.astype(np.float32)
        print(f"Loaded cached MPNet embeddings: {text_emb.shape}")
if text_emb is None:
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(MPNET_CHECKPOINT)
        text_emb = m.encode(df['comment_clean'].tolist(), batch_size=32,
                            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(EMB_CACHE, text_emb)
    except Exception as e:
        print(f"[WARN] MPNet failed ({e}); TF-IDF+SVD fallback.")
        mat = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                              analyzer="char_wb", min_df=2).fit_transform(df['comment_clean'])
        text_emb = TruncatedSVD(n_components=256, random_state=RANDOM_STATE
                                ).fit_transform(mat).astype(np.float32)
        encoder_name = "TF-IDF + SVD (fallback)"

FEATURE_SETS = {
    "Text (MPNet)":          text_emb,
    "Behavioral (no h1-h4)": behav_noleak,
    "Behavioral (+leak)":    behav,
    "Hybrid (no h1-h4)":     np.hstack([text_emb, behav_noleak]),
}
print(f"Encoder: {encoder_name}\n")


# ── 3. Autoencoder helper ────────────────────────────────────────────────────
def train_ae_errors(X_tr, X_te):
    if HAS_TORCH:
        dim = X_tr.shape[1]
        mid = max(dim // 4, 128)

        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(dim, mid), nn.ReLU(),
                                         nn.Linear(mid, 64), nn.ReLU())
                self.dec = nn.Sequential(nn.Linear(64, mid), nn.ReLU(),
                                         nn.Linear(mid, dim))

            def forward(self, x):
                return self.dec(self.enc(x))

        net = AE()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.MSELoss()
        Xt = torch.tensor(X_tr, dtype=torch.float32)
        net.train()
        for _ in range(80):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 32):
                b = Xt[perm[i:i + 32]]
                opt.zero_grad()
                crit(net(b), b).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            et = torch.mean((Xt - net(Xt)) ** 2, dim=1).numpy()
            Xe = torch.tensor(X_te, dtype=torch.float32)
            ee = torch.mean((Xe - net(Xe)) ** 2, dim=1).numpy()
        return et, ee
    else:
        from sklearn.neural_network import MLPRegressor
        mid = max(X_tr.shape[1] // 4, 128)
        net = MLPRegressor(hidden_layer_sizes=(mid, 64, mid), activation='relu',
                           solver='adam', alpha=1e-5, max_iter=80,
                           random_state=RANDOM_STATE).fit(X_tr, X_tr)
        et = np.mean((X_tr - net.predict(X_tr)) ** 2, axis=1)
        ee = np.mean((X_te - net.predict(X_te)) ** 2, axis=1)
        return et, ee


def iqr_threshold(errors, k=1.5):
    q1, q3 = np.percentile(errors, [25, 75])
    return q3 + k * (q3 - q1)


def norm01(train_vals, test_vals):
    lo, hi = train_vals.min(), train_vals.max()
    return np.clip((test_vals - lo) / (hi - lo + 1e-9), 0, 1)


# ── 4. Shared 10-fold one-class CV ───────────────────────────────────────────
normal_idx = np.where(y_all == 0)[0]
anom_idx = np.where(y_all == 1)[0]
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

DETECTORS = ["Autoencoder", "One-Class SVM", "Isolation Forest",
             "Local Outlier Factor", "Fusion (IF+AE)"]
acc = {(fs, det): {"Precision": [], "Recall": [], "F1": [], "AUC-ROC": [], "PR-AUC": []}
       for fs in FEATURE_SETS for det in DETECTORS}

print(f"Running {N_FOLDS}-fold one-class CV across {len(FEATURE_SETS)} feature sets...")

for fold, (tr_pos, te_pos) in enumerate(kf.split(normal_idx), 1):
    tr_norm = normal_idx[tr_pos]
    test_idx = np.concatenate([normal_idx[te_pos], anom_idx])
    y_te = y_all[test_idx]

    for fs_name, X_full in FEATURE_SETS.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_full[tr_norm])
        X_te = scaler.transform(X_full[test_idx])

        # AE
        et, ee = train_ae_errors(X_tr, X_te)
        ae_pred = (ee >= iqr_threshold(et)).astype(int)
        ae_score = ee

        # OC-SVM
        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X_tr)
        oc_pred = (oc.predict(X_te) == -1).astype(int)
        oc_score = -oc.decision_function(X_te)

        # IF
        if_ = IsolationForest(n_estimators=200, contamination=0.1,
                              random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr)
        if_pred = (if_.predict(X_te) == -1).astype(int)
        if_score = -if_.score_samples(X_te)
        if_train_score = -if_.score_samples(X_tr)

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                                 contamination=0.1).fit(X_tr)
        lof_pred = (lof.predict(X_te) == -1).astype(int)
        lof_score = -lof.decision_function(X_te)

        # Fusion (your run_phase4 method): 0.4*IF + 0.6*AE, normalized on train
        if_n = norm01(if_train_score, if_score)
        ae_n = norm01(et, ee)
        fus_score = 0.4 * if_n + 0.6 * ae_n
        fus_train = 0.4 * norm01(if_train_score, if_train_score) + 0.6 * norm01(et, et)
        fus_thr = np.percentile(fus_train, 90)  # contamination ~0.1
        fus_pred = (fus_score >= fus_thr).astype(int)

        bundle = {
            "Autoencoder":          (ae_pred, ae_score),
            "One-Class SVM":        (oc_pred, oc_score),
            "Isolation Forest":     (if_pred, if_score),
            "Local Outlier Factor": (lof_pred, lof_score),
            "Fusion (IF+AE)":       (fus_pred, fus_score),
        }
        for det, (pred, score) in bundle.items():
            d = acc[(fs_name, det)]
            d["Precision"].append(precision_score(y_te, pred, zero_division=0))
            d["Recall"].append(recall_score(y_te, pred, zero_division=0))
            d["F1"].append(f1_score(y_te, pred, zero_division=0))
            d["AUC-ROC"].append(roc_auc_score(y_te, score))
            d["PR-AUC"].append(average_precision_score(y_te, score))

    print(f"  fold {fold:02d} done")


# ── 5. Unified table ─────────────────────────────────────────────────────────
rows, num_rows = [], []
for fs_name in FEATURE_SETS:
    for det in DETECTORS:
        d = acc[(fs_name, det)]
        row = {"Feature set": fs_name, "Detector": det}
        nrow = {"Feature set": fs_name, "Detector": det}
        for met in ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]:
            v = np.array(d[met])
            row[met] = f"{v.mean():.3f}+/-{v.std():.3f}"
            nrow[met + "_mean"] = round(float(v.mean()), 4)
            nrow[met + "_std"] = round(float(v.std()), 4)
        rows.append(row)
        num_rows.append(nrow)

bench = pd.DataFrame(rows)
print("\n" + "=" * 90)
print(f"UNIFIED {N_FOLDS}-FOLD ONE-CLASS CV BENCHMARK (anomalous = positive)   "
      f"Encoder: {encoder_name}")
print("=" * 90)
print(bench.to_string(index=False))

bench.to_csv(os.path.join(OUT_DIR, "unified_benchmark.csv"),
             index=False, encoding="utf-8-sig")
pd.DataFrame(num_rows).to_csv(os.path.join(OUT_DIR, "unified_benchmark_numeric.csv"),
                              index=False, encoding="utf-8-sig")

# best F1 per feature set
print("\nBest F1 per feature set:")
nb = pd.DataFrame(num_rows)
for fs_name in FEATURE_SETS:
    sub = nb[nb["Feature set"] == fs_name]
    best = sub.loc[sub["F1_mean"].idxmax()]
    print(f"  {fs_name:14s}: {best['Detector']:22s} F1={best['F1_mean']:.3f}")

print(f"\nSaved: {os.path.join(OUT_DIR, 'unified_benchmark.csv')}")
print("Done.")
