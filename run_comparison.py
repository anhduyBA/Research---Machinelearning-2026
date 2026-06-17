# -*- coding: utf-8 -*-
"""
Faithful comparison of the paper's text-only anomaly detectors on the Shopee
dataset, using the paper's evaluation protocol (k-fold CV, train on NORMAL only).

    Novoa-Paradela et al. (2024), EAAI 133:108065
    "Explained anomaly detection in text reviews"

Protocol (paper Section 5.1):
  * Encode reviews with MPNet (multilingual checkpoint for Vietnamese).
  * Split the NORMAL reviews into K folds. In each run, K-1 folds of normal
    data train the detector; the held-out normal fold is tested together with
    the anomalous reviews. Anomalies are never used for training.
  * Report mean F1 +/- std (anomalous class = positive), as in paper Tables 2-3.

Models compared (all trained one-class on normal embeddings):
  - Autoencoder (DAEF-style), threshold = Q3 + 1.5*IQR of train recon errors
  - One-Class SVM
  - Isolation Forest
  - Local Outlier Factor
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


# ── 1. Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['comment_clean'] = df['comment'].fillna("").apply(clean_comment)
y_all = df['is_suspicious'].fillna(0).astype(int).values
print(f"Reviews: {len(df)} | normal={int((y_all==0).sum())} | "
      f"anomalous={int((y_all==1).sum())}")


# ── 2. MPNet embeddings (use cache if available) ─────────────────────────────
embeddings = None
encoder_name = f"MPNet ({MPNET_CHECKPOINT})"
if os.path.exists(EMB_CACHE):
    cached = np.load(EMB_CACHE)
    if len(cached) == len(df):
        embeddings = cached.astype(np.float32)
        print(f"Loaded cached MPNet embeddings: {embeddings.shape}")

if embeddings is None:
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Encoding with {MPNET_CHECKPOINT} ...")
        m = SentenceTransformer(MPNET_CHECKPOINT)
        embeddings = m.encode(df['comment_clean'].tolist(), batch_size=32,
                              show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(EMB_CACHE, embeddings)
    except Exception as e:
        print(f"[WARN] MPNet failed ({e}); TF-IDF+SVD fallback.")
        mat = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                              analyzer="char_wb", min_df=2).fit_transform(df['comment_clean'])
        embeddings = TruncatedSVD(n_components=256, random_state=RANDOM_STATE
                                  ).fit_transform(mat).astype(np.float32)
        encoder_name = "TF-IDF + SVD (fallback)"
print(f"Encoder: {encoder_name}\n")


# ── 3. Autoencoder definition (DAEF-style proxy) ─────────────────────────────
def train_ae_errors(X_tr, X_te):
    """Train an AE on normal X_tr; return recon errors on X_tr and X_te."""
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
                loss = crit(net(b), b)
                loss.backward()
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
                           random_state=RANDOM_STATE)
        net.fit(X_tr, X_tr)
        et = np.mean((X_tr - net.predict(X_tr)) ** 2, axis=1)
        ee = np.mean((X_te - net.predict(X_te)) ** 2, axis=1)
        return et, ee


def iqr_threshold(errors, k=1.5):
    q1, q3 = np.percentile(errors, [25, 75])
    return q3 + k * (q3 - q1)


# ── 4. K-fold CV over normal data (paper protocol) ───────────────────────────
normal_idx = np.where(y_all == 0)[0]
anom_idx = np.where(y_all == 1)[0]
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

MODELS = ["Autoencoder (DAEF-style)", "One-Class SVM",
          "Isolation Forest", "Local Outlier Factor"]
metrics_acc = {m: {"Precision": [], "Recall": [], "F1": [],
                   "AUC-ROC": [], "PR-AUC": []} for m in MODELS}

print(f"Running {N_FOLDS}-fold CV (train on normal only, test = held-out "
      f"normal + all anomalies)...")

for fold, (tr_pos, te_pos) in enumerate(kf.split(normal_idx), 1):
    tr_norm = normal_idx[tr_pos]
    te_norm = normal_idx[te_pos]
    test_idx = np.concatenate([te_norm, anom_idx])
    y_te = y_all[test_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embeddings[tr_norm])
    X_te = scaler.transform(embeddings[test_idx])

    fold_preds = {}  # name -> (pred, score)

    # Autoencoder
    et, ee = train_ae_errors(X_tr, X_te)
    thr = iqr_threshold(et, 1.5)
    fold_preds["Autoencoder (DAEF-style)"] = ((ee >= thr).astype(int), ee)

    # One-Class SVM
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X_tr)
    fold_preds["One-Class SVM"] = ((oc.predict(X_te) == -1).astype(int),
                                   -oc.decision_function(X_te))

    # Isolation Forest
    if_ = IsolationForest(n_estimators=200, contamination=0.1,
                          random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr)
    fold_preds["Isolation Forest"] = ((if_.predict(X_te) == -1).astype(int),
                                      -if_.score_samples(X_te))

    # Local Outlier Factor (novelty)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                             contamination=0.1).fit(X_tr)
    fold_preds["Local Outlier Factor"] = ((lof.predict(X_te) == -1).astype(int),
                                          -lof.decision_function(X_te))

    for name, (pred, score) in fold_preds.items():
        metrics_acc[name]["Precision"].append(precision_score(y_te, pred, zero_division=0))
        metrics_acc[name]["Recall"].append(recall_score(y_te, pred, zero_division=0))
        metrics_acc[name]["F1"].append(f1_score(y_te, pred, zero_division=0))
        metrics_acc[name]["AUC-ROC"].append(roc_auc_score(y_te, score))
        metrics_acc[name]["PR-AUC"].append(average_precision_score(y_te, score))

    print(f"  fold {fold:02d} done "
          f"(test: {int((y_te==0).sum())} normal + {int((y_te==1).sum())} anomalous)")


# ── 5. Aggregate mean +/- std (paper Tables 2-3 style) ───────────────────────
rows = []
for name in MODELS:
    row = {"Model": name}
    for met in ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]:
        vals = np.array(metrics_acc[name][met])
        row[met] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
    rows.append(row)

bench = pd.DataFrame(rows)
print("\n" + "=" * 70)
print(f"{N_FOLDS}-FOLD CV BENCHMARK  (anomalous = positive class)")
print(f"Encoder: {encoder_name}")
print("=" * 70)
print(bench.to_string(index=False))

out_csv = os.path.join(OUT_DIR, "cv_benchmark_results.csv")
bench.to_csv(out_csv, index=False, encoding="utf-8-sig")

# numeric (mean-only) table for easy plotting
num_rows = []
for name in MODELS:
    r = {"Model": name}
    for met in ["Precision", "Recall", "F1", "AUC-ROC", "PR-AUC"]:
        r[met + "_mean"] = round(float(np.mean(metrics_acc[name][met])), 4)
        r[met + "_std"] = round(float(np.std(metrics_acc[name][met])), 4)
    num_rows.append(r)
pd.DataFrame(num_rows).to_csv(
    os.path.join(OUT_DIR, "cv_benchmark_numeric.csv"),
    index=False, encoding="utf-8-sig")

print(f"\nSaved: {out_csv}")
print("Done.")
