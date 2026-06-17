# -*- coding: utf-8 -*-
"""
Replicate the anomaly-detection pipeline of:

    Novoa-Paradela, Fontenla-Romero & Guijarro-Berdiñas (2024)
    "Explained anomaly detection in text reviews: Can subjective
     scenarios be correctly evaluated?"
    Engineering Applications of Artificial Intelligence 133 (2024) 108065

Paper pipeline:
  1. Text encoding  -> pretrained MPNet transformer (768-d sentence embeddings)
  2. Anomaly detection -> autoencoder (DAEF) trained ONLY on normal data,
     reconstruction error (MSE) as the normality score.
     Baselines compared in the paper: OS-ELM, OC-SVM, Isolation Forest, LOF.
  3. Threshold -> Interquartile Range (IQR) of training reconstruction errors:
        outlier_IQR = Q3 + 1.5 * IQR  (also percentile thresholds tested)

This script applies that pipeline to the crawled Shopee (Vietnamese) dataset.

Adaptations for this dataset:
  * Reviews are Vietnamese, so the multilingual MPNet checkpoint
    `paraphrase-multilingual-mpnet-base-v2` is used as the encoder (same MPNet
    architecture as the paper, multilingual weights). Falls back to TF-IDF+SVD
    if the model cannot be downloaded.
  * One-class protocol (paper Section 5.1): the detector is trained on NORMAL
    reviews only (is_suspicious == 0); the test set mixes held-out normal
    reviews with the anomalous (is_suspicious == 1) ones. F1 is reported with
    the anomalous class as the positive class.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

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

OUT_DIR = "p4_outputs_paper"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = "labeled_shopee_dataset.csv"
MPNET_CHECKPOINT = "paraphrase-multilingual-mpnet-base-v2"


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_comment(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\\n', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def evaluate(y_true, y_pred, scores):
    """Anomalous class (=1) is the positive class, as in the paper."""
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC":   roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else float('nan'),
        "PR-AUC":    average_precision_score(y_true, scores) if len(set(y_true)) > 1 else float('nan'),
        "CM":        confusion_matrix(y_true, y_pred),
    }


def print_eval(name, m):
    print(f"\n--- {name} ---")
    print(f"Precision : {m['Precision']:.4f}")
    print(f"Recall    : {m['Recall']:.4f}")
    print(f"F1        : {m['F1']:.4f}")
    print(f"AUC-ROC   : {m['AUC-ROC']:.4f}")
    print(f"PR-AUC    : {m['PR-AUC']:.4f}")
    print(f"Confusion matrix [[TN FP],[FN TP]]:\n{m['CM']}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: LOAD SHOPEE DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df['comment'] = df['comment'].fillna("")
df['comment_clean'] = df['comment'].apply(clean_comment)
y_all = df['is_suspicious'].fillna(0).astype(int).values

n_normal = int((y_all == 0).sum())
n_anom = int((y_all == 1).sum())
print(f"Total reviews    : {len(df)}")
print(f"Normal   (y=0)   : {n_normal}")
print(f"Anomalous(y=1)   : {n_anom}  ({n_anom/len(df)*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Text encoding with MPNet (paper module 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: TEXT ENCODING (MPNet transformer)")
print("=" * 60)

embeddings = None
encoder_name = None
try:
    from sentence_transformers import SentenceTransformer
    print(f"Loading MPNet checkpoint: {MPNET_CHECKPOINT} ...")
    st_model = SentenceTransformer(MPNET_CHECKPOINT)
    embeddings = st_model.encode(
        df['comment_clean'].tolist(),
        batch_size=32, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    encoder_name = f"MPNet ({MPNET_CHECKPOINT})"
    np.save(os.path.join(OUT_DIR, "mpnet_embeddings.npy"), embeddings)
    print(f"MPNet embeddings shape: {embeddings.shape}")
except Exception as e:
    print(f"[WARN] MPNet encoding failed ({e}). Falling back to TF-IDF + SVD.")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                            analyzer="char_wb", min_df=2)
    mat = tfidf.fit_transform(df['comment_clean'])
    svd = TruncatedSVD(n_components=256, random_state=RANDOM_STATE)
    embeddings = svd.fit_transform(mat).astype(np.float32)
    encoder_name = "TF-IDF + TruncatedSVD (fallback)"
    np.save(os.path.join(OUT_DIR, "tfidf_embeddings.npy"), embeddings)
    print(f"TF-IDF SVD embeddings shape: {embeddings.shape}  "
          f"(explained var {svd.explained_variance_ratio_.sum():.3f})")

print(f"Encoder used: {encoder_name}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — One-class split (train on NORMAL only, paper Section 5.1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: ONE-CLASS TRAIN/TEST SPLIT")
print("=" * 60)

rng = np.random.RandomState(RANDOM_STATE)
normal_idx = np.where(y_all == 0)[0]
anom_idx = np.where(y_all == 1)[0]

rng.shuffle(normal_idx)
n_train = int(0.8 * len(normal_idx))
train_idx = normal_idx[:n_train]                       # normal-only training
test_idx = np.concatenate([normal_idx[n_train:], anom_idx])  # held-out normal + all anomalies
rng.shuffle(test_idx)

X_train = embeddings[train_idx]
X_test = embeddings[test_idx]
y_test = y_all[test_idx]

# Standardize using training (normal) statistics only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_all_s = scaler.transform(embeddings)

print(f"Train (normal only): {len(train_idx)}")
print(f"Test set           : {len(test_idx)}  "
      f"(normal={int((y_test==0).sum())}, anomalous={int((y_test==1).sum())})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Anomaly detectors trained on normal data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: ANOMALY DETECTION MODELS")
print("=" * 60)

results = {}          # name -> metrics dict
score_columns = {}    # name -> anomaly score over ALL rows (for export)


def iqr_threshold(train_errors, k=1.5):
    q1, q3 = np.percentile(train_errors, [25, 75])
    return q3 + k * (q3 - q1)


# ---- 4a. Autoencoder (DAEF-style, trained on normal embeddings) -------------
print("\n>> Model 1: Autoencoder (DAEF-style reconstruction)")

if HAS_TORCH:
    class AE(nn.Module):
        def __init__(self, dim, latent=64):
            super().__init__()
            mid = max(dim // 4, 128)
            self.enc = nn.Sequential(
                nn.Linear(dim, mid), nn.ReLU(),
                nn.BatchNorm1d(mid), nn.Dropout(0.1),
                nn.Linear(mid, latent), nn.ReLU(),
            )
            self.dec = nn.Sequential(
                nn.Linear(latent, mid), nn.ReLU(),
                nn.BatchNorm1d(mid),
                nn.Linear(mid, dim),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    class DS(data.Dataset):
        def __init__(self, a):
            self.a = torch.tensor(a, dtype=torch.float32)

        def __len__(self):
            return len(self.a)

        def __getitem__(self, i):
            return self.a[i]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # internal val split inside the normal-only training data
    perm = rng.permutation(len(X_train_s))
    cut = int(0.85 * len(perm))
    tr_in, va_in = perm[:cut], perm[cut:]
    tr_loader = data.DataLoader(DS(X_train_s[tr_in]), batch_size=32, shuffle=True)
    va_loader = data.DataLoader(DS(X_train_s[va_in]), batch_size=32, shuffle=False)

    ae = AE(X_train_s.shape[1]).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best, best_state, patience = float('inf'), None, 0
    for epoch in range(1, 101):
        ae.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(ae(b), b)
            loss.backward()
            opt.step()
        ae.eval()
        vl = 0.0
        with torch.no_grad():
            for b in va_loader:
                b = b.to(device)
                vl += crit(ae(b), b).item() * len(b)
        vl /= len(va_in)
        sched.step(vl)
        if vl < best:
            best, best_state, patience = vl, {k: v.clone() for k, v in ae.state_dict().items()}, 0
        else:
            patience += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"   epoch {epoch:03d}  val_loss={vl:.6f}")
        if patience >= 10:
            print(f"   early stopping at epoch {epoch}")
            break
    ae.load_state_dict(best_state)
    ae.eval()

    def recon_err(arr):
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32).to(device)
            return torch.mean((t - ae(t)) ** 2, dim=1).cpu().numpy()

    err_train = recon_err(X_train_s)
    err_test = recon_err(X_test_s)
    err_all = recon_err(X_all_s)
else:
    from sklearn.neural_network import MLPRegressor
    mid = max(X_train_s.shape[1] // 4, 128)
    ae = MLPRegressor(hidden_layer_sizes=(mid, 64, mid), activation='relu',
                      solver='adam', alpha=1e-5, max_iter=100,
                      random_state=RANDOM_STATE)
    ae.fit(X_train_s, X_train_s)
    err_train = np.mean((X_train_s - ae.predict(X_train_s)) ** 2, axis=1)
    err_test = np.mean((X_test_s - ae.predict(X_test_s)) ** 2, axis=1)
    err_all = np.mean((X_all_s - ae.predict(X_all_s)) ** 2, axis=1)

thr_iqr = iqr_threshold(err_train, 1.5)
y_pred_ae = (err_test >= thr_iqr).astype(int)
m_ae = evaluate(y_test, y_pred_ae, err_test)
results["Autoencoder (DAEF-style) @ IQR"] = m_ae
score_columns["ae_recon_error"] = err_all
print(f"   IQR threshold (Q3+1.5*IQR) = {thr_iqr:.6f}")
print_eval("Autoencoder (DAEF-style) @ IQR threshold", m_ae)

# also report best percentile threshold (paper tests Q50..Q95)
print("   Percentile-threshold sweep on AE (train-error percentiles):")
for p in [50, 60, 70, 80, 90, 95]:
    thr = np.percentile(err_train, p)
    pred = (err_test >= thr).astype(int)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(f"      Q{p}: thr={thr:.5f}  F1={f1:.4f}")


# ---- 4b. One-Class SVM ------------------------------------------------------
print("\n>> Model 2: One-Class SVM")
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
ocsvm.fit(X_train_s)
# decision_function: higher = more normal -> anomaly score = -decision
sc_test = -ocsvm.decision_function(X_test_s)
pred = (ocsvm.predict(X_test_s) == -1).astype(int)
m = evaluate(y_test, pred, sc_test)
results["One-Class SVM"] = m
score_columns["ocsvm_score"] = -ocsvm.decision_function(X_all_s)
print_eval("One-Class SVM", m)


# ---- 4c. Isolation Forest ---------------------------------------------------
print("\n>> Model 3: Isolation Forest")
iforest = IsolationForest(n_estimators=200, contamination=0.1,
                          random_state=RANDOM_STATE, n_jobs=-1)
iforest.fit(X_train_s)
sc_test = -iforest.score_samples(X_test_s)
pred = (iforest.predict(X_test_s) == -1).astype(int)
m = evaluate(y_test, pred, sc_test)
results["Isolation Forest"] = m
score_columns["iforest_score"] = -iforest.score_samples(X_all_s)
print_eval("Isolation Forest", m)


# ---- 4d. Local Outlier Factor (novelty) -------------------------------------
print("\n>> Model 4: Local Outlier Factor")
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
lof.fit(X_train_s)
sc_test = -lof.decision_function(X_test_s)
pred = (lof.predict(X_test_s) == -1).astype(int)
m = evaluate(y_test, pred, sc_test)
results["Local Outlier Factor"] = m
score_columns["lof_score"] = -lof.decision_function(X_all_s)
print_eval("Local Outlier Factor", m)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Benchmark table + exports
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: BENCHMARK COMPARISON (anomalous = positive class)")
print("=" * 60)

bench = pd.DataFrame([
    {"Model": name, "Precision": m["Precision"], "Recall": m["Recall"],
     "F1": m["F1"], "AUC-ROC": m["AUC-ROC"], "PR-AUC": m["PR-AUC"]}
    for name, m in results.items()
])
print(bench.to_string(index=False))
bench.to_csv(os.path.join(OUT_DIR, "benchmark_results.csv"),
             index=False, encoding="utf-8-sig")

# full per-review scores export
export = df[["user_id", "comment_clean", "rating", "timestamp", "is_suspicious"]].copy()
for col, arr in score_columns.items():
    export[col] = np.round(arr, 6)
export.to_csv(os.path.join(OUT_DIR, "review_scores.csv"),
              index=False, encoding="utf-8-sig")

# text summary
with open(os.path.join(OUT_DIR, "summary_report.txt"), "w", encoding="utf-8") as f:
    f.write("PAPER-MODEL REPLICATION ON SHOPEE DATASET\n")
    f.write("Novoa-Paradela et al. (2024) EAAI 133:108065\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Encoder: {encoder_name}\n")
    f.write(f"Total reviews: {len(df)} | normal={n_normal} | anomalous={n_anom}\n")
    f.write(f"One-class protocol: train on {len(train_idx)} normal reviews; "
            f"test on {len(test_idx)} (normal={int((y_test==0).sum())}, "
            f"anomalous={int((y_test==1).sum())}).\n\n")
    f.write("Benchmark (anomalous class = positive):\n")
    f.write(bench.to_string(index=False) + "\n")

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("Done.")
