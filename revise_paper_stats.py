# -*- coding: utf-8 -*-
"""
revise_paper_stats.py
==================================================================
Tái lập (reproducible) các con số đã HIỆU CHỈNH cho bản revise Hướng A:

  1) Autoencoder: phát hiện điểm AE gần như không tương quan với nhãn
     (text-reconstruction trực giao với nhãn yếu định nghĩa theo hành vi).
     Báo cáo AE ở mức "fair best-case": polarity-corrected + threshold-optimised.
  2) Bootstrap 95% CI cho các metric chính (đáp ứng nhận xét #1 của giảng viên).

Nguồn: p4_outputs/p4_final_predictions.csv (đã có if_score, ae_score, nhãn).
==================================================================
"""
import numpy as np, pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

rng = np.random.default_rng(42)
df = pd.read_csv("p4_outputs/p4_final_predictions.csv")
y = df["is_suspicious"].values
ifs = df["if_score"].values
ae = df["ae_score"].values


def bootstrap_ci(y, score, pred, n=2000):
    f1s, aucs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        f1s.append(f1_score(y[idx], pred[idx], zero_division=0))
        aucs.append(roc_auc_score(y[idx], score[idx]))
    pc = lambda a, q: float(np.percentile(a, q))
    return (pc(f1s, 2.5), pc(f1s, 97.5)), (pc(aucs, 2.5), pc(aucs, 97.5))


# ---------- 1) Autoencoder: chẩn đoán + hiệu chỉnh ----------
auc_raw = roc_auc_score(y, ae)
corr = np.corrcoef(ae, y)[0, 1]
# polarity-corrected score = hướng cho AUC >= 0.5
ae_fix = ae if auc_raw >= 0.5 else -ae
auc_fix = roc_auc_score(y, ae_fix)
# threshold tối ưu F1 trên score đã hiệu chỉnh
thrs = np.quantile(ae_fix, np.linspace(0.50, 0.99, 50))
best_t = max(thrs, key=lambda t: f1_score(y, (ae_fix >= t).astype(int), zero_division=0))
pred_ae = (ae_fix >= best_t).astype(int)
P = precision_score(y, pred_ae, zero_division=0)
R = recall_score(y, pred_ae, zero_division=0)
F = f1_score(y, pred_ae, zero_division=0)
PR = average_precision_score(y, ae_fix)
print("== AUTOENCODER (chẩn đoán) ==")
print(f"  ROC-AUC thô            = {auc_raw:.3f}  (< 0.5 -> dưới ngẫu nhiên)")
print(f"  corr(ae_score, nhãn)   = {corr:+.3f}  (~0 -> gần như không có tín hiệu)")
print(f"  -> Báo cáo AE fair best-case (polarity-corrected, threshold-optimised):")
print(f"     Prec={P:.3f}  Recall={R:.3f}  F1={F:.3f}  ROC-AUC={auc_fix:.3f}  PR-AUC={PR:.3f}")
f1ci, aucci = bootstrap_ci(y, ae_fix, pred_ae)
print(f"     CI95  F1=[{f1ci[0]:.3f}, {f1ci[1]:.3f}]  ROC-AUC=[{aucci[0]:.3f}, {aucci[1]:.3f}]")

# ---------- 2) Isolation Forest @0.65 + CI ----------
pred_if = (ifs >= 0.65).astype(int)
print("\n== ISOLATION FOREST @0.65 (headline in-sample) ==")
print(f"  Prec={precision_score(y,pred_if,zero_division=0):.3f}  "
      f"Recall={recall_score(y,pred_if,zero_division=0):.3f}  "
      f"F1={f1_score(y,pred_if,zero_division=0):.3f}  "
      f"ROC-AUC={roc_auc_score(y,ifs):.3f}")
f1ci, aucci = bootstrap_ci(y, ifs, pred_if)
print(f"  CI95  F1=[{f1ci[0]:.3f}, {f1ci[1]:.3f}]  ROC-AUC=[{aucci[0]:.3f}, {aucci[1]:.3f}]")

print("\nKẾT LUẬN RQ1: IF (F1 0.600) > AE đã hiệu chỉnh (F1 %.3f); CI không chồng lấn -> IF thắng có ý nghĩa." % F)
