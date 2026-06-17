# -*- coding: utf-8 -*-
"""
run_all_and_compare.py
==================================================================
Pipeline đối sánh đầu-cuối cho dự án Fake Review Detection:

  BƯỚC 1 — Chạy lần lượt các mô hình (subprocess):
     a) run_phase4.py        -> Mô hình của tôi   trên Shopee
     b) run_paper_model.py   -> Mô hình paper     trên Shopee
     c) run_full_comparison.py -> Benchmark hợp nhất 10-fold (kiểm soát rò rỉ)
     d) run_amazon_reviews.py -> Mô hình của tôi   trên Amazon (unsupervised)

  BƯỚC 2 — Đọc output thật của từng mô hình, lập BẢNG SO SÁNH
           (in ra màn hình + lưu figures/comparison_table.csv).

  BƯỚC 3 — VẼ biểu đồ từ chính số liệu vừa parse (không hardcode):
     fig1_shopee_best.png        - 4 chỉ số ở điểm vận hành tốt nhất
     fig2_same_algorithm_if.png  - cùng Isolation Forest, chỉ khác đặc trưng
     fig3_leakage_controlled.png - bản 10-fold đã kiểm soát rò rỉ nhãn
     fig4_amazon_generalization.png - flag rate trên Amazon

Cách dùng:
     python run_all_and_compare.py             # chạy đủ 4 model rồi vẽ
     python run_all_and_compare.py --no-run    # bỏ qua bước chạy, dùng output có sẵn
     python run_all_and_compare.py --timeout 1800
==================================================================
"""

import os
import sys
import argparse
import subprocess

# Ép stdout/stderr sang UTF-8 để in được tiếng Việt trên console Windows (cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ------------------------------------------------------------------
# Cấu hình
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

rcParams["font.family"] = "DejaVu Sans"   # hỗ trợ Unicode tiếng Việt
rcParams["font.size"] = 11
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["figure.dpi"] = 150

COLOR_MINE = "#1D9E75"   # teal  - mô hình của tôi
COLOR_PAPER = "#EF9F27"  # amber - mô hình paper
DPI_SAVE = 300

# (nhãn, script, file output mốc để kiểm tra thành công)
STEPS = [
    ("Mô hình của tôi  -> Shopee",      "run_phase4.py",
     "p4_outputs/p4_benchmark_results.csv"),
    ("Mô hình paper    -> Shopee",      "run_paper_model.py",
     "p4_outputs_paper/benchmark_results.csv"),
    ("Benchmark 10-fold (kiểm soát rò rỉ)", "run_full_comparison.py",
     "p4_outputs_paper/unified_benchmark_numeric.csv"),
    ("Mô hình của tôi  -> Amazon",      "run_amazon_reviews.py",
     "p4_outputs_amazon/detection_summary.csv"),
]


# ==================================================================
# BƯỚC 1 — chạy các model
# ==================================================================
def run_models(timeout):
    print("=" * 70)
    print("BƯỚC 1 — CHẠY LẦN LƯỢT CÁC MÔ HÌNH")
    print("=" * 70)
    for label, script, _ in STEPS:
        path = os.path.join(ROOT, script)
        print(f"\n>>> {label}\n    python {script}")
        if not os.path.exists(path):
            print(f"    [BỎ QUA] không tìm thấy {script}")
            continue
        try:
            subprocess.run([sys.executable, path], cwd=ROOT,
                           check=True, timeout=timeout)
            print(f"    [OK] {script} chạy xong.")
        except subprocess.CalledProcessError as e:
            print(f"    [LỖI] {script} kết thúc với mã {e.returncode}.")
        except subprocess.TimeoutExpired:
            print(f"    [QUÁ GIỜ] {script} vượt {timeout}s, bỏ qua.")


# ==================================================================
# BƯỚC 2 — đọc output, lập bảng so sánh
# ==================================================================
def _read_csv(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"    [thiếu] {rel}")
        return None
    return pd.read_csv(p)


def _best_row(df, by="F1"):
    """Trả về dòng có chỉ số `by` cao nhất."""
    return df.loc[df[by].idxmax()]


def collect_results():
    """Đọc toàn bộ output -> trả về dict số liệu đã chuẩn hoá."""
    print("\n" + "=" * 70)
    print("BƯỚC 2 — ĐỌC OUTPUT & LẬP BẢNG SO SÁNH")
    print("=" * 70)
    R = {}

    # --- Mô hình của tôi trên Shopee (best F1 trong các cấu hình) ---
    mine = _read_csv("p4_outputs/p4_benchmark_results.csv")
    if mine is not None:
        best = _best_row(mine, "F1")
        R["mine_shopee"] = dict(model=best["Model"], P=best["Precision"],
                                Rec=best["Recall"], F1=best["F1"],
                                AUC=best["AUC-ROC"])
        # Isolation Forest tốt nhất của tôi (cho fig2 cùng-thuật-toán)
        mine_if = mine[mine["Model"].str.contains("Isolation Forest", case=False)]
        if len(mine_if):
            r = _best_row(mine_if, "F1")
            R["mine_if"] = dict(F1=r["F1"], AUC=r["AUC-ROC"])

    # --- Mô hình paper trên Shopee (best F1 trong 4 detector) ---
    paper = _read_csv("p4_outputs_paper/benchmark_results.csv")
    if paper is not None:
        best = _best_row(paper, "F1")
        R["paper_shopee"] = dict(model=best["Model"], P=best["Precision"],
                                 Rec=best["Recall"], F1=best["F1"],
                                 AUC=best["AUC-ROC"])
        paper_if = paper[paper["Model"].str.contains("Isolation Forest", case=False)]
        if len(paper_if):
            r = paper_if.iloc[0]
            R["paper_if"] = dict(F1=r["F1"], AUC=r["AUC-ROC"])

    # --- Benchmark hợp nhất 10-fold (kiểm soát rò rỉ) ---
    uni = _read_csv("p4_outputs_paper/unified_benchmark_numeric.csv")
    if uni is not None:
        def best_of(feature_set):
            sub = uni[uni["Feature set"] == feature_set]
            if not len(sub):
                return None
            return dict(
                F1=sub["F1_mean"].max(),
                AUC=sub["AUC-ROC_mean"].max(),
            )
        R["uni_mine"] = best_of("Behavioral (no h1-h4)")   # đặc trưng của tôi (sạch)
        R["uni_paper"] = best_of("Text (MPNet)")           # đặc trưng paper

    # --- Amazon (unsupervised, chỉ flag rate) ---
    amz = _read_csv("p4_outputs_amazon/detection_summary.csv")
    if amz is not None:
        R["amazon"] = amz

    return R


def build_table(R):
    """Tạo bảng so sánh tổng hợp -> in ra + lưu CSV."""
    rows = []
    if "mine_shopee" in R:
        m = R["mine_shopee"]
        rows.append(["Của tôi (best)", "Shopee", "Hành vi + văn bản + tín hiệu yếu",
                     m["P"], m["Rec"], m["F1"], m["AUC"]])
    if "paper_shopee" in R:
        m = R["paper_shopee"]
        rows.append(["Paper (best)", "Shopee", "Chỉ văn bản MPNet",
                     m["P"], m["Rec"], m["F1"], m["AUC"]])
    if "uni_mine" in R and R["uni_mine"]:
        m = R["uni_mine"]
        rows.append(["Của tôi (10-fold, sạch)", "Shopee", "Hành vi (bỏ h1-h4)",
                     None, None, m["F1"], m["AUC"]])
    if "uni_paper" in R and R["uni_paper"]:
        m = R["uni_paper"]
        rows.append(["Paper (10-fold)", "Shopee", "Văn bản MPNet",
                     None, None, m["F1"], m["AUC"]])
    if "amazon" in R:
        rows.append(["Của tôi (unsupervised)", "Amazon", "TF-IDF (không nhãn)",
                     None, None, None, None])

    table = pd.DataFrame(
        rows, columns=["Mô hình", "Tập dữ liệu", "Đặc trưng",
                       "Precision", "Recall", "F1", "AUC-ROC"])
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\nBẢNG SO SÁNH TỔNG HỢP")
    print("-" * 70)
    print(table.to_string(index=False, na_rep="—",
                          float_format=lambda x: f"{x:.3f}"))

    out = os.path.join(OUT_DIR, "comparison_table.csv")
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[đã lưu] {out}")

    if "amazon" in R:
        print("\nGhi chú Amazon (không có nhãn -> chỉ flag rate):")
        print(R["amazon"].to_string(index=False))
    return table


# ==================================================================
# BƯỚC 3 — vẽ biểu đồ từ số liệu đã parse
# ==================================================================
def _bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h),
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


def fig1_shopee_best(R):
    if "mine_shopee" not in R or "paper_shopee" not in R:
        print("    [bỏ fig1] thiếu số liệu Shopee")
        return
    metrics = ["Precision", "Recall", "F1", "AUC-ROC"]
    mn, pp = R["mine_shopee"], R["paper_shopee"]
    mine = [mn["P"], mn["Rec"], mn["F1"], mn["AUC"]]
    paper = [pp["P"], pp["Rec"], pp["F1"], pp["AUC"]]

    x = range(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], mine, w,
                label="Mô hình của tôi (hành vi + văn bản + tín hiệu yếu)",
                color=COLOR_MINE)
    b2 = ax.bar([i + w / 2 for i in x], paper, w,
                label="Mô hình paper (chỉ văn bản MPNet)", color=COLOR_PAPER)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Giá trị")
    ax.set_title("Shopee — so sánh ở điểm vận hành tốt nhất của mỗi mô hình", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_shopee_best.png")
    fig.savefig(p, dpi=DPI_SAVE, bbox_inches="tight"); plt.close(fig)
    print("    [vẽ]", p)


def fig2_same_algorithm_if(R):
    if "mine_if" not in R or "paper_if" not in R:
        print("    [bỏ fig2] thiếu số liệu Isolation Forest")
        return
    metrics = ["F1", "AUC-ROC"]
    mine = [R["mine_if"]["F1"], R["mine_if"]["AUC"]]
    paper = [R["paper_if"]["F1"], R["paper_if"]["AUC"]]
    y = range(len(metrics)); h = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    b1 = ax.barh([i + h / 2 for i in y], mine, h,
                 label="IF + đặc trưng của tôi", color=COLOR_MINE)
    b2 = ax.barh([i - h / 2 for i in y], paper, h,
                 label="IF + văn bản (paper)", color=COLOR_PAPER)
    for bars in (b1, b2):
        for b in bars:
            wv = b.get_width()
            ax.annotate(f"{wv:.3f}", xy=(wv, b.get_y() + b.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.1); ax.set_xlabel("Giá trị")
    ax.set_title("Cùng thuật toán Isolation Forest — đóng góp từ đặc trưng", fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_same_algorithm_if.png")
    fig.savefig(p, dpi=DPI_SAVE, bbox_inches="tight"); plt.close(fig)
    print("    [vẽ]", p)


def fig3_leakage_controlled(R):
    if not R.get("uni_mine") or not R.get("uni_paper"):
        print("    [bỏ fig3] thiếu số liệu 10-fold")
        return
    metrics = ["F1 (tốt nhất)", "AUC-ROC (tốt nhất)"]
    mine = [R["uni_mine"]["F1"], R["uni_mine"]["AUC"]]
    paper = [R["uni_paper"]["F1"], R["uni_paper"]["AUC"]]
    x = range(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], mine, w,
                label="Đặc trưng hành vi của tôi (không h1–h4)", color=COLOR_MINE)
    b2 = ax.bar([i + w / 2 for i in x], paper, w,
                label="Đặc trưng văn bản paper (MPNet)", color=COLOR_PAPER)
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.9); ax.set_ylabel("Giá trị")
    ax.set_title("Shopee — 10-fold CV đã kiểm soát rò rỉ nhãn", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig3_leakage_controlled.png")
    fig.savefig(p, dpi=DPI_SAVE, bbox_inches="tight"); plt.close(fig)
    print("    [vẽ]", p)


def fig4_amazon_generalization(R):
    if "amazon" not in R:
        print("    [bỏ fig4] thiếu detection_summary Amazon")
        return
    amz = R["amazon"]
    thr = [str(t) for t in amz["Threshold"].tolist()]
    flagged = amz["Flagged"].tolist()
    pct = amz["Flagged_%"].tolist()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(thr, flagged, color=COLOR_MINE, width=0.55)
    for b, p in zip(bars, pct):
        ax.annotate(f"{int(b.get_height()):,}\n({p:.2f}%)",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Ngưỡng fusion score"); ax.set_ylabel("Số review bị gắn cờ")
    ax.set_ylim(0, max(flagged) * 1.18)
    ax.set_title("Amazon (unsupervised) — khả năng tổng quát hóa\n"
                 "Không có nhãn ground-truth → chỉ báo cáo flag rate, không tính P/R/F1",
                 fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig4_amazon_generalization.png")
    fig.savefig(p, dpi=DPI_SAVE, bbox_inches="tight"); plt.close(fig)
    print("    [vẽ]", p)


def make_plots(R):
    print("\n" + "=" * 70)
    print("BƯỚC 3 — VẼ BIỂU ĐỒ")
    print("=" * 70)
    fig1_shopee_best(R)
    fig2_same_algorithm_if(R)
    fig3_leakage_controlled(R)
    fig4_amazon_generalization(R)


# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true",
                    help="bỏ qua bước chạy model, chỉ đọc output có sẵn rồi vẽ")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="timeout (giây) cho mỗi model")
    args = ap.parse_args()

    if not args.no_run:
        run_models(args.timeout)
    else:
        print("[--no-run] Bỏ qua bước chạy model, dùng output có sẵn.")

    R = collect_results()
    build_table(R)
    make_plots(R)
    print("\nHOÀN TẤT. Hình & bảng nằm trong:", OUT_DIR)


if __name__ == "__main__":
    main()
