# -*- coding: utf-8 -*-
"""
plot_comparison.py
==================================================================
Vẽ các biểu đồ so sánh "Mô hình của tôi" vs "Mô hình paper"
(Novoa-Paradela et al., 2024) phục vụ paper Fake Review Detection.

Số liệu lấy trực tiếp từ output thật của dự án:
  - p4_outputs/p4_benchmark_results.csv        (mô hình của tôi - Shopee)
  - p4_outputs_paper/benchmark_results.csv     (mô hình paper - Shopee)
  - p4_outputs_paper/unified_benchmark_numeric.csv (10-fold CV, kiểm soát rò rỉ)
  - p4_outputs_amazon/detection_summary.csv    (flag rate - Amazon, unsupervised)

Xuất ra thư mục figures/ :
  fig1_shopee_best.png        - so kè 4 chỉ số ở điểm vận hành tốt nhất
  fig2_same_algorithm_if.png  - cùng Isolation Forest, chỉ khác đặc trưng
  fig3_leakage_controlled.png - bản đã kiểm soát rò rỉ (an toàn cho phản biện)
  fig4_amazon_generalization.png - flag rate trên Amazon (định tính)
==================================================================
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ------------------------------------------------------------------
# Cấu hình style chung (paper-ready)
# ------------------------------------------------------------------
rcParams["font.family"] = "DejaVu Sans"   # hỗ trợ Unicode tiếng Việt
rcParams["font.size"] = 11
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["figure.dpi"] = 150

COLOR_MINE = "#1D9E75"   # teal  - mô hình của tôi
COLOR_PAPER = "#EF9F27"  # amber - mô hình paper
DPI_SAVE = 300

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def _bar_labels(ax, bars, fmt="{:.3f}"):
    """Ghi giá trị lên đầu mỗi cột."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h),
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


# ==================================================================
# FIG 1 - Shopee: điểm vận hành tốt nhất của mỗi mô hình
#   Mô hình của tôi : Isolation Forest @0.65  (p4_benchmark_results.csv)
#   Mô hình paper   : Autoencoder DAEF @IQR   (benchmark_results.csv)
# ==================================================================
def fig1_shopee_best():
    metrics = ["Precision", "Recall", "F1", "AUC-ROC"]
    mine = [0.720, 0.514, 0.600, 0.966]
    paper = [0.257, 0.800, 0.389, 0.422]

    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], mine, w,
                label="Mô hình của tôi (hành vi + văn bản + tín hiệu yếu)",
                color=COLOR_MINE)
    b2 = ax.bar([i + w / 2 for i in x], paper, w,
                label="Mô hình paper (chỉ văn bản MPNet)",
                color=COLOR_PAPER)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Giá trị")
    ax.set_title("Shopee — so sánh ở điểm vận hành tốt nhất của mỗi mô hình",
                 fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1,
              frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_shopee_best.png")
    fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ==================================================================
# FIG 2 - Cùng thuật toán Isolation Forest, chỉ khác đặc trưng
#   của tôi : IF + đặc trưng hành vi/tín hiệu (F1 .600, AUC .966)
#   paper   : IF + văn bản MPNet            (F1 .269, AUC .565)
# ==================================================================
def fig2_same_algorithm_if():
    metrics = ["F1", "AUC-ROC"]
    mine = [0.600, 0.966]
    paper = [0.269, 0.565]

    y = range(len(metrics))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    b1 = ax.barh([i + h / 2 for i in y], mine, h,
                 label="IF + đặc trưng của tôi", color=COLOR_MINE)
    b2 = ax.barh([i - h / 2 for i in y], paper, h,
                 label="IF + văn bản (paper)", color=COLOR_PAPER)
    for bars in (b1, b2):
        for b in bars:
            wv = b.get_width()
            ax.annotate("{:.3f}".format(wv),
                        xy=(wv, b.get_y() + b.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9)

    ax.set_yticks(list(y))
    ax.set_yticklabels(metrics)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Giá trị")
    ax.set_title("Cùng thuật toán Isolation Forest — đóng góp từ đặc trưng",
                 fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_same_algorithm_if.png")
    fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ==================================================================
# FIG 3 - Bản KIỂM SOÁT RÒ RỈ (an toàn cho phản biện)
#   10-fold CV, bỏ h1-h4 (unified_benchmark_numeric.csv)
#   Hành vi của tôi (best): F1 .597 (IF), AUC .751 (LOF)
#   Văn bản paper (best)  : F1 .519 (Fusion), AUC .552 (IF)
# ==================================================================
def fig3_leakage_controlled():
    metrics = ["F1 (tốt nhất)", "AUC-ROC (tốt nhất)"]
    mine = [0.597, 0.751]
    paper = [0.519, 0.552]

    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar([i - w / 2 for i in x], mine, w,
                label="Đặc trưng hành vi của tôi (không h1–h4)", color=COLOR_MINE)
    b2 = ax.bar([i + w / 2 for i in x], paper, w,
                label="Đặc trưng văn bản paper (MPNet)", color=COLOR_PAPER)
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Giá trị")
    ax.set_title("Shopee — 10-fold CV đã kiểm soát rò rỉ nhãn",
                 fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_leakage_controlled.png")
    fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ==================================================================
# FIG 4 - Amazon (unsupervised, KHÔNG có nhãn => chỉ flag rate)
#   detection_summary.csv : ngưỡng / số review bị flag / %
# ==================================================================
def fig4_amazon_generalization():
    thresholds = ["0.35", "0.50", "0.65"]
    flagged = [2543, 176, 11]
    pct = [12.08, 0.84, 0.05]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(thresholds, flagged, color=COLOR_MINE, width=0.55)
    for b, p in zip(bars, pct):
        ax.annotate("{:,}\n({:.2f}%)".format(int(b.get_height()), p),
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Ngưỡng fusion score")
    ax.set_ylabel("Số review bị gắn cờ")
    ax.set_ylim(0, max(flagged) * 1.18)
    ax.set_title("Amazon (unsupervised) — khả năng tổng quát hóa\n"
                 "Không có nhãn ground-truth → chỉ báo cáo flag rate, không tính P/R/F1",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_amazon_generalization.png")
    fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    fig1_shopee_best()
    fig2_same_algorithm_if()
    fig3_leakage_controlled()
    fig4_amazon_generalization()
    print("\nDone! All figures saved in:", os.path.abspath(OUT_DIR))
