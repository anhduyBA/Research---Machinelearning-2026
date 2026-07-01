# -*- coding: utf-8 -*-
"""
shopee_multi_import.py
==================================================================
Gộp NHIỀU file JSON review Shopee (mỗi file = 1 sản phẩm, export từ trình
duyệt) thành MỘT raw CSV có thêm cột product_id — sẵn sàng cho pipeline
weak_labeling.py -> clean_dataset.py.

Vì sao dùng JSON export thay vì gọi API trực tiếp?
  Shopee có WAF (lỗi 90309999 / 403). Cách ổn định nhất là mở trang sản phẩm,
  DevTools > Network > lọc "get_ratings", copy MẢNG data.ratings (hoặc cả
  response) lưu thành 1 file .json cho mỗi sản phẩm.

CÁCH LẤY JSON (làm 1 lần cho mỗi sản phẩm):
  1. Mở trang sản phẩm trên shopee.vn, kéo tới phần Đánh giá.
  2. F12 > tab Network > gõ lọc: get_ratings
  3. Bấm sang các trang đánh giá để sinh request; với mỗi request:
     chuột phải > Copy > Copy response  (hoặc mở, copy JSON).
  4. Dán vào 1 file, ví dụ raw_json/product_<itemid>.json
     (có thể dán nhiều mảng ratings nối nhau — script tự gộp; xem README bên dưới).
  5. Lặp lại cho 4-6 sản phẩm ở các ngành hàng khác nhau.

CÁCH CHẠY:
  # gộp tất cả .json trong thư mục raw_json/
  python shopee_multi_import.py --dir raw_json --out raw_shopee_dataset.csv
  # hoặc chỉ định từng file
  python shopee_multi_import.py --files a.json b.json c.json

Định dạng JSON chấp nhận (script tự nhận diện):
  - Một LIST các object rating  (giống file shopee_reviews_*.json bạn đang có)
  - Một DICT response đầy đủ: sẽ tự đào tới data.ratings
  - Một LIST chứa nhiều response/dict lồng nhau
==================================================================
"""
import os, re, csv, json, glob, hashlib, argparse
from datetime import datetime


def hash_username(username):
    if not username:
        return "anonymous_user"
    return hashlib.sha256(str(username).encode("utf-8")).hexdigest()


def extract_ratings(obj):
    """Đào đệ quy để lấy mọi object rating (có 'cmtid' hoặc 'rating_star')."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "rating_star" in o and ("comment" in o or "cmtid" in o):
                found.append(o)
                return
            # response Shopee: data.ratings
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return found


def normalize_comment(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\r\n]+", " ", text).strip()


def record_from_rating(r):
    ctime = r.get("ctime")
    ts = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S") if ctime else ""
    return {
        "cmtid": r.get("cmtid"),                       # dùng để khử trùng lặp, không xuất
        "user_id": hash_username(r.get("author_username") or r.get("userid")),
        "comment": normalize_comment(r.get("comment") or ""),
        "rating": r.get("rating_star"),
        "timestamp": ts,
        "image_count": len(r.get("images") or []),
        "is_purchased": True,
        "product_id": r.get("itemid") or r.get("origin_itemid"),
        "shop_id": r.get("shopid") or r.get("origin_shopid"),
    }


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    # Cho phép nhiều mảng/đối tượng JSON nối nhau trong 1 file
    try:
        return [json.loads(txt)]
    except json.JSONDecodeError:
        dec = json.JSONDecoder()
        objs, i, n = [], 0, len(txt)
        while i < n:
            while i < n and txt[i] in " \t\r\n,":
                i += 1
            if i >= n:
                break
            obj, end = dec.raw_decode(txt, i)
            objs.append(obj)
            i = end
        return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None, help="Thư mục chứa các file .json")
    ap.add_argument("--files", nargs="*", default=None, help="Danh sách file .json cụ thể")
    ap.add_argument("--out", type=str, default="raw_shopee_dataset.csv", help="File raw CSV đầu ra")
    args = ap.parse_args()

    paths = []
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if args.files:
        paths += args.files
    if not paths:
        print("[Lỗi] Chưa chỉ định --dir hoặc --files. Ví dụ: --dir raw_json")
        return

    seen = set()
    rows = []
    per_product = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"[Bỏ qua] không thấy {p}")
            continue
        ratings = []
        for obj in load_json_file(p):
            ratings += extract_ratings(obj)
        added = 0
        for r in ratings:
            rec = record_from_rating(r)
            key = rec["cmtid"] or (rec["user_id"], rec["comment"], rec["timestamp"])
            if key in seen:
                continue
            seen.add(key)
            if not rec["comment"]:
                continue
            rows.append(rec)
            per_product[rec["product_id"]] = per_product.get(rec["product_id"], 0) + 1
            added += 1
        print(f"[OK] {os.path.basename(p)}: +{added} review (tổng thô {len(ratings)})")

    if not rows:
        print("[Lỗi] Không trích được review nào.")
        return

    fields = ["user_id", "comment", "rating", "timestamp", "image_count", "is_purchased", "product_id", "shop_id"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n===== TỔNG KẾT =====")
    print(f"Số sản phẩm: {len(per_product)}")
    for pid, c in per_product.items():
        print(f"  product_id={pid}: {c} review")
    print(f"Tổng review (đã khử trùng lặp): {len(rows)}")
    print(f"Đã lưu: {args.out}")


if __name__ == "__main__":
    main()
