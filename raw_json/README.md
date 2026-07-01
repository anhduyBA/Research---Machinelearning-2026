# Thu thập thêm dữ liệu Shopee (nhiều sản phẩm)

Mục tiêu: gom review của **4–6 sản phẩm ở các ngành hàng khác nhau** để mở rộng
corpus từ 1 sản phẩm (500) lên nhiều sản phẩm — nâng bài từ pilot (Q4) lên Q3.

Đặt tất cả file `.json` bạn thu thập được vào **chính thư mục này** (`raw_json/`),
mỗi sản phẩm 1 file, đặt tên `product_<itemid>.json`.

---

## Cách 1 — Console tự động (KHUYẾN NGHỊ, nhanh nhất)

1. Đăng nhập shopee.vn trên trình duyệt.
2. Mở trang 1 sản phẩm. URL có dạng: `https://shopee.vn/ten-san-pham-i.SHOPID.ITEMID`
   (2 số cuối là shopid và itemid).
3. Cuộn xuống phần **Đánh giá** một lần (để trang nạp API).
4. Nhấn **F12** → tab **Console**.
5. Mở file `collect_reviews_console.js` (trong thư mục này), copy TOÀN BỘ,
   dán vào Console → **Enter**.
6. Chờ tới khi hiện `DONE` → trình duyệt tự tải `product_<itemid>.json`.
7. Kéo file vừa tải vào thư mục `raw_json/`.
8. Lặp lại cho các sản phẩm khác (mỗi sản phẩm mở tab mới, làm lại bước 3–7).

> Script chạy bằng cookie đăng nhập của chính bạn nên **không bị WAF chặn**
> như gọi API từ Python. Có thể chỉnh `TARGET` (mặc định 600 review/sản phẩm).

---

## Cách 2 — Copy tay từ DevTools (dự phòng nếu Cách 1 bị chặn)

1. Trang sản phẩm → **F12** → tab **Network** → ô lọc gõ: `get_ratings`
2. Bấm chuyển qua các **trang đánh giá** để sinh request `get_ratings`.
3. Với mỗi request: chuột phải → **Copy → Copy response**.
4. Dán vào 1 file text, lưu thành `raw_json/product_<itemid>.json`.
   - Dán **nhiều response nối tiếp nhau** trong cùng 1 file cũng được —
     script gộp `shopee_multi_import.py` tự tách và khử trùng lặp.

---

## Sau khi có đủ file JSON → gộp thành raw CSV

Từ thư mục gốc project, chạy:

```bash
python shopee_multi_import.py --dir raw_json --out raw_shopee_dataset.csv
```

Script sẽ:
- Đào mảng `ratings` (nhận cả list thuần lẫn response `{data:{ratings:[...]}}`).
- Ẩn danh username bằng SHA-256.
- Thêm cột `product_id`, `shop_id`.
- **Khử trùng lặp theo `cmtid`** (kể cả khi bạn copy chồng nhiều trang).
- In số review theo từng sản phẩm.

Sau đó báo cho Claude để chạy `weak_labeling.py` → `clean_dataset.py`
(H3 burst đã được sửa để gom theo **[sản phẩm × giờ]** cho dữ liệu đa sản phẩm).

---

## Mẹo chọn sản phẩm (để dataset "khỏe")

- Chọn sản phẩm **có nhiều đánh giá có chữ** (vài trăm trở lên).
- Đa dạng ngành hàng: thời trang, điện tử, mỹ phẩm, gia dụng, đồ ăn...
- Ưu tiên sản phẩm hay bị "cày xu"/seeding để có tín hiệu spam thật.
- Ghi lại (tên SP, itemid, ngành hàng) — sẽ cần cho phần mô tả dataset trong paper.
