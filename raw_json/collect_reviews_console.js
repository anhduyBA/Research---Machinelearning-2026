/* ==================================================================
   collect_reviews_console.js
   Tự động tải review của MỘT sản phẩm Shopee -> file product_<itemid>.json
   Chạy bằng session trình duyệt của BẠN (dùng cookie đang đăng nhập) nên
   không bị WAF chặn như gọi API từ Python.

   CÁCH DÙNG:
   1. Mở trang sản phẩm trên shopee.vn (URL có dạng .../ten-sp-i.SHOPID.ITEMID)
   2. Cuộn xuống phần Đánh giá ít nhất 1 lần (để trang nạp API get_ratings).
   3. Nhấn F12 -> tab Console -> dán TOÀN BỘ file này -> Enter.
   4. Chờ log "DONE" -> trình duyệt tự tải product_<itemid>.json.
   5. Chuyển file đó vào thư mục raw_json/ của project.
   6. Lặp lại cho 4-6 sản phẩm ở các ngành hàng khác nhau.

   Chỉnh TARGET nếu muốn lấy nhiều/ít hơn.
================================================================== */
(async () => {
  const TARGET = 600;      // số review tối đa muốn lấy cho sản phẩm này
  const LIMIT  = 50;       // số review mỗi request
  const DELAY  = 800;      // nghỉ giữa các request (ms) - lịch sự, tránh bị chặn

  const m = location.href.match(/i\.(\d+)\.(\d+)/);
  if (!m) { console.error('Không đọc được shopid/itemid từ URL. Hãy mở đúng trang sản phẩm.'); return; }
  const shopid = m[1], itemid = m[2];
  console.log(`shopid=${shopid} itemid=${itemid} — bắt đầu thu thập...`);

  const seen = new Set();
  const all = [];
  let offset = 0;
  while (all.length < TARGET) {
    const url = `https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid=${itemid}`
              + `&limit=${LIMIT}&offset=${offset}&shopid=${shopid}&type=0`;
    let data;
    try {
      const res = await fetch(url, {
        credentials: 'include',
        headers: { 'x-api-source': 'pc', 'x-requested-with': 'XMLHttpRequest' }
      });
      data = await res.json();
    } catch (e) { console.error('Lỗi mạng, dừng lại:', e); break; }

    const ratings = data && data.data && data.data.ratings;
    if (!ratings || ratings.length === 0) { console.log('Hết review.'); break; }

    for (const r of ratings) {
      if (!seen.has(r.cmtid)) { seen.add(r.cmtid); all.push(r); }
    }
    offset += ratings.length;
    console.log(`Đã lấy ${all.length} review...`);
    await new Promise(r => setTimeout(r, DELAY));
  }

  if (all.length === 0) { console.warn('Không lấy được review nào (có thể sản phẩm chưa có đánh giá).'); return; }

  const blob = new Blob([JSON.stringify(all)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `product_${itemid}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`DONE — đã tải product_${itemid}.json với ${all.length} review. Chuyển file vào thư mục raw_json/.`);
})();
