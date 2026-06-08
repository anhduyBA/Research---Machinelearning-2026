import requests
import time
import hashlib
import csv
import os
import argparse
import json
from datetime import datetime

def hash_username(username):
    """Anonymize username using SHA-256 hashing."""
    if not username:
        return "anonymous_user"
    return hashlib.sha256(username.encode("utf-8")).hexdigest()

def load_cookies():
    """Load raw cookies from shopee_cookies.txt if it exists."""
    cookie_file = "shopee_cookies.txt"
    if os.path.exists(cookie_file):
        print(f"[Info] Loading cookies from {cookie_file}...")
        with open(cookie_file, "r", encoding="utf-8") as f:
            cookie_content = f.read().strip()
            # If the user copied the whole header "Cookie: ...", extract the value
            if cookie_content.lower().startswith("cookie:"):
                cookie_content = cookie_content[7:].strip()
            return cookie_content
    return None

def crawl_shopee_reviews(shop_id, item_id, max_reviews=100, limit_per_request=20):
    print(f"[Crawler] Starting collection for Shop ID: {shop_id}, Item ID: {item_id}")
    print(f"[Crawler] Target reviews to collect: {max_reviews}")
    
    url = "https://shopee.vn/api/v2/item/get_ratings"
    session = requests.Session()
    
    # Establish realistic headers to look like a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        "Referer": f"https://shopee.vn/product-i.{shop_id}.{item_id}",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Sec-Ch-Ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "X-Requested-With": "XMLHttpRequest",
        "X-Api-Source": "pc"
    }
    
    # Load cookies if available to bypass the 90309999 error
    cookie_str = load_cookies()
    if cookie_str:
        headers["Cookie"] = cookie_str
        # Parse cookie string to extract csrftoken for the X-Csrftoken header
        cookies_dict = {}
        for item in cookie_str.split(';'):
            item = item.strip()
            if '=' in item:
                k, v = item.split('=', 1)
                cookies_dict[k] = v
        if 'csrftoken' in cookies_dict:
            headers['X-Csrftoken'] = cookies_dict['csrftoken']
    else:
        print("[Warning] No shopee_cookies.txt file found. The request might be blocked by Shopee WAF (Error 90309999).")
        print("[Warning] If you get a 403 error, please export your Shopee browser cookies to shopee_cookies.txt and re-run.")
        
    session.headers.update(headers)
    
    offset = 0
    all_reviews = []
    
    while len(all_reviews) < max_reviews:
        print(f"[Crawler] Fetching offset {offset} (Current total: {len(all_reviews)})...")
        
        params = {
            "exclude_filter": 1,
            "filter": 0,
            "filter_size": 0,
            "flag": 1,
            "fold_filter": 0,
            "itemid": item_id,
            "limit": min(limit_per_request, max_reviews - len(all_reviews)),
            "offset": offset,
            "relevant_reviews": "false",
            "request_source": 2,
            "shopid": shop_id,
            "tag_filter": "",
            "type": 5,  # 5: Reviews with comments (needed for text analysis)
            "variation_filters": "",
            "need_translation": 1,
            "fe_toggle": "[2,3]",
            "preferred_item_shop_id": shop_id,
            "preferred_item_item_id": item_id,
            "preferred_item_include_type": 1
        }
        
        try:
            response = session.get(url, params=params, timeout=15)
            
            if response.status_code == 403:
                print("\n[Error 403] Forbidden! Shopee has blocked the request.")
                print(f"Response Headers: {response.headers}")
                print(f"Response Body (First 500 chars): {response.text[:500]}")
                print("[Resolution] Please capture cookies from a logged-in browser session at shopee.vn, save them to 'shopee_cookies.txt' in this directory, and try again.")
                break
                
            response.raise_for_status()
            data = response.json()
            
            # Check for Shopee-specific anti-bot error code
            if data.get("error") == 90309999:
                print("\n[Error 90309999] Anti-bot block triggered! Shopee requires verification/cookies.")
                print("[Resolution] Please capture cookies from a logged-in browser session at shopee.vn, save them to 'shopee_cookies.txt' in this directory, and try again.")
                break
                
            ratings = data.get("data", {}).get("ratings", [])
            if not ratings:
                print("[Crawler] No more reviews returned.")
                break
                
            for r in ratings:
                username = r.get("author_username")
                anonymized_user = hash_username(username)
                
                comment_text = r.get("comment") or ""
                # Replace newlines with spaces to maintain single-line comment format in CSV
                comment_text = comment_text.replace("\n", " ").replace("\r", " ")
                
                rating_star = r.get("rating_star")
                ctime = r.get("ctime")
                
                # Convert timestamp
                timestamp_str = ""
                if ctime:
                    timestamp_str = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
                
                image_count = len(r.get("images") or [])
                
                # Check purchase status. Shopee reviews generally require buying, default to True
                is_purchased = True
                
                all_reviews.append({
                    "user_id": anonymized_user,
                    "comment": comment_text,
                    "rating": rating_star,
                    "timestamp": timestamp_str,
                    "image_count": image_count,
                    "is_purchased": is_purchased
                })
                
                if len(all_reviews) >= max_reviews:
                    break
            
            # Update offset for next batch
            offset += len(ratings)
            
            # Strict Rate Limiting to maintain ethical data collection guidelines (No DDoS behaviors)
            print(f"[Crawler] Sleeping for 3 seconds to respect rate limits...")
            time.sleep(3)
            
        except requests.exceptions.HTTPError as he:
            print(f"\n[HTTP Error] {he}")
            break
        except Exception as e:
            print(f"\n[Unexpected Error] {e}")
            break
            
    # Save results to raw_shopee_dataset.csv
    csv_file = "raw_shopee_dataset.csv"
    if all_reviews:
        fields = ["user_id", "comment", "rating", "timestamp", "image_count", "is_purchased"]
        with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in all_reviews:
                writer.writerow(r)
        print(f"\n[Success] Collected {len(all_reviews)} reviews and saved to '{csv_file}'.")
    else:
        print("\n[Crawler] No reviews were extracted. No CSV file generated.")
        
    return all_reviews

def process_local_json(json_file_path):
    if not os.path.exists(json_file_path):
        print(f"[Error] File '{json_file_path}' does not exist.")
        return
        
    print(f"[Importer] Loading raw reviews from '{json_file_path}'...")
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            ratings = json.load(f)
            
        all_reviews = []
        for r in ratings:
            username = r.get("author_username")
            anonymized_user = hash_username(username)
            comment_text = r.get("comment") or ""
            rating_star = r.get("rating_star")
            ctime = r.get("ctime")
            
            timestamp_str = ""
            if ctime:
                timestamp_str = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
            
            image_count = len(r.get("images") or [])
            is_purchased = True
            
            all_reviews.append({
                "user_id": anonymized_user,
                "comment": comment_text,
                "rating": rating_star,
                "timestamp": timestamp_str,
                "image_count": image_count,
                "is_purchased": is_purchased
            })
            
        csv_file = "raw_shopee_dataset.csv"
        if all_reviews:
            fields = ["user_id", "comment", "rating", "timestamp", "image_count", "is_purchased"]
            with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in all_reviews:
                    writer.writerow(r)
            print(f"\n[Success] Imported {len(all_reviews)} reviews from JSON and saved to '{csv_file}'.")
        else:
            print("\n[Importer] No reviews were found in JSON. No CSV file generated.")
            
    except Exception as e:
        print(f"[Error] Failed to import JSON: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shopee Reviews Crawler Pipeline")
    parser.add_argument("--shopid", type=str, default="88201679", help="Shopee Shop ID")
    parser.add_argument("--itemid", type=str, default="10753341705", help="Shopee Item ID")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of reviews to collect")
    parser.add_argument("--import_json", type=str, default=None, help="Import a raw reviews JSON file downloaded from the browser")
    
    args = parser.parse_args()
    if args.import_json:
        process_local_json(args.import_json)
    else:
        crawl_shopee_reviews(args.shopid, args.itemid, max_reviews=args.limit)
