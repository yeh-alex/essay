import requests
import time
import datetime
import pandas as pd
import random
import os

# ==========================================
# 1. 設定
# ==========================================
START_DATE = datetime.date(2018, 1, 1)
END_DATE = datetime.date(2018, 1, 31)
SAVE_DIR = "news_data"
FINAL_FILE = "news_test.xlsx"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==========================================
# 2. 隨機 Header 產生器 (偽裝術)
# ==========================================
def get_random_headers():
    user_agents = [
        # Chrome Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        # Chrome Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Edge
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        # Firefox
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Referer': 'https://news.cnyes.com/news/cat/tw_stock',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Origin': 'https://news.cnyes.com',
        'Connection': 'keep-alive'
    }

# ==========================================
# 3. 請求函式 (含重試 + 動態 Header)
# ==========================================
def fetch_url_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            # 每次請求都換一個新的 Header
            current_headers = get_random_headers()
            
            response = requests.get(url, headers=current_headers, timeout=20)
            
            if 500 <= response.status_code < 600:
                response.raise_for_status() # 觸發例外，進入 retry
            
            if response.status_code == 200:
                return response.json()
            
            # 429 Too Many Requests (被擋了)
            if response.status_code == 429:
                print("  [被限制] 請求太快，休息久一點...")
                time.sleep(30) # 休息 30 秒
                continue

            return None

        except Exception as e:
            wait_time = (attempt + 1) * 10
            print(f"  [連線異常] {e}")
            print(f"  -> 換個身分，等待 {wait_time} 秒後重試 ({attempt+1}/{max_retries})...")
            time.sleep(wait_time)
    return None

# ==========================================
# 4. 爬蟲函式
# ==========================================
def scrape_tw_stock_news(target_date):
    start_dt = datetime.datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
    end_dt = datetime.datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)
    
    BASE_API_URL = f"https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30&startAt={int(start_dt.timestamp())}&endAt={int(end_dt.timestamp())}"
    
    daily_titles = []
    current_page = 1
    max_pages = 50 

    while current_page <= max_pages:
        # 注意：這裡不再傳入固定的 headers，而是讓函式內部自己生成
        data = fetch_url_with_retry(f"{BASE_API_URL}&page={current_page}")
        
        if not data: break

        if 'items' in data and 'data' in data['items'] and isinstance(data['items']['data'], list):
            news_list = data['items']['data']
            if not news_list: break 
            
            for item in news_list:
                t = item.get('title', '').strip()
                if t: daily_titles.append(t)
            
            current_page += 1
            # 隨機延遲 (模擬人類閱讀速度)
            time.sleep(random.uniform(1.0, 3.0)) 
        else:
            break
            
    return daily_titles

# ==========================================
# 5. 主程式 (含分月存檔與合併)
# ==========================================
if __name__ == "__main__":
    current_date = START_DATE
    delta = datetime.timedelta(days=1)
    monthly_data = [] 

    print(f"===== 1. 開始爬蟲任務 (偽裝版) ({START_DATE} ~ {END_DATE}) =====")

    while current_date <= END_DATE:
        print(f"進度: {current_date} ...", end="\r")
        
        titles = scrape_tw_stock_news(current_date)
        if titles:
            for t in titles:
                monthly_data.append({'Date': current_date.isoformat(), 'Title': t})
        
        next_date = current_date + delta
        
        # 分月存檔
        if next_date.month != current_date.month or current_date == END_DATE:
            if monthly_data:
                fname = f"{SAVE_DIR}/news_{current_date.year}_{current_date.month}.xlsx"
                pd.DataFrame(monthly_data).to_excel(fname, index=False)
                print(f"\n[備份] 已存檔 {fname} ({len(monthly_data)} 則)")
                monthly_data = []
        
        current_date += delta

    print("\n\n===== 2. 爬取完成，開始合併 =====")
    
    all_files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith('.xlsx')]
    
    if all_files:
        df_list = []
        for f in all_files:
            try:
                df_list.append(pd.read_excel(f))
            except: pass

        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            final_df = final_df.sort_values(by='Date')
            final_df.to_excel(FINAL_FILE, index=False)
            print(f"SUCCESS! 總資料筆數: {len(final_df)}")
            print(f"檔案已儲存為: {FINAL_FILE}")