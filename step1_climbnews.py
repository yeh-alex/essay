# import requests
# import time
# import datetime
# import pandas as pd
# import random
# # ==========================================
# # 核心爬蟲函式
# # ==========================================
# def scrape_tw_stock_news(target_date):
#     """
#     抓取鉅亨網指定日期的「台股」新聞標題。
#     """
#     print(f"--- [函式執行] 開始抓取 {target_date} ---")
    
#     # 1. 計算當天的開始與結束時間戳記 (Unix Timestamp)
#     start_dt = datetime.datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0)
#     end_dt = datetime.datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59)
#     start_timestamp = int(start_dt.timestamp())
#     end_timestamp = int(end_dt.timestamp())

#     # 2. (關鍵修改) 將 category 改為 'tw_stock'
#     BASE_API_URL = f"https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30&startAt={start_timestamp}&endAt={end_timestamp}"
    
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#         # (關鍵修改) Referer 改為台股頁面
#         'Referer': 'https://news.cnyes.com/news/cat/tw_stock' 
#     }
    
#     daily_titles = []
#     current_page = 1
#     max_pages = 50 # 台股新聞量較大，上限設高一點 (例如 50 頁)

#     try:
#         while current_page <= max_pages:
#             api_url = f"{BASE_API_URL}&page={current_page}"
            
#             # 發送請求
#             response = requests.get(api_url, headers=headers)
#             response.raise_for_status() 
#             data = response.json()

#             # 解析資料結構
#             if 'items' in data and 'data' in data['items'] and isinstance(data['items']['data'], list):
#                 news_list = data['items']['data']
                
#                 # 如果該頁沒資料，代表抓完了
#                 if not news_list:
#                     break 

#                 for item in news_list:
#                     title = item.get('title', '').strip()
#                     if title:
#                         daily_titles.append(title)
                
#                 # 準備抓下一頁
#                 current_page += 1
#                 time.sleep(0.5) # 稍微加速，但仍保持禮貌 (0.5秒)
#             else:
#                 print(f"  [警告] {target_date} 第 {current_page} 頁結構異常，跳過。")
#                 break
                
#     except Exception as e:
#         print(f"  [錯誤] {target_date} 發生錯誤: {e}")
#         return []

#     print(f"  -> {target_date} 抓取完成，共 {len(daily_titles)} 則標題。")
#     return daily_titles


# # ==========================================
# # 主程式執行區
# # ==========================================
# if __name__ == "__main__":
    
#     # --- 1. 設定您要抓取的日期範圍 ---
#     # 建議先抓個 1 個月試試看，確認資料沒問題再擴大到幾年
#     start_date = datetime.date(2018, 1, 1) 
#     end_date = datetime.date(2024, 12, 31)   
#     # --------------------------------

#     all_data_list = [] # 用來存 (日期, 標題) 的列表
    
#     current_date = start_date
#     delta = datetime.timedelta(days=1)

#     print(f"===== 開始爬取台股新聞 ({start_date} ~ {end_date}) =====")

#     while current_date <= end_date:
        
#         # 呼叫爬蟲函式
#         titles = scrape_tw_stock_news(current_date)
        
#         # 將抓到的標題存入列表
#         if titles:
#             for t in titles:
#                 all_data_list.append({
#                     'Date': current_date.isoformat(), # 轉成字串 '2024-10-01'
#                     'Title': t
#                 })
        
#         # 往下一天
#         current_date += delta



# # 隨機暫停 1 到 3 秒之間的小數點時間 (例如 1.25 秒, 2.8 秒)
#         time.sleep(random.uniform(1, 3))
#         # time.sleep(1) # 每天之間休息 1 秒

#     # --- 2. 儲存結果 ---
#     print("\n===== 爬取結束，正在儲存檔案 =====")
    
#     if all_data_list:
#         df = pd.DataFrame(all_data_list)
        
#         filename = "cnyes_tw_stock_news.xlsx"
#         df.to_excel(filename, index=False)
        
#         print(f"成功！已儲存 {len(df)} 筆資料至 {filename}")
#         print("預覽前 5 筆：")
#         print(df.head())
#     else:
#         print("沒有抓到任何資料。")

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
END_DATE = datetime.date(2025, 12, 10)
SAVE_DIR = "news_data"
FINAL_FILE = "cnyes_tw_stock_news.xlsx"

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