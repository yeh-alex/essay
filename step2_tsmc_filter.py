import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# ==========================================
# 1. 設定
# ==========================================
input_excel = "cnyes_tw_stock_news.xlsx"
output_csv = "daily_news_features_tsmc_ecosystem.csv" # 檔名改為台積電生態系
check_file = "tsmc_filtered_check.xlsx"     # 用來檢查過濾結果

# SBERT 門檻 (因為錨點很精準，門檻可以設 0.45 左右)
THRESHOLD = 0.45 

# ==========================================
# 2. 定義「台積電生態系」白名單
# ==========================================
# 只要出現這些名字，絕對保留
whitelist_companies = [
    # --- 本尊 ---
    "台積電", "2330", "TSMC", "魏哲家", "張忠謀",
    
    # --- 關鍵大客戶 (股價連動最高) ---
    "輝達", "Nvidia", "黃仁勳", "NVDA",
    "蘋果", "Apple", "iPhone", 
    "超微", "AMD", "蘇姿丰", 
    "聯發科", "高通", "Broadcom", "博通",

    # --- CoWoS 先進封裝設備供應鏈 (近期飆股群) ---
    "日月光", "辛耘", "萬潤", "弘塑", "家登", "中砂", 
    "鈦昇", "志聖", "均華", "均豪",
    
    # --- IP 矽智財 (台積電聯盟) ---
    "創意", "世芯", "M31", "力旺"
]

# ==========================================
# 3. 定義「台積電專屬」錨點 (Anchors)
# ==========================================
# 這裡我們只描述「跟晶圓代工相關」的概念
tsmc_anchors = [
    # 核心業務
    "台積電晶圓代工先進製程與產能利用率",
    "3奈米2奈米製程技術與良率突破",
    
    # 封測與設備 (CoWoS)
    "CoWoS先進封裝產能擴充與半導體設備廠",
    "矽光子CPO技術與封裝測試供應鏈",
    
    # 客戶訂單 (需求面)
    "輝達AI晶片訂單與伺服器需求強勁",
    "蘋果iPhone處理器拉貨與HPC運算晶片",
    
    # 產業景氣
    "半導體庫存去化與晶圓代工報價調漲"
]

# 負向錨點 (用來踢掉雜訊)
non_tech_anchors = [
    "金融保險銀行金控配息", 
    "傳統產業水泥鋼鐵塑化紡織", 
    "航運貨櫃航空運輸", 
    "營建房地產與觀光旅遊"
]

# ==========================================
# 主程式邏輯 (維持不變)
# ==========================================
print(f"--- 啟動台積電生態系過濾器 ---")

try:
    df = pd.read_excel(input_excel)
    df = df.dropna(subset=['Title'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    print(f"原始資料: {len(df)} 則")
except FileNotFoundError:
    print("找不到檔案，請先執行 Step 1 爬蟲。")
    exit()

print("載入 AI 模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 1. 向量化
print("計算語意相似度...")
titles = df['Title'].astype(str).tolist()
title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=True)
tech_anchor_embeddings = model.encode(tsmc_anchors, convert_to_tensor=True)
non_tech_anchor_embeddings = model.encode(non_tech_anchors, convert_to_tensor=True)

# 2. 計算分數
# 標題 vs 台積電錨點
tech_scores = util.cos_sim(title_embeddings, tech_anchor_embeddings)
max_tech_scores = torch.max(tech_scores, dim=1).values.cpu().numpy()

# 標題 vs 非科技錨點
non_tech_scores = util.cos_sim(title_embeddings, non_tech_anchor_embeddings)
max_non_tech_scores = torch.max(non_tech_scores, dim=1).values.cpu().numpy()

# 3. 白名單檢查
pattern = '|'.join(whitelist_companies)
mask_whitelist = df['Title'].str.contains(pattern, case=False, na=False)

# 4. 綜合篩選邏輯
# 保留條件：(是白名單) OR (SBERT分數夠高 且 比傳產分數高)
mask_sbert = (max_tech_scores >= THRESHOLD) & (max_tech_scores > max_non_tech_scores)
final_mask = mask_whitelist | mask_sbert

# 分割與檢查
filtered_df = df[final_mask].copy()
filtered_embeddings = title_embeddings[final_mask]

print(f"\n=== 過濾結果 ===")
print(f"原始: {len(df)}")
print(f"保留: {len(filtered_df)} (台積電生態系)")

# 存一份讓您檢查 (看看是不是真的很純)
filtered_df[['Date', 'Title']].to_excel(check_file, index=False)
print(f"已儲存檢查檔: {check_file} (請務必打開檢查)")

if len(filtered_df) == 0:
    print("警告：沒有新聞符合條件。")
    exit()

# 5. 每日彙整與存檔
print("\n正在彙整每日向量特徵...")
emb_numpy = filtered_embeddings.cpu().numpy()
emb_cols = [f'emb_{i}' for i in range(emb_numpy.shape[1])]
df_emb = pd.DataFrame(emb_numpy, columns=emb_cols, index=filtered_df.index)
df_final = pd.concat([filtered_df[['Date']], df_emb], axis=1)
daily_features = df_final.groupby('Date')[emb_cols].mean()

daily_features.to_csv(output_csv)
print(f"成功！台積電專屬特徵已存至 {output_csv}")