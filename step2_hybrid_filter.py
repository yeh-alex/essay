import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# ==========================================
# 1. 設定
# ==========================================
input_excel = "cnyes_tw_stock_news.xlsx"
output_csv = "daily_news_features_high_precision.csv"
check_file = "dropped_titles_check.xlsx" # 存那些被刪掉的，讓您檢查有沒有誤殺

# --- 關鍵修改：高門檻 ---
# 設定為 0.55 或 0.60，確保只有非常明確的科技新聞才會被留下
HIGH_THRESHOLD = 0.5

# ==========================================
# 2. 定義錨點 (正向 vs 負向)
# ==========================================

# A. 正向錨點 (科技業) - 您想要的
tech_anchors = [
    "半導體晶圓代工與封測技術台積電聯電",
    "IC設計晶片研發與矽智財聯發科",
    "AI伺服器與電腦硬體組裝鴻海廣達",
    "電子零組件PCB被動元件與散熱模組",
    "光電面板顯示器與光學鏡頭",
    "DRAM記憶體與快閃記憶體",
    "網通設備與5G通訊技術"
]

# B. 負向錨點 (非科技業) - 您絕對不要的
# 用來把那些「沾到邊」但本質不是科技的新聞踢掉
non_tech_anchors = [
    "金融保險銀行金控壽險配息與法說會",  # 踢掉中信金、富邦金
    "傳統產業水泥鋼鐵塑化紡織玻璃造紙",  # 踢掉中鋼、台泥
    "航運海運航空貨櫃運輸",             # 踢掉長榮、陽明
    "營建資產房地產與營造工程",         # 踢掉營建股
    "觀光餐飲旅遊飯店與零售百貨",       # 踢掉觀光股
    "生技醫療製藥與疫苗開發"            # 踢掉生技股
]

# ==========================================
# 主程式
# ==========================================
print(f"--- 啟動高精度過濾 (High Precision Filter) ---")

try:
    df = pd.read_excel(input_excel)
    df = df.dropna(subset=['Title'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    print(f"原始資料: {len(df)} 則")
except FileNotFoundError:
    print("找不到檔案。")
    exit()

print("\n正在載入模型...")
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 1. 向量化
print("正在計算相似度...")
titles = df['Title'].astype(str).tolist()
title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=True)
tech_anchor_embeddings = model.encode(tech_anchors, convert_to_tensor=True)
non_tech_anchor_embeddings = model.encode(non_tech_anchors, convert_to_tensor=True)

# 2. 計算分數
# 每個標題 vs 科技錨點 (取最大值)
tech_scores = util.cos_sim(title_embeddings, tech_anchor_embeddings)
max_tech_scores = torch.max(tech_scores, dim=1).values.cpu().numpy()

# 每個標題 vs 非科技錨點 (取最大值)
non_tech_scores = util.cos_sim(title_embeddings, non_tech_anchor_embeddings)
max_non_tech_scores = torch.max(non_tech_scores, dim=1).values.cpu().numpy()

# ==========================================
# 3. 嚴格篩選邏輯
# ==========================================
print(f"\n執行篩選 (門檻 > {HIGH_THRESHOLD} 且 科技分 > 傳產分)...")

# 條件 A: 科技分數夠高
cond_high_score = max_tech_scores >= HIGH_THRESHOLD

# 條件 B: 科技分數 必須大於 傳產分數 (避免模糊地帶)
# 例如: "中鋼導入AI" -> 科技分 0.4, 傳產分 0.7 -> 剔除
cond_is_tech_dominant = max_tech_scores > max_non_tech_scores

# 綜合條件
final_mask = cond_high_score & cond_is_tech_dominant

# 分割資料
filtered_df = df[final_mask].copy()
dropped_df = df[~final_mask].copy() # 被丟掉的，存起來檢查用

# 加入分數讓您檢查
filtered_df['Tech_Score'] = max_tech_scores[final_mask]
dropped_df['Tech_Score'] = max_tech_scores[~final_mask]
dropped_df['NonTech_Score'] = max_non_tech_scores[~final_mask]

print(f"\n=== 過濾結果 ===")
print(f"原始: {len(df)}")
print(f"保留: {len(filtered_df)} (純科技)")
print(f"剔除: {len(dropped_df)} (非科技或分數不足)")

# 儲存被剔除的檔案供檢查 (強烈建議看一下)
dropped_df[['Date', 'Title', 'Tech_Score', 'NonTech_Score']].sort_values(by='Tech_Score', ascending=False).to_excel(check_file, index=False)
print(f"\n[檢查] 已將剔除的新聞存至 {check_file}，請檢查是否有誤殺。")

if len(filtered_df) == 0:
    print("警告：過濾太嚴格，沒有新聞留下。請降低 HIGH_THRESHOLD。")
    exit()

# ==========================================
# 4. 每日彙整與存檔
# ==========================================
print("\n正在彙整每日向量特徵...")
emb_numpy = title_embeddings[final_mask].cpu().numpy()
emb_cols = [f'emb_{i}' for i in range(emb_numpy.shape[1])]
df_emb = pd.DataFrame(emb_numpy, columns=emb_cols, index=filtered_df.index)
df_final = pd.concat([filtered_df[['Date']], df_emb], axis=1)
daily_features = df_final.groupby('Date')[emb_cols].mean()

daily_features.to_csv(output_csv)
print(f"成功！高純度特徵已存至 {output_csv}")