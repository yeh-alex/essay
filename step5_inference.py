import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定與參數
# ==========================================
model_path = "best_tech_stock_model.h5"
news_file = "daily_news_features_high_precision.csv" # 您的新聞特徵
stock_file = "stock_labels_2330.csv"                 # 您的股價資料

TIME_STEPS = 10  # 模型訓練時設定的窗口 (必須一樣)
CONFIDENCE_THRESHOLD = 0.6 # 信心門檻 (超過 60% 才動作)

# ==========================================
# 2. 載入模型
# ==========================================
print("--- 步驟 5: AI 操盤手實戰預測 ---")
print(f"正在載入模型 {model_path} ...")
try:
    model = tf.keras.models.load_model(model_path)
    print("模型載入成功！")
except Exception as e:
    print(f"模型載入失敗: {e}")
    exit()

# ==========================================
# 3. 準備「最近」的資料
# ==========================================
print("正在讀取最近的市場數據...")

# A. 讀取股價
df_stock = pd.read_csv(stock_file)
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock = df_stock.set_index('Date').sort_index()
# 只需要這 5 個特徵，順序必須跟訓練時一樣
df_stock = df_stock[['Open', 'High', 'Low', 'Close', 'Volume']]

# B. 讀取新聞
df_news = pd.read_csv(news_file)
df_news['Date'] = pd.to_datetime(df_news['Date'])
df_news = df_news.set_index('Date').sort_index()
news_cols = [c for c in df_news.columns if c.startswith('emb_')]
df_news = df_news[news_cols]

# C. 合併資料
# 我們取「最後 60 天」的資料來做正規化 (Scaler 擬合需要一段時間的數據才準)
# 但我們最後只會取「最後 10 天」進去預測
LOOKBACK_WINDOW = 60 
df_merged = df_stock.join(df_news, how='left').fillna(0)
recent_data = df_merged.iloc[-LOOKBACK_WINDOW:].copy()

if len(recent_data) < TIME_STEPS:
    print("錯誤：資料不足，無法進行預測。")
    exit()

print(f"已載入最近 {len(recent_data)} 天的市場數據。")
last_date = recent_data.index[-1].strftime('%Y-%m-%d')
print(f"最新資料日期: {last_date}")

# ==========================================
# 4. 資料前處理 (跟訓練時必須一模一樣!)
# ==========================================
# A. 正規化 (Scaling)
# 注意：我們要在這 60 天的數據上 fit scaler，以反映近期的相對高低點
scaler = MinMaxScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
recent_data[price_cols] = scaler.fit_transform(recent_data[price_cols])

# B. 擷取最後 N 天 (Model Input)
# 我們要預測「明天」，所以拿「包含今天在內的過去 10 天」
input_price = recent_data[price_cols].values[-TIME_STEPS:]
input_news = recent_data[news_cols].values[-TIME_STEPS:]

# 增加一個維度 (Batch Size) -> (1, 10, 5)
input_price = np.expand_dims(input_price, axis=0)
input_news = np.expand_dims(input_news, axis=0)

# ==========================================
# 5. 進行預測
# ==========================================
print("\n正在進行 AI 運算...")
# prediction 會回傳 [[prob_hold, prob_buy, prob_sell]]
probs = model.predict([input_price, input_news])[0]

prob_hold = probs[0]
prob_buy = probs[1]
prob_sell = probs[2]

# ==========================================
# 6. 輸出決策報告
# ==========================================
print("\n" + "="*30)
print(f"   🤖 AI 交易員預測報告 ({last_date})")
print("="*30)

print(f"盤整機率 (Hold): {prob_hold:.2%}")
print(f"買進機率 (Buy) : {prob_buy:.2%}  <-- 關注這個")
print(f"賣出機率 (Sell): {prob_sell:.2%}")

print("-" * 30)
print("【AI 最終建議】")

if prob_buy > CONFIDENCE_THRESHOLD:
    print(f"🚀 強力買進訊號 (Strong Buy)！ (信心度 > {CONFIDENCE_THRESHOLD*100}%)")
    print("原因：股價形態與新聞情緒同時出現轉折向上的特徵。")
    
elif prob_sell > CONFIDENCE_THRESHOLD:
    print(f"📉 強力賣出訊號 (Strong Sell)！ (信心度 > {CONFIDENCE_THRESHOLD*100}%)")
    print("原因：偵測到利空情緒或股價頭部訊號。")
    
else:
    print("☕ 觀望 / 續抱 (Hold)")
    print("原因：訊號不明顯，建議多觀察幾天。")
    
print("="*30)

# (選擇性) 畫出這 10 天的走勢給你看
# 注意：這是「正規化後」的走勢，主要看型態
plt.figure(figsize=(5, 3))
plt.plot(input_price[0, :, 3], label='Normalized Close') # Index 3 is Close
plt.title("Pattern Used for Prediction (Last 10 Days)")
plt.legend()
plt.show()