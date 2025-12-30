import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns

# ==========================================
# 1. 設定 (必須與 Step 4 完全一致)
# ==========================================
stock_symbol = "2330"
# 注意：這裡要讀取跟 Step 4 一樣的檔案
stock_file = f"stock_labels_{stock_symbol}_8years_expanded.csv"
news_file = "daily_news_features_tsmc_ecosystem.csv"
model_path = "best_tsmc_model_binary.h5"

TIME_STEPS = 10
# 設定你要用多少信心才相信模型 (稍後程式會幫你算出最佳值，這裡先給預設)
DEFAULT_THRESHOLD = 0.6 

# ==========================================
# 2. 資料重建 (Data Pipeline Reconstruction)
# ==========================================
print("--- 啟動 Step 5: 模型回測與視覺化 ---")
print("正在重建資料管線...")

# A. 讀取
try:
    df_stock = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    df_news = pd.read_csv(news_file, index_col=0, parse_dates=True)
    
    # 篩選 Embedding 欄位
    news_cols = [c for c in df_news.columns if c.startswith('emb_') or c.startswith('bert_')]
    if not news_cols:
        news_cols = df_news.select_dtypes(include=[np.number]).columns.tolist()
    df_news = df_news[news_cols]
    
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案 {e}，請確認 Step 4 是否執行成功。")
    exit()

# B. 合併
df_merged = df_stock.join(df_news, how='left').fillna(0)

# C. 前處理 (必須與 Step 4 的 Scaler 邏輯一致)
scaler = MinMaxScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
existing_price_cols = [c for c in price_cols if c in df_merged.columns]

# 注意：這裡我們對全體資料做 Transform，以確保數據分佈一致
# (在嚴謹的回測中，Scaler 應該只 fit 在 training set，但為了視覺化方便，這裡沿用全域 fit)
df_merged[existing_price_cols] = scaler.fit_transform(df_merged[existing_price_cols])

data_price = df_merged[existing_price_cols].values
data_news = df_merged[news_cols].values
data_y = df_merged['Label'].values

# 保留「原始收盤價」以便畫圖 (還原用)
# 我們需要重新讀一次原始檔案來拿未正規化的價格
df_raw = pd.read_csv(stock_file, index_col=0, parse_dates=True)
raw_close = df_raw['Close'].values

# D. 製作時間序列
def create_sequences(prices, news, labels, raw_prices, time_steps=10):
    X_p, X_n, Y, P_raw = [], [], [], []
    for i in range(time_steps, len(prices)):
        X_p.append(prices[i-time_steps : i])
        X_n.append(news[i-time_steps : i])
        Y.append(labels[i])
        # 我們存下第 i 天的原始價格，畫圖用
        P_raw.append(raw_prices[i]) 
    return np.array(X_p), np.array(X_n), np.array(Y), np.array(P_raw)

# 這裡多回傳一個 raw_prices 序列
X_price, X_news, Y, Y_raw_price = create_sequences(data_price, data_news, data_y, raw_close, TIME_STEPS)

# E. 切分測試集 (只取最後 20% 做驗證)
split = int(len(Y) * 0.8)
X_test_p = X_price[split:]
X_test_n = X_news[split:]
y_test = Y[split:]
price_test = Y_raw_price[split:] # 測試集的真實股價
dates_test = df_merged.index[TIME_STEPS:][split:] # 測試集的日期

print(f"測試集資料筆數: {len(y_test)}")
print(f"回測期間: {dates_test[0].date()} 到 {dates_test[-1].date()}")

# ==========================================
# 3. 載入模型與預測
# ==========================================
print(f"\n正在載入模型: {model_path} ...")
model = load_model(model_path)

print("正在進行推論 (Inference)...")
y_pred_prob = model.predict([X_test_p, X_test_n], verbose=1)

# ==========================================
# 4. 尋找最佳閥值 (Threshold Tuning)
# ==========================================
print("\n--- [關鍵分析] 最佳閥值尋找 ---")
print("目標：找到 Precision (準度) > 0.6 且 Recall (抓到率) 不太差的點")
thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

best_threshold = DEFAULT_THRESHOLD
max_f1 = 0

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")
print("-" * 45)

for thresh in thresholds:
    y_pred_temp = (y_pred_prob > thresh).astype(int)
    prec = precision_score(y_test, y_pred_temp, zero_division=0)
    rec = recall_score(y_test, y_pred_temp, zero_division=0)
    acc = accuracy_score(y_test, y_pred_temp)
    
    # 簡單的 F1 Score 近似，用來選最佳解
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    if f1 > max_f1 and prec > 0.5: # 條件：準度至少要過半
        max_f1 = f1
        best_threshold = thresh
        
    print(f"{thresh:<10} {prec:.4f}     {rec:.4f}     {acc:.4f}")

print("-" * 45)
print(f"系統建議最佳閥值: {best_threshold}")

# 使用最佳閥值產生最終預測
y_pred = (y_pred_prob > best_threshold).astype(int)

# ==========================================
# 5. 視覺化驗證 (Visualization)
# ==========================================
# 只畫最後一年 (約 250 天)，不然點會擠在一起看不清楚
plot_len = 250 
if len(price_test) > plot_len:
    start_idx = len(price_test) - plot_len
    p_dates = dates_test[start_idx:]
    p_prices = price_test[start_idx:]
    p_true = y_test[start_idx:]
    p_pred = y_pred[start_idx:].flatten()
else:
    p_dates = dates_test
    p_prices = price_test
    p_true = y_test
    p_pred = y_pred.flatten()

# ==========================================
# 6. 微調分析：針對 0.50 ~ 0.60 進行顯微鏡檢查
# ==========================================
print("\n--- [微調分析] 針對 0.50~0.60 進行細部掃描 ---")
# 我們切得更細：0.50, 0.51, 0.52 ... 0.60
fine_thresholds = np.arange(0.50, 0.61, 0.01)

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'Accuracy':<10}")
print("-" * 45)

for thresh in fine_thresholds:
    y_pred_temp = (y_pred_prob > thresh).astype(int)
    
    prec = precision_score(y_test, y_pred_temp, zero_division=0)
    rec = recall_score(y_test, y_pred_temp, zero_division=0)
    acc = accuracy_score(y_test, y_pred_temp)
    
    # 標記出推薦的點 (準度 > 60% 且 還有抓到東西)
    mark = ""
    if prec > 0.60 and rec > 0.2:
        mark = "<-- 潛力點"
    elif prec > 0.58 and rec > 0.5:
        mark = "<-- 穩健點"
        
    print(f"{thresh:<10.2f} {prec:.4f}     {rec:.4f}     {acc:.4f}   {mark}")


plt.figure(figsize=(14, 7))

# 1. 畫股價線
plt.plot(p_dates, p_prices, label='Stock Price', color='gray', alpha=0.5, linewidth=1.5)

# 2. 畫模型預測的變盤點 (紅三角)
pred_idx = np.where(p_pred == 1)[0]
if len(pred_idx) > 0:
    plt.scatter(p_dates[pred_idx], p_prices[pred_idx], 
                color='red', marker='^', s=80, label=f'AI Predicted (Conf>{best_threshold})', zorder=5)

# 3. 畫真實的反轉點 (藍點)
true_idx = np.where(p_true == 1)[0]
if len(true_idx) > 0:
    plt.scatter(p_dates[true_idx], p_prices[true_idx], 
                color='blue', marker='.', s=30, label='Actual Reversal (Truth)', alpha=0.5)

plt.title(f"{stock_symbol} Reversal Prediction (Last {len(p_prices)} Days)\nThreshold={best_threshold}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("step5_prediction_result.png")
print("\n視覺化完成！請查看: step5_prediction_result.png")

# 畫混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Consolidation', 'Reversal'],
            yticklabels=['Consolidation', 'Reversal'])
plt.title(f'Confusion Matrix (Thresh={best_threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("step5_confusion_matrix.png")

plt.show()