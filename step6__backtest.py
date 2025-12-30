import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 設定
# ==========================================
stock_symbol = "2330"
stock_file = f"stock_labels_{stock_symbol}_8years_expanded.csv"
news_file = "daily_news_features_tsmc_ecosystem.csv"
model_path = "best_tsmc_model_binary.h5"

# === 策略參數 ===
BEST_THRESHOLD = 0.53
TIME_STEPS = 10
HOLD_DAYS = 5        

# RSI 參數 (關鍵！)
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 45  # 寬鬆一點：RSI 低於 45 且 AI 說變盤 -> 買
RSI_SELL_THRESHOLD = 75 # 嚴格一點：RSI 高於 75 且 AI 說變盤 -> 空 (避免逆勢被軋)

# ==========================================
# 2. 資料準備 (加入 RSI 計算)
# ==========================================
print("--- 啟動 Step 6 (V3): RSI 策略回測 ---")

df_raw = pd.read_csv(stock_file, index_col=0, parse_dates=True)
df_news = pd.read_csv(news_file, index_col=0, parse_dates=True)

# --- 手刻 RSI 函數 (不依賴額外套件) ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 計算 RSI
df_raw['RSI'] = calculate_rsi(df_raw['Close'], RSI_PERIOD)

# 合併與清理
news_cols = [c for c in df_news.columns if c.startswith('emb_') or c.startswith('bert_')]
if not news_cols:
    news_cols = df_news.select_dtypes(include=[np.number]).columns.tolist()
df_news = df_news[news_cols]

df_merged = df_raw.join(df_news, how='left').fillna(0).dropna()

# 正規化 (只針對模型輸入)
scaler = MinMaxScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_scaled = df_merged.copy()
df_scaled[price_cols] = scaler.fit_transform(df_scaled[price_cols])

data_price_scaled = df_scaled[price_cols].values
data_news = df_merged[news_cols].values
data_y = df_merged['Label'].values

# 保留回測用原始數據
raw_close = df_merged['Close'].values
raw_rsi = df_merged['RSI'].values
dates = df_merged.index

# 製作序列
def create_sequences(prices_scaled, news, labels, p_raw, rsi_raw, time_steps=10):
    X_p, X_n, Y = [], [], []
    P_raw_seq, RSI_seq = [], []
    
    for i in range(time_steps, len(prices_scaled)):
        X_p.append(prices_scaled[i-time_steps : i])
        X_n.append(news[i-time_steps : i])
        Y.append(labels[i])
        P_raw_seq.append(p_raw[i])
        RSI_seq.append(rsi_raw[i])
        
    return np.array(X_p), np.array(X_n), np.array(Y), np.array(P_raw_seq), np.array(RSI_seq)

X_price, X_news, Y, Y_raw, Y_rsi = create_sequences(
    data_price_scaled, data_news, data_y, raw_close, raw_rsi, TIME_STEPS
)

# 切分測試集
split = int(len(Y) * 0.8)
X_test_p = X_price[split:]
X_test_n = X_news[split:]
price_test = Y_raw[split:]   
rsi_test = Y_rsi[split:]       
test_dates = dates[TIME_STEPS:][split:]

# ==========================================
# 3. 預測
# ==========================================
print("載入模型並預測...")
model = load_model(model_path)
y_prob = model.predict([X_test_p, X_test_n], verbose=1)
y_pred = (y_prob > BEST_THRESHOLD).astype(int).flatten()

# ==========================================
# 4. 回測邏輯 (RSI Filter)
# ==========================================
print(f"\n--- 開始回測 (RSI Buy<{RSI_BUY_THRESHOLD}, Sell>{RSI_SELL_THRESHOLD}) ---")

# 向量化回測
pos_array = np.zeros(len(price_test))
current_hold_days = 0
current_pos = 0 

for i in range(len(y_pred)):
    # 1. 持倉天數檢查
    if current_pos != 0:
        current_hold_days += 1
        if current_hold_days >= HOLD_DAYS:
            current_pos = 0 # 時間到出場
            current_hold_days = 0
    
    # 2. 進場邏輯
    if current_pos == 0 and y_pred[i] == 1:
        # === 關鍵修改：用 RSI 判斷位階 ===
        
        # 情境 A: AI 說變盤 + RSI 在低檔 (抄底)
        if rsi_test[i] < RSI_BUY_THRESHOLD:
            current_pos = 1
            current_hold_days = 0
            
        # 情境 B: AI 說變盤 + RSI 在高檔 (摸頭)
        # 注意：台股強勢時，RSI > 70 是常態，所以我們要設很高(75)才敢空
        elif rsi_test[i] > RSI_SELL_THRESHOLD:
            current_pos = -1 
            current_hold_days = 0
            
        # 情境 C: RSI 在中間 (45~75) -> 即使 AI 喊單，我們也觀望 (過濾雜訊)
            
    pos_array[i] = current_pos

# 計算損益
daily_pct_change = np.diff(price_test) / price_test[:-1]
daily_pct_change = np.insert(daily_pct_change, 0, 0)
strategy_pct_change = pos_array[:-1] * daily_pct_change[1:]
strategy_pct_change = np.insert(strategy_pct_change, 0, 0)

cum_market = (1 + daily_pct_change).cumprod()
cum_strategy = (1 + strategy_pct_change).cumprod()
total_ret = cum_strategy[-1] - 1

# 統計
n_long = np.sum((pos_array == 1) & (np.roll(pos_array, 1) == 0))
n_short = np.sum((pos_array == -1) & (np.roll(pos_array, 1) == 0))

print(f"回測期間: {test_dates[0].date()} ~ {test_dates[-1].date()}")
print(f"策略總報酬率: {total_ret*100:.2f}%")
print(f"做多次數: {n_long}")
print(f"放空次數: {n_short}")
print(f"市場基準報酬: {(cum_market[-1]-1)*100:.2f}%")

# ==========================================
# 5. 畫圖
# ==========================================
plt.figure(figsize=(14, 8))

# 上圖：股價 + RSI 點位
ax1 = plt.subplot(2, 1, 1)
ax1.plot(test_dates, price_test, label='Stock Price', color='gray', alpha=0.5)
long_idx = np.where((pos_array == 1) & (np.roll(pos_array, 1) == 0))[0]
ax1.scatter(test_dates[long_idx], price_test[long_idx], color='red', marker='^', s=80, label='Buy (Low RSI)', zorder=5)
short_idx = np.where((pos_array == -1) & (np.roll(pos_array, 1) == 0))[0]
ax1.scatter(test_dates[short_idx], price_test[short_idx], color='green', marker='v', s=80, label='Sell (High RSI)', zorder=5)
ax1.set_title(f'Trades with RSI Filter (Long<{RSI_BUY_THRESHOLD}, Short>{RSI_SELL_THRESHOLD})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 下圖：資產曲線
plt.subplot(2, 1, 2)
plt.plot(test_dates, cum_strategy, label='AI + RSI Strategy', color='red', linewidth=2)
plt.plot(test_dates, cum_market, label='Buy & Hold', color='blue', alpha=0.3)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
plt.title(f'Cumulative Return: {total_ret*100:.2f}% vs Market: {(cum_market[-1]-1)*100:.2f}%')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step6_backtest_rsi.png")
plt.show()
