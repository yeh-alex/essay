import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定
# ==========================================
stock_symbol = "2330.TW"  # 台積電
start_date = "2024-01-01" # 建議抓長一點，PIP 效果比較好
end_date = "2024-12-31"   # 或用 datetime.now()

# PIP 的數量：這決定了您要抓多「大」的波段
# 數量越少 -> 只抓大波段 (長線)
# 數量越多 -> 連小波段都抓 (短線)
# 建議：約總天數的 5% ~ 10%
PIP_COUNT = 60

# ==========================================
# 2. PIP 演算法核心 (數學部分)
# ==========================================
def calculate_vertical_distance(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if x1 == x2: return abs(x3 - x1)
    m = (y2 - y1) / (x2 - x1)
    A, B, C = m, -1, y1 - m * x1
    return abs(A * x3 + B * y3 + C) / np.sqrt(A**2 + B**2)

def find_pips(data, n_pips):
    # pips 存放格式: [(index, price), (index, price)...]
    pips = [(0, data[0]), (len(data)-1, data[-1])]
    
    while len(pips) < n_pips:
        max_dist = -1
        max_point = None
        insert_index = -1
        
        for i in range(len(pips) - 1):
            p1, p2 = pips[i], pips[i+1]
            # 搜尋兩點區間內的所有點
            for j in range(p1[0] + 1, p2[0]):
                p3 = (j, data[j])
                dist = calculate_vertical_distance(p1, p2, p3)
                if dist > max_dist:
                    max_dist, max_point, insert_index = dist, p3, i + 1
        
        if max_point: pips.insert(insert_index, max_point)
        else: break
    return pips

# ==========================================
# 主程式
# ==========================================
print(f"--- 步驟 3: 抓取 {stock_symbol} 並標記 PIP 反轉點 ---")

# 1. 下載股價
print("正在下載股價資料...")
df = yf.download(stock_symbol, start=start_date, end=end_date)
# 清理 MultiIndex (yfinance 新版問題)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 只要收盤價
close_prices = df['Close'].values
dates = df.index
print(f"下載完成，共 {len(df)} 個交易日。")

# 2. 執行 PIP
print(f"正在計算 PIP (目標找出 {PIP_COUNT} 個關鍵點)...")
pip_points = find_pips(close_prices, PIP_COUNT)
# pip_points 是一個 list: [(0, 450.0), (15, 500.0), ...]

# 3. 轉換為標籤 (Labeling)
# 0: 無訊號, 1: 買點(Valley), 2: 賣點(Peak)
labels = np.zeros(len(close_prices), dtype=int)
pip_indices = [] # 存下來畫圖用

# 我們只看 PIP 列表的中間點 (排除頭尾)
for i in range(1, len(pip_points) - 1):
    prev_p = pip_points[i-1]
    curr_p = pip_points[i]
    next_p = pip_points[i+1]
    
    idx = curr_p[0]
    price = curr_p[1]
    pip_indices.append(idx)
    
    # 判斷波峰還是波谷
    if price > prev_p[1] and price > next_p[1]:
        labels[idx] = 2 # 賣點 (高點)
    elif price < prev_p[1] and price < next_p[1]:
        labels[idx] = 1 # 買點 (低點)

# 將標籤加回 DataFrame
df['Label'] = labels
print(f"標記完成！")
print(f"買點 (Label 1): {sum(labels==1)} 個")
print(f"賣點 (Label 2): {sum(labels==2)} 個")

# 4. 存檔 (這是之後要跟新聞合併的檔案)
output_csv = f"stock_labels_{stock_symbol.split('.')[0]}.csv"
df.to_csv(output_csv)
print(f"已儲存股價與標籤至: {output_csv}")

# ==========================================
# 5. 視覺化驗證 (非常重要!)
# ==========================================
plt.figure(figsize=(14, 7))
plt.plot(dates, close_prices, label='Stock Price', color='gray', alpha=0.5)

# 畫出 PIP 連線 (紅色趨勢線)
pip_x = [dates[p[0]] for p in pip_points]
pip_y = [p[1] for p in pip_points]
plt.plot(pip_x, pip_y, color='blue', linestyle='--', linewidth=1, label='PIP Trend', alpha=0.6)

# 畫出買賣點
buy_indices = df[df['Label'] == 1].index
buy_prices = df[df['Label'] == 1]['Close']
plt.scatter(buy_indices, buy_prices, color='red', marker='^', s=100, label='Buy (Valley)', zorder=5)

sell_indices = df[df['Label'] == 2].index
sell_prices = df[df['Label'] == 2]['Close']
plt.scatter(sell_indices, sell_prices, color='green', marker='v', s=100, label='Sell (Peak)', zorder=5)

plt.title(f"{stock_symbol} PIP Turning Points (n={PIP_COUNT})")
plt.legend()
plt.grid(True, alpha=0.3)

# 存圖片
img_name = "pip_verification.png"
plt.savefig(img_name)
print(f"驗證圖片已儲存至: {img_name} (請打開檢查)")
plt.show()