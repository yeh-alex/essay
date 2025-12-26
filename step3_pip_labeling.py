import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定
# ==========================================
stock_symbol = "2330.TW"  # 台積電
start_date = "2018-01-01" 
end_date = "2025-12-10"   

# PIP 的數量：
# 因為只標記反轉點的前一天，這裡的數量大約就是最後 Label=1 的總數
PIP_COUNT = 60 

# ==========================================
# 2. PIP 演算法核心 (數學部分 - 不變)
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
    pips = [(0, data[0]), (len(data)-1, data[-1])]
    
    while len(pips) < n_pips:
        max_dist = -1
        max_point = None
        insert_index = -1
        
        for i in range(len(pips) - 1):
            p1, p2 = pips[i], pips[i+1]
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
print(f"--- 步驟 3: 抓取 {stock_symbol} 並標記 PIP 反轉點 (前一天模式) ---")

# 1. 下載股價
print("正在下載股價資料...")
df = yf.download(stock_symbol, start=start_date, end=end_date)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

close_prices = df['Close'].values
dates = df.index
print(f"下載完成，共 {len(df)} 個交易日。")

# 2. 執行 PIP
print(f"正在計算 PIP (目標找出 {PIP_COUNT} 個關鍵點)...")
pip_points = find_pips(close_prices, PIP_COUNT)

# ==========================================
# 3. [修改重點] 轉換為二元標籤且提前一天
# ==========================================
# 0: 盤整/無訊號
# 1: 反轉預警 (代表明天就是 PIP 頂點或底點)
labels = np.zeros(len(close_prices), dtype=int)

# 我們只看 PIP 列表的中間點
for i in range(1, len(pip_points) - 1):
    curr_p = pip_points[i]
    idx = curr_p[0] # 這是 PIP 發生的當天 (第 t 天)
    
    # === 關鍵修改：標記在 idx - 1 (第 t-1 天) ===
    target_idx = idx - 1
    
    # 確保索引沒有超出範圍 (也就是第一天無法標記前一天，需忽略)
    if target_idx >= 0:
        labels[target_idx] = 1 

# 將標籤加回 DataFrame
df['Label'] = labels

print(f"標記完成！")
print(f"總資料筆數: {len(df)}")
print(f"反轉預警點 (Label 1): {sum(labels==1)} 個")
print(f"盤整/無訊號 (Label 0): {sum(labels==0)} 個")
print(f"資料比例 (1 : 0) -> 1 : {sum(labels==0)/sum(labels==1):.2f}")

# 4. 存檔
output_csv = f"stock_labels_{stock_symbol.split('.')[0]}_binary_lag1.csv"
df.to_csv(output_csv)
print(f"已儲存股價與標籤至: {output_csv}")

# ==========================================
# 5. [修改重點] 視覺化驗證
# ==========================================
plt.figure(figsize=(14, 7))

# 畫股價
plt.plot(dates, close_prices, label='Stock Price', color='gray', alpha=0.5)

# 畫 PIP 連線 (雖然標籤提前，但為了方便人類理解，連線還是畫在實際轉折點)
pip_x = [dates[p[0]] for p in pip_points]
pip_y = [p[1] for p in pip_points]
plt.plot(pip_x, pip_y, color='blue', linestyle='--', linewidth=1, label='Actual Trends', alpha=0.4)

# 畫出「標籤點」(這些點會在轉折發生前一天出現)
signal_indices = df[df['Label'] == 1].index
signal_prices = df[df['Label'] == 1]['Close']

# 使用醒目的橘色圓點表示「預警訊號」
plt.scatter(signal_indices, signal_prices, color='orange', marker='o', s=80, label='Reversal Warning (t-1)', zorder=5)

plt.title(f"{stock_symbol} Reversal Prediction (Label=1 at Day t-1)")
plt.legend()
plt.grid(True, alpha=0.3)

# 存圖片
img_name = "pip_verification_binary_lag1.png"
plt.savefig(img_name)
print(f"驗證圖片已儲存至: {img_name}")
plt.show()
