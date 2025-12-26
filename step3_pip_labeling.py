import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定 (針對 8 年資料優化)
# ==========================================
stock_symbol = "2330.TW"  # 台積電

# 設定 8 年區間 (視你的新聞資料而定，這裡設為 2016~2024)
start_date = "2016-01-01"
end_date = "2024-12-31"

# === 關鍵修改：針對 8 年資料調整 PIP 數量 ===
# 假設一年約有 60 個重要反轉點
YEARS_ESTIMATE = 8
PIPS_PER_YEAR = 60
PIP_COUNT = YEARS_ESTIMATE * PIPS_PER_YEAR  # 總共約 480 個點

print(f"--- 設定確認 ---")
print(f"股票代號: {stock_symbol}")
print(f"時間範圍: {start_date} ~ {end_date}")
print(f"預計抓取 PIP 數量: {PIP_COUNT} (基於 {YEARS_ESTIMATE} 年資料估算)")

# ==========================================
# 2. PIP 演算法核心 (數學運算)
# ==========================================
def calculate_vertical_distance(p1, p2, p3):
    """計算點 p3 到直線 p1-p2 的垂直距離"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if x1 == x2: return abs(x3 - x1)
    m = (y2 - y1) / (x2 - x1)
    A, B, C = m, -1, y1 - m * x1
    return abs(A * x3 + B * y3 + C) / np.sqrt(A**2 + B**2)

def find_pips(data, n_pips):
    """
    PIP 演算法主程式
    :param data: 股價 list 或 array
    :param n_pips: 要找出的關鍵點數量
    """
    # pips 初始包含頭尾兩點: [(index, price), ...]
    pips = [(0, data[0]), (len(data)-1, data[-1])]
    
    while len(pips) < n_pips:
        max_dist = -1
        max_point = None
        insert_index = -1
        
        # 遍歷目前所有的線段，找出距離最遠的點
        for i in range(len(pips) - 1):
            p1, p2 = pips[i], pips[i+1]
            # 搜尋該線段區間內的所有點
            for j in range(p1[0] + 1, p2[0]):
                p3 = (j, data[j])
                dist = calculate_vertical_distance(p1, p2, p3)
                if dist > max_dist:
                    max_dist, max_point, insert_index = dist, p3, i + 1
        
        if max_point:
            pips.insert(insert_index, max_point)
        else:
            break # 找不到點了 (區間內無資料)
            
    return pips

# ==========================================
# 主程式流程
# ==========================================
print(f"\n--- 步驟 3: 下載股價並執行 PIP 標記 ---")

# 1. 下載股價
print("正在從 Yahoo Finance 下載資料...")
df = yf.download(stock_symbol, start=start_date, end=end_date)

# 資料清理 (處理 yfinance 新版 MultiIndex 問題)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
    
# 移除空值 (以免 PIP 報錯)
df = df.dropna()

close_prices = df['Close'].values
dates = df.index
print(f"下載完成，共 {len(df)} 個交易日。")

# 2. 執行 PIP
print(f"正在計算 PIP...")
pip_points = find_pips(close_prices, PIP_COUNT)

# ==========================================
# 3. [核心修改] 標籤擴展策略 (t & t-1)
# ==========================================
# 0: 盤整 / 無訊號
# 1: 變盤訊號 (反轉當天 + 反轉前一天)
labels = np.zeros(len(close_prices), dtype=int)

# 遍歷 PIP 點 (排除頭尾)
for i in range(1, len(pip_points) - 1):
    curr_p = pip_points[i]
    idx = curr_p[0] # 反轉發生的當天 index (t)
    
    # 標記 A: 反轉當天 (t)
    labels[idx] = 1
    
    # 標記 B: 反轉前一天 (t-1) -> 這讓模型學習「徵兆」
    if idx - 1 >= 0:
        labels[idx - 1] = 1

# 將標籤加回 DataFrame
df['Label'] = labels

# ==========================================
# 4. 資料平衡檢查
# ==========================================
count_0 = sum(labels == 0)
count_1 = sum(labels == 1)
ratio = count_0 / count_1 if count_1 > 0 else 0

print(f"\n--- 標記結果統計 ---")
print(f"變盤訊號 (Label 1): {count_1} 筆 (包含 t 和 t-1)")
print(f"盤整區間 (Label 0): {count_0} 筆")
print(f"資料比例 (0 vs 1): {ratio:.2f} : 1")

if 0.8 <= ratio <= 1.5:
    print(">> [完美] 資料比例非常平衡，適合直接訓練！")
elif ratio > 3:
    print(">> [警告] 盤整資料還是太多，可能需要增加 PIP_COUNT 或調整權重。")
else:
    print(">> [良好] 資料比例可接受。")

# 存檔
output_csv = f"stock_labels_{stock_symbol.split('.')[0]}_8years_expanded.csv"
df.to_csv(output_csv)
print(f"\n已儲存標記後的資料至: {output_csv}")

# ==========================================
# 5. 視覺化驗證 (抽樣最近 1 年來畫圖，不然 8 年擠在一起看不清)
# ==========================================
print("正在繪製驗證圖表 (顯示最近 300 天)...")

# 只取最後 300 天來畫圖驗證
plot_len = 300
if len(dates) > plot_len:
    plot_dates = dates[-plot_len:]
    plot_prices = close_prices[-plot_len:]
    plot_labels = labels[-plot_len:]
    
    # 篩選這段期間內的 PIP 點用來畫連線
    start_idx = len(dates) - plot_len
    plot_pips = [p for p in pip_points if p[0] >= start_idx]
    # 校正 index 以符合 plot_dates
    plot_pip_x = [dates[p[0]] for p in plot_pips]
    plot_pip_y = [p[1] for p in plot_pips]
else:
    plot_dates = dates
    plot_prices = close_prices
    plot_labels = labels
    plot_pip_x = [dates[p[0]] for p in pip_points]
    plot_pip_y = [p[1] for p in pip_points]

plt.figure(figsize=(14, 7))
plt.plot(plot_dates, plot_prices, label='Stock Price', color='gray', alpha=0.5)

# 畫 PIP 連線
plt.plot(plot_pip_x, plot_pip_y, color='blue', linestyle='--', linewidth=1, label='Actual Trends', alpha=0.5)

# 畫出訊號點 (Label 1)
# 找出 Label 為 1 的那些天的 index
label_1_indices = [i for i, x in enumerate(plot_labels) if x == 1]
if label_1_indices:
    # 對應回原本的日期和價格
    scatter_x = [plot_dates[i] for i in label_1_indices]
    scatter_y = [plot_prices[i] for i in label_1_indices]
    plt.scatter(scatter_x, scatter_y, color='red', s=30, label='Signal (t & t-1)', zorder=5)

plt.title(f"{stock_symbol} Labeling Verification (Last {plot_len} Days)")
plt.legend()
plt.grid(True, alpha=0.3)

img_name = "pip_verification_8years.png"
plt.savefig(img_name)
print(f"驗證圖片已儲存至: {img_name}")
plt.show()
