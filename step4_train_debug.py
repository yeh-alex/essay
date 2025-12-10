import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定
# ==========================================
# 請確認檔名正確
news_file = "daily_news_features_tsmc_ecosystem.csv" 
stock_file = "stock_labels_2330.csv"                 
model_save_path = "best_tsmc_model_robust.h5"

TIME_STEPS = 10
BATCH_SIZE = 16  # 資料少，Batch Size 改小一點
EPOCHS = 60

# ==========================================
# 2. 資料讀取與檢查 (Data Health Check)
# ==========================================
print("--- 啟動修復版訓練程序 ---")

# A. 讀取
try:
    df_stock = pd.read_csv(stock_file, index_col='Date', parse_dates=True)
    df_news = pd.read_csv(news_file, index_col='Date', parse_dates=True)
    
    # 篩選 Embedding 欄位
    news_cols = [c for c in df_news.columns if c.startswith('emb_')]
    df_news = df_news[news_cols]
    
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案 {e}")
    exit()

# B. 合併
df_merged = df_stock.join(df_news, how='left').fillna(0)

# C. 檢查類別分佈 (這步最重要！)
label_counts = df_merged['Label'].value_counts().sort_index()
print("\n[數據健康檢查] 您的標籤分佈：")
print(label_counts)

# 如果買點(1)或賣點(2)少於 30 筆，強烈建議增加資料年份
min_class_count = label_counts.min()
if min_class_count < 30:
    print(f"\n⚠️ 警告：稀有類別樣本數僅 {min_class_count} 筆，極易造成過擬合！")
    print("建議：請回到 Step 1 & 3，將抓取時間拉長到 3-5 年 (例如 2020-2024)。")
    # 但我們還是繼續跑，嘗試救救看

# ==========================================
# 3. 前處理
# ==========================================
# 正規化
scaler = MinMaxScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_merged[price_cols] = scaler.fit_transform(df_merged[price_cols])

data_price = df_merged[price_cols].values
data_news = df_merged[news_cols].values
data_y = df_merged['Label'].values

# 製作序列
def create_sequences(prices, news, labels, time_steps=10):
    X_p, X_n, Y = [], [], []
    for i in range(len(prices) - time_steps):
        X_p.append(prices[i : i+time_steps])
        X_n.append(news[i : i+time_steps])
        Y.append(labels[i + time_steps])
    return np.array(X_p), np.array(X_n), np.array(Y)

X_price, X_news, Y = create_sequences(data_price, data_news, data_y, TIME_STEPS)

# 切分 (不打亂時間順序)
split = int(len(Y) * 0.8)
X_train_p, X_test_p = X_price[:split], X_price[split:]
X_train_n, X_test_n = X_news[:split], X_news[split:]
y_train, y_test = Y[:split], Y[split:]

# 轉 One-hot
y_train_cat = tf.keras.utils.to_categorical(y_train, 3)
y_test_cat = tf.keras.utils.to_categorical(y_test, 3)

# ==========================================
# 4. 手動設定溫和權重 (Softer Weights)
# ==========================================
# 原本 balanced 可能會給出 {0:0.5, 1:10, 2:10} 這種極端值
# 我們改用比較溫和的倍率：盤整=1, 買賣點=3~5倍
counts = np.bincount(y_train)
total = sum(counts)
weight_0 = 1.0
weight_1 = (counts[0] / counts[1]) * 0.3 if counts[1] > 0 else 10.0 # 只補償 30% 的不平衡
weight_2 = (counts[0] / counts[2]) * 0.3 if counts[2] > 0 else 10.0

# 限制最大權重，避免梯度爆炸
weight_1 = min(weight_1, 5.0)
weight_2 = min(weight_2, 5.0)

class_weight_dict = {0: weight_0, 1: weight_1, 2: weight_2}
print(f"\n[權重設定] 使用溫和權重: {class_weight_dict}")

# ==========================================
# 5. 簡化版模型 (Simple Model)
# ==========================================
input_price = Input(shape=(TIME_STEPS, 5), name='price')
# 簡化 LSTM: 64 -> 32
x_price = LSTM(32, return_sequences=False)(input_price)
x_price = Dropout(0.2)(x_price) # 降低 Dropout

input_news = Input(shape=(TIME_STEPS, 768), name='news')
# 簡化架構: Dense 128 -> 64, LSTM 64 -> 32
x_news = Dense(64, activation='relu')(input_news)
x_news = LSTM(32, return_sequences=False)(x_news)
x_news = Dropout(0.2)(x_news)

combined = concatenate([x_price, x_news])
z = Dense(32, activation='relu')(combined)
z = BatchNormalization()(z) # 加 BatchNorm 穩定訓練
output = Dense(3, activation='softmax')(z)

model = Model(inputs=[input_price, input_news], outputs=output)

# ==========================================
# 6. 訓練配置 (加上梯度裁剪)
# ==========================================
# clipnorm=1.0 是關鍵！它會限制梯度最大值，防止崩潰
optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 加入 ReduceLROnPlateau: 如果 Loss 卡住，自動再降低學習率
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

print("\n開始訓練...")
history = model.fit(
    [X_train_p, X_train_n], 
    y_train_cat,
    validation_data=([X_test_p, X_test_n], y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict, 
    callbacks=[checkpoint, early_stop, lr_reducer],
    verbose=1
)

# 畫圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy (Fixed)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss (Fixed)')
plt.legend()
plt.savefig("training_history_fixed.png")
print("\n修復版訓練完成！請檢查 training_history_fixed.png")