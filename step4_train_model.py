import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定檔案與參數
# ==========================================
# news_file = "daily_news_features_high_precision.csv" # 步驟二產生的新聞特徵
news_file = "daily_news_features_tsmc_ecosystem.csv"  # <--- 改成這個！
stock_file = "stock_labels_2330.csv"                 # 步驟三產生的股價與標籤
model_save_path = "best_tech_stock_model.h5"         # 模型存檔位置

TIME_STEPS = 10  # AI 要看過去幾天的新聞/股價來預測？(建議 10~20)
BATCH_SIZE = 32
EPOCHS = 50

# ==========================================
# 2. 資料讀取與合併
# ==========================================
print("--- 步驟 4: 資料準備與模型訓練 ---")

# A. 讀取股價
try:
    df_stock = pd.read_csv(stock_file)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock = df_stock.set_index('Date')
    # 只需要 OHLCV 和 Label
    df_stock = df_stock[['Open', 'High', 'Low', 'Close', 'Volume', 'Label']]
    print(f"讀取股價資料: {len(df_stock)} 筆")
except FileNotFoundError:
    print(f"找不到 {stock_file}，請先執行步驟三。")
    exit()

# B. 讀取新聞特徵
try:
    df_news = pd.read_csv(news_file)
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    df_news = df_news.set_index('Date')
    # 篩選出 embedding 欄位 (emb_0 ~ emb_767)
    news_cols = [c for c in df_news.columns if c.startswith('emb_')]
    df_news = df_news[news_cols]
    print(f"讀取新聞特徵: {len(df_news)} 筆")
except FileNotFoundError:
    print(f"找不到 {news_file}，請先執行步驟二。")
    exit()

# C. 合併 (Merge)
# 以股價的日期為準 (left join)，因為股市沒開盤的日子不需要預測
# 如果某天有開盤但沒新聞，fillna(0) 補零向量
print("正在合併資料...")
df_merged = df_stock.join(df_news, how='left').fillna(0)

# ==========================================
# 3. 資料前處理 (Preprocessing)
# ==========================================
# A. 正規化 (Normalization)
# LSTM 對數值很敏感，股價 (例如 500元) 必須縮放到 0~1 之間
scaler = MinMaxScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df_merged[price_cols] = scaler.fit_transform(df_merged[price_cols])

# B. 準備特徵矩陣 (X) 和 標籤 (Y)
data_price = df_merged[price_cols].values
data_news = df_merged[news_cols].values
data_y = df_merged['Label'].values

# C. 製作序列資料 (Sliding Window)
# 這是 LSTM 訓練的關鍵：把 "過去 N 天" 打包成一筆資料
def create_sequences(prices, news, labels, time_steps=10):
    X_p, X_n, Y = [], [], []
    for i in range(len(prices) - time_steps):
        # 輸入: 第 i 天 到 第 i+N-1 天
        X_p.append(prices[i : i+time_steps])
        X_n.append(news[i : i+time_steps])
        # 預測: 第 i+N 天的狀態 (Label)
        # 注意: 這裡是預測 "這段期間結束後的那一天" 是不是轉折點
        Y.append(labels[i + time_steps])
    return np.array(X_p), np.array(X_n), np.array(Y)

print(f"製作序列資料 (Time Steps={TIME_STEPS})...")
X_price, X_news, Y = create_sequences(data_price, data_news, data_y, TIME_STEPS)

# 切分訓練集 (80%) 與 測試集 (20%)
# 這裡不隨機打亂 (Shuffle=False)，因為時間序列必須保持順序
split = int(len(Y) * 0.8)
X_train_p, X_test_p = X_price[:split], X_price[split:]
X_train_n, X_test_n = X_news[:split], X_news[split:]
y_train, y_test = Y[:split], Y[split:]

print(f"訓練集: {len(y_train)}, 測試集: {len(y_test)}")

# ==========================================
# 4. 計算 Class Weights (關鍵!)
# ==========================================
# 因為轉折點 (1, 2) 很少，我們要給它們更高的權重
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"類別權重 (解決不平衡): {class_weight_dict}")
# 預期結果類似: {0: 0.35, 1: 5.2, 2: 5.2} -> 代表猜對買賣點的分數是猜對盤整的 15 倍

# 確保 Y 轉為 One-hot encoding (給 Softmax 用)
y_train_cat = tf.keras.utils.to_categorical(y_train, 3)
y_test_cat = tf.keras.utils.to_categorical(y_test, 3)

# ==========================================
# 5. 建立雙模態 LSTM 模型
# ==========================================
print("建立模型架構...")

# --- 分支 A: 股價塔 (Price Tower) ---
input_price = Input(shape=(TIME_STEPS, 5), name='price_input')
x_price = LSTM(64, return_sequences=False)(input_price)
x_price = Dropout(0.3)(x_price)

# --- 分支 B: 新聞塔 (News Tower) ---
input_news = Input(shape=(TIME_STEPS, 768), name='news_input')
# 因為 768 維很大，先用 Dense 降維，減輕 LSTM 負擔
x_news = Dense(128, activation='relu')(input_news) 
x_news = LSTM(64, return_sequences=False)(x_news)
x_news = Dropout(0.3)(x_news)

# --- 合併 (Fusion) ---
combined = concatenate([x_price, x_news])
z = Dense(64, activation='relu')(combined)
z = BatchNormalization()(z)
z = Dropout(0.3)(z)

# --- 輸出層 (3個類別: 0盤整, 1買, 2賣) ---
output = Dense(3, activation='softmax', name='prediction')(z)

model = Model(inputs=[input_price, input_news], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 6. 訓練模型
# ==========================================
print("\n開始訓練...")

# 設定回調函數：只存最好的模型，且如果沒進步就提早停止
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = model.fit(
    [X_train_p, X_train_n], 
    y_train_cat,
    validation_data=([X_test_p, X_test_n], y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict, # 這裡套用了加權！
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ==========================================
# 7. 訓練結果視覺化
# ==========================================
# 畫出 Accuracy 和 Loss 曲線
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.savefig("training_history.png")
print("\n訓練完成！歷史曲線已儲存為 training_history.png")
print(f"最佳模型已儲存為 {model_save_path}")

# ==========================================
# 8. 簡單驗證 (預測測試集)
# ==========================================
print("\n--- 測試集驗證 ---")
# 載入最好的模型 (不是最後一輪的，而是驗證分數最高的)
best_model = tf.keras.models.load_model(model_save_path)
pred_probs = best_model.predict([X_test_p, X_test_n])
pred_classes = np.argmax(pred_probs, axis=1)

# 印出混淆矩陣
from sklearn.metrics import confusion_matrix, classification_report
print("\n混淆矩陣 (Confusion Matrix):")
print(confusion_matrix(y_test, pred_classes))
print("\n分類報告:")
print(classification_report(y_test, pred_classes, target_names=['Hold', 'Buy', 'Sell']))