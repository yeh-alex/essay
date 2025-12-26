import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定
# ==========================================
# 請確認檔名正確 (對應 Step 3 輸出的新檔名)
stock_symbol = "2330"
stock_file = f"stock_labels_{stock_symbol}_8years_expanded.csv"
news_file = "daily_news_features_tsmc_ecosystem.csv"  # 假設新聞檔名沒變
model_save_path = "best_tsmc_model_binary.h5"

TIME_STEPS = 10
BATCH_SIZE = 32  # 8 年資料量較大，可以設 32 或 64
EPOCHS = 60

# ==========================================
# 2. 資料讀取與合併 (Data Pipeline)
# ==========================================
print("--- 啟動 Step 4: 訓練二元分類模型 (變盤預測) ---")

# A. 讀取
try:
    print(f"讀取股價標籤檔: {stock_file}")
    df_stock = pd.read_csv(stock_file, index_col=0, parse_dates=True) # 假設第一欄是 Date
    
    print(f"讀取新聞特徵檔: {news_file}")
    df_news = pd.read_csv(news_file, index_col=0, parse_dates=True)
    
    # 篩選 Embedding 欄位 (假設新聞向量欄位以 emb_ 開頭)
    # 如果你的欄位名不同，請在這裡修改，例如 col.startswith('bert_')
    news_cols = [c for c in df_news.columns if c.startswith('emb_') or c.startswith('bert_')]
    if not news_cols:
        # 如果找不到 emb_ 開頭，嘗試讀取所有數值欄位作為特徵 (除了 Date)
        print("警告：找不到 'emb_' 開頭欄位，將使用新聞檔中所有數值欄位。")
        news_cols = df_news.select_dtypes(include=[np.number]).columns.tolist()
        
    df_news = df_news[news_cols]
    
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案 {e}")
    exit()

# B. 合併 (Left Join 以股價日曆為主)
# fillna(0) 代表當天沒新聞就補 0 向量
df_merged = df_stock.join(df_news, how='left').fillna(0)

# C. 檢查類別分佈
label_counts = df_merged['Label'].value_counts().sort_index()
print("\n[數據健康檢查] 標籤分佈 (0:盤整, 1:變盤訊號):")
print(label_counts)

# 自動計算不平衡比例
count_0 = label_counts.get(0, 0)
count_1 = label_counts.get(1, 0)
ratio = count_0 / count_1 if count_1 > 0 else 0
print(f"目前比例 (0 : 1) = {ratio:.2f} : 1")

# ==========================================
# 3. 前處理 (Preprocessing)
# ==========================================
# 正規化股價數據
scaler = MinMaxScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# 確保這些欄位存在
existing_price_cols = [c for c in price_cols if c in df_merged.columns]
df_merged[existing_price_cols] = scaler.fit_transform(df_merged[existing_price_cols])

data_price = df_merged[existing_price_cols].values
data_news = df_merged[news_cols].values
data_y = df_merged['Label'].values

# 製作時間序列視窗 (Sliding Window)
def create_sequences(prices, news, labels, time_steps=10):
    X_p, X_n, Y = [], [], []
    # 注意：我们要用過去 time_steps 天的數據，預測「第 i 天」的標籤
    # 由於我們在 Step 3 已經把標籤 shift (t-1) 好了，這裡直接對齊即可
    for i in range(time_steps, len(prices)):
        X_p.append(prices[i-time_steps : i])
        X_n.append(news[i-time_steps : i])
        Y.append(labels[i]) 
    return np.array(X_p), np.array(X_n), np.array(Y)

X_price, X_news, Y = create_sequences(data_price, data_news, data_y, TIME_STEPS)

# 切分訓練/測試集 (時間序列不可打亂！)
split = int(len(Y) * 0.8) # 80% Train, 20% Test
X_train_p, X_test_p = X_price[:split], X_price[split:]
X_train_n, X_test_n = X_news[:split], X_news[split:]
y_train, y_test = Y[:split], Y[split:]

print(f"\n訓練集樣本數: {len(y_train)}, 測試集樣本數: {len(y_test)}")

# ==========================================
# 4. 權重策略 (Class Weight) - 自動判斷
# ==========================================
# 如果比例在 1:1.5 以內，我們就不加權重 (設為 None)，效果通常比較好
if 0.6 <= ratio <= 1.5:
    print("[權重設定] 資料已經很平衡，不使用 Class Weight。")
    class_weight_dict = None
else:
    # 如果還是不平衡，計算 class_weight
    # 使用 sklearn 自動計算平衡權重
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(weights))
    print(f"[權重設定] 偵測到不平衡，使用自動計算權重: {class_weight_dict}")

# ==========================================
# 5. 雙輸入模型架構 (Binary Classification)
# ==========================================
# 輸入 A: 股價技術面
input_price = Input(shape=(TIME_STEPS, len(existing_price_cols)), name='price')
x_price = LSTM(64, return_sequences=True)(input_price)
x_price = Dropout(0.3)(x_price)
x_price = LSTM(32, return_sequences=False)(x_price)

# 輸入 B: 新聞語意面
# 如果沒有新聞特徵 (news_cols 為空)，這部分會報錯，所以要檢查
input_news = Input(shape=(TIME_STEPS, len(news_cols)), name='news')
x_news = Dense(64, activation='relu')(input_news) # 先降維
x_news = LSTM(32, return_sequences=False)(x_news)
x_news = Dropout(0.3)(x_news)

# 合併
combined = concatenate([x_price, x_news])
z = Dense(32, activation='relu')(combined)
z = BatchNormalization()(z)
z = Dropout(0.3)(z)

# 輸出層：二元分類使用 Sigmoid (輸出機率 0~1)
output = Dense(1, activation='sigmoid', name='prediction')(z)

model = Model(inputs=[input_price, input_news], outputs=output)

# ==========================================
# 6. 訓練配置
# ==========================================
# Binary Classification 必須使用 binary_crossentropy
# Metrics 加入 Precision 和 Recall 以便觀察是否真的抓到變盤點
optimizer = Adam(learning_rate=0.001, clipnorm=1.0) # clipnorm 防止梯度爆炸

model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

# Callbacks
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

print("\n開始訓練...")
history = model.fit(
    [X_train_p, X_train_n], 
    y_train, # 二元標籤不需要 to_categorical
    validation_data=([X_test_p, X_test_n], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict, 
    callbacks=[checkpoint, early_stop, lr_reducer],
    verbose=1
)

# ==========================================
# 7. 畫圖與結果分析
# ==========================================
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss (Binary Crossentropy)')
plt.legend()

# Recall (重點指標)
plt.subplot(1, 3, 3)
plt.plot(history.history['recall'], label='Train')
plt.plot(history.history['val_recall'], label='Val')
plt.title('Recall (Sensitivity)')
plt.legend()

plt.savefig("training_history_binary.png")
print("\n訓練完成！結果圖已儲存為 training_history_binary.png")
print(f"最佳模型已儲存為: {model_save_path}")

# 最後做一個簡單的評估
print("\n--- 最終測試集評估 ---")
loss, acc, prec, rec = model.evaluate([X_test_p, X_test_n], y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f} (預測變盤準確度)")
print(f"Recall: {rec:.4f} (實際變盤抓到率)")
