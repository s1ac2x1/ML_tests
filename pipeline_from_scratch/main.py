import pandas as pd
import numpy as np
from scipy import stats

# 1. Загрузка данных
df = pd.read_csv(
    "AES_hourly.csv",
    parse_dates=["datetime"],
    index_col="datetime"
)

# 2. Проверка и заполнение пропусков таймстемпов (непрерывный ряд часов)
idx_full = pd.date_range(df.index.min(), df.index.max(), freq="H")
missing = idx_full.difference(df.index)
print(f"Пропущено баров: {len(missing)}")

df = df.reindex(idx_full)
df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].interpolate()
df["volume"] = df["volume"].fillna(0)

# 3. Обработка выбросов по скользящему Z-score по Close
rolling_mean = df["close"].rolling(window=100, min_periods=1).mean()
rolling_std  = df["close"].rolling(window=100, min_periods=1).std()
z_score = (df["close"] - rolling_mean) / rolling_std
mask = (z_score.abs() < 5) | z_score.isna()
df = df.loc[mask]

# 4. Расчёт ATR (Average True Range) за 14 баров
high_low        = df["high"] - df["low"]
high_prev_close = (df["high"] - df["close"].shift()).abs()
low_prev_close  = (df["low"]  - df["close"].shift()).abs()
tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
df["ATR"] = tr.rolling(window=14, min_periods=14).mean()

# 5. Установка динамического стоп-лосса
k = 1.2  # коэффициент (обычно 1.0–1.5)
df["stop_price"] = df["close"] - k * df["ATR"]

# 6. Разметка целевой переменной y
H = 5  # горизонт прогнозирования в барах
df["future_min_low"] = (
    df["low"]
    .shift(-1)
    .rolling(window=H, min_periods=H)
    .min()
)
df["y"] = (df["future_min_low"] > df["stop_price"]).astype(int)

# 7. Инженерия признаков

# 7.1 Скользящие средние, EMA и волатильность
for window in [5, 10, 20]:
    df[f"SMA_{window}"] = df["close"].rolling(window, min_periods=1).mean()
    df[f"EMA_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
    df[f"STD_{window}"] = df["close"].rolling(window, min_periods=1).std()

# 7.2 Лаговые доходности и моментум
df["return_1h"] = df["close"].pct_change(1)
df["return_5h"] = df["close"].pct_change(5)
df["momentum_5h"] = df["close"] - df["close"].shift(5)

# 7.3 RSI (14)
delta = df["close"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
roll_up = up.rolling(14, min_periods=14).mean()
roll_down = down.rolling(14, min_periods=14).mean()
rs = roll_up / roll_down
df["RSI_14"] = 100 - (100 / (1 + rs))

# 7.4 Объёмные осцилляторы
df["vol_ma_20"] = df["volume"].rolling(20, min_periods=1).mean()
df["vol_ratio"] = df["volume"] / df["vol_ma_20"]

# 7.5 Циклическое кодирование времени
df["hour"] = df.index.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["weekday"] = df.index.dayofweek
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

# === 7.6. Rate of Change (ROC) ===
for n in [1, 5, 10]:
    df[f"ROC_{n}"] = (df["close"] - df["close"].shift(n)) / df["close"].shift(n)

# === 7.7. MACD + Signal + Histogram ===
ema_short = df["close"].ewm(span=12, adjust=False).mean()
ema_long  = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"]        = ema_short - ema_long
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

# === 7.8. Отношения цен ===
df["HL_ratio"] = df["high"] / df["low"]
df["OC_ratio"] = df["open"] / df["close"]

# === 7.9. Окна агрегации (среднее, медиана, квантиль) для close и volume ===
for w in [3, 6, 12]:
    df[f"close_ma_{w}"]     = df["close"].rolling(w).mean()
    df[f"close_med_{w}"]    = df["close"].rolling(w).median()
    df[f"close_q25_{w}"]    = df["close"].rolling(w).quantile(0.25)
    df[f"close_q75_{w}"]    = df["close"].rolling(w).quantile(0.75)

    df[f"vol_ma_{w}"]       = df["volume"].rolling(w).mean()
    df[f"vol_med_{w}"]      = df["volume"].rolling(w).median()
    df[f"vol_q25_{w}"]      = df["volume"].rolling(w).quantile(0.25)
    df[f"vol_q75_{w}"]      = df["volume"].rolling(w).quantile(0.75)

# === 7.10. Скользящая Z-нормализация всех числовых признаков (кроме таргета) ===
#    окно 500 баров (можно увеличить до 1000 при достаточных данных)
num_cols = df.select_dtypes(include=[np.number]).columns.drop("y")
for col in num_cols:
    roll_mean = df[col].rolling(500, min_periods=1).mean()
    roll_std  = df[col].rolling(500, min_periods=1).std()
    df[f"{col}_z"] = (df[col] - roll_mean) / roll_std

# 8. Окончательная очистка — убрать все NaN
df = df.dropna()

# === 9. Обучение через LGBMClassifier ===

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# 9.1. Формируем X и y
# убираем таргет и все производные от future_min_low и stop_price
drop_cols = [
    "y",
    "future_min_low", "future_min_low_z",
    "stop_price",     "stop_price_z",
]
X = df.drop(columns=drop_cols)
y = df["y"]

# 9.2. TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
aucs, pr_aucs = [], []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 9.3. Инициализируем scikit-like модель
    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=1000,
        random_state=42
    )

    # 9.4. Тренируем с ранней остановкой
    from lightgbm import early_stopping, log_evaluation

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=0)
        ]
    )

    # 9.5. Калибруем модель на валидационном сете
    from sklearn.calibration import CalibratedClassifierCV

    calibrator = CalibratedClassifierCV(
        estimator=clf,      # <-- заменили base_estimator на estimator
        method="isotonic",  # или "sigmoid"
        cv="prefit"
    )
    calibrator.fit(X_val, y_val)

    # 9.6. Предсказываем уже откалиброванную вероятность
    y_pred = calibrator.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, y_pred)
    prec, rec, _ = precision_recall_curve(y_val, y_pred)
    pr_auc = auc(rec, prec)

    print(f"Fold {fold+1}: Calibrated ROC AUC={roc:.4f}, PR AUC={pr_auc:.4f}")
    aucs.append(roc)
    pr_aucs.append(pr_auc)

print(f"\nMean ROC AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Mean PR  AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")

# 9.6. Топ-20 признаков по importance
imp = pd.Series(clf.feature_importances_, index=X.columns)
print("\nTop-20 features:\n", imp.sort_values(ascending=False).head(20))

