import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# === 1. Загрузка и валидация данных ===
df = pd.read_csv("AES_hourly.csv", parse_dates=["datetime"], index_col="datetime")

# === 2. Таймстемпы и пропуски ===
idx_full = pd.date_range(df.index.min(), df.index.max(), freq="H")
df = df.reindex(idx_full)
df[["open","high","low","close"]] = df[["open","high","low","close"]].interpolate()
df["volume"] = df["volume"].fillna(0)

# === 3. Фильтрация выбросов по скользящему Z-score ===
roll_mean = df["close"].rolling(100, min_periods=1).mean()
roll_std  = df["close"].rolling(100, min_periods=1).std()
z_score   = (df["close"] - roll_mean) / roll_std
df = df[(z_score.abs() < 5) | z_score.isna()]

# === 4. ATR и стоп-лосс ===
tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - df["close"].shift()).abs(),
    (df["low"]  - df["close"].shift()).abs()
], axis=1).max(axis=1)
df["ATR"] = tr.rolling(14, min_periods=14).mean()
k = 1.2
df["stop_price"] = df["close"] - k * df["ATR"]

# === 5. Разметка таргета y ===
H = 5
df["future_min_low"] = df["low"].shift(-1).rolling(H, min_periods=H).min()
df["y"] = (df["future_min_low"] > df["stop_price"]).astype(int)

# === 6. Инженерия признаков ===
# 6.1 SMA, EMA, STD
for w in [5,10,20]:
    df[f"SMA_{w}"] = df["close"].rolling(w, min_periods=1).mean()
    df[f"EMA_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    df[f"STD_{w}"] = df["close"].rolling(w, min_periods=1).std()

# 6.2 Лаговые доходности и моментум
df["return_1h"]   = df["close"].pct_change(1)
df["return_5h"]   = df["close"].pct_change(5)
df["momentum_5h"] = df["close"] - df["close"].shift(5)

# 6.3 RSI
delta    = df["close"].diff()
up       = delta.clip(lower=0)
down     = -delta.clip(upper=0)
roll_up  = up.rolling(14, min_periods=14).mean()
roll_down= down.rolling(14, min_periods=14).mean()
rs       = roll_up / roll_down
df["RSI_14"] = 100 - (100 / (1 + rs))

# 6.4 Объёмные осцилляторы
df["vol_ma_20"] = df["volume"].rolling(20, min_periods=1).mean()
df["vol_ratio"] = df["volume"] / df["vol_ma_20"]

# 6.5 Временные признаки
df["hour"]      = df.index.hour
df["hour_sin"]  = np.sin(2*np.pi * df["hour"]/24)
df["hour_cos"]  = np.cos(2*np.pi * df["hour"]/24)
df["weekday"]   = df.index.dayofweek
df["weekday_sin"]= np.sin(2*np.pi * df["weekday"]/7)
df["weekday_cos"]= np.cos(2*np.pi * df["weekday"]/7)

# 6.6 ROC
for n in [1,5,10]:
    df[f"ROC_{n}"] = (df["close"] - df["close"].shift(n)) / df["close"].shift(n)

# 6.7 MACD
ema_s = df["close"].ewm(span=12,adjust=False).mean()
ema_l = df["close"].ewm(span=26,adjust=False).mean()
df["MACD"]        = ema_s - ema_l
df["MACD_signal"] = df["MACD"].ewm(span=9,adjust=False).mean()
df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

# 6.8 Отношения цен
df["HL_ratio"] = df["high"] / df["low"]
df["OC_ratio"] = df["open"] / df["close"]

# 6.9 Окна агрегации
for w in [3,6,12]:
    df[f"close_ma_{w}"]  = df["close"].rolling(w).mean()
    df[f"close_med_{w}"] = df["close"].rolling(w).median()
    df[f"close_q25_{w}"] = df["close"].rolling(w).quantile(0.25)
    df[f"close_q75_{w}"] = df["close"].rolling(w).quantile(0.75)
    df[f"vol_ma_{w}"]    = df["volume"].rolling(w).mean()
    df[f"vol_med_{w}"]   = df["volume"].rolling(w).median()
    df[f"vol_q25_{w}"]   = df["volume"].rolling(w).quantile(0.25)
    df[f"vol_q75_{w}"]   = df["volume"].rolling(w).quantile(0.75)

# 6.10 Скользящая Z-нормализация (окно 500)
num_cols = df.select_dtypes(include=[np.number]).columns.drop("y")
for col in num_cols:
    m = df[col].rolling(500, min_periods=1).mean()
    s = df[col].rolling(500, min_periods=1).std()
    df[f"{col}_z"] = (df[col] - m) / s

df = df.dropna()

# === 7. Кросс-валидация + калибровка + threshold tuning ===
drop_cols = ["y","future_min_low","future_min_low_z","stop_price","stop_price_z"]
X = df.drop(columns=drop_cols)
y = df["y"]

tscv = TimeSeriesSplit(n_splits=5)
best_thresholds = []
for fold,(tr_idx,va_idx) in enumerate(tscv.split(X)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    clf = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        learning_rate=0.05, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        n_estimators=1000, random_state=42
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va,y_va)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)]
    )

    cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
    cal.fit(X_va, y_va)

    y_proba = cal.predict_proba(X_va)[:,1]
    p, r, thr = precision_recall_curve(y_va, y_proba)
    f1 = 2*p*r/(p+r+1e-8)
    idx = f1[:-1].argmax()
    bt = thr[idx]
    best_thresholds.append(bt)

    roc = roc_auc_score(y_va, y_proba)
    pr_auc = auc(r, p)
    print(f"Fold {fold+1}: ROC={roc:.4f}, PR AUC={pr_auc:.4f}, best p_min={bt:.4f}")

final_threshold = np.mean(best_thresholds)
print(f"\nChosen p_min = {final_threshold:.4f}")

# === 8. Финальное обучение и калибровка на всём датасете ===
final_clf = lgb.LGBMClassifier(
    objective="binary", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=31,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    n_estimators=clf.best_iteration_, random_state=42
)
final_clf.fit(X, y)
final_cal = CalibratedClassifierCV(estimator=final_clf, method="isotonic", cv="prefit")
final_cal.fit(X, y)

df["calibrated_prob"] = final_cal.predict_proba(X)[:,1]

# === 9. Бэктест стратегии ===
df["signal"] = (df["calibrated_prob"] > final_threshold).astype(int)
df["entry"]  = df["signal"].diff().fillna(0)==1
df["pnl"]    = 0.0

for t in df.index[df["entry"]]:
    ep = df.at[t,"close"]
    sp = df.at[t,"stop_price"]
    period = df.loc[t:].iloc[1:H+1]
    if (period["low"] <= sp).any():
        ex_idx = period["low"].le(sp).idxmax()
        ex_pr  = sp
    else:
        ex_idx = period.index[-1]
        ex_pr  = df.at[ex_idx,"close"]
    pnl = ex_pr/ep - 1
    df.at[ex_idx,"pnl"] += pnl

df["equity"] = 1.0 + df["pnl"].cumsum()
df["ret"]    = df["equity"].pct_change().fillna(0)
sharpe = df["ret"].mean()/df["ret"].std()*np.sqrt(252*6.5)
dd = (df["equity"] - df["equity"].cummax())/df["equity"].cummax()

print(f"Final equity: {df['equity'].iloc[-1]:.4f}")
print(f"Sharpe (annualized): {sharpe:.2f}")
print(f"Max drawdown: {dd.min():.2%}")

plt.figure(figsize=(12,4))
plt.plot(df["equity"], label="Equity")
plt.fill_between(dd.index, dd, 0, color="red", alpha=0.3, label="Drawdown")
plt.legend(); plt.title("Backtest Results")
plt.show()
