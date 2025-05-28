import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, recall_score
from sklearn.feature_selection import SelectFromModel
import joblib
import optuna
import warnings

# === Параметры ===
INPUT_CSV      = "AES.csv"        # ← замените на своё имя файла
NAN_THRESHOLD  = 0.3               # максимально допустимая доля NaN в признаке
TEST_SIZE      = 0.2               # доля тестовой выборки
OPTUNA_TRIALS  = 50                # количество испытаний Optuna
OPTUNA_TIMEOUT = 600               # таймаут Optuna (сек)
HORIZON        = 15                # прогноз на 15 дней вперёд
MIN_RETURN     = 0.10              # целевой рост +10%

warnings.simplefilter(action='ignore', category=FutureWarning)

# Параметры для GPU в XGBoost
GPU_PARAMS = {
    'tree_method': 'gpu_hist',
    'predictor':    'gpu_predictor',
    'gpu_id':       0
}


def main():
    # --- Диагностика ---
    print("Рабочая директория:", os.getcwd())
    print("Список файлов:", os.listdir())

    # --- Загрузка данных ---
    df = pd.read_csv(INPUT_CSV)
    df = df.iloc[2:].copy()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
    df.set_index('Date', inplace=True)
    print("Исходная форма:", df.shape)

    # --- Индикаторы ---
    df.ta.strategy("all")
    print("После расчёта индикаторов:", df.shape)

    # --- Фильтрация по NaN ---
    keep = df.isna().mean() < NAN_THRESHOLD
    df = df.loc[:, keep].copy()
    print(f"После удаления фич с >{NAN_THRESHOLD*100:.0f}% NaN:", df.shape)
    df.dropna(inplace=True)
    print("После dropna:", df.shape)

    # --- Таргет ---
    for i in range(1, HORIZON+1):
        df[f'Close_fut_{i}'] = df['Close'].shift(-i)
    fut_cols = [f'Close_fut_{i}' for i in range(1, HORIZON+1)]
    df['future_max'] = df[fut_cols].max(axis=1)
    df['target'] = ((df['future_max'] / df['Close'] - 1) >= MIN_RETURN).astype(int)
    df.drop(columns=fut_cols + ['future_max'], inplace=True)
    df.dropna(subset=['target'], inplace=True)
    print("После формирования таргета:", df.shape)

    # --- Признаки и таргет ---
    X = df.drop(columns=['target'])
    y = df['target']
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=TEST_SIZE
    )

    # --- 1) Модель на всех фичах (GPU) ---
    model_all = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        **GPU_PARAMS,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_all.fit(X_train, y_train)
    print("\n=== Отчёт: модель на всех фичах (GPU) ===")
    print(classification_report(y_test, model_all.predict(X_test)))

    # --- 2) Optuna + xgb.train (GPU) ---
    def objective(trial):
        params = {
            'booster': trial.suggest_categorical('booster', ['gbtree','dart']),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10.0),
            'eval_metric': 'logloss',
            'objective': 'binary:logistic',
            **GPU_PARAMS
        }
        n_rounds = trial.suggest_int('n_estimators', 50, 1000)

        tscv = TimeSeriesSplit(n_splits=5)
        recalls = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            bst = xgb.train(
                params,
                dtr,
                num_boost_round=n_rounds,
                evals=[(dval, 'validation')],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            preds = bst.predict(dval, iteration_range=(0, bst.best_iteration+1))
            preds_label = (preds >= 0.5).astype(int)
            recalls.append(recall_score(y_val, preds_label))
        return sum(recalls) / len(recalls)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    print("\n=== Optuna результаты ===")
    print("Best recall:", study.best_value)
    print("Best params:", study.best_params)

    # --- Финальная модель Optuna (GPU) ---
    best = study.best_params
    model_opt = xgb.XGBClassifier(
        **{k:v for k,v in best.items() if k!='n_estimators'},
        n_estimators=best['n_estimators'],
        **GPU_PARAMS,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_opt.fit(X_train, y_train)
    print("\n=== Отчёт: модель Optuna (GPU) ===")
    print(classification_report(y_test, model_opt.predict(X_test)))
    joblib.dump(model_opt, "xgb_model_optuna_gpu.joblib")
    print("Сохранена: xgb_model_optuna_gpu.joblib")

    # --- 3) SelectFromModel (GPU) ---
    selector = SelectFromModel(model_all, threshold='median').fit(X_train, y_train)
    sel_feats = X_train.columns[selector.get_support()].tolist()
    print(f"\nSelectFromModel(median): оставлено {len(sel_feats)}/{X.shape[1]} признаков")
    X_train_med = selector.transform(X_train)
    X_test_med  = selector.transform(X_test)

    model_med = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        **GPU_PARAMS,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model_med.fit(X_train_med, y_train)
    print("\n=== Отчёт: после SelectFromModel(median) (GPU) ===")
    print(classification_report(y_test, model_med.predict(X_test_med)))
    joblib.dump(model_med, "xgb_model_selectfrommodel_median_gpu.joblib")
    print("Сохранена: xgb_model_selectfrommodel_median_gpu.joblib")

if __name__ == "__main__":
    main()
