import os
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, recall_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import joblib
import optuna
import warnings

# === Параметры ===
INPUT_CSV = "AES.csv"       # ← замените на своё имя файла
NAN_THRESHOLD = 0.3                # максимум доли NaN в признаке
IMPORTANCE_THRESHOLD = 0.01        # минимальная относительная важность (1%)
TEST_SIZE = 0.2                    # доля тестовой выборки
OPTUNA_TRIALS = 50                 # число испытаний для Optuna
OPTUNA_TIMEOUT = 600               # таймаут в секундах (None для без таймаута)

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    # --- Диагностика путей ---
    print("Рабочая директория:", os.getcwd())
    print("Файлы в ней:", os.listdir())

    # --- Загрузка и подготовка данных ---
    df = pd.read_csv(INPUT_CSV)
    df = df.iloc[2:].copy()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
    df.set_index('Date', inplace=True)
    print("Исходная форма:", df.shape)

    # --- Расчёт всех индикаторов ---
    df.ta.strategy("all")
    print("После расчёта индикаторов:", df.shape)

    # --- Отбор по пропускам ---
    nan_ratio = df.isna().mean()
    keep_cols = nan_ratio[nan_ratio < NAN_THRESHOLD].index
    df = df[keep_cols].copy()
    print(f"После удаления признаков с >{NAN_THRESHOLD*100:.0f}% NaN:", df.shape)

    # --- Полное удаление NaN ---
    df.dropna(inplace=True)
    print("После dropna:", df.shape)

    # --- Цель: рост Close на следующий день ---
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    # --- Формируем X и y ---
    features = [c for c in df.columns if c != 'target']
    X = df[features]
    y = df['target']
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --- Разбиение на train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=TEST_SIZE
    )

    # --- 1) Базовая модель на всех признаках ---
    model_all = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model_all.fit(X_train, y_train)
    print("\n=== Отчёт: модель на всех признаках ===")
    y_pred_all = model_all.predict(X_test)
    print(classification_report(y_test, y_pred_all))

    # --- 2) Optuna для поиска гиперпараметров ---
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        tscv = TimeSeriesSplit(n_splits=5)
        recalls = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            recalls.append(recall_score(y_val, preds))
        return sum(recalls) / len(recalls)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    print("\n=== Optuna результаты ===")
    print("Best recall:", study.best_value)
    print("Best params:", study.best_params)

    # Обучаем финальную модель с лучшими параметрами
    best_params = study.best_params
    best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss'})
    model_opt = xgb.XGBClassifier(**best_params)
    model_opt.fit(X_train, y_train)
    print("\n=== Отчёт: модель Optuna ===")
    y_pred_opt = model_opt.predict(X_test)
    print(classification_report(y_test, y_pred_opt))
    joblib.dump(model_opt, "xgb_model_optuna.joblib")
    print("Модель Optuna сохранена: xgb_model_optuna.joblib")

    # --- 3) Отбор через SelectFromModel (медиана) ---
    selector = SelectFromModel(model_all, threshold='median').fit(X_train, y_train)
    sel_feats_med = X_train.columns[selector.get_support()].tolist()
    print(f"\nSelectFromModel(median): оставлено {len(sel_feats_med)}/{X.shape[1]} признаков")
    X_train_med = selector.transform(X_train)
    X_test_med  = selector.transform(X_test)
    model_med = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model_med.fit(X_train_med, y_train)
    print("\n=== Отчёт: после SelectFromModel(median) ===")
    print(classification_report(y_test, model_med.predict(X_test_med)))
    joblib.dump(model_med, "xgb_model_selectfrommodel_median.joblib")
    print("Модель SelectFromModel сохр.: xgb_model_selectfrommodel_median.joblib")

    # --- Визуализация важности оптимизированных моделей ---
    print("\n=== Важность модели Optuna ===")
    xgb.plot_importance(model_opt, max_num_features=20)
    plt.tight_layout()
    plt.show()

    print("\n=== Важность после SelectFromModel(median) ===")
    xgb.plot_importance(model_med, max_num_features=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
