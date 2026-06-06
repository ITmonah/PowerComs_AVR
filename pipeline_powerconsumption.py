import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

from statsforecast import StatsForecast
from statsforecast.models import (
    ARIMA,
    AutoARIMA,
    AutoETS,
    Theta,
    AutoTheta,
    SeasonalNaive,
)
from statsforecast.utils import ConformalIntervals

warnings.filterwarnings("ignore")

# Настройки
DATA_PATH = "powerconsumption.csv"
TARGET_COL = "PowerConsumption_Zone1"
FREQ = "10min"
SEASON_LENGTH = 144          # суточная сезонность для ряда 10 минут
H = 144                      # горизонт прогноза: 1 сутки
N_WINDOWS = 3                # backtesting окна
TRAIN_DAYS = 90              # ограничиваем историю для ускорения


# Загрузка и подготовка данных
df = pd.read_csv(DATA_PATH)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

# Оставляем только одну целевую переменную для пайплайна
series_df = df[["Datetime", TARGET_COL]].rename(
    columns={"Datetime": "ds", TARGET_COL: "y"}
)

# Приводим к регулярной сетке 10 минут
series_df = (
    series_df.set_index("ds")
    .resample(FREQ)
    .mean()
    .interpolate("time")
    .reset_index()
)

# Ограничиваем длину ряда для ускорения
series_df = series_df.tail(int(TRAIN_DAYS * 24 * 6)).copy()
series_df["unique_id"] = "zone_1"

# train / test split
train = series_df.iloc[:-H].copy()
test = series_df.iloc[-H:].copy()

print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Train range:", train["ds"].min(), "->", train["ds"].max())
print("Test range :", test["ds"].min(), "->", test["ds"].max())


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(out) * 100)


def metrics_table(y_true, pred_df, model_cols):
    rows = []
    y_true = np.asarray(y_true)

    for model in model_cols:
        pred = np.asarray(pred_df[model])
        rows.append({
            "model": model,
            "MAE": mean_absolute_error(y_true, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
            "MAPE_%": float(np.nanmean(
                np.abs((y_true - pred) / np.where(y_true == 0, np.nan, y_true))
            ) * 100),
            "sMAPE_%": smape(y_true, pred),
        })

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


stat_models = [
    ARIMA(order=(1, 1, 1)),
    AutoARIMA(season_length=SEASON_LENGTH),
    AutoETS(season_length=SEASON_LENGTH),
    Theta(season_length=SEASON_LENGTH, decomposition_type="additive"),
    AutoTheta(season_length=SEASON_LENGTH),
    SeasonalNaive(season_length=SEASON_LENGTH),
]

model_names = [
    "ARIMA",
    "AutoARIMA",
    "AutoETS",
    "Theta",
    "AutoTheta",
    "SeasonalNaive",
]

sf = StatsForecast(
    models=stat_models,
    freq=FREQ,
    n_jobs=-1
)

prediction_intervals = ConformalIntervals(
    h=H,
    n_windows=N_WINDOWS
)

t0 = time.perf_counter()
cv = sf.cross_validation(
    df=train,
    h=H,
    n_windows=N_WINDOWS,
    step_size=H,
    level=[95],
    prediction_intervals=prediction_intervals,
)
bt_time = time.perf_counter() - t0

print("Backtesting time (sec):", round(bt_time, 2))
print("CV shape:", cv.shape)

forecast_cols = [
    c for c in cv.columns
    if c not in ["unique_id", "ds", "cutoff", "y"]
    and not c.endswith("-lo-95")
    and not c.endswith("-hi-95")
]

cv_metrics = metrics_table(
    y_true=cv["y"],
    pred_df=cv,
    model_cols=forecast_cols
)

print("\nBacktesting metrics:")
print(cv_metrics)


best_model = cv_metrics.iloc[0]["model"]
print("\nBest model:", best_model)

best_model_map = {
    "ARIMA": ARIMA(order=(1, 1, 1)),
    "AutoARIMA": AutoARIMA(season_length=SEASON_LENGTH),
    "AutoETS": AutoETS(season_length=SEASON_LENGTH),
    "Theta": Theta(season_length=SEASON_LENGTH, decomposition_type="additive"),
    "AutoTheta": AutoTheta(season_length=SEASON_LENGTH),
    "SeasonalNaive": SeasonalNaive(season_length=SEASON_LENGTH),
}

chosen_sf = StatsForecast(
    models=[best_model_map[best_model]],
    freq=FREQ,
    n_jobs=-1
)

t0 = time.perf_counter()
forecast = chosen_sf.forecast(
    df=train,
    h=H,
    level=[95],
    prediction_intervals=prediction_intervals
)
fit_forecast_time = time.perf_counter() - t0

forecast_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
lo_col = f"{forecast_col}-lo-95"
hi_col = f"{forecast_col}-hi-95"

print("Forecast time (sec):", round(fit_forecast_time, 2))
print("\nForecast head:")
print(forecast.head())


test_metrics = {
    "model": best_model,
    "MAE": mean_absolute_error(test["y"], forecast[forecast_col]),
    "RMSE": float(np.sqrt(mean_squared_error(test["y"], forecast[forecast_col]))),
    "MAPE_%": float(np.nanmean(
        np.abs((test["y"].values - forecast[forecast_col].values) / np.where(test["y"].values == 0, np.nan, test["y"].values))
    ) * 100),
    "sMAPE_%": smape(test["y"].values, forecast[forecast_col].values),
}
test_metrics = pd.DataFrame([test_metrics])

print("\nTest metrics:")
print(test_metrics)

# Берем прогноз на обучающей части для анализа остатков
train_pred = chosen_sf.forecast(
    df=train,
    h=H,
    level=[95],
    fitted=True,
    prediction_intervals=prediction_intervals
)


residual_source = cv.copy()
residual_model_cols = [
    c for c in residual_source.columns
    if c not in ["unique_id", "ds", "cutoff", "y"]
    and not c.endswith("-lo-95")
    and not c.endswith("-hi-95")
]

resid_model_col = residual_model_cols[0]
resid = (residual_source["y"] - residual_source[resid_model_col]).dropna()

lb_test = acorr_ljungbox(resid, lags=[24], return_df=True)
dw_test = durbin_watson(resid)

print("\nResidual mean:", float(np.nanmean(resid)))
print("Residual std :", float(np.nanstd(resid)))
print("Durbin-Watson:", float(dw_test))
print("\nLjung-Box test:")
print(lb_test)

# График прогноза
fig, ax = plt.subplots(figsize=(14, 4))

tail_len = 7 * 24 * 6
train_plot = train[["ds", "y"]].tail(tail_len)
test_plot = pd.concat([train[["ds", "y"]].tail(1), test[["ds", "y"]]], ignore_index=True)

ax.plot(train_plot["ds"], train_plot["y"], color="blue", label="Train tail")
ax.plot(test_plot["ds"], test_plot["y"], label="Actual")
ax.plot(forecast["ds"], forecast[forecast_col], label=f"Forecast: {best_model}")

if lo_col in forecast.columns and hi_col in forecast.columns:
    ax.fill_between(forecast["ds"], forecast[lo_col], forecast[hi_col], alpha=0.2)

ax.set_title(f"Final forecast: {best_model}")
ax.set_xlabel("Datetime")
ax.set_ylabel("Power consumption")
ax.legend()
plt.tight_layout()
plt.show()

# Сохранение результатов
os.makedirs("results", exist_ok=True)

cv_metrics.to_csv("results/task4_backtesting_metrics.csv", index=False)
test_metrics.to_csv("results/task4_test_metrics.csv", index=False)
forecast.to_csv("results/task4_forecast.csv", index=False)

with open("results/task4_pipeline_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Best model: {best_model}\n")
    f.write(f"Backtesting time (sec): {bt_time:.2f}\n")
    f.write(f"Forecast time (sec): {fit_forecast_time:.2f}\n")
    f.write(f"Durbin-Watson: {float(dw_test):.4f}\n")
    f.write("\nBacktesting metrics:\n")
    f.write(cv_metrics.to_string(index=False))
    f.write("\n\nTest metrics:\n")
    f.write(test_metrics.to_string(index=False))