"""
Pipeline train và đánh giá model dự báo điện năng tiêu thụ.

Quy trình:
    train      -> fit model
    validation -> chọn hyperparameter (không bao giờ dùng test ở bước này)
    test       -> chỉ đánh giá một lần duy nhất ở cuối

Chạy:
    python src/train.py              # chạy cả 3 horizon: 15min, 1h, 24h
    python src/train.py 1h 24h       # chỉ chạy horizon được chỉ định
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import config
import evaluate as ev
from features import get_datasets, CANDIDATE_LAGS

# Sau khi đã chọn được hyperparameter tốt nhất trên validation, fit lại model
# trên train + validation rồi mới dự báo test. Làm vậy model được học thêm
# giai đoạn tháng 10-11, gần với tháng 12 (test) hơn.
# Đặt False nếu muốn bám đúng nguyên văn README (chỉ fit trên train).
REFIT_ON_TRAIN_VAL = True


# ============================================================
# Lưới hyperparameter (nhỏ, đủ để so sánh mà không tốn thời gian)
# ============================================================

RF_GRID = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
]

# Lưu ý quan trọng: mặc định XGBoost/RandomForest tối ưu squared error, tức là
# học để dự báo GIÁ TRỊ TRUNG BÌNH. Với dữ liệu lệch mạnh như dataset này
# (mean 27.4 nhưng median chỉ 4.6 kWh) thì mục tiêu đó kéo dự báo lên cao ở
# những khoảng nhà máy chạy nhẹ và làm MAE xấu đi. Vì vậy lưới dưới đây có cả
# cấu hình dùng objective "reg:absoluteerror" — tối ưu trực tiếp MAE.
XGB_GRID = [
    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 900, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 900, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
     "objective": "reg:absoluteerror"},
    {"n_estimators": 900, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9,
     "objective": "reg:absoluteerror"},
]


def make_rf(params):
    return RandomForestRegressor(
        random_state=config.RANDOM_STATE, n_jobs=-1, **params
    )


def make_xgb(params):
    params = {"objective": "reg:squarederror", **params}
    return XGBRegressor(
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def tune(factory, grid, Xtr, ytr, Xva, yva, name):
    """Fit từng cấu hình trên train, chọn cấu hình có MAE thấp nhất trên validation."""
    best = None
    for params in grid:
        model = factory(params)
        model.fit(Xtr, ytr)
        score = ev.mae(yva, model.predict(Xva))
        print(f"    {name} {params} -> val MAE = {score:.4f}")
        if best is None or score < best[0]:
            best = (score, params, model)
    print(f"    -> chọn {name}: {best[1]} (val MAE = {best[0]:.4f})")
    return best[1], best[2]


def run_horizon(label: str, horizon: int, variant: str = "A") -> pd.DataFrame:
    """
    variant "A" = Historical Forecast (calendar + cyclical + lag).
    variant "B" = Schedule-Aware: Model A + Load_Type, với giả định nhà máy
                  đã biết trước kế hoạch tải tại thời điểm forecast.
    """
    use_load_type = variant == "B"
    tag = f"{label}_{variant}"

    print("\n" + "=" * 70)
    print(f"MODEL {variant} — HORIZON {label}  (H = {horizon} bước x 15 phút)")
    print("=" * 70)

    train, val, test, feats = get_datasets(horizon, use_load_type)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  features={len(feats)}")

    Xtr, ytr = train[feats].values, train[config.TARGET].values
    Xva, yva = val[feats].values, val[config.TARGET].values
    Xte, yte = test[feats].values, test[config.TARGET].values

    # Mẫu số MASE luôn tính trên tập train, so với seasonal naive theo ngày
    scale = ev.seasonal_naive_scale(ytr, season=config.STEPS_PER_DAY)

    rows, preds = [], {}

    # ---------- Baseline: không cần train ----------
    # Persistence = lấy luôn quan sát mới nhất mà mình còn được phép biết,
    # tức là mức tiêu thụ tại t - H.
    fresh_k = min(k for k in CANDIDATE_LAGS if k >= horizon)
    baselines = {"Naive (persistence)": CANDIDATE_LAGS[fresh_k]}
    if CANDIDATE_LAGS[fresh_k] != "lag_24h":
        baselines["Seasonal naive (24h)"] = "lag_24h"
    baselines["Seasonal naive (7 ngày)"] = "lag_7d"

    for name, col in baselines.items():
        yhat = test[col].values
        rows.append({"model": name, **ev.compute_metrics(yte, yhat, scale)})
        preds[name] = yhat
        print(f"  {name:<26} MAE = {rows[-1]['MAE']:.4f}")

    # ---------- Các model học máy ----------
    fitted = {}

    print("  Linear Regression...")
    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(Xtr, ytr)
    print(f"    val MAE = {ev.mae(yva, lin.predict(Xva)):.4f}")
    fitted["Linear Regression"] = (lin, None)

    print("  Random Forest...")
    t0 = time.time()
    rf_params, rf = tune(make_rf, RF_GRID, Xtr, ytr, Xva, yva, "RF")
    print(f"    ({time.time() - t0:.1f}s)")
    fitted["Random Forest"] = (rf, rf_params)

    print("  XGBoost...")
    t0 = time.time()
    xgb_params, xgb = tune(make_xgb, XGB_GRID, Xtr, ytr, Xva, yva, "XGB")
    print(f"    ({time.time() - t0:.1f}s)")
    fitted["XGBoost"] = (xgb, xgb_params)

    # ---------- Refit trên train + validation rồi đánh giá test ----------
    Xfull = np.vstack([Xtr, Xva])
    yfull = np.concatenate([ytr, yva])

    best_params = {}
    for name, (model, params) in fitted.items():
        if REFIT_ON_TRAIN_VAL:
            if name == "Linear Regression":
                final = make_pipeline(StandardScaler(), LinearRegression())
            elif name == "Random Forest":
                final = make_rf(params)
            else:
                final = make_xgb(params)
            final.fit(Xfull, yfull)
        else:
            final = model

        yhat = final.predict(Xte)
        rows.append({"model": name, **ev.compute_metrics(yte, yhat, scale)})
        preds[name] = yhat
        best_params[name] = params
        print(f"  {name:<26} test MAE = {rows[-1]['MAE']:.4f}")
        fitted[name] = (final, params)

    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    metrics.insert(0, "horizon", label)
    metrics.insert(1, "variant", variant)

    # ---------- Lưu kết quả ----------
    metrics.to_csv(config.METRICS_DIR / f"metrics_{tag}.csv", index=False)

    pred_df = pd.DataFrame({config.TIME_COL: test[config.TIME_COL], "actual": yte})
    for name, yhat in preds.items():
        pred_df[name] = yhat
    pred_df.to_csv(config.PREDICTIONS_DIR / f"predictions_{tag}.csv", index=False)

    with open(config.METRICS_DIR / f"best_params_{tag}.json", "w") as f:
        json.dump({k: v for k, v in best_params.items() if v}, f, indent=2)

    # ---------- Biểu đồ ----------
    ev.plot_model_comparison(metrics, label, config.FIGURES_DIR / f"comparison_{tag}.png")
    ev.plot_predictions(
        pd.to_datetime(test[config.TIME_COL]).values, yte,
        {k: preds[k] for k in ["Naive (persistence)", "Random Forest", "XGBoost"] if k in preds},
        label, config.FIGURES_DIR / f"forecast_{tag}.png",
    )

    best_name = metrics.iloc[0]["model"]
    ev.plot_residuals(yte, preds[best_name], best_name, label,
                      config.FIGURES_DIR / f"residuals_{tag}.png")

    xgb_model = fitted["XGBoost"][0]
    ev.plot_feature_importance(feats, xgb_model.feature_importances_,
                               "XGBoost", config.FIGURES_DIR / f"importance_{tag}.png")

    print(f"\n  Model tốt nhất: {best_name}")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return metrics


def compare_a_vs_b(summary: pd.DataFrame) -> pd.DataFrame:
    """Trả lời câu hỏi của README: biết trước kế hoạch tải giúp cải thiện bao nhiêu?"""
    ml = ["Linear Regression", "Random Forest", "XGBoost"]
    d = summary[summary["model"].isin(ml)]

    pivot = d.pivot_table(index=["horizon", "model"], columns="variant", values="MAE")
    if not {"A", "B"} <= set(pivot.columns):
        return pd.DataFrame()

    pivot = pivot.rename(columns={"A": "MAE_A", "B": "MAE_B"}).reset_index()
    pivot["cải_thiện_kWh"] = pivot["MAE_A"] - pivot["MAE_B"]
    pivot["cải_thiện_%"] = 100 * pivot["cải_thiện_kWh"] / pivot["MAE_A"]

    order = {lb: i for i, lb in enumerate(config.HORIZONS)}
    return pivot.sort_values(["horizon", "model"], key=lambda c: c.map(order) if c.name == "horizon" else c)


def main():
    labels = sys.argv[1:] or list(config.HORIZONS)
    all_metrics = [
        run_horizon(lb, config.HORIZONS[lb], variant)
        for lb in labels
        for variant in ("A", "B")
    ]

    summary = pd.concat(all_metrics, ignore_index=True)
    summary.to_csv(config.METRICS_DIR / "summary_all_horizons.csv", index=False)

    print("\n" + "=" * 70)
    print("TỔNG HỢP TẤT CẢ HORIZON (tập test)")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ab = compare_a_vs_b(summary)
    if not ab.empty:
        ab.to_csv(config.METRICS_DIR / "model_a_vs_b.csv", index=False)
        print("\n" + "=" * 70)
        print("MODEL A vs MODEL B — biết trước kế hoạch tải giúp được bao nhiêu? (MAE, kWh)")
        print("=" * 70)
        print(ab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nKết quả đã lưu vào: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
