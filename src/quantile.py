"""
Prediction interval bằng quantile regression.

VÌ SAO CẦN?
-----------
Dự báo điểm ở horizon 24 giờ có MAE khoảng 7 kWh. Với một con số đơn lẻ, người
vận hành không biết nên chuẩn bị cho kịch bản nào. Khoảng dự báo P10-P90 trả lời
được câu hỏi thực tế hơn: "ngày mai lúc 14h nhà máy tiêu thụ trong khoảng nào?"

XGBoost hỗ trợ objective "reg:quantileerror", tối ưu trực tiếp pinball loss —
đúng hàm mất mát của bài toán dự báo phân vị.

HIỆU CHỈNH BẰNG CONFORMAL PREDICTION
------------------------------------
Quantile regression thường cho khoảng HẸP HƠN mức đáng ra phải có: model tự tin
quá mức, coverage thực tế thấp hơn 80% danh nghĩa. Đây là hiện tượng đã biết,
không phải lỗi cài đặt.

Cách sửa dùng ở đây là Conformalized Quantile Regression (CQR):

    1. Fit các phân vị CHỈ trên tập train.
    2. Trên tập validation, tính điểm lệch của từng quan sát:
           E_i = max(P10_i - y_i,  y_i - P90_i)
       E_i > 0 nghĩa là giá trị thực nằm ngoài khoảng, và lệch bao nhiêu.
    3. Lấy phân vị 80% của các E_i làm lượng nới rộng, áp cho tập test:
           [P10 - E*,  P90 + E*]

Cách này có bảo đảm lý thuyết: coverage trên test sẽ xấp xỉ 80% mà không cần giả
định gì về phân bố sai số. Tập validation đóng vai trò calibration set, và test
vẫn không bị đụng tới.

CÁCH ĐỌC KẾT QUẢ
----------------
    coverage_%   : bao nhiêu phần trăm giá trị thực rơi vào khoảng P10-P90.
                   (bản thô, chưa hiệu chỉnh)
                   Lý tưởng là 80%. Thấp hơn nhiều = khoảng quá hẹp, model
                   tự tin quá mức. Cao hơn nhiều = khoảng quá rộng, ít hữu ích.
    avg_width    : độ rộng trung bình của khoảng (kWh). Càng hẹp càng tốt,
                   với điều kiện coverage vẫn đạt.
    pinball_loss : hàm mất mát chuẩn cho dự báo phân vị, càng nhỏ càng tốt.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import config
import evaluate as ev
from features import get_datasets

QUANTILES = [0.1, 0.5, 0.9]


def pinball_loss(y, yhat, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def fit_quantiles(Xtr, ytr, Xte):
    preds = {}
    for q in QUANTILES:
        m = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            n_estimators=700, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=config.RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )
        m.fit(Xtr, ytr)
        preds[q] = m.predict(Xte)

    # Ép thứ tự phân vị không được cắt nhau (monotonic)
    stacked = np.sort(np.vstack([preds[q] for q in QUANTILES]), axis=0)
    return {q: stacked[i] for i, q in enumerate(QUANTILES)}


def conformal_delta(y_cal, lo_cal, hi_cal, alpha: float = 0.2) -> float:
    """Lượng nới rộng khoảng, lấy từ phân vị (1-alpha) của điểm lệch trên validation."""
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # hiệu chỉnh mẫu hữu hạn
    return float(np.quantile(scores, level))


def run(label: str, horizon: int, variant: str):
    train, val, test, feats = get_datasets(horizon, include_load_type=(variant == "B"))

    Xtr, ytr = train[feats].values, train[config.TARGET].values
    Xva, yva = val[feats].values, val[config.TARGET].values
    Xte, yte = test[feats].values, test[config.TARGET].values

    # Fit trên train, dự báo cho cả validation (để hiệu chỉnh) và test
    preds_va = fit_quantiles(Xtr, ytr, Xva)
    preds_te = fit_quantiles(Xtr, ytr, Xte)

    lo, mid, hi = preds_te[0.1], preds_te[0.5], preds_te[0.9]

    # --- Hiệu chỉnh conformal, dùng validation làm calibration set ---
    delta = conformal_delta(yva, preds_va[0.1], preds_va[0.9], alpha=0.2)
    lo_c = np.maximum(lo - delta, 0.0)   # điện năng không âm
    hi_c = hi + delta

    cov_raw = float(((yte >= lo) & (yte <= hi)).mean() * 100)
    cov_cal = float(((yte >= lo_c) & (yte <= hi_c)).mean() * 100)
    scale = ev.seasonal_naive_scale(ytr)

    row = {
        "horizon": label, "variant": variant,
        "coverage_thô_%": round(cov_raw, 2),
        "coverage_hiệu_chỉnh_%": round(cov_cal, 2),
        "độ_rộng_thô": round(float(np.mean(hi - lo)), 3),
        "độ_rộng_hiệu_chỉnh": round(float(np.mean(hi_c - lo_c)), 3),
        "nới_rộng_kWh": round(delta, 3),
        "pinball_P10": round(pinball_loss(yte, lo, 0.1), 4),
        "pinball_P50": round(pinball_loss(yte, mid, 0.5), 4),
        "pinball_P90": round(pinball_loss(yte, hi, 0.9), 4),
        "MAE_P50": round(ev.mae(yte, mid), 4),
        "MASE_P50": round(ev.mae(yte, mid) / scale, 4),
    }
    print(f"  {label}/{variant}: coverage {cov_raw:.1f}% -> {cov_cal:.1f}% sau hiệu chỉnh"
          f"  (nới {delta:.2f} kWh)  MAE(P50)={row['MAE_P50']:.3f}")

    pd.DataFrame({config.TIME_COL: test[config.TIME_COL], "actual": yte,
                  "P10": lo, "P50": mid, "P90": hi,
                  "P10_hiệu_chỉnh": lo_c, "P90_hiệu_chỉnh": hi_c}).to_csv(
        config.PREDICTIONS_DIR / f"quantile_{label}_{variant}.csv", index=False)

    return row, (pd.to_datetime(test[config.TIME_COL]).values, yte, lo_c, mid, hi_c)


def plot_interval(dates, y, lo, mid, hi, label, variant, path, days=5):
    n = days * config.STEPS_PER_DAY
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.fill_between(dates[:n], lo[:n], hi[:n], color="#4C78A8", alpha=0.25,
                    label="Khoảng P10-P90 (đã hiệu chỉnh)")
    ax.plot(dates[:n], mid[:n], color="#4C78A8", lw=1.4, label="Dự báo P50")
    ax.plot(dates[:n], y[:n], color="#333333", lw=1.5, label="Thực tế", zorder=5)
    ax.set_ylabel("Usage_kWh")
    ax.set_title(f"Khoảng dự báo P10-P90 đã hiệu chỉnh conformal — {days} ngày đầu "
                 f"tập test (horizon {label}, Model {variant})")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Prediction interval — quantile regression\n" + "-" * 70)
    rows = []
    for label, h in config.HORIZONS.items():
        for variant in ("A", "B"):
            row, plot_data = run(label, h, variant)
            rows.append(row)
            if label == "24h":
                plot_interval(*plot_data, label, variant,
                              config.FIGURES_DIR / f"interval_{label}_{variant}.png")

    out = pd.DataFrame(rows)
    out.to_csv(config.METRICS_DIR / "quantile_results.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nĐã lưu: {config.METRICS_DIR / 'quantile_results.csv'}")


if __name__ == "__main__":
    main()
