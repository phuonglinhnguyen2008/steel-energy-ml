"""
Model C — Hybrid hai tầng (phân loại chạy/nghỉ, rồi hồi quy).

VÌ SAO CẦN MODEL NÀY?
---------------------
Phân bố Usage_kWh của dataset này là bimodal rất rõ: khoảng 60% quan sát nằm
dưới 10 kWh (nhà máy nghỉ hoặc chạy rất nhẹ), phần còn lại tập trung ở vùng
30-80 kWh (đang sản xuất). Vùng giữa 10-20 kWh gần như trống.

Một model hồi quy đơn lẻ bị buộc phải đi qua khoảng trống đó, nên sai nhiều nhất
ở đúng các điểm bật/tắt máy — cũng là chỗ sai đắt nhất về mặt vận hành.

CÁCH LÀM
--------
Tách bài toán thành hai tầng:

    Tầng 1 (phân loại) : nhà máy có đang sản xuất không?  ->  p = P(active)
    Tầng 2 (hồi quy)   : hai model riêng cho hai chế độ   ->  y_idle, y_active

    Dự báo cuối = p * y_active + (1 - p) * y_idle

Đây là kỳ vọng của một mô hình hỗn hợp (mixture), nên vẫn là dự báo điểm hợp lệ,
đồng thời mỗi regressor chỉ phải học một chế độ tải thay vì cả hai.

Ngưỡng phân chia lấy từ đáy phân bố bimodal, và CHỈ tính trên tập train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import config
import evaluate as ev
from features import get_datasets

# Ngưỡng mặc định (kWh) phân chia "nghỉ" và "đang sản xuất".
# Xem find_threshold(): đây là đáy của phân bố bimodal trên tập train.
DEFAULT_THRESHOLD = 15.0


def find_threshold(y_train: np.ndarray, lo: float = 8.0, hi: float = 28.0) -> float:
    """Tìm đáy phân bố (bin thưa nhất) trong khoảng [lo, hi], chỉ dùng tập train."""
    hist, edges = np.histogram(y_train, bins=120, range=(0, float(y_train.max())))
    centers = (edges[:-1] + edges[1:]) / 2
    mask = (centers >= lo) & (centers <= hi)
    return float(centers[mask][np.argmin(hist[mask])])


def fit_predict(Xtr, ytr, Xte, threshold: float, seed: int = config.RANDOM_STATE):
    """Huấn luyện hybrid trên (Xtr, ytr) rồi dự báo cho Xte."""
    active = (ytr > threshold).astype(int)

    common = dict(n_estimators=600, max_depth=6, learning_rate=0.05,
                  subsample=0.9, colsample_bytree=0.9,
                  random_state=seed, n_jobs=-1, tree_method="hist")

    # Tầng 1 — xác suất nhà máy đang sản xuất
    clf = XGBClassifier(eval_metric="logloss", **common)
    clf.fit(Xtr, active)
    p_active = clf.predict_proba(Xte)[:, 1]

    # Tầng 2 — hai regressor riêng, đều tối ưu MAE
    reg_kwargs = dict(objective="reg:absoluteerror", **common)

    reg_idle = XGBRegressor(**reg_kwargs)
    reg_idle.fit(Xtr[active == 0], ytr[active == 0])

    reg_active = XGBRegressor(**reg_kwargs)
    reg_active.fit(Xtr[active == 1], ytr[active == 1])

    y_idle = reg_idle.predict(Xte)
    y_active = reg_active.predict(Xte)

    return p_active * y_active + (1 - p_active) * y_idle, clf, p_active


def run(label: str, horizon: int, variant: str) -> dict:
    train, val, test, feats = get_datasets(horizon, include_load_type=(variant == "B"))

    Xtr, ytr = train[feats].values, train[config.TARGET].values
    Xva, yva = val[feats].values, val[config.TARGET].values
    Xte, yte = test[feats].values, test[config.TARGET].values

    threshold = find_threshold(ytr)
    scale = ev.seasonal_naive_scale(ytr)

    # Chọn ngưỡng trên validation (quanh đáy phân bố), không đụng test
    best = None
    for t in [threshold, threshold * 0.8, threshold * 1.2, 10.0, 20.0]:
        yhat_va, _, _ = fit_predict(Xtr, ytr, Xva, t)
        score = ev.mae(yva, yhat_va)
        if best is None or score < best[0]:
            best = (score, t)
    val_mae, threshold = best

    # Fit lại trên train + validation rồi dự báo test
    Xfull = np.vstack([Xtr, Xva])
    yfull = np.concatenate([ytr, yva])
    yhat, clf, p_active = fit_predict(Xfull, yfull, Xte, threshold)

    metrics = ev.compute_metrics(yte, yhat, scale)
    acc = float(((p_active > 0.5).astype(int) == (yte > threshold).astype(int)).mean())

    print(f"  {label}/{variant}: ngưỡng={threshold:.1f} kWh  val MAE={val_mae:.4f}  "
          f"test MAE={metrics['MAE']:.4f}  độ chính xác phân loại={acc:.3f}")

    return {"horizon": label, "variant": variant, "model": "Hybrid 2 tầng",
            "threshold_kWh": round(threshold, 2), "classifier_accuracy": round(acc, 4),
            **metrics, "_pred": yhat, "_dates": test[config.TIME_COL].values,
            "_actual": yte}


def main():
    print("Model C — Hybrid hai tầng\n" + "-" * 70)
    rows = []
    for label, h in config.HORIZONS.items():
        for variant in ("A", "B"):
            rows.append(run(label, h, variant))

    out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    out.to_csv(config.METRICS_DIR / "hybrid_results.csv", index=False)

    for r in rows:
        pd.DataFrame({config.TIME_COL: r["_dates"], "actual": r["_actual"],
                      "Hybrid": r["_pred"]}).to_csv(
            config.PREDICTIONS_DIR / f"hybrid_{r['horizon']}_{r['variant']}.csv", index=False)

    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nĐã lưu: {config.METRICS_DIR / 'hybrid_results.csv'}")


if __name__ == "__main__":
    main()
