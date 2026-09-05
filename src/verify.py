"""
Kiểm chứng độc lập: đảm bảo không có data leakage và các con số khớp README.

Chạy: python src/verify.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from features import get_datasets, build_base_frame, add_horizon_features, BURN_IN

RNG = np.random.default_rng(0)
ok = lambda msg: print(f"  [OK] {msg}")


def check_leakage(horizon: int, label: str):
    """
    Kiểm tra kiểu brute-force: với một số dòng lấy ngẫu nhiên, tính lại từng
    feature bằng tay từ chuỗi Usage_kWh gốc, và xác nhận mọi giá trị được dùng
    đều nằm ở thời điểm <= t - horizon.
    """
    base = build_base_frame()
    usage = base[config.TARGET].values
    df, feats = add_horizon_features(base, horizon)

    idx_offset = BURN_IN  # df bị cắt BURN_IN dòng đầu so với base
    rows = RNG.choice(len(df), size=200, replace=False)

    for i in rows:
        g = i + idx_offset  # vị trí tương ứng trong chuỗi gốc
        assert df[config.TARGET].iloc[i] == usage[g]

        # lag_k phải bằng đúng usage[g - k], và k luôn >= horizon
        for name in feats:
            if name.startswith("lag_"):
                k = {"lag_15min": 1, "lag_30min": 2, "lag_1h": 4, "lag_2h": 8,
                     "lag_24h": 96, "lag_48h": 192, "lag_7d": 672}[name]
                assert k >= horizon, f"{name} vi phạm horizon {horizon}"
                assert np.isclose(df[name].iloc[i], usage[g - k])

        # rolling mean 24h: cửa sổ [g-horizon-95, g-horizon]
        w = config.STEPS_PER_DAY
        expected = usage[g - horizon - w + 1: g - horizon + 1].mean()
        assert np.isclose(df["roll_mean_24h"].iloc[i], expected)

        # same-slot: chỉ dùng các bội số của 1 ngày và đều >= horizon
        if "same_slot_mean_7d" in feats:
            ks = [k for k in range(w, config.STEPS_PER_WEEK + 1, w) if k >= horizon]
            expected = np.mean([usage[g - k] for k in ks])
            assert np.isclose(df["same_slot_mean_7d"].iloc[i], expected)

    ok(f"{label}: 200 dòng ngẫu nhiên — mọi feature chỉ dùng dữ liệu tại t-{horizon} trở về trước")

    lag_names = [n for n in feats if n.startswith("lag_")]
    ok(f"{label}: lag được giữ lại = {lag_names}")


def check_splits(horizon: int, label: str):
    train, val, test, _ = get_datasets(horizon)
    for name, part, lo, hi in [
        ("train", train, "2018-01-08", "2018-09-30"),
        ("validation", val, "2018-10-01", "2018-11-30"),
        ("test", test, "2018-12-01", "2018-12-31"),
    ]:
        d = pd.to_datetime(part[config.TIME_COL])
        assert d.min().strftime("%Y-%m-%d") == lo, (name, d.min())
        assert d.max().strftime("%Y-%m-%d") == hi, (name, d.max())

    assert (len(train), len(val), len(test)) == (25536, 5856, 2976), \
        f"Số dòng không khớp README: {len(train), len(val), len(test)}"

    assert pd.to_datetime(train[config.TIME_COL]).max() < pd.to_datetime(val[config.TIME_COL]).min()
    assert pd.to_datetime(val[config.TIME_COL]).max() < pd.to_datetime(test[config.TIME_COL]).min()
    ok(f"{label}: split 25,536 / 5,856 / 2,976 — đúng như README, không overlap")


def check_against_repo_processed():
    """Đối chiếu feature tự dựng lại với file processed sẵn có trong repo."""
    repo = pd.concat([
        pd.read_csv(config.PROCESSED_DIR / f, parse_dates=[config.TIME_COL])
        for f in ("train.csv", "validation.csv", "test.csv")
    ], ignore_index=True)

    mine, _ = add_horizon_features(build_base_frame(), horizon=1)
    merged = repo.merge(mine, on=config.TIME_COL, suffixes=("_repo", "_mine"))
    assert len(merged) == len(repo) == 34368

    for col in ["hour", "day_of_week", "is_weekend", "time_sin", "month_cos",
                "lag_15min", "lag_24h", "lag_7d", config.TARGET]:
        a, b = merged[f"{col}_repo"], merged[f"{col}_mine"]
        assert np.allclose(a, b), f"Lệch ở cột {col}"
    ok("Feature dựng lại khớp 100% với data/processed sẵn có (34,368 dòng)")


def check_baseline_identity():
    """Persistence baseline phải đúng bằng giá trị thực dịch đi H bước."""
    for label, h in config.HORIZONS.items():
        _, _, test, _ = get_datasets(h)
        col = {1: "lag_15min", 4: "lag_1h", 96: "lag_24h"}[h]
        base = build_base_frame()
        u = base.set_index(config.TIME_COL)[config.TARGET]
        t = pd.to_datetime(test[config.TIME_COL])
        expected = u.reindex(t - pd.Timedelta(minutes=15 * h)).values
        assert np.allclose(test[col].values, expected)
    ok("Baseline persistence = giá trị thực dịch đúng H bước (cả 3 horizon)")


if __name__ == "__main__":
    print("Kiểm chứng pipeline\n" + "-" * 60)
    for label, h in config.HORIZONS.items():
        check_leakage(h, label)
        check_splits(h, label)
    check_against_repo_processed()
    check_baseline_identity()
    print("-" * 60)
    print("Tất cả kiểm tra đều PASS.")
