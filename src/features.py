"""
Tạo feature cho bài toán forecasting nhiều bước (multi-step ahead).

VÌ SAO PHẢI VIẾT LẠI PHẦN FEATURE?
----------------------------------
Bộ processed data trong repo được thiết kế cho bài toán "one-step ahead":
mỗi dòng dự báo Usage_kWh tại thời điểm t và được phép dùng lag_15min,
tức là mức tiêu thụ tại t - 15 phút.

Khi chuyển sang dự báo xa hơn (1 giờ hoặc 24 giờ), giả định đó không còn đúng.
Đứng ở thời điểm t để dự báo cho t + 24 giờ thì mình KHÔNG thể biết mức tiêu
thụ tại t + 23h45. Nếu vẫn đưa lag_15min vào model thì đó là data leakage:
metric sẽ rất đẹp nhưng không dùng được ngoài thực tế.

QUY TẮC DUY NHẤT ÁP DỤNG Ở ĐÂY
------------------------------
Để dự báo Usage_kWh(t) trước H bước (H bước = H * 15 phút), model chỉ được
dùng những thông tin đã biết tại thời điểm t - H:

  1. Feature lịch (hour, day_of_week, sin/cos...) của thời điểm t
     -> luôn hợp lệ, vì lịch là thứ mình biết trước bao lâu cũng được.

  2. Usage_kWh(t - k) với k >= H
     -> chỉ những lag đủ "cũ" mới thực sự nằm trong quá khứ tại lúc forecast.

  3. Thống kê trượt (rolling mean/std/max/min) tính trên cửa sổ KẾT THÚC
     tại t - H, không bao giờ chạm vào khoảng [t-H+1, t].

Toàn bộ phần còn lại (làm sạch, calendar, cyclical, mốc chia train/val/test)
tái sử dụng đúng pipeline đã có trong src/preprocessing.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from preprocessing import (
    load_data,
    clean_data,
    add_calendar_features,
    add_cyclical_features,
)

# ------------------------------------------------------------
# Các lag ứng viên, tính theo số bước 15 phút
# ------------------------------------------------------------
# Giữ nguyên bộ lag của bước preprocessing gốc, thêm lag_48h vì với horizon 24h
# thì các lag ngắn đều bị loại, model cần thêm thông tin lịch sử.
CANDIDATE_LAGS = {
    1: "lag_15min",
    2: "lag_30min",
    4: "lag_1h",
    8: "lag_2h",
    96: "lag_24h",
    192: "lag_48h",
    672: "lag_7d",
}

# Model B: Load_Type được one-hot thành 3 cột.
# CHỈ hợp lệ khi giả định nhà máy đã biết trước kế hoạch tải tại thời điểm
# forecast. Nếu giả định đó sai thì đây là leakage — xem mục 12 README gốc.
LOAD_TYPE_FEATURES = ["load_light", "load_medium", "load_maximum"]

CALENDAR_FEATURES = ["hour", "minute", "day_of_week", "month", "is_weekend"]
CYCLICAL_FEATURES = ["time_sin", "time_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos"]

# Số dòng đầu bị bỏ: bằng lag dài nhất (7 ngày = 672 dòng).
# Cố định cho mọi horizon để các horizon so sánh được với nhau trên cùng tập dòng.
BURN_IN = max(CANDIDATE_LAGS)


def build_base_frame() -> pd.DataFrame:
    """Đọc raw data và dựng lại các feature thời gian (dùng lại pipeline gốc)."""
    df = load_data()
    df = clean_data(df)
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    return df


def add_horizon_features(df: pd.DataFrame, horizon: int,
                         include_load_type: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Tạo lag + rolling feature hợp lệ cho một horizon cụ thể.

    horizon: số bước 15 phút dự báo trước (1 = 15 phút, 4 = 1 giờ, 96 = 24 giờ).
    include_load_type: True = Model B (Schedule-Aware), thêm Load_Type của
        thời điểm t. Chỉ dùng khi nhà máy biết trước kế hoạch tải.
    Trả về (dataframe, danh sách tên feature).
    """
    df = df.copy()
    usage = df[config.TARGET]

    feature_names: list[str] = CALENDAR_FEATURES + CYCLICAL_FEATURES

    # --- Lag features: chỉ giữ lag có độ trễ >= horizon ---
    for k, name in sorted(CANDIDATE_LAGS.items()):
        if k >= horizon:
            df[name] = usage.shift(k)
            feature_names.append(name)

    # --- Rolling features trên cửa sổ kết thúc tại t - horizon ---
    # shift(horizon) đưa gốc thời gian về đúng thời điểm forecast,
    # rolling() sau đó chỉ nhìn ngược về quá khứ.
    base = usage.shift(horizon)

    windows = {
        "1h": config.STEPS_PER_HOUR,     # 4 bước
        "24h": config.STEPS_PER_DAY,     # 96 bước
    }
    for label, w in windows.items():
        df[f"roll_mean_{label}"] = base.rolling(w).mean()
        df[f"roll_std_{label}"] = base.rolling(w).std()
        feature_names += [f"roll_mean_{label}", f"roll_std_{label}"]

    df["roll_max_24h"] = base.rolling(config.STEPS_PER_DAY).max()
    df["roll_min_24h"] = base.rolling(config.STEPS_PER_DAY).min()
    feature_names += ["roll_max_24h", "roll_min_24h"]

    # --- Same-slot profile: hồ sơ tiêu thụ tại ĐÚNG khung giờ này trong 7 ngày qua ---
    # Ví dụ dự báo cho 14:30 thứ Ba thì lấy mức tiêu thụ lúc 14:30 của 7 ngày
    # trước đó rồi tính trung bình / độ lệch. Đây là feature mạnh cho horizon dài,
    # khi các lag ngắn đều đã bị loại vì lý do leakage.
    slot_lags = [k for k in range(config.STEPS_PER_DAY, config.STEPS_PER_WEEK + 1,
                                  config.STEPS_PER_DAY) if k >= horizon]
    if slot_lags:
        slots = pd.concat([usage.shift(k) for k in slot_lags], axis=1)
        df["same_slot_mean_7d"] = slots.mean(axis=1)
        df["same_slot_median_7d"] = slots.median(axis=1)
        df["same_slot_std_7d"] = slots.std(axis=1)
        df["same_slot_max_7d"] = slots.max(axis=1)
        feature_names += ["same_slot_mean_7d", "same_slot_median_7d",
                          "same_slot_std_7d", "same_slot_max_7d"]

    # --- Model B: kế hoạch tải đã biết trước tại thời điểm forecast ---
    if include_load_type:
        df["load_light"] = (df["Load_Type"] == "Light_Load").astype(int)
        df["load_medium"] = (df["Load_Type"] == "Medium_Load").astype(int)
        df["load_maximum"] = (df["Load_Type"] == "Maximum_Load").astype(int)
        assert df[LOAD_TYPE_FEATURES].sum(axis=1).eq(1).all(), \
            "Load_Type có giá trị ngoài 3 nhãn đã biết."
        feature_names += LOAD_TYPE_FEATURES

    # Bỏ phần đầu chuỗi chưa đủ lịch sử (cố định 672 dòng cho mọi horizon)
    df = df.iloc[BURN_IN:].reset_index(drop=True)

    assert df[feature_names + [config.TARGET]].isnull().sum().sum() == 0, \
        "Còn giá trị thiếu sau khi tạo feature."

    return df, feature_names


def split_chronological(df: pd.DataFrame):
    """Chia theo thời gian, giữ đúng mốc của bước preprocessing gốc."""
    d = df[config.TIME_COL]
    train = df[d < config.VAL_START].reset_index(drop=True)
    val = df[(d >= config.VAL_START) & (d < config.TEST_START)].reset_index(drop=True)
    test = df[d >= config.TEST_START].reset_index(drop=True)

    assert train[config.TIME_COL].max() < val[config.TIME_COL].min()
    assert val[config.TIME_COL].max() < test[config.TIME_COL].min()
    assert len(train) + len(val) + len(test) == len(df)

    return train, val, test


def get_datasets(horizon: int, include_load_type: bool = False):
    """Trả về (train, val, test, feature_names) đã sẵn sàng để train."""
    base = build_base_frame()
    df, feature_names = add_horizon_features(base, horizon, include_load_type)
    train, val, test = split_chronological(df)
    return train, val, test, feature_names


if __name__ == "__main__":
    for label, h in config.HORIZONS.items():
        for variant, use_lt in [("A", False), ("B", True)]:
            tr, va, te, feats = get_datasets(h, use_lt)
            print(f"\nModel {variant} — horizon {label} (H={h}): {len(feats)} features")
            print(f"  train={len(tr)}  val={len(va)}  test={len(te)}")
            print(f"  features: {feats}")
