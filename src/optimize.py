"""
Vế "Tối ưu hóa" của đề tài: dịch chuyển tải theo khung giá điện.

BỐI CẢNH
--------
Tên đề tài là "Dự báo VÀ tối ưu hóa điện năng tiêu thụ", nhưng toàn bộ phần trước
mới chỉ làm vế dự báo. Module này là bản demo cho vế còn lại, đồng thời trả lời
một câu hỏi mà bảng MAE không trả lời được:

    Dự báo chính xác hơn thì tiết kiệm được bao nhiêu TIỀN?

MÔ HÌNH
-------
Giá điện sản xuất của Việt Nam chia ba khung giờ (Thông tư 16/2014/TT-BCT):

    Cao điểm    : T2-T7  09:30-11:30 và 17:00-20:00   (Chủ nhật không có)
    Thấp điểm   : mọi ngày  22:00-04:00
    Bình thường : phần còn lại

Giả định vận hành: một tỉ lệ ALPHA của điện năng trong ngày là "dịch chuyển
được" (nấu luyện theo mẻ, sấy, nạp lò...), phần còn lại là tải nền bắt buộc.

Bài toán quy hoạch tuyến tính cho mỗi ngày:

    min  sum_t  giá_t * x_t
    với  sum_t x_t   = tổng điện năng dịch chuyển được trong ngày
         0 <= x_t    <= trần công suất mỗi khung 15 phút

Ba kịch bản được so sánh:

    1. Không tối ưu   : nhà máy chạy đúng như thực tế
    2. Tối ưu lý tưởng: biết trước chính xác nhu cầu ngày mai (cận trên)
    3. Tối ưu theo dự báo: dùng dự báo 24h của model, phần lệch so với thực tế
       phải mua bù ở giá giờ bình thường

Tỉ lệ (2) mà (3) đạt được chính là "giá trị kinh tế của độ chính xác dự báo".

LƯU Ý VỀ GIÁ
------------
Đơn giá dưới đây là giá THAM KHẢO để minh hoạ, không phải biểu giá hiện hành.
Trước khi đưa vào báo cáo, hãy thay bằng biểu giá EVN đang áp dụng cho cấp điện
áp của nhà máy. Cấu trúc ba khung giờ thì ổn định, chỉ có đơn giá thay đổi.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

import config

# Đơn giá tham khảo (đồng/kWh) — THAY BẰNG BIỂU GIÁ THỰC TẾ
PRICE = {"thấp điểm": 1_100.0, "bình thường": 1_800.0, "cao điểm": 3_300.0}

# Tỉ lệ điện năng trong ngày có thể dịch chuyển sang khung giờ khác
ALPHA = 0.30

# Trần công suất mỗi khung 15 phút, tính theo bội số của đỉnh thực tế trong ngày.
# Để 1.00 nghĩa là không được tạo đỉnh mới cao hơn đỉnh vốn có — phản ánh ràng
# buộc công suất đăng ký của nhà máy. Tăng lên nếu nhà máy còn dư công suất.
PEAK_CAP_RATIO = 1.00


def tariff_period(ts: pd.Timestamp) -> str:
    """Xác định khung giờ của một mốc thời gian."""
    h = ts.hour + ts.minute / 60
    if h >= 22 or h < 4:
        return "thấp điểm"
    if ts.dayofweek == 6:            # Chủ nhật không có giờ cao điểm
        return "bình thường"
    if 9.5 <= h < 11.5 or 17 <= h < 20:
        return "cao điểm"
    return "bình thường"


def day_prices(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([PRICE[tariff_period(ts)] for ts in index])


def optimise_day(shiftable_kwh: float, prices: np.ndarray, cap: float) -> np.ndarray:
    """
    Phân bổ `shiftable_kwh` vào 96 khung giờ sao cho chi phí nhỏ nhất.
    Ràng buộc: tổng đúng bằng shiftable_kwh, mỗi khung không vượt trần `cap`.
    """
    n = len(prices)
    res = linprog(
        c=prices,
        A_eq=np.ones((1, n)), b_eq=[shiftable_kwh],
        bounds=[(0, cap)] * n,
        method="highs",
    )
    if not res.success:                       # trần quá chặt -> chia đều
        return np.full(n, shiftable_kwh / n)
    return res.x


def realise(shape: np.ndarray, energy: float, base: np.ndarray,
            prices: np.ndarray, cap: float):
    """
    Thực thi một kế hoạch trong thực tế.

    `shape` là hình dạng lịch do bước tối ưu đề xuất. Lượng điện thực sự phải
    tiêu thụ là `energy` (theo thực tế, không theo dự báo), được phân bổ theo
    hình dạng đó rồi cắt theo trần công suất thật. Phần không nhét vừa sẽ dồn
    sang các khung giờ rẻ nhất còn chỗ trống.

    Nhờ vậy kế hoạch lý tưởng luôn là cận trên của mọi kế hoạch khác — một
    kế hoạch dựa trên dự báo sai không bao giờ "rẻ hơn" kế hoạch biết trước.
    """
    headroom = np.maximum(cap - base, 0.0)
    w = shape / max(shape.sum(), 1e-12)

    alloc = np.minimum(energy * w, headroom)
    spill = energy - alloc.sum()

    if spill > 1e-9:
        remaining = headroom - alloc
        for i in np.argsort(prices, kind="stable"):
            take = min(spill, remaining[i])
            alloc[i] += take
            spill -= take
            if spill <= 1e-9:
                break

    cost = float((prices * (base + alloc)).sum())
    if spill > 1e-9:                      # vượt trần công suất: phải chạy quá tải
        cost += spill * float(prices.max())
    return alloc, cost


def run(pred_file: str, model_name: str, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(config.PREDICTIONS_DIR / pred_file, parse_dates=[config.TIME_COL])
    if pred_col not in df.columns:
        raise KeyError(f"{pred_file} không có cột '{pred_col}'. Có: {list(df.columns)}")
    df = df.set_index(config.TIME_COL)

    rows = []
    for day, g in df.groupby(df.index.date):
        if len(g) != config.STEPS_PER_DAY:
            continue

        actual = g["actual"].values
        forecast = np.maximum(g[pred_col].values, 0.0)
        prices = day_prices(g.index)

        # Tải nền luôn diễn ra theo thực tế; chỉ phần dịch chuyển được là do
        # kế hoạch quyết định. Trần công suất tính trên đỉnh thực tế của ngày.
        base = (1 - ALPHA) * actual
        cap = actual.max() * PEAK_CAP_RATIO
        shift_actual = ALPHA * actual.sum()

        # Kế hoạch lập từ DỰ BÁO: nhà máy chỉ có forecast trong tay khi xếp lịch
        base_hat = (1 - ALPHA) * forecast
        cap_hat = forecast.max() * PEAK_CAP_RATIO
        plan = optimise_day(ALPHA * forecast.sum(), prices,
                            float(np.maximum(cap_hat - base_hat, 0).max()))

        # Kế hoạch lý tưởng: biết trước chính xác phụ tải ngày mai
        oracle = optimise_day(shift_actual, prices,
                              float(np.maximum(cap - base, 0).max()))

        cost_none = float((prices * actual).sum())
        alloc_plan, cost_plan = realise(plan, shift_actual, base, prices, cap)
        alloc_oracle, cost_oracle = realise(oracle, shift_actual, base, prices, cap)

        rows.append({
            "ngày": str(day),
            "chi_phí_không_tối_ưu": cost_none,
            "chi_phí_lý_tưởng": cost_oracle,
            "chi_phí_theo_dự_báo": cost_plan,
            "đỉnh_gốc_kWh": float(actual.max()),
            "đỉnh_sau_tối_ưu_kWh": float((base + alloc_plan).max()),
        })

    out = pd.DataFrame(rows)
    out.insert(0, "model", model_name)
    return out


def summarise(out: pd.DataFrame, model_name: str) -> dict:
    none_ = out["chi_phí_không_tối_ưu"].sum()
    oracle = out["chi_phí_lý_tưởng"].sum()
    fcast = out["chi_phí_theo_dự_báo"].sum()

    saving_oracle = none_ - oracle
    saving_fcast = none_ - fcast

    return {
        "model": model_name,
        "chi_phí_không_tối_ưu_triệu_đ": round(none_ / 1e6, 2),
        "chi_phí_theo_dự_báo_triệu_đ": round(fcast / 1e6, 2),
        "chi_phí_lý_tưởng_triệu_đ": round(oracle / 1e6, 2),
        "tiết_kiệm_%": round(100 * saving_fcast / none_, 2),
        "tiết_kiệm_tối_đa_%": round(100 * saving_oracle / none_, 2),
        "%_lợi_ích_đạt_được": round(100 * saving_fcast / saving_oracle, 1)
        if saving_oracle > 0 else float("nan"),
        "thay_đổi_đỉnh_%": round(100 * (out["đỉnh_sau_tối_ưu_kWh"].sum()
                                        / out["đỉnh_gốc_kWh"].sum() - 1), 2),
    }


def plot_day(pred_file: str, pred_col: str, path, day_index: int = 3):
    """Minh hoạ một ngày: biểu đồ tải trước và sau khi dịch chuyển."""
    df = pd.read_csv(config.PREDICTIONS_DIR / pred_file, parse_dates=[config.TIME_COL])
    df = df.set_index(config.TIME_COL)

    days = sorted(set(df.index.date))
    g = df[df.index.date == days[day_index]]
    actual, forecast = g["actual"].values, np.maximum(g[pred_col].values, 0)
    prices = day_prices(g.index)

    base = (1 - ALPHA) * actual
    headroom = np.maximum(actual.max() * PEAK_CAP_RATIO - base, 0)
    x = optimise_day(ALPHA * forecast.sum(), prices, headroom.max())
    x = np.minimum(x, headroom)
    x *= (ALPHA * forecast.sum()) / max(x.sum(), 1e-9)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    for period, color in [("cao điểm", "#E45756"), ("thấp điểm", "#54A24B")]:
        mask = np.array([tariff_period(t) == period for t in g.index])
        ax.fill_between(g.index, 0, actual.max() * 1.25, where=mask,
                        color=color, alpha=0.12, label=f"Giờ {period}")
    ax.plot(g.index, actual, color="#333333", lw=1.8, label="Tải gốc")
    ax.plot(g.index, base + x, color="#4C78A8", lw=1.8, label="Tải sau dịch chuyển")
    ax.set_ylabel("Usage_kWh")
    ax.set_ylim(0, actual.max() * 1.25)
    ax.set_title(f"Dịch chuyển tải theo khung giá điện — ngày {days[day_index]}")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def sensitivity(caps=(0.45, 0.55, 0.65, 0.80, 1.00)) -> pd.DataFrame:
    """
    Khi nào độ chính xác dự báo mới thực sự có giá trị kinh tế?

    Nếu nhà máy còn nhiều dư địa công suất thì lịch tối ưu gần như luôn là
    "dồn hết vào giờ thấp điểm", gần như không phụ thuộc dự báo. Dự báo chỉ bắt
    đầu có giá khi trần công suất siết lại, lúc đó phải biết trước phụ tải nền
    rơi vào đâu mới xếp được lịch tốt.

    Hàm này quét trần công suất từ chặt đến rộng và đo phần lợi ích mỗi model
    giữ được so với kế hoạch lý tưởng.
    """
    global PEAK_CAP_RATIO
    original = PEAK_CAP_RATIO

    sources = [
        ("predictions_24h_B.csv", "XGBoost 24h (Model B)", "XGBoost"),
        ("predictions_24h_A.csv", "XGBoost 24h (Model A)", "XGBoost"),
        ("predictions_24h_A.csv", "Seasonal naive 7 ngày", "Seasonal naive (7 ngày)"),
        ("predictions_24h_A.csv", "Naive (persistence)", "Naive (persistence)"),
    ]

    rows = []
    for cap in caps:
        PEAK_CAP_RATIO = cap
        for f, name, col in sources:
            if not (config.PREDICTIONS_DIR / f).exists():
                continue
            summary = summarise(run(f, name, col), name)
            rows.append({"trần_công_suất": cap, "model": name,
                         "tiết_kiệm_%": summary["tiết_kiệm_%"],
                         "%_lợi_ích_đạt_được": summary["%_lợi_ích_đạt_được"]})

    PEAK_CAP_RATIO = original
    return pd.DataFrame(rows)


def plot_sensitivity(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, (name, g) in enumerate(df.groupby("model", sort=False)):
        ax.plot(g["trần_công_suất"] * 100, g["%_lợi_ích_đạt_được"],
                marker="o", lw=1.8, color=colors[i % len(colors)], label=name)
    ax.axhline(100, color="#888888", ls="--", lw=1)
    ax.set_xlabel("Trần công suất (% đỉnh phụ tải thực tế)")
    ax.set_ylabel("% lợi ích đạt được so với kế hoạch lý tưởng")
    ax.set_title("Dự báo chính xác có giá trị kinh tế khi nào?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Demo tối ưu hoá — dịch chuyển tải theo khung giá điện")
    print(f"ALPHA = {ALPHA:.0%} điện năng dịch chuyển được, "
          f"trần công suất = {PEAK_CAP_RATIO:.0%} đỉnh thực tế\n" + "-" * 70)

    sources = [
        ("predictions_24h_B.csv", "XGBoost 24h (Model B)", "XGBoost"),
        ("hybrid_24h_B.csv", "Hybrid 24h (Model B)", "Hybrid"),
        ("predictions_24h_A.csv", "XGBoost 24h (Model A)", "XGBoost"),
        ("predictions_24h_A.csv", "Seasonal naive 7 ngày", "Seasonal naive (7 ngày)"),
    ]

    summaries, details = [], []
    for f, name, col in sources:
        if not (config.PREDICTIONS_DIR / f).exists():
            continue
        out = run(f, name, col)
        details.append(out)
        summaries.append(summarise(out, name))
        print(f"  {name}: tiết kiệm {summaries[-1]['tiết_kiệm_%']}% "
              f"(tối đa {summaries[-1]['tiết_kiệm_tối_đa_%']}%, "
              f"đạt {summaries[-1]['%_lợi_ích_đạt_được']}% lợi ích)")

    pd.concat(details).to_csv(config.METRICS_DIR / "optimization_daily.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(config.METRICS_DIR / "optimization_summary.csv", index=False)

    plot_day("predictions_24h_B.csv", "XGBoost", config.FIGURES_DIR / "load_shifting.png")

    print("\n" + summary.to_string(index=False))

    print("\n" + "-" * 70)
    print("Độ nhạy: dự báo chính xác có giá trị khi trần công suất siết lại?")
    sens = sensitivity()
    sens.to_csv(config.METRICS_DIR / "optimization_sensitivity.csv", index=False)
    plot_sensitivity(sens, config.FIGURES_DIR / "optimization_sensitivity.png")
    print(sens.pivot(index="trần_công_suất", columns="model",
                     values="%_lợi_ích_đạt_được").to_string())
    print(f"\nĐã lưu: {config.METRICS_DIR / 'optimization_summary.csv'}")


if __name__ == "__main__":
    main()
