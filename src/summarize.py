"""
Gộp kết quả của tất cả model vào một bảng xếp hạng duy nhất.

Chạy sau khi đã chạy train.py, hybrid.py, quantile.py và deep.py.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import config

COLS = ["horizon", "variant", "model", "MAE", "RMSE", "sMAPE_%", "R2", "MASE"]


def load(name: str, rename: dict | None = None) -> pd.DataFrame:
    path = config.METRICS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)
    return df[[c for c in COLS if c in df.columns]]


def main():
    parts = [
        load("summary_all_horizons.csv"),
        load("hybrid_results.csv"),
        load("lstm_results.csv"),
    ]

    q = config.METRICS_DIR / "quantile_results.csv"
    if q.exists():
        d = pd.read_csv(q)
        parts.append(pd.DataFrame({
            "horizon": d["horizon"], "variant": d["variant"],
            "model": "Quantile P50", "MAE": d["MAE_P50"], "MASE": d["MASE_P50"],
        }))

    all_ = pd.concat([p for p in parts if not p.empty], ignore_index=True)

    order = {lb: i for i, lb in enumerate(config.HORIZONS)}
    all_ = all_.sort_values(
        ["horizon", "MAE"],
        key=lambda c: c.map(order) if c.name == "horizon" else c,
    ).reset_index(drop=True)

    all_.to_csv(config.METRICS_DIR / "leaderboard.csv", index=False)

    print("BẢNG XẾP HẠNG TOÀN BỘ MODEL (tập test, sắp theo MAE)\n" + "=" * 78)
    for label in config.HORIZONS:
        d = all_[all_["horizon"] == label]
        print(f"\n--- Horizon {label} ---")
        print(d.drop(columns="horizon").to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))

    # Biểu đồ: MAE của model tốt nhất mỗi họ, theo horizon
    fig, ax = plt.subplots(figsize=(9, 4.6))
    best = (all_.loc[all_.groupby(["horizon", "model"])["MAE"].idxmin()]
                .pivot(index="horizon", columns="model", values="MAE")
                .reindex(list(config.HORIZONS)))
    best.plot(ax=ax, marker="o", lw=1.8)
    ax.set_ylabel("MAE (kWh) — thấp hơn là tốt hơn")
    ax.set_xlabel("Horizon dự báo")
    ax.set_title("Sai số theo horizon, mọi họ model (lấy cấu hình tốt nhất)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "leaderboard.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nĐã lưu: {config.METRICS_DIR / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
