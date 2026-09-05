"""
Metric và biểu đồ đánh giá cho bài toán forecasting.

Lưu ý về MAPE: dataset có 1 observation Usage_kWh = 0 và median chỉ ~4.57 kWh,
nên MAPE rất dễ bị thổi phồng bởi các mẫu có giá trị thực gần 0. Vì vậy ở đây
MAPE được tính trên các mẫu khác 0 và luôn báo cáo kèm sMAPE + MASE, là hai
metric ổn định hơn cho dữ liệu lệch mạnh như thế này.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


# ============================================================
# Metrics
# ============================================================

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat):
    """MAPE (%), bỏ qua các mẫu có giá trị thực bằng 0."""
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat):
    """Symmetric MAPE (%) — không vỡ khi giá trị thực gần 0."""
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


def seasonal_naive_scale(train_y: np.ndarray, season: int = config.STEPS_PER_DAY) -> float:
    """
    Mẫu số của MASE: sai số trung bình của seasonal naive trên tập train.
    MASE < 1 nghĩa là model tốt hơn cách dự báo "lấy y hệt hôm qua".
    """
    return float(np.mean(np.abs(train_y[season:] - train_y[:-season])))


def compute_metrics(y, yhat, scale: float) -> dict:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAPE_%": mape(y, yhat),
        "sMAPE_%": smape(y, yhat),
        "R2": r2(y, yhat),
        "MASE": mae(y, yhat) / scale,
    }


# ============================================================
# Biểu đồ
# ============================================================

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def plot_model_comparison(metrics_df: pd.DataFrame, horizon_label: str, path):
    """Bar chart so sánh MAE và RMSE giữa các model trên tập test."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        d = metrics_df.sort_values(metric)
        bars = ax.barh(d["model"], d[metric], color=PALETTE[: len(d)])
        ax.set_xlabel(f"{metric} (kWh)")
        ax.set_title(f"{metric} — test set (horizon {horizon_label})")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlim(0, d[metric].max() * 1.2)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(dates, y_true, preds: dict, horizon_label: str, path, days: int = 7):
    """Vẽ giá trị thực và dự báo trong `days` ngày đầu của tập test."""
    n = days * config.STEPS_PER_DAY
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(dates[:n], y_true[:n], color="#333333", lw=1.6, label="Thực tế", zorder=5)
    for i, (name, yhat) in enumerate(preds.items()):
        ax.plot(dates[:n], yhat[:n], lw=1.2, alpha=0.85,
                color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_ylabel("Usage_kWh")
    ax.set_title(f"Dự báo vs thực tế — {days} ngày đầu tập test (horizon {horizon_label})")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, yhat, model_name: str, horizon_label: str, path):
    """Phân bố sai số và biểu đồ thực tế vs dự báo của model tốt nhất."""
    resid = y_true - yhat
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].hist(resid, bins=60, color=PALETTE[0], edgecolor="white")
    axes[0].axvline(0, color="#E45756", lw=1.4)
    axes[0].set_title(f"Phân bố residual — {model_name}")
    axes[0].set_xlabel("Thực tế - Dự báo (kWh)")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(yhat, y_true, s=6, alpha=0.3, color=PALETTE[0])
    lim = [0, max(y_true.max(), yhat.max()) * 1.02]
    axes[1].plot(lim, lim, color="#E45756", lw=1.2)
    axes[1].set_xlabel("Dự báo (kWh)")
    axes[1].set_ylabel("Thực tế (kWh)")
    axes[1].set_title(f"Thực tế vs dự báo — horizon {horizon_label}")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(names, importances, model_name: str, path, top: int = 15):
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1.4))
    ax.barh([names[i] for i in order], [importances[i] for i in order], color=PALETTE[0])
    ax.set_title(f"Top {top} feature quan trọng — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
