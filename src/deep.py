"""
Model D — LSTM học thẳng từ chuỗi thô.

Ý ĐỒ SO SÁNH
------------
Các model trước đều dựa vào lag feature do mình tự đắp: lag_1h, lag_24h,
roll_mean_24h, same_slot_mean_7d... Nói cách khác, phần "hiểu chuỗi thời gian"
là do con người làm, model chỉ việc hồi quy.

LSTM đảo ngược việc đó: đưa thẳng 24 giờ dữ liệu thô gần nhất vào mạng và để nó
tự học xem đoạn nào quan trọng. Đầu vào gồm hai luồng:

    1. Chuỗi 96 bước Usage_kWh, kết thúc tại t - H  (đúng quy tắc chống leakage)
    2. Feature lịch của thời điểm t (+ Load_Type nếu là Model B)

Mục đích không phải để chắc chắn thắng XGBoost — với dataset cỡ 25 nghìn dòng
thì gradient boosting thường vẫn rất mạnh — mà để trả lời một câu hỏi phương
pháp: việc đắp lag feature bằng tay có thực sự cần thiết không, hay mạng tự học
được?
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import evaluate as ev
from features import (build_base_frame, add_horizon_features, split_chronological,
                      BURN_IN, CALENDAR_FEATURES, CYCLICAL_FEATURES, LOAD_TYPE_FEATURES)

SEQ_LEN = 96          # 24 giờ lịch sử gần nhất
HIDDEN = 64
EPOCHS = 40
PATIENCE = 6
BATCH = 256
LR = 1e-3

torch.manual_seed(config.RANDOM_STATE)


class LSTMForecaster(nn.Module):
    def __init__(self, n_static: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(-1)


def build_arrays(horizon: int, variant: str):
    """Dựng (chuỗi, feature tĩnh, target) cho train/val/test."""
    base = build_base_frame()
    usage_full = base[config.TARGET].values.astype(np.float32)

    df, _ = add_horizon_features(base, horizon, include_load_type=(variant == "B"))
    static_cols = CALENDAR_FEATURES + CYCLICAL_FEATURES
    if variant == "B":
        static_cols = static_cols + LOAD_TYPE_FEATURES

    # Chuỗi cho dòng i kết thúc tại t - horizon, dài SEQ_LEN bước
    g = np.arange(len(df)) + BURN_IN
    starts = g - horizon - SEQ_LEN + 1
    assert starts.min() >= 0, "Không đủ lịch sử cho SEQ_LEN — tăng BURN_IN."
    idx = starts[:, None] + np.arange(SEQ_LEN)[None, :]
    seqs = usage_full[idx]

    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    train, val, test = split_chronological(df)

    def pack(part):
        rows = part["_row"].values
        return (seqs[rows], part[static_cols].values.astype(np.float32),
                part[config.TARGET].values.astype(np.float32),
                part[config.TIME_COL].values)

    return pack(train), pack(val), pack(test), static_cols


def run(label: str, horizon: int, variant: str) -> dict:
    (Str, Xtr, ytr, _), (Sva, Xva, yva, _), (Ste, Xte, yte, dte), cols = \
        build_arrays(horizon, variant)

    # Chuẩn hoá bằng thống kê của tập train
    s_mu, s_sd = Str.mean(), Str.std() + 1e-8
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0) + 1e-8

    def tens(S, X, y=None):
        t = [torch.tensor((S - s_mu) / s_sd).unsqueeze(-1),
             torch.tensor((X - x_mu) / x_sd)]
        if y is not None:
            t.append(torch.tensor(y))
        return t

    tr = tens(Str, Xtr, ytr)
    va = tens(Sva, Xva, yva)
    te = tens(Ste, Xte)

    model = LSTMForecaster(n_static=len(cols))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.L1Loss()

    n = len(ytr)
    best = (float("inf"), 0, None)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            j = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(model(tr[0][j], tr[1][j]), tr[2][j])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_mae = float(lossf(model(va[0], va[1]), va[2]))

        if val_mae < best[0] - 1e-4:
            best = (val_mae, epoch, {k: v.clone() for k, v in model.state_dict().items()})
        elif epoch - best[1] >= PATIENCE:
            print(f"    dừng sớm ở epoch {epoch}")
            break

        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:2d}  val MAE = {val_mae:.4f}")

    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        yhat = model(te[0], te[1]).numpy()

    scale = ev.seasonal_naive_scale(ytr)
    metrics = ev.compute_metrics(yte, yhat, scale)
    print(f"  {label}/{variant}: val MAE={best[0]:.4f} (epoch {best[1]})  "
          f"test MAE={metrics['MAE']:.4f}  [{time.time() - t0:.0f}s]")

    pd.DataFrame({config.TIME_COL: dte, "actual": yte, "LSTM": yhat}).to_csv(
        config.PREDICTIONS_DIR / f"lstm_{label}_{variant}.csv", index=False)

    return {"horizon": label, "variant": variant, "model": "LSTM",
            "best_epoch": best[1], **metrics}


def main():
    print(f"Model D — LSTM (chuỗi {SEQ_LEN} bước, hidden {HIDDEN})\n" + "-" * 70)
    rows = []
    for label, h in config.HORIZONS.items():
        for variant in ("A", "B"):
            rows.append(run(label, h, variant))

    out = pd.DataFrame(rows)
    out.to_csv(config.METRICS_DIR / "lstm_results.csv", index=False)
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nĐã lưu: {config.METRICS_DIR / 'lstm_results.csv'}")


if __name__ == "__main__":
    main()
