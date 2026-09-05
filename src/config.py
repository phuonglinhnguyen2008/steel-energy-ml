"""
Cấu hình dùng chung cho toàn bộ project.

File này được README nhắc tới nhưng chưa tồn tại trong repo, nên bổ sung ở đây
để phần implementation không phải hard-code đường dẫn ở nhiều chỗ.
"""

from pathlib import Path

# ============================================================
# Đường dẫn
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Steel_industry_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

for _d in (RESULTS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Thông số dữ liệu
# ============================================================

# Dataset lấy mẫu mỗi 15 phút
FREQ_MINUTES = 15
STEPS_PER_HOUR = 60 // FREQ_MINUTES        # 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR        # 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY         # 672

TARGET = "Usage_kWh"
TIME_COL = "date"

# Mốc chia train/validation/test — giữ nguyên đúng như bước preprocessing của chị
VAL_START = "2018-10-01"
TEST_START = "2018-12-01"


# ============================================================
# Horizon dự báo
# ============================================================
# H là số bước 15 phút mà mình dự báo trước.
# Ví dụ H = 4 nghĩa là: đứng ở thời điểm t, dự báo mức tiêu thụ tại t + 1 giờ.

HORIZONS = {
    "15min": 1,
    "1h": STEPS_PER_HOUR,    # 4
    "24h": STEPS_PER_DAY,    # 96
}

RANDOM_STATE = 42
