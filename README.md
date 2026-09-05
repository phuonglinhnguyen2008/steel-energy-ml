# Steel Industry Energy Consumption – Data Preparation

## 1. Project này đang làm gì?

Đề tài hiện tại của team là:

**Dự báo và tối ưu hóa điện năng tiêu thụ trong công nghiệp nặng ứng dụng Machine Learning**  
(*Machine Learning-based Energy Consumption Forecasting and Optimization in Heavy Manufacturing*)

Dataset đang sử dụng là **Steel Industry Energy Consumption Dataset**.

Phần data understanding và preprocessing chị đã xử lý trước để khi chuyển sang phần implementation thì không cần bắt đầu lại từ raw data.

README này chủ yếu để ghi lại:

- Dataset có gì và mình đã hiểu nó như thế nào.
- Những vấn đề đã kiểm tra trong dữ liệu.
- Mình đã preprocessing những gì.
- Feature nào hiện tại được giữ lại cho forecasting.
- Feature nào mình chủ động chưa đưa vào model.
- Train / validation / test đã được chia như thế nào.
- Và Linh tiếp tục phần implementation thì nên bắt đầu từ đâu.

---

# 2. Cấu trúc project hiện tại

```text
steel-energy-ml/
│
├── data/
│   ├── raw/
│   │   └── Steel_industry_data.csv
│   │
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_data_preprocessing.ipynb
│
├── src/
│   ├── config.py
│   └── preprocessing.py
│
├── README.md
└── requirements.txt
```

Hai notebook có mục đích khác nhau:

### `01_data_understanding.ipynb`

File này mình dùng để đọc và kiểm tra dataset trước.

Có thể xem đây là phần tìm hiểu xem:

- dữ liệu có sạch không;
- timestamp có vấn đề không;
- biến nào đáng chú ý;
- mức tiêu thụ điện có pattern gì;
- có outlier hay giá trị bất thường nào không;
- các giá trị trong quá khứ có liên quan tới mức tiêu thụ hiện tại không.

### `02_data_preprocessing.ipynb`

Sau khi hiểu dataset rồi thì file này mới thực sự xử lý data và tạo feature để chuẩn bị cho model.

---

# 3. Dataset đã được kiểm tra như thế nào?

Raw data nằm ở:

```text
data/raw/Steel_industry_data.csv
```

Dataset ban đầu có:

```text
35,040 rows
11 columns
```

Dữ liệu chạy từ:

```text
01/01/2018 00:00
->
31/12/2018 23:45
```

Mỗi observation cách nhau đúng **15 phút**.

Như vậy một ngày sẽ có:

```text
24 giờ × 4 observations/giờ = 96 observations
```

và:

```text
365 × 96 = 35,040 observations
```

nên số lượng dữ liệu cũng khớp với một năm đầy đủ.

---

# 4. Data quality

Mình đã kiểm tra các vấn đề cơ bản trước khi preprocessing.

Kết quả hiện tại:

```text
Missing values          : 0
Duplicate timestamps    : 0
Negative Usage_kWh      : 0
Irregular time interval : 0
```

Toàn bộ timestamp liên tục theo interval 15 phút.

Có đúng **1 observation `Usage_kWh = 0`**.

Mình có kiểm tra riêng observation này và thấy các biến điện liên quan cũng bằng 0, nên hiện tại chị **giữ lại**, không tự động coi đây là lỗi dữ liệu.

---

# 5. Về các giá trị Usage_kWh cao

Ban đầu nhìn histogram và boxplot sẽ thấy khá nhiều giá trị ở vùng 120–157 kWh và nhìn giống outlier.

Mình kiểm tra riêng các observation:

```text
Usage_kWh >= 120
```

thì có:

```text
447 observations
```

và chúng xuất hiện ở nhiều thời điểm cũng như nhiều trạng thái tải khác nhau.

Vì vậy mình **không xóa các giá trị này chỉ dựa trên IQR/boxplot**.

Đối với dữ liệu nhà máy, mức điện cao hoàn toàn có thể xuất hiện khi nhà máy hoạt động ở tải lớn. Nếu xóa chỉ vì chúng nằm xa median thì có khả năng mình lại xóa đúng những giai đoạn sản xuất quan trọng mà model cần học.

---

# 6. Một điều khá rõ sau khi xem data: Usage_kWh không phân bố đều

Mean của `Usage_kWh` khoảng:

```text
27.39 kWh
```

nhưng median chỉ khoảng:

```text
4.57 kWh
```

Nghĩa là distribution lệch khá mạnh.

Lý do là dataset có những khoảng thời gian nhà máy hoạt động rất nhẹ và những khoảng thời gian mức tiêu thụ tăng lên khoảng 40–60 kWh hoặc cao hơn.

Điều này cũng thể hiện khá rõ khi xem theo `Load_Type`.

| Load Type | Mean Usage |
|---|---:|
| Light Load | ~8.63 kWh |
| Medium Load | ~38.45 kWh |
| Maximum Load | ~59.27 kWh |

Nên không nên nhìn toàn bộ distribution rồi kết luận các giá trị lớn đều là anomaly.

---

# 7. Pattern theo thời gian

Chị kiểm tra mức tiêu thụ theo giờ và thấy pattern trong ngày khá rõ.

Nói đơn giản là:

> Điện năng nhà máy tiêu thụ phụ thuộc khá nhiều vào thời điểm trong ngày.

Có những khoảng thời gian mức tiêu thụ thấp, sau đó tăng rất mạnh vào giờ hoạt động sản xuất, giảm xuống ở một số khoảng nghỉ rồi tăng trở lại.

Ngoài ra weekday và weekend cũng khác nhau khá rõ.

Mean Usage:

```text
Weekday ≈ 33.62 kWh
Weekend ≈ 11.73 kWh
```

Nên thông tin về thời gian chắc chắn đáng để giữ làm feature.

---

# 8. Historical Usage cũng rất quan trọng

Mình có kiểm tra correlation giữa `Usage_kWh` hiện tại và các thời điểm trước đó.

Kết quả:

| Previous time | Correlation |
|---|---:|
| 15 phút trước | 0.912 |
| 30 phút trước | 0.834 |
| 1 giờ trước | 0.687 |
| 2 giờ trước | 0.540 |
| 6 giờ trước | 0.227 |
| 12 giờ trước | -0.243 |
| 24 giờ trước | 0.606 |
| 7 ngày trước | 0.677 |

Cái này khá quan trọng.

15 phút và 30 phút trước có correlation rất cao, tức là mức tiêu thụ vừa xảy ra có khả năng cung cấp khá nhiều thông tin cho mức tiêu thụ tiếp theo.

Ngoài ra:

```text
24h  → 0.606
7d   → 0.677
```

cũng khá cao.

Nó cho thấy dataset có pattern lặp lại theo ngày và theo tuần.

Vì vậy mình đã tạo lag features thay vì chỉ đưa các biến thời gian vào model.

---

# 9. Feature chị đã tạo

## Calendar features

Từ `date`, tạo:

```text
hour
minute
day_of_week
month
is_weekend
```

Rồi tạo lại trực tiếp từ timestamp thay vì phụ thuộc hoàn toàn vào các cột text có sẵn trong raw dataset.

---

## Cyclical features

Mình tạo thêm:

```text
time_sin
time_cos

dow_sin
dow_cos

month_sin
month_cos
```

Lý do là thời gian có tính vòng tròn.

Ví dụ:

```text
23:45
00:00
```

thực tế chỉ cách nhau 15 phút.

Nếu chỉ biểu diễn thời gian bằng số bình thường thì model có thể xem hai giá trị này là hai đầu rất xa nhau.

Sin/cos giúp giữ được tính tuần hoàn đó.

Tuy nhiên khi implementation **không cần mặc định cyclical features chắc chắn tốt hơn raw calendar features**. Có thể benchmark để xem model nào thực sự hưởng lợi từ chúng.

---

# 10. Lag features

Dựa trên phần data understanding, tạo:

```text
lag_15min
lag_30min
lag_1h
lag_2h
lag_24h
lag_7d
```

Ví dụ:

```text
lag_15min
```

là lượng điện đã tiêu thụ 15 phút trước observation hiện tại.

Còn:

```text
lag_7d
```

là mức tiêu thụ tại cùng thời điểm cách đó 7 ngày.

Do `lag_7d` cần dữ liệu của 7 ngày trước nên 7 ngày đầu dataset không thể có feature này.

Một ngày có 96 observations:

```text
96 × 7 = 672
```

nên sau khi tạo lag features, chị bỏ **672 rows đầu tiên**.

Dataset từ:

```text
35,040
```

còn:

```text
34,368 observations
```

Đây không phải xóa bad data.

Chỉ đơn giản là 672 observations đầu tiên chưa có đủ historical information để tạo `lag_7d`.

---

# 11. Feature set hiện tại

Primary forecasting dataset hiện tại có 17 input features.

### Calendar

```text
hour
minute
day_of_week
month
is_weekend
```

### Cyclical

```text
time_sin
time_cos
dow_sin
dow_cos
month_sin
month_cos
```

### Historical Usage

```text
lag_15min
lag_30min
lag_1h
lag_2h
lag_24h
lag_7d
```

Target:

```text
Usage_kWh
```

File processed sẽ có:

```text
date
+ 17 features
+ Usage_kWh
```

tổng cộng **19 columns**.

---

# 12. Một chỗ cần lưu ý: đừng tự động lấy hết raw columns để train

Trong raw dataset còn có:

```text
CO2(tCO2)

Lagging_Current_Reactive.Power_kVarh
Leading_Current_Reactive_Power_kVarh

Lagging_Current_Power_Factor
Leading_Current_Power_Factor

NSM
WeekStatus
Day_of_week
Load_Type
```

Chị chủ động **chưa đưa tất cả các cột này vào primary forecasting dataset**.

Không phải vì chúng không có giá trị.

Vấn đề nằm ở câu hỏi:

> Tại thời điểm mình cần forecast Usage_kWh trong tương lai, những thông tin đó đã có sẵn cho model hay chưa?

Nếu model đang dự báo:

```text
Usage_kWh(t)
```

nhưng lại được cung cấp một sensor measurement cũng được đo tại:

```text
time = t
```

thì cần kiểm tra rất kỹ vì trong deployment thực tế có thể lúc forecast mình chưa biết giá trị đó.

Nếu không cẩn thận thì performance của model có thể rất đẹp nhưng không phản ánh forecasting thực tế.

---

# 13. Đặc biệt chú ý CO2

Trong EDA chị thấy:

```text
corr(CO2, Usage_kWh) ≈ 0.988
```

Correlation cực kỳ cao.

Nhìn qua thì rất dễ nghĩ:

> Cho CO2 vào chắc model dự báo rất tốt.

Nhưng chính vì correlation cao như vậy nên càng phải kiểm tra.

Hiện tại chị **không đưa CO2 vào primary forecasting model** để tránh trường hợp model học một biến gần như phản ánh trực tiếp mức tiêu thụ điện tại cùng thời điểm.

Sau này nếu team xác định được CO2 có sẵn trước prediction time và có ý nghĩa đúng với deployment scenario thì có thể thử thêm một experiment riêng.

---

# 14. NSM là gì?

Mình cũng đã kiểm tra `NSM`.

Nó chạy như sau:

```text
00:00 → 0
00:15 → 900
00:30 → 1800
00:45 → 2700
...
23:45 → 85500
```

Tức là số giây tính từ đầu ngày.

Dataset có đúng:

```text
96 unique NSM values
```

nên về cơ bản nó đang encode thông tin thời gian mà mình đã lấy được từ `date`.

Vì vậy primary feature set không cần giữ raw `NSM`.

---

# 15. Còn Load_Type thì sao?

`Load_Type` hơi khác các biến phía trên.

Nó có:

```text
Light_Load
Medium_Load
Maximum_Load
```

và liên quan khá rõ tới mức tiêu thụ điện.

Mình chưa đưa nó vào primary model vì muốn tách thành hai trường hợp để test sau.

### Model A – Historical Forecast

```text
Calendar
+ Cyclical
+ Lag Usage
```

Model này dự báo chủ yếu từ lịch sử tiêu thụ và thời gian.

### Model B – Schedule-Aware Forecast

```text
Model A
+ Load_Type
```

Model B sẽ phù hợp nếu mình giả định nhà máy đã biết production/load schedule trước khi forecast.

Nếu làm được cả hai thì khá hay, vì mình có thể trả lời thêm:

> Biết trước kế hoạch tải sản xuất giúp forecasting tốt hơn bao nhiêu?

Nên phần implementation có thể cân nhắc experiment này sau khi baseline chính đã chạy ổn.

---

# 16. Mình đã chia train / validation / test rồi

Vì đây là forecasting nên chị **không random split**.

Dữ liệu được chia theo thời gian:

### Train

```text
08/01/2018 00:00
→
30/09/2018 23:45

25,536 rows
```

### Validation

```text
01/10/2018 00:00
→
30/11/2018 23:45

5,856 rows
```

### Test

```text
01/12/2018 00:00
→
31/12/2018 23:45

2,976 rows
```

Tổng:

```text
25,536 + 5,856 + 2,976
= 34,368 rows
```

Mình cũng đã check để đảm bảo ba khoảng thời gian không overlap nhau.

Lý do chia như vậy là để gần với tình huống thực tế hơn:

```text
học từ quá khứ
      ↓
tuning trên giai đoạn sau
      ↓
test trên tương lai chưa thấy
```

Nên phần implementation **không shuffle lại ba file này rồi chia random từ đầu**.

---

# 17. Data để implementation nằm ở đâu?

Em có thể bắt đầu trực tiếp từ:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

Không cần preprocessing lại từ raw data để bắt đầu train model.

Nếu muốn xem chị đã xử lý từng bước như thế nào thì xem:

```text
notebooks/01_data_understanding.ipynb
notebooks/02_data_preprocessing.ipynb
```

Còn pipeline để tái tạo processed data nằm ở:

```text
src/preprocessing.py
```

---

# 18. Nếu muốn chạy lại preprocessing

Đứng ở root của project rồi chạy:

```bash
python src/preprocessing.py
```

Pipeline sẽ đi theo luồng:

```text
Steel_industry_data.csv
          ↓
      load data
          ↓
data quality checks
          ↓
   datetime sorting
          ↓
 calendar features
          ↓
 cyclical features
          ↓
    lag features
          ↓
 remove first 672 rows
          ↓
 select model features
          ↓
 chronological split
          ↓
 ┌────────┼──────────┐
 ↓        ↓          ↓
train   validation   test
```

---

# 19. Phần implementation tiếp theo

Em nhận phần model thì chị nghĩ nên bắt đầu đơn giản trước, chưa cần nhảy ngay vào model phức tạp.

Có thể đi theo thứ tự:

```text
Naive/Persistence baseline
          ↓
Linear Regression
          ↓
Random Forest
          ↓
XGBoost
```

Sau đó mới so sánh.

Metric có thể dùng:

```text
MAE
RMSE
MAPE
R²
```

Nhớ dùng:

```text
train
```

để fit model,

```text
validation
```

để lựa chọn model / tuning,

và:

```text
test
```

chỉ dùng cho final evaluation.

Đừng tuning dựa trên test set.

---

# 20. Tóm lại phần chị đã xử lý

Hiện tại chị đã làm xong:

```text
✓ Đọc và kiểm tra raw dataset
✓ Kiểm tra missing values
✓ Kiểm tra duplicate timestamps
✓ Kiểm tra interval 15 phút
✓ Kiểm tra zero / negative Usage
✓ Xem distribution và high-consumption observations
✓ Phân tích theo Load Type
✓ Phân tích pattern theo giờ
✓ Phân tích weekday/weekend
✓ Phân tích theo tháng
✓ Kiểm tra lag correlation
✓ Parse datetime
✓ Tạo calendar features
✓ Tạo cyclical features
✓ Tạo lag features
✓ Xử lý rows không đủ lag history
✓ Chọn primary forecasting feature set
✓ Tách các feature có nguy cơ leakage
✓ Chronological train/validation/test split
✓ Export processed datasets
✓ Viết preprocessing pipeline
```

Vậy phần implementation có thể bắt đầu từ processed data luôn.

Nếu trong lúc làm model em muốn thêm feature nào từ raw dataset thì cứ thêm experiment riêng, nhưng nhớ kiểm tra trước:

> Feature này có thực sự tồn tại tại thời điểm mình cần đưa ra forecast hay không?

Nếu câu trả lời là không thì dù model score tăng mạnh cũng chưa chắc đó là improvement thật.