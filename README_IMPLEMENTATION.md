# Phần Implementation — Model Forecasting

Nối tiếp phần data understanding và preprocessing chị đã làm trong `README.md`.

File này ghi lại:

- Chỗ em phải sửa lại trước khi train được.
- Feature em thêm vào và lý do.
- Các model đã chạy và kết quả.
- Vế tối ưu hoá trong tên đề tài.
- Những gì em đã kiểm chứng lại.

---

# 1. Chỗ đầu tiên em phải sửa: horizon chưa được định nghĩa

Bộ processed data trong repo đang được làm cho bài toán dự báo trước một bước, tức là 15 phút.

Mỗi dòng dự báo:

```text
Usage_kWh(t)
```

và được dùng:

```text
lag_15min = Usage_kWh(t - 15 phút)
```

Cái đó chỉ đúng nếu mình đang đứng ở thời điểm `t - 15 phút` để dự báo.

Nhưng nếu đề tài cần dự báo trước 24 giờ để nhà máy lên kế hoạch vận hành, thì tại lúc ra quyết định mình chưa hề biết `Usage_kWh(t - 15 phút)`. Nó còn chưa xảy ra.

Đưa nó vào model là data leakage. Metric sẽ rất đẹp nhưng không dùng được ngoài thực tế.

Đây đúng là loại lỗi chị cảnh báo ở mục 12, chỉ khác là nó nằm ở lag feature chứ không phải ở mấy cột sensor.

## Quy tắc em áp dụng xuyên suốt

Để dự báo `Usage_kWh(t)` trước H bước, với 1 bước = 15 phút, model chỉ được dùng những gì đã biết tại thời điểm `t - H`:

| Loại thông tin | Dùng được không | Lý do |
|---|---|---|
| Calendar / cyclical của `t` | Được | Lịch thì biết trước bao lâu cũng được |
| `Usage_kWh(t - k)` với `k >= H` | Được | Đã thực sự xảy ra tại lúc forecast |
| `Usage_kWh(t - k)` với `k < H` | Không | Chưa tồn tại tại lúc forecast |
| Rolling stats trên cửa sổ kết thúc tại `t - H` | Được | Không chạm vào tương lai |

Nên mỗi horizon có một feature set khác nhau:

| Horizon | H | Lag còn giữ được | Số feature |
|---|---:|---|---:|
| 15 phút | 1 | 15min, 30min, 1h, 2h, 24h, 48h, 7d | 28 |
| 1 giờ | 4 | 1h, 2h, 24h, 48h, 7d | 26 |
| 24 giờ | 96 | 24h, 48h, 7d | 24 |

Horizon 15 phút em vẫn giữ lại, nhưng chỉ để làm mốc tham chiếu. Có nó thì mới thấy độ khó tăng lên bao nhiêu khi dự báo xa dần.

---

# 2. Feature em thêm vào

Ngoài calendar, cyclical và lag của pipeline gốc, em thêm hai nhóm. Cả hai đều tuân thủ quy tắc ở trên.

## Rolling statistics

Cửa sổ luôn kết thúc tại `t - H`:

```text
roll_mean_1h
roll_std_1h

roll_mean_24h
roll_std_24h
roll_max_24h
roll_min_24h
```

## Same-slot profile

Mức tiêu thụ tại đúng khung giờ đó trong 7 ngày qua:

```text
same_slot_mean_7d
same_slot_median_7d
same_slot_std_7d
same_slot_max_7d
```

Ví dụ dự báo cho 14:30 thứ Ba thì lấy giá trị lúc 14:30 của 7 ngày trước đó rồi tính trung bình và độ lệch.

Nhóm này quan trọng nhất ở horizon 24 giờ, vì lúc đó mọi lag ngắn đều đã bị loại. Riêng nó kéo MAE của horizon 24h từ:

```text
11.97 kWh
->
10.87 kWh
```

Số dòng cắt ở đầu chuỗi vẫn là 672 như cũ, tức lag dài nhất là 7 ngày. Nên dataset vẫn còn 34.368 dòng và ba mốc split vẫn đúng như chị chia:

```text
25.536 / 5.856 / 2.976
```

---

# 3. Một chuyện em thấy đáng nói: chọn objective theo MAE

Mặc định RandomForest và XGBoost tối ưu squared error. Nghĩa là chúng học để dự báo giá trị trung bình.

Với dataset lệch mạnh như thế này:

```text
mean   = 27.39 kWh
median =  4.57 kWh
```

thì mục tiêu đó kéo dự báo lên cao ở những khoảng nhà máy chạy nhẹ, và làm MAE xấu đi rõ.

Em đổi XGBoost sang:

```text
objective = "reg:absoluteerror"
```

tức tối ưu trực tiếp MAE. Đây là thay đổi có tác động lớn nhất trong cả phần implementation:

```text
Horizon 24h, XGBoost:
    squared error   ->  MAE = 11.87 kWh
    absolute error  ->  MAE =  9.12 kWh
```

Giảm được 23% chỉ bằng một dòng.

---

# 4. Kết quả của bộ model trong README

Đơn vị MAE và RMSE là kWh. MASE dưới 1 nghĩa là tốt hơn cách dự báo "lấy y hệt hôm qua".

## Horizon 15 phút

| Model | MAE | RMSE | sMAPE % | R² | MASE |
|---|---:|---:|---:|---:|---:|
| Random Forest | 2.874 | 6.587 | 11.59 | 0.941 | 0.175 |
| XGBoost | 2.929 | 6.497 | 12.91 | 0.942 | 0.179 |
| Naive persistence | 3.819 | 9.612 | 12.83 | 0.873 | 0.233 |
| Linear Regression | 5.515 | 9.073 | 54.83 | 0.887 | 0.337 |
| Seasonal naive 7 ngày | 10.560 | 21.314 | 36.86 | 0.378 | 0.644 |
| Seasonal naive 24h | 12.049 | 23.794 | 44.30 | 0.224 | 0.735 |

## Horizon 1 giờ

| Model | MAE | RMSE | sMAPE % | R² | MASE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 4.696 | 9.770 | 23.84 | 0.869 | 0.287 |
| Random Forest | 4.707 | 10.207 | 21.00 | 0.857 | 0.287 |
| Naive persistence | 9.022 | 19.757 | 31.51 | 0.465 | 0.551 |
| Linear Regression | 9.922 | 14.370 | 85.23 | 0.717 | 0.605 |
| Seasonal naive 7 ngày | 10.560 | 21.314 | 36.86 | 0.378 | 0.644 |

## Horizon 24 giờ

| Model | MAE | RMSE | sMAPE % | R² | MASE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 9.122 | 17.167 | 41.20 | 0.596 | 0.557 |
| Seasonal naive 7 ngày | 10.560 | 21.314 | 36.86 | 0.378 | 0.644 |
| Random Forest | 10.871 | 19.952 | 47.26 | 0.455 | 0.663 |
| Naive persistence | 12.049 | 23.794 | 44.30 | 0.224 | 0.735 |
| Linear Regression | 12.249 | 17.676 | 88.57 | 0.572 | 0.747 |

## Hai baseline em thêm vào

README chỉ yêu cầu naive persistence. Em thêm seasonal naive theo ngày và theo tuần, vì mục 8 của chị đã chỉ ra dataset có pattern lặp khá rõ:

```text
24h  -> 0.606
7d   -> 0.677
```

Hoá ra thêm là đúng. Ở horizon 24h, seasonal naive 7 ngày đạt MAE 10.56, đánh bại cả Random Forest lẫn Linear Regression.

Nếu không có baseline này thì báo cáo sẽ kết luận sai rằng Random Forest tốt.

---

# 5. Model B — biết trước lịch tải thì được gì

Chị có đề xuất ở mục 15 là tách hai trường hợp. Em chạy cả hai.

Giả định của Model B là nhà máy đã biết trước kế hoạch tải tại thời điểm forecast. Với giả định đó thì `Load_Type` của thời điểm `t` là thông tin hợp lệ. Em one-hot nó thành ba cột rồi thêm vào Model A.

> Nếu giả định trên không đúng, tức nhà máy không có lịch tải trước, thì toàn bộ kết quả Model B là leakage và phải bỏ. Chỗ này cần chốt với nhóm trước khi viết báo cáo.

## Kết quả

| Horizon | Model | Model A | Model B | Chênh lệch |
|---|---|---:|---:|---:|
| 15 phút | Random Forest | 2.874 | 2.913 | -1.3% |
| 15 phút | XGBoost | 2.929 | 2.820 | +3.7% |
| 1 giờ | XGBoost | 4.696 | 4.659 | +0.8% |
| 1 giờ | Random Forest | 4.707 | 4.659 | +1.0% |
| 24 giờ | XGBoost | 9.122 | 7.050 | +22.7% |
| 24 giờ | Random Forest | 10.871 | 11.321 | -4.1% |
| 24 giờ | Linear Regression | 12.249 | 11.935 | +2.6% |

Câu trả lời khá rõ.

Ở horizon 15 phút và 1 giờ, chênh lệch chỉ trong khoảng ba phần trăm, nằm trong nhiễu. Coi như không có tác dụng. Lý do dễ hiểu, vì khi đã có `lag_15min` rồi thì mức tiêu thụ vừa xảy ra đã nói lên nhà máy đang chạy chế độ nào, `Load_Type` không thêm được gì mới.

Ở horizon 24 giờ thì ngược lại hẳn:

```text
MAE   9.12  ->  7.05 kWh
R2    0.596 ->  0.746
RMSE 17.17  -> 13.61
```

Khi mọi lag ngắn đã bị loại vì lý do leakage, model không còn cách nào biết ngày mai nhà máy có chạy hay không. `Load_Type` trả lời đúng câu hỏi đó.

Có một chi tiết lạ. Random Forest được cho thêm thông tin nhưng lại kém đi 4.1%. Em nghĩ là do RF vẫn tối ưu squared error nên bị kéo về giá trị trung bình. Nó phân biệt được chế độ tải nhưng không chuyển được thành MAE tốt hơn.

Chuyện chỉ XGBoost hưởng lợi củng cố nhận định ở mục 3: với dữ liệu lệch thế này thì chọn objective quan trọng ngang chọn feature.

## Trả lời câu hỏi chị đặt ra

> Biết trước kế hoạch tải sản xuất giúp forecasting tốt hơn bao nhiêu?

Không giúp gì ở dự báo ngắn hạn dưới 1 giờ, nhưng giảm khoảng 23% sai số ở dự báo trước 24 giờ.

Nói cách khác, chốt lịch sản xuất sớm chỉ có giá trị khi nhà máy cần dự báo xa. Mà đó cũng là lúc lịch sản xuất khó chốt nhất. Đây là một đánh đổi thật, em nghĩ nên đưa vào phần kết luận.

---

# 6. Ba model em chạy thêm

Bộ Linear, Random Forest, XGBoost trong README là công thức mặc định của gần như mọi tutorial forecasting. Chạy đúng ba cái đó thì phần đóng góp của mình nhìn hơi mỏng.

Ba phần dưới đây trả lời những câu hỏi mà bộ model đó không trả lời được.

## Hybrid hai tầng

Nhìn lại phân bố `Usage_kWh` thì thấy nó bimodal khá rõ:

```text
dưới 10 kWh   : 59.6% quan sát   (nhà máy nghỉ)
30-80 kWh     : phần lớn còn lại (đang sản xuất)
10-20 kWh     : gần như trống
```

Một model hồi quy đơn lẻ bị buộc phải đi qua khoảng trống đó. Nên nó sai nhiều nhất ở đúng các điểm bật và tắt máy, cũng là chỗ sai đắt nhất về mặt vận hành.

Em tách bài toán thành hai tầng:

```text
Tầng 1  phân loại : nhà máy có đang sản xuất không?  -> p
Tầng 2  hồi quy   : hai model riêng cho hai chế độ   -> y_idle, y_active

Dự báo = p * y_active + (1 - p) * y_idle
```

Ngưỡng phân chia lấy từ đáy phân bố, tính chỉ trên tập train, rồi tinh chỉnh trên validation.

| Horizon | Model | Hybrid MAE | Tốt nhất trước đó | Độ chính xác tầng 1 |
|---|---|---:|---:|---:|
| 15 phút | A | 2.690 | 2.874 | 99.0% |
| 1 giờ | B | 4.359 | 4.659 | 96.7% |
| 24 giờ | B | 6.978 | 7.050 | 93.9% |
| 24 giờ | A | 10.018 | 9.122 | 87.3% |

Hybrid thắng ở mọi trường hợp mà tầng phân loại đạt trên 93%.

Dòng cuối là ngoại lệ đáng chú ý. Ở horizon 24h mà không có kế hoạch tải, độ chính xác phân loại tụt xuống 87.3%, và hybrid trở nên kém hơn model đơn lẻ gần 10%.

Cấu trúc hai tầng khuếch đại lỗi của tầng đầu. Đoán sai chế độ thì tầng hai dự báo bằng model sai hoàn toàn.

Em nghĩ chỗ này đáng viết vào báo cáo, vì biết một phương pháp hỏng trong điều kiện nào thì có ích hơn là chỉ biết nó thắng.

## LSTM

Câu hỏi em muốn trả lời: việc tự tay đắp lag feature có thực sự cần thiết, hay mạng nơ-ron tự học được từ chuỗi thô?

Nên em cố tình không cho LSTM bất kỳ lag hay rolling nào. Đầu vào chỉ có:

```text
1. Chuỗi 96 bước Usage_kWh, kết thúc tại t - H
2. Feature lịch của thời điểm t
```

| Horizon | LSTM | Tốt nhất | Kém hơn |
|---|---:|---:|---:|
| 15 phút | 3.590 | 2.690 | 33% |
| 1 giờ | 6.079 | 4.359 | 39% |
| 24 giờ | 8.750 | 6.978 | 25% |

LSTM thua ở cả ba horizon.

Kết quả này không có gì bất thường và em nghĩ nên trình bày đúng như vậy. Với dataset 25 nghìn dòng và cấu trúc lặp theo ngày, theo tuần rất rõ, feature do con người thiết kế đang mã hoá đúng thứ cần thiết. Còn mạng thì phải tự học lại từ đầu với lượng dữ liệu không đủ lớn.

Deep learning không tự động tốt hơn chỉ vì nó hiện đại hơn.

Có một cách đọc khác cũng hay: LSTM là model duy nhất không cần feature engineering, nên khoảng cách 25 đến 39% chính là giá trị định lượng của toàn bộ công sức đắp feature ở mục 2.

## Khoảng dự báo

MAE 7 kWh ở horizon 24 giờ nghĩa là một con số đơn lẻ không đủ để người vận hành ra quyết định. Họ cần biết khoảng.

Em dùng XGBoost với `objective="reg:quantileerror"` để lấy P10, P50, P90.

Nhưng gặp vấn đề: coverage thực tế chỉ 67 đến 79% so với 80% danh nghĩa. Model tự tin quá mức. Đây là hiện tượng đã biết của quantile regression chứ không phải lỗi cài đặt.

Cách sửa em dùng là conformalized quantile regression:

```text
1. Fit các phân vị chỉ trên tập train.
2. Trên validation, đo xem giá trị thực lệch ra ngoài khoảng bao nhiêu.
3. Nới khoảng đúng bằng lượng đó rồi áp cho test.
```

Cách này có bảo đảm lý thuyết về coverage mà không cần giả định gì về phân bố sai số. Tập validation đóng vai calibration set, còn test vẫn không bị đụng tới.

| Horizon | Model | Coverage thô | Sau hiệu chỉnh | Nới thêm |
|---|---|---:|---:|---:|
| 15 phút | A | 77.6% | 81.5% | 0.07 kWh |
| 1 giờ | B | 71.6% | 82.5% | 0.27 kWh |
| 24 giờ | B | 79.4% | 85.2% | 0.44 kWh |

Ngoài ra P50 của model này đạt MAE 6.935 ở horizon 24h, tốt ngang nhóm dẫn đầu. Nên nó vừa là khoảng dự báo vừa là dự báo điểm dùng được.

---

# 7. Vế tối ưu hoá

Tên đề tài là "Dự báo và tối ưu hoá điện năng tiêu thụ", nhưng cả project mới chỉ làm vế dự báo. Em làm thêm một bản demo cho vế còn lại.

Phần này không phải model. Nó là một tầng riêng nằm sau dự báo:

```text
dự báo 24h        ->   optimize.py          ->   lịch dịch chuyển tải
(model học máy)        (quy hoạch tuyến tính)
```

## Mô hình

Giá điện sản xuất của Việt Nam chia ba khung giờ theo Thông tư 16/2014/TT-BCT:

```text
Cao điểm     T2-T7  09:30-11:30 và 17:00-20:00   (Chủ nhật không có)
Thấp điểm    mọi ngày  22:00-04:00
Bình thường  phần còn lại
```

Giả định vận hành: 30% điện năng trong ngày là dịch chuyển được, ví dụ nấu luyện theo mẻ, sấy, nạp lò. Phần còn lại là tải nền bắt buộc. Trần công suất bằng đỉnh phụ tải thực tế, tức không được tạo đỉnh mới.

Mỗi ngày giải một bài quy hoạch tuyến tính xếp lịch tải rẻ nhất.

> Đơn giá trong code là giá tham khảo để minh hoạ, không phải biểu giá hiện hành. Trước khi đưa vào báo cáo phải thay bằng biểu giá EVN đang áp dụng cho cấp điện áp của nhà máy. Cấu trúc ba khung giờ thì ổn định, chỉ đơn giá là thay đổi.

## Kết quả

Trên tháng 12/2018:

```text
tiết kiệm chi phí điện   khoảng 15%
giảm đỉnh tiêu thụ       khoảng 18%
```

Nhưng chỗ thú vị hơn không nằm ở hai con số đó.

Khi so các model với nhau, mọi model đều đạt 100% lợi ích của kế hoạch lý tưởng. Kể cả model dở.

Lý do là khi nhà máy còn nhiều dư địa công suất thì lịch tối ưu gần như luôn là dồn hết vào giờ thấp điểm, gần như không phụ thuộc dự báo.

Nên em quét thử trần công suất từ chặt đến rộng:

| Trần công suất | Naive | Seasonal naive | XGBoost A | XGBoost B |
|---|---:|---:|---:|---:|
| 45% đỉnh | 92.1% | 96.0% | 94.6% | 92.9% |
| 55% đỉnh | 91.0% | 97.7% | 99.7% | 99.0% |
| 65% đỉnh | 89.7% | 97.8% | 100% | 100% |
| 80% đỉnh | 92.9% | 98.0% | 100% | 100% |
| 100% đỉnh | 95.9% | 98.7% | 100% | 100% |

Số trong bảng là phần lợi ích giữ được so với kế hoạch lý tưởng.

Ở vùng 55 đến 65% đỉnh, khoảng cách giữa dự báo tốt và naive là 9 đến 10% lợi ích. Đó là vùng mà đầu tư vào model có lãi. Ngoài vùng đó thì gần như không.

Em nghĩ đây là kết luận trung thực và đáng giá hơn so với việc chỉ nói model tiết kiệm được 15% chi phí, vì nó chỉ rõ điều kiện để con số đó có ý nghĩa.

---

# 8. Bảng xếp hạng chung

Tổng cộng 9 họ model, mỗi họ chạy trên 3 horizon và 2 biến thể A/B, thành 52 lần chạy.

| Horizon | Model tốt nhất | MAE | Baseline tốt nhất | Cải thiện |
|---|---|---:|---:|---:|
| 15 phút | Hybrid hai tầng A | 2.690 | 3.819 | -29.6% |
| 1 giờ | Hybrid hai tầng B | 4.359 | 9.022 | -51.7% |
| 24 giờ | Quantile P50 B | 6.935 | 10.560 | -34.3% |

Bảng đầy đủ nằm ở `results/metrics/leaderboard.csv`.

---

# 9. Mấy điều em nghĩ nên đưa vào báo cáo

Khoảng cách với baseline mới là kết quả, không phải con số R² tuyệt đối. Ở horizon 15 phút, naive persistence đã đạt MAE 3.82. Mọi model phải so với mốc đó chứ không phải so với 0. Một model R² bằng 0.94 nghe rất ấn tượng nhưng thực ra chỉ hơn naive khoảng 30%.

Độ khó tăng rất nhanh theo horizon. MAE của model tốt nhất đi từ 2.69 lên 4.36 rồi 6.94 kWh. Dự báo trước 24 giờ vẫn dùng được để lên kế hoạch, nhưng sai số đủ lớn để nên dùng khoảng tin cậy chứ không phải một con số điểm.

Linear Regression kém ở mọi horizon. Quan hệ giữa thời gian và mức tiêu thụ là dạng bậc thang, nhà máy chạy hoặc không chạy, không tuyến tính. Nhưng nó lại có RMSE khá tốt ở horizon 24h là 17.68, nghĩa là nó ít bị lệch lớn, chỉ sai đều ở mọi điểm. Chi tiết này cũng đáng nhắc.

MAPE không đáng tin trên dataset này. Median chỉ 4.57 kWh và có một observation bằng 0, nên MAPE bị thổi lên 120 đến 200% dù model không hề tệ. Trong báo cáo nên dùng sMAPE và MASE, MAPE chỉ để tham khảo.

Một điểm sai đặc trưng, xem `results/figures/forecast_24h_A.png`: ngày 02/12 model dự báo một đợt sản xuất mà thực tế không xảy ra. Model học pattern lịch nên không biết lịch dừng máy đột xuất. Sang Model B thì chỗ này bám sát hơn hẳn, đúng như kỳ vọng.

---

# 10. Cấu trúc code em thêm

```text
src/
├── config.py       đường dẫn, hằng số, định nghĩa horizon
├── features.py     tạo feature theo horizon, đảm bảo không leakage
├── evaluate.py     MAE, RMSE, MAPE, sMAPE, R2, MASE và biểu đồ
├── train.py        Model A/B: Linear, Random Forest, XGBoost, baseline
├── hybrid.py       hybrid hai tầng
├── deep.py         LSTM
├── quantile.py     khoảng dự báo P10-P90 và hiệu chỉnh conformal
├── optimize.py     dịch chuyển tải theo giá điện
├── summarize.py    gộp mọi kết quả thành một bảng xếp hạng
└── verify.py       kiểm chứng lại: leakage, split, đối chiếu README

results/
├── metrics/        leaderboard.csv và metrics từng phần
├── predictions/    dự báo của từng model trên tập test
└── figures/        biểu đồ so sánh, dự báo, residual, feature importance
```

Chị có nhắc `src/config.py` trong README nhưng file đó chưa tồn tại trong repo, nên em viết bổ sung.

Ngoài ra `requirements.txt` đang lưu ở encoding UTF-16 nên `pip install -r` sẽ lỗi. Em chuyển sang UTF-8 và thêm xgboost, scipy, torch.

## Cách chạy

```bash
pip install -r requirements.txt

python src/train.py       # Model A/B, 3 horizon       khoảng 22 phút trên 2 core
python src/hybrid.py      # hybrid hai tầng            khoảng 6 phút
python src/quantile.py    # khoảng dự báo              khoảng 8 phút
python src/deep.py        # LSTM                       khoảng 25 phút
python src/optimize.py    # demo tối ưu hoá            khoảng 2 phút
python src/summarize.py   # gộp thành bảng xếp hạng
python src/verify.py      # kiểm chứng không có leakage
```

Muốn chạy riêng một horizon:

```bash
python src/train.py 24h
python src/train.py 15min 1h
```

## Quy trình train

```text
train       T1-T9    ->  fit model
validation  T10-T11  ->  chọn hyperparameter, không đụng test
test        T12      ->  đánh giá một lần duy nhất
```

Sau khi chốt hyperparameter trên validation, em fit lại model trên train cộng validation rồi mới dự báo test. Làm vậy model được học thêm giai đoạn tháng 10 và 11, vốn gần tháng 12 hơn.

Nếu chị muốn bám đúng nguyên văn README thì đặt `REFIT_ON_TRAIN_VAL = False` trong `src/train.py`.

---

# 11. Em đã kiểm chứng những gì

Chạy `python src/verify.py`, pass toàn bộ:

```text
✓ Với 200 dòng ngẫu nhiên mỗi horizon, tính lại từng feature bằng tay
  từ chuỗi Usage_kWh gốc, xác nhận mọi giá trị được dùng đều nằm tại
  t - H trở về trước
✓ Feature em dựng lại khớp 100% với data/processed sẵn có, cả 34.368 dòng
✓ Split đúng 25.536 / 5.856 / 2.976 và ba khoảng không overlap
✓ Baseline persistence đúng bằng giá trị thực dịch đi H bước
```
