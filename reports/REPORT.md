# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Thị Thùy Linh

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Camera giám sát đặt cố định trên cao tốc, góc nhìn gần như không đổi suốt toàn bộ video.
Một chiếc xe di chuyển qua khung hình mất khoảng 2–4 giây, nghĩa là có thể xuất hiện trong
hàng chục frame liên tiếp với hình dạng, vị trí và ánh sáng gần như giống hệt nhau.

Nếu chia tập ngẫu nhiên (random split), rất có thể frame thứ 100 của chiếc xe đó nằm trong tập
train và frame thứ 102 nằm trong tập test. Khi đó mô hình đã "nhìn thấy" xe này lúc huấn luyện —
nó chỉ đang ghi nhớ xe quen (memorisation) chứ không học được cách phát hiện xe lạ. Điểm AP50
đo được trên test lúc đó bị ảo và cao hơn thực tế (data leakage). Bài lab khắc phục bằng cách
chia theo trục thời gian (temporal split): pool lấy từ phần đầu và giữa video, tập test lấy
từ phần cuối, có thêm vùng đệm thời gian ở giữa để đảm bảo không frame nào của cùng một xe
vượt biên. Cách chia này phản ánh đúng điều kiện triển khai thực tế: mô hình sẽ phải nhận dạng
xe mới ở thời điểm tương lai mà nó chưa từng thấy trong quá trình học.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `rounds_table.md` (tập test: 20 ảnh, 403 box tham chiếu, bỏ qua 14 box dưới 16 px):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Mô hình cold start là YOLOv8n đã được pre-train trên COCO với các lớp car, bus và truck, không
qua bất kỳ bước fine-tune nào trên dữ liệu đêm của bài lab. Precision đạt 0.925 — những gì nó
khoanh thường là xe thật — nhưng recall chỉ 0.489, tức là bỏ sót tới 51% số xe (FN = 206 trên
403 box tham chiếu, theo `outputs/metrics_round0.json`).

Phân tích theo kích thước cho thấy khoảng cách rõ rệt: recall xe nhỏ chỉ đạt **0.182** (66 box
tham chiếu), trong khi xe trung và lớn lần lượt là 0.547 và 0.561. Nguyên nhân: ban đêm, xe ở
xa chỉ còn hai chấm đèn nhỏ trên nền tối, đặc trưng hình dạng thân xe gần như biến mất, khiến
mô hình COCO — vốn quen với ảnh ban ngày rõ nét — không nhận ra. Quan sát trực quan từ
`outputs/compare_round0.jpg` xác nhận: nhiều xe ở làn đối diện và xe ở cuối tầm nhìn bị bỏ
sót hoàn toàn (không có box), trong khi FP = 16 cho thấy mô hình thỉnh thoảng khoanh nhầm
vào vệt đèn pha rọi trên mặt đường hoặc biển báo phát sáng.

Lưu ý quan trọng về chất lượng nhãn tham chiếu: nhãn tập test được tạo tự động bởi mô hình
tham chiếu, chưa được rà soát thủ công toàn bộ. Theo `reports/BLIND_SCAN.MD`, chỉ riêng
frame_0099.jpg người quan sát đếm được 25 xe bằng mắt thường, trong đó "2 xe bị khuất bên
trái chỉ nhìn thấy 1 phần đèn" và "2 xe ở góc xa thiếu sáng" — những xe này có thể đã bị bỏ
qua trong nhãn tham chiếu. Do đó, điểm AP50 thấp chưa hẳn hoàn toàn do lỗi của mô hình; một
phần có thể do nhãn tham chiếu chưa đầy đủ.

## 3. Chiến lược chọn mẫu

Thuật toán chọn mẫu tính điểm ưu tiên cho mỗi frame theo công thức:

```
score = W_U · U  +  W_A · A  +  W_D · D
```

Trong đó:
- **U — Uncertainty (50%)**: Đo mức độ mô hình không chắc chắn. Frame có nhiều box với
  confidence score thấp (gần ngưỡng phân loại) sẽ có U cao — đây là những vùng mô hình đang
  "phân vân" nhất và sẽ học được nhiều nhất nếu được sửa nhãn đúng.
- **A — Ambiguity (30%)**: Tỉ lệ box mơ hồ trên tổng số box (`n_ambiguous / n_boxes`). Frame
  có nhiều box chồng lấp, độ tin cậy lưỡng lự hoặc không rõ ranh giới lớp sẽ có A cao.
- **D — Diversity (20%)**: Phần thưởng cho frame có cảnh quay chưa được đại diện trong lô đã
  chọn — ưu tiên mẫu nằm xa về mặt thời gian so với các frame đã được chọn trước đó.

**Điều kiện lọc trùng (MIN_GAP_S)**: Hai frame được chọn trong cùng một lô phải cách nhau tối
thiểu một khoảng thời gian quy định. Nếu camera gần như đứng yên, hai frame cách nhau dưới 2
giây cho cùng một cảnh đường phố — việc sửa thêm frame đó tốn công ngang bằng sửa một frame
mới nhưng mô hình không học thêm được đặc trưng mới nào.

**Ba frame được chọn** (từ `reports/SELECTION.md`):
1. **frame_0182.jpg** — hạng 1, score = 0.9591 (t = 72.8 s): U cao nhất (0.9182), 18/28 box
   mơ hồ — cảnh nhiều xe xếp hàng, ánh đèn pha giao thoa phức tạp.
2. **frame_0369.jpg** — hạng 2, score = 0.9324 (t = 147.6 s): n_ambiguous = 16/43 box — cảnh
   đông xe nhất trong lô, nhiều xe ở xa chỉ thấy đèn, đã thêm 24 box sau khi sửa.
3. **frame_0326.jpg** — hạng 4, score = 0.9155 (t = 130.4 s): đã xóa 2 box FP (AI vẽ nhầm
   vệt đèn mặt đường) và thêm 21 box FN xe bị che khuất.

**Một frame bị loại dù điểm cao**: **frame_0372.jpg** (hạng 6, score = 0.9101, t = 148.8 s)
nằm kẹp giữa frame_0369 (t = 147.6 s) và frame_0380 (t = 152.0 s), cả ba cách nhau dưới 5
giây. Diversity filter loại frame_0372 vì cảnh quay gần như trùng nhau — sửa thêm sẽ không
mang lại thông tin học mới.

**Lưu ý quan trọng**: Điểm bất định cao chỉ cho thấy mô hình đang bối rối nhiều nhất tại
những frame đó, không đảm bảo rằng việc sửa nhãn chắc chắn sẽ làm tăng AP50. Nếu frame bất
định cao xuất phát từ cảnh quay đặc biệt hiếm gặp hoặc lỗi nhãn không đại diện cho lỗi phổ
biến của mô hình, hiệu quả cải thiện sẽ rất hạn chế.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp các vòng (từ `rounds_table.md`):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 331 | 0.413 | −0.359 | 1.000 | 0.052 | 0.099 | 0.000 | 0.024 | 0.342 |

**Thống kê sửa nhãn vòng 1** (từ `outputs/round1_diff.md`):

Model đề xuất **169 box** trên 12 ảnh. Sau khi sửa còn **331 box** — tăng gần gấp đôi. Chi tiết:

| Hành động | Số box | Ý nghĩa |
|---|---:|---|
| Accepted (giữ nguyên) | 151 (89%) | Box AI đúng, không cần chỉnh |
| Edited (kéo chỉnh) | 10 | Box lệch viền, chưa ôm sát thân xe/gương/đèn |
| Deleted (xóa) | 8 | FP: AI khoanh nhầm vệt đèn đường, biển báo |
| Added (vẽ thêm) | 170 (51%) | FN: AI bỏ sót xe ở xa, xe bị che, xe mép ảnh |

**Phân tích thay đổi AP50**: Sau fine-tune vòng 1, AP50 giảm từ 0.771 xuống 0.413 (Δ = −0.359).
Đây là kết quả của sự đánh đổi rõ ràng: Precision tăng lên **1.000** — mô hình không còn
khoanh nhầm vệt đèn hay biển báo — nhưng Recall sụt thê thảm từ 0.489 xuống **0.052**. Mô hình
trở nên quá thận trọng: nó chỉ dự đoán box khi rất chắc chắn, bỏ sót gần như toàn bộ xe
trên tập test (FN tăng mạnh). Recall xe nhỏ = **0.000** — mô hình hoàn toàn không nhận ra xe
nhỏ ở xa sau fine-tune. Xe lớn recall = 0.342, xe trung = 0.024.

Nguyên nhân có thể: (a) chỉ 12 ảnh train quá ít so với 331 box mới — tỉ lệ box/ảnh cao (27,6
box/ảnh) khiến mô hình overfit vào đặc trưng cảnh đêm phức tạp của lô này; (b) learning rate
hoặc số epoch fine-tune quá lớn, khiến mô hình "quên" đặc trưng COCO ban đầu (catastrophic
forgetting); (c) ngưỡng conf = 0.25 có thể không còn phù hợp sau fine-tune.

**Phân biệt ba góc nhìn**:
1. *Mắt người thấy gì ban đầu* (`BLIND_SCAN.MD`): frame_0099.jpg có 25 xe nhìn thấy bằng mắt
   thường trước khi xem pre-label, trong đó "2 xe bị khuất bên trái chỉ nhìn thấy 1 phần đèn"
   và "2 xe ở góc xa thiếu sáng" — đây là quan sát độc lập, chưa bị ảnh hưởng bởi gợi ý AI.
2. *Thao tác sửa nhãn* (`REVIEW_LOG.csv`): 5 ca được ghi lại gồm xóa box FP vệt đèn mặt đường
   (frame_0326), thêm box xe bị che khuất (frame_0326, frame_0369), chỉnh box lệch gương/đèn
   (frame_0099), và giữ nguyên box đúng (frame_0187) — đây là can thiệp của người gán nhãn.
3. *Hành vi mô hình sau fine-tune*: AP50 giảm, P = 1.000, R = 0.052 — mô hình phản ứng với
   dữ liệu mới theo hướng bảo thủ hơn, giảm FP nhưng tăng FN đáng kể.

## 5. Kết luận và giới hạn

**Kết quả vòng 1 so với cold start**: AP50 giảm −0.359, Recall giảm từ 0.489 xuống 0.052.
Tuy nhiên không nên kết luận vội rằng fine-tune thất bại: cold start dùng toàn bộ tri thức
COCO (nhiều ảnh ban ngày rõ nét), trong khi vòng 1 chỉ fine-tune trên 12 ảnh đêm. Với dữ liệu
nhiều hơn và chiến lược fine-tune cẩn thận hơn, kết quả hoàn toàn có thể cải thiện.

**Quyết định: tiếp tục vòng 2** — với điều chỉnh chiến lược:
- Tăng số ảnh train (mục tiêu 24–36 ảnh) để giảm nguy cơ overfit.
- Ưu tiên chọn frame có nhiều xe nhỏ ở xa (R_small = 0.000 sau vòng 1) và xe bị cắt mép ảnh —
  hai nhóm xe yếu nhất hiện tại.
- Kiểm tra và giảm learning rate hoặc số epoch fine-tune để tránh catastrophic forgetting.
- Xem xét lại ngưỡng conf sau fine-tune: nếu mô hình quá thận trọng, hạ ngưỡng từ 0.25 xuống
  0.15–0.20 trước khi kết luận Recall thấp.

**Hai điểm yếu cốt lõi cần ưu tiên ở vòng 2**:
1. **Xe nhỏ ở xa (R_small = 0.000)**: Cần bổ sung frame từ vùng t = 60–70 s và t = 100–110 s
   chưa được khai thác. Lưu ý tránh frame cách nhau dưới 2 giây (ví dụ frame_0370 tại t=148.0s
   cách frame_0369 tại t=147.6s chỉ 0,4 giây — chi phí gán nhãn cao, lợi ích gần như bằng không).
2. **Xe bị cắt ở mép khung hình**: Theo guideline, chỉ vẽ box cho phần nằm trong ảnh — nhưng
   nhiều pre-label của AI hoặc bỏ sót hoàn toàn hoặc vẽ box tràn ra ngoài biên. Cần thêm ảnh
   có xe mép trái/phải để mô hình học đặc trưng này.

**Giới hạn của kết quả hiện tại**:
- *Tập test nhỏ (20 ảnh)*: Biến động AP50 lớn; một vài frame sai/đúng thêm có thể làm lệch
  kết quả vài điểm phần trăm — chưa đủ ổn định thống kê để kết luận chắc chắn.
- *Nhãn tham chiếu chưa rà thủ công*: Nhãn test do mô hình tham chiếu tạo, có thể còn FN;
  điểm AP50 thấp chưa chắc hoàn toàn do lỗi của mô hình fine-tune.
- *Luật bỏ qua box dưới 16 px*: 14 box tham chiếu bị loại; nếu mô hình phát hiện được xe
  rất nhỏ này, chúng sẽ bị tính là FP — bóp méo chỉ số Precision.

**Quy trình QC khi AP50 giảm**: Trước khi nạp thêm dữ liệu, cần tự kiểm tra theo thứ tự:
(a) Rà lại các box đã sửa trong `REVIEW_LOG.csv` xem có box nào vẽ quá khắt khe hoặc quá
rộng không; (b) Xem histogram confidence của model sau fine-tune để xem ngưỡng conf = 0.25
còn phù hợp không; (c) Giảm learning rate / số epoch rồi fine-tune lại trước khi thêm dữ liệu.
