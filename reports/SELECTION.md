# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box: Nếu chỉ có ngân sách rà đúng 5 ảnh,
tôi sẽ chọn 5 frame đứng đầu bảng xếp hạng theo điểm bất định: **frame_0182.jpg** (hạng 1,
score = 0.9591, t = 72.8 s), **frame_0369.jpg** (hạng 2, score = 0.9324, t = 147.6 s),
**frame_0380.jpg** (hạng 3, score = 0.9170, t = 152.0 s), **frame_0326.jpg** (hạng 4,
score = 0.9155, t = 130.4 s) và **frame_0331.jpg** (hạng 5, score = 0.9154, t = 132.4 s).
Lý do ưu tiên: 5 frame này có điểm U (uncertainty) cao nhất, cho thấy mô hình đang phân vân
nhiều nhất và khả năng sửa nhãn sẽ mang lại thông tin học mới lớn nhất. Tuy nhiên cần lưu ý:
frame_0326.jpg (t = 130.4 s) và frame_0331.jpg (t = 132.4 s) chỉ cách nhau 2 giây — hai cảnh
gần như giống nhau. Nếu muốn tránh trùng lặp, có thể thay frame_0331.jpg bằng frame_0312.jpg
(hạng 7, score = 0.9100, t = 124.8 s) để đảm bảo đa dạng cảnh quay hơn trong ngân sách 5 ảnh.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet: Ba frame tiêu
biểu trong lô 12 ảnh được chọn (selected = True) với điểm số cao nhất là **frame_0182.jpg**
(score = 0.9591, n_ambiguous = 18/28 box), **frame_0369.jpg** (score = 0.9324,
n_ambiguous = 16/43 box) và **frame_0380.jpg** (score = 0.9170, n_ambiguous = 15/40 box). Đặc
điểm chung của ba frame này: tỉ lệ box mơ hồ trên tổng số box cao (trên 37%), chứng tỏ mô hình
đang đề xuất nhiều khung với độ tin cậy thấp trên cùng một cảnh đêm phức tạp — nhiều xe xếp
thành hàng, ánh đèn pha giao thoa, xe bị che khuất một phần. Bằng chứng trong CSV: cột U đều
trên 0.91 và cột n_ambiguous đều lớn hơn 15, khớp với vị trí hiển thị trên ảnh contact sheet
`outputs/selection_round1.jpg`.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
**frame_0372.jpg** (hạng 6, score = 0.9101, t = 148.8 s) có điểm cao hơn 6 trong số 12 frame
được chọn vào lô, nhưng cột selected = False. Lý do: bộ lọc đa dạng thời gian (diversity
filter) của thuật toán loại bỏ những frame quá gần nhau về mặt thời gian — frame_0372.jpg
(t = 148.8 s) nằm kẹp giữa frame_0369.jpg (t = 147.6 s) và frame_0380.jpg (t = 152.0 s), cả
ba cách nhau chưa đến 5 giây, cảnh quay gần như trùng nhau. Nếu đưa frame_0372.jpg vào sửa,
mô hình sẽ nhận thêm dữ liệu của cùng một cảnh đường phố đó mà không học thêm được đặc trưng
mới — chi phí gán nhãn tốn thêm nhưng lợi ích huấn luyện gần như bằng không.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: Điểm bất định (uncertainty score)
cao chỉ phản ánh mức độ mô hình đang phân vân nhiều nhất ở những frame đó, chứ không tự bảo
đảm rằng việc sửa nhãn các ảnh này chắc chắn sẽ giúp mô hình tăng AP50 sau khi fine-tune. Có
thể mô hình phân vân vì cảnh đặc biệt khó (xe bị mờ, ánh sáng ngược) vốn ít xuất hiện trong
tập kiểm thử, hoặc lỗi nhãn ở những frame đó có thể không đại diện cho lỗi phổ biến nhất mà
mô hình mắc phải trên toàn bộ video. Để xác nhận hiệu quả thực sự cần so sánh AP50 trước và
sau khi fine-tune trên tập blind test độc lập.
