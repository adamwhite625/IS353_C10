# Bàn giao kỹ thuật PN1

## Đầu vào chính thức cho cuối kỳ

- **BA-Shapes** dùng đúng dữ liệu T06 đã khôi phục từ ZIP gốc, gồm đặc trưng và split gốc.
- **Cora** dùng dữ liệu khớp manifest T06 và checkpoint được huấn luyện mới. Checkpoint `gcn_s1000.pt` là mốc cố định. Lưới đầy đủ dùng cả năm seed.
- Gói `PN1_H1.zip` có manifest để kiểm tra SHA-256 trước khi chạy. Quyết định này áp dụng cho lượt cuối kỳ. Biên bản GATE 0 ghi riêng các đối chiếu với checkpoint giữa kỳ.

## H1 chuyển cho PN2

- Dữ liệu chính ở `results/data/{bashapes,cora}/data.pt`. Cả hai tệp khớp manifest T06.
- Checkpoint ở `results/checkpoints/{dataset}/gcn_s{seed}.pt`, kèm JSON và `selection.json`. Cả hai dataset dùng seed 1000–1004, với checkpoint seed 1000 làm mốc cố định.
- BA-Shapes seed 1000 và 1001 đạt recall lớp 3 là 81,25%, dưới mục tiêu 85% nhưng trên ngưỡng thay seed 50%. Cả hai vượt ngưỡng accuracy tối thiểu. Giữ đủ năm seed theo kế hoạch.
- Danh sách node test và hash dữ liệu nằm ở `data/{dataset}/test_nodes.json`. BA-Shapes có 140 node, Cora có 100 node.
- Kiểm tra gói đã giải nén bằng `python scripts/download_drive_assets.py --root /path/to/extracted/H1 --manifest /path/to/extracted/H1/manifest.json`. Gói giữ các đường dẫn tương đối từ thư mục gốc repo.
- Giữ nguyên thứ tự cột cạnh và dùng `target_class = pred[node]` của checkpoint. Đầu ra H2 cần hash checkpoint khớp JSON tương ứng và các cột theo kế hoạch.

## T2–T3 chuyển cho PN3

Dùng `results/analysis/a3_t2_t3/t2_t3_by_node.csv`, ghép với đầu ra A3 theo
`dataset, gnn_seed, node_id, checkpoint_sha256`. File chứa hai loại can thiệp
trên mỗi node test và mỗi checkpoint, gồm Drop với đặc trưng cố định (T2),
Drop sau tính lại bậc (T3), số cặp cạnh bị xóa và phần chồng lấn. Hai can thiệp
có ngân sách khác nhau. Trong 225 phép can thiệp không xóa cạnh, chênh lệch T3 bằng 0.
Node lịch sử 692 thuộc validation và không nằm trong bảng test. Cohort node mâu thuẫn
và case study chính thức cần đầu ra A3 của PN3.

## Mốc nghiệm thu

- GATE 0 đã kiểm tra hash, dự đoán, bảng T06 và đối chiếu điểm Saliency/GNNExplainer seed 42.
- Phần checkpoint, danh sách test và smoke của PN1 đã có. Kiểm tra H2 với nhánh PN2
  tại `7736114` pass trên 2 node mỗi dataset. GATE 1 toàn nhóm vẫn cần tích hợp nhánh PN2
  và bộ đánh giá v2 của PN3. GATE 2 còn cần bảng A2/A3. Các thí nghiệm tùy chọn
  Tầng B chưa được mở.
