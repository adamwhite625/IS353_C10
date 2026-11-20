# Bàn giao kỹ thuật PN1

## H1 chuyển cho PN2

- Dữ liệu chính ở `results/data/{bashapes,cora}/data.pt`. BA-Shapes dựng lại và Cora khớp manifest T06.
- Checkpoint ở `results/checkpoints/{dataset}/gcn_s{seed}.pt`, kèm JSON và `selection.json`. Cả hai dataset dùng seed 1000–1004, với checkpoint seed 1000 làm mốc cố định.
- Danh sách node test và hash dữ liệu nằm ở `data/{dataset}/test_nodes.json`. BA-Shapes có 140 node, Cora có 100 node.
- Kiểm tra gói đã giải nén bằng `python scripts/download_drive_assets.py --root /path/to/extracted/H1 --manifest /path/to/extracted/H1/manifest.json`. Gói giữ các đường dẫn tương đối từ thư mục gốc repo.
- Giữ nguyên thứ tự cột cạnh và dùng `target_class = pred[node]` của checkpoint. Đầu ra H2 cần hash checkpoint khớp JSON tương ứng và các cột theo kế hoạch.

## T2–T3 chuyển cho PN3

Dùng `results/analysis/a3_t2_t3/t2_t3_by_node.csv`, ghép với đầu ra A3 theo
`dataset, gnn_seed, node_id, checkpoint_sha256`. File chứa hai loại can thiệp
trên mỗi node test và mỗi checkpoint, gồm Drop với đặc trưng cố định (T2),
Drop sau tính lại bậc (T3), số cặp cạnh bị xóa và phần chồng lấn. Hai can thiệp
có ngân sách khác nhau. Node lịch sử 692 không thuộc tập test mới nên nằm ngoài
bảng này. Cohort node mâu thuẫn và case study chính thức cần đầu ra A3 của PN3.

## Mốc nghiệm thu

- GATE 0 còn bị chặn bởi asset lịch sử, theo biên bản riêng.
- Phần checkpoint, danh sách test và smoke của PN1 đã có. GATE 1 toàn nhóm còn
  cần F1–F4/F8 từ PN2/PN3 và kiểm thử pipeline chính thức. GATE 2 còn cần bảng A2/A3
  từ hai nhóm đó. Hiện trạng chưa mở các thí nghiệm tùy chọn Tầng B.
