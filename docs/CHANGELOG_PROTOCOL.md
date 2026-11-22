# Giao thức cuối kỳ

- 20/11/2026 — PN1. Tái dựng BA-Shapes vì dữ liệu giữa kỳ đã mất và bản trên Drive dùng tiền xử lý khác. Bản mới dùng PyG BAGraph(300, 5), 80 HouseMotif, seed đồ thị 0, đặc trưng bậc không chuẩn hóa và split phân tầng hai bước với seed 0. Đây là lựa chọn triển khai cho phần thuật toán chia split chưa mô tả chi tiết trong kế hoạch.
- 20/11/2026 — PN1. Dùng cấu hình YAML làm nguồn train. Với BA-Shapes, notebook 01 là căn cứ cho 600 epoch, weight decay 0,0001, trọng số lớp tính từ train và chọn checkpoint theo macro-F1 validation. Metadata JSON giữa kỳ lưu riêng để kiểm tra nguồn T06.
- 20/11/2026 — PN1. Dùng bản Cora có SHA-256 khớp T06. Không có checkpoint Cora giữa kỳ trong file bàn giao. Checkpoint cuối kỳ được huấn luyện mới.
- 20/11/2026 — PN1. Smoke dùng 1 seed GNN, 5 node có cạnh ứng viên mỗi dataset và ít epoch. Lượt đầy đủ dùng các entry point bàn giao của PN2/PN3. Đánh dấu giao thức chính thức sau khi các phần F1–F8 qua GATE 1.
- 20/11/2026 — Vũ xác nhận BA-Shapes tái dựng và checkpoint Cora huấn luyện mới là đầu vào chính thức cho cuối kỳ. Dữ liệu Cora giữ bản khớp manifest T06.
- 22/11/2026 — Theo xác nhận của Vũ, thay BA-Shapes tái dựng bằng đúng BA-Shapes giữa kỳ từ `data_goc_giuaki_BAShapes_Cora.zip`. Khóa SHA-256 của dữ liệu, giữ split gốc, tạo lại danh sách test và huấn luyện lại đủ hai bộ checkpoint. File Cora giữ SHA-256 cũ vì đã đúng bản T06. Checkpoint trong ZIP chỉ dùng để kiểm tra giữa kỳ.
