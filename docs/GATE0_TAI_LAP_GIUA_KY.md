# GATE 0 — kiểm tra khả năng tái lập giữa kỳ

Đối chiếu file bàn giao với SHA-256 trong [manifest T06](../notebooks/05_duc_kha_evaluation/results/manifest.json).

| Đầu vào | Kết quả |
| --- | --- |
| BA-Shapes giữa kỳ | Bộ gốc đã mất. File ZIP chứa biến thể chuẩn hóa đặc trưng và chia theo motif, với dữ liệu, cấu hình và danh sách validation khác manifest. |
| Cora giữa kỳ | `data.pt` và `val_nodes.json` khớp manifest. `config.json` trong ZIP khác hash, và ZIP thiếu `gcn_ckpt.pt`. |
| CSV điểm T05 | Nội dung trong Git khớp SHA-256 của T06 sau khi khôi phục xuống dòng CRLF. |

BA-Shapes tái dựng bằng PyG với seed 0 cho 700 node và 3.972 cột cạnh. Split mới phân tầng theo nhãn node, tỷ lệ 60/20/20. Chỉ 17/100 node trong danh sách validation BA-Shapes cũ thuộc tập validation mới. Cấu hình mới và kết quả đo được lưu cùng checkpoint cuối kỳ.

GATE 0 còn thiếu xác nhận dự đoán của checkpoint gốc, đối chiếu bảng T06 đến bốn chữ số và so sánh điểm Saliency/GNNExplainer seed 42 với scores cũ. Không có số `max_diff` để báo cáo từ đúng bộ gốc. PGExplainer giữa kỳ còn có lỗi gieo seed sau khởi tạo MLP như phần F1 của kế hoạch nêu. Theo quy tắc GATE 0, công việc cuối kỳ tiếp tục với lượt thực nghiệm mới.

```bash
python scripts/download_drive_assets.py --root /path/to/historical/assets
```
