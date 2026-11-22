# GATE 0 — kiểm tra khả năng tái lập giữa kỳ

Ngày 22/11/2026, xác minh hai bộ dữ liệu và hai checkpoint gốc theo [manifest T06](../notebooks/05_duc_kha_evaluation/results/manifest.json).

- ZIP `data_goc_giuaki_BAShapes_Cora.zip` cung cấp bốn tệp BA-Shapes và checkpoint Cora, với năm SHA-256 khớp trực tiếp.
- ZIP `02_Du_lieu.zip` cung cấp ba tệp Cora còn lại. `data.pt` và `val_nodes.json` khớp trực tiếp. Khôi phục xuống dòng CRLF của `config.json` và CSV điểm T05 để khớp hash lưu trong T06.
- Nạp lại checkpoint trên CPU cho đúng **700/700 dự đoán BA-Shapes** và **2.708/2.708 dự đoán Cora**.
- Dùng CSV điểm T05 để chạy lại T06. Thu được 2.080 dòng metric, 6.080 dòng agreement và 0 lỗi. Bảng tổng hợp metric và agreement khớp đến **4 chữ số thập phân**.

Chạy lại Saliency và GNNExplainer với seed 42 trên 20 node validation gốc của mỗi dataset, dùng script kiểm tra của PN2 tại commit `7736114`. Sai lệch tuyệt đối lớn nhất so với điểm thô đã lưu:

| Dataset | Saliency | GNNExplainer |
| --- | ---: | ---: |
| BA-Shapes | 8,82 × 10⁻⁷ | 5,75 × 10⁻⁷ |
| Cora | 6,64 × 10⁻⁷ | 8,23 × 10⁻⁷ |

Checkpoint gốc dùng để đối chiếu giữa kỳ. Giao thức cuối kỳ dùng lại dữ liệu đã xác minh, với các checkpoint đa seed huấn luyện mới. Xem [bằng chứng kiểm tra](../results/gate0/verification.json).
