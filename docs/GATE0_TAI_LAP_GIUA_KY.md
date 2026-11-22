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

---

## Phụ lục — Chi tiết Bước 3 và phát hiện PGExplainer (Khôi, PN2)

> Môi trường: Windows, Python 3.11.6, PyTorch `2.14.0+cpu`, PyG `2.8.0.post1`, CPU. Commit đối chiếu `cf55fb3`.
> Lệnh: `python scripts/verify_gate0_khoi.py --seed 42 --n-nodes 20 --check-pg-f1 --pg-epochs 40`

### A.1. Saliency và GNNExplainer, seed 42, trước và sau F8

GNNExplainer: `epochs=200, lr=0.01, coeff_size=0.005, coeff_ent=0.1, size_loss_mode='mean'`. Đối chiếu từng cạnh `(node_id, edge_id, u, v)` với `notebooks/01_midterm/edge_scores.csv`:

| Dataset | Phương pháp | Số node | Số cạnh | `max_diff` thô vs CSV giữa kỳ | `max_diff` làm tròn 6dp vs CSV | `max_diff` trước vs sau F8 | Kết luận |
|---|---|---:|---:|---:|---:|---:|---|
| `bashapes` | `saliency` | 20 | 3,473 | `4.9998e-07` | `0.0` | `0.0` | **PASS** |
| `bashapes` | `gnnexplainer` | 20 | 3,473 | `4.9995e-07` | `0.0` | `0.0` | **PASS** |
| `cora` | `saliency` | 20 | 703 | `4.9937e-07` | `0.0` | `0.0` | **PASS** |
| `cora` | `gnnexplainer` | 20 | 703 | `4.9963e-07` | `0.0` | `0.0` | **PASS** |

Sai khác `~5e-7` ở cột thô là do notebook giữa kỳ lưu điểm đã làm tròn 6 chữ số (`round(float(sc), 6)`). Sau khi làm tròn, **8,352 / 8,352** dòng khớp 0.0. Điểm trước và sau F8 (chuyển sang `model(x, edge_index, edge_weight=w)`) khớp `max_diff = 0.0`.

### A.2. PGExplainer giữa kỳ không tái lập được (lỗi F1)

Chạy 2 lượt `PGExplainerModel(model)` → `train_on_graph(data, epochs=40, lr=0.005, seed=42)` theo code tại `cf55fb3`, BA-Shapes, 20 node dev, 3,473 cạnh:

- `max_diff` giữa 2 lượt cùng `seed = 42`: **`0.885080`** (`mean_diff = 0.415227`).
- `max_diff` giữa lượt 1 và CSV giữa kỳ: **`0.323309`**.
- Nguyên nhân: MLP được khởi tạo trong `PGExplainerModel.__init__` **trước khi** `train_on_graph` gọi `torch.manual_seed(seed)`.
- Đã sửa bằng F1 (Thiện, nhánh `feat/pn2-thien-pgexplainer`): seed đặt trước khi tạo MLP, huấn luyện ở chế độ tất định. Ghi vào Bảng thay đổi v1 → v2.
