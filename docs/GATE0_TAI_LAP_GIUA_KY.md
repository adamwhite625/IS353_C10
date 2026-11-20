# BIÊN BẢN GATE 0 - XÁC MINH TÁI LẬP KẾT QUẢ GIỮA KỲ

> **Đề tài:** C.10 - Making Graph Machine Learning Models Explainable (Nhóm 09)
> **Phạm vi:** GATE 0 (Mục 6.7 Kế hoạch chốt Nhóm 09, Ngày 1) - trọng tâm Bước 3 (Nguyễn Trọng Khôi - PN2) và đối chiếu chéo Bước 1, Bước 2
> **Môi trường kiểm chứng:** Windows, Python 3.11.6, PyTorch `2.14.0+cpu`, PyTorch Geometric `2.8.0.post1`, thiết bị `cpu`
> **Commit gốc đối chiếu:** `cf55fb3` (nhánh làm việc `feat/pn2-khoi-explainers`)
> **Lệnh tái lập:** `python scripts/verify_gate0_khoi.py --seed 42 --n-nodes 20 --check-pg-f1 --pg-epochs 40`

---

## 1. Xác minh SHA-256 tài sản gốc và dự đoán Checkpoint (Đối chiếu Bước 1)

Nạp trực tiếp `data.pt` và `gcn_ckpt.pt` trên CPU (`model.eval()`, `torch.no_grad()`), tính SHA-256 và đối chiếu `argmax(model(x, edge_index))` với `ck['pred']`:

| Dataset | Tệp tài sản | SHA-256 | Khớp `ck['pred']` |
|---|---|---|---|
| `bashapes` | `data/bashapes/data.pt` | `8fbf94a7ce5850dbdae4a0cd13b178aaa3db1b7009a12efcef8e47306b2c3c73` | 700 / 700 (100%) |
| `bashapes` | `data/bashapes/gcn_ckpt.pt` | `e751d9e112e5def02e0e80401afb132553ad9071036b4da34d6673e8bebe7b9c` | Khớp tuyệt đối |
| `bashapes` | `data/bashapes/config.json` | `acb5621d082c08fd5676e58d445fe15a369ec4ba8794b8f9e468d7b066137303` | - |
| `bashapes` | `data/bashapes/val_nodes.json` | `d24f19579a02c45aa4ac2a2832c53a885ee0d2287ba17e1fc1ca6ca84ac4cf45` | 20 dev nodes đầu |
| `cora` | `data/cora/data.pt` | `f2846a82cdd1e5b5a0804baad24c997a7720b27155a6c9655730bd6c42836268` | 2708 / 2708 (100%) |
| `cora` | `data/cora/gcn_ckpt.pt` | `145fc3bf22cdf1d6ce27df19b6ca42cd291f83f185f400e186414f1d7a67fe52` | Khớp tuyệt đối |
| `cora` | `data/cora/config.json` | `4709552b815c87bc68297f0662a54b1554ef21f96da03897f01c80ec1fa28fb8` | - |
| `cora` | `data/cora/val_nodes.json` | `29bc6f13efc3c8368153f30c8140bcc82ea1a8f1aa3db6b414c3a8c13dcd0796` | 20 dev nodes đầu |
| `midterm` | `notebooks/01_midterm/edge_scores.csv` | `102b49ffa3cc0a8e1acd187a0f2c2535fbf5c5b9d59dcaccc6e12076336152c2` | Khớp hash T06 |

---

## 2. Xác minh bộ đánh giá T06 trên CSV giữa kỳ (Đối chiếu Bước 2)

Chạy lại `python -m sna.evaluation.runner --device cpu` trên `notebooks/01_midterm/edge_scores.csv`:
- Trạng thái: `completed`, 120 dòng audit, 2,080 dòng metric, **0 lỗi**.
- Đối chiếu với `notebooks/05_duc_kha_evaluation/results/summary.csv` (240 dòng): `max_mean_diff = 1.3411e-08`, `max_std_diff = 1.5762e-08` (khớp chặt chẽ vượt xa ngưỡng 4 chữ số thập phân `1e-4`).
- Đối chiếu với `notebooks/05_duc_kha_evaluation/results/agreement_summary.csv` (160 dòng): `max_mean_diff = 0.0`, `max_std_diff = 0.0`.

---

## 3. Tái lập điểm Saliency và GNNExplainer với seed 42 (Bước 3 - Khôi)

Sinh lại điểm bằng code package `sna` (`seed = 42`, CPU, `GNNExplainer: epochs=200, lr=0.01, coeff_size=0.005, coeff_ent=0.1, size_loss_mode='mean'`) trên đúng 20 node dev của BA-Shapes và 20 node dev của Cora, đối chiếu từng cạnh `(node_id, edge_id, u, v)` với `notebooks/01_midterm/edge_scores.csv` và kiểm tra trước/sau sửa lỗi F8:

| Dataset | Phương pháp | Số node | Số cạnh | `max_diff` thô vs CSV giữa kỳ | `max_diff` làm tròn 6dp vs CSV | `max_diff` trước vs sau F8 | Kết luận |
|---|---|---:|---:|---:|---:|---:|---|
| `bashapes` | `saliency` | 20 | 3,473 | `4.9998e-07` | `0.0` | `0.0` | **PASS** |
| `bashapes` | `gnnexplainer` | 20 | 3,473 | `4.9995e-07` | `0.0` | `0.0` | **PASS** |
| `cora` | `saliency` | 20 | 703 | `4.9937e-07` | `0.0` | `0.0` | **PASS** |
| `cora` | `gnnexplainer` | 20 | 703 | `4.9963e-07` | `0.0` | `0.0` | **PASS** |

**Giải thích sai khác `~5e-7` ở cột thô:** Notebook giữa kỳ `04_khoi_saliency_gnnexplainer_t05.ipynb` lưu điểm đã làm tròn 6 chữ số thập phân (`round(float(sc), 6)`). Khi làm tròn 6 chữ số thập phân, toàn bộ **8,352 / 8,352** dòng điểm Saliency và GNNExplainer khớp **0.0 tuyệt đối**. Đồng thời, so sánh điểm float64 thô trước và sau khi sửa F8 (bỏ gọi cứng `c1/c2`, chuyển sang `model(x, edge_index, edge_weight=w)`) đạt `max_diff = 0.0` trên cả hai tập dữ liệu (đạt yêu cầu `< 1e-6`).

---

## 4. Phát hiện về PGExplainer không tái lập được do lỗi F1 (Ghi nhận trung thực)

Thực nghiệm chạy 2 lượt liên tiếp `PGExplainerModel(model)` -> `train_on_graph(data, epochs=40, lr=0.005, seed=42)` theo đúng thứ tự code hiện hành tại commit `cf55fb3` trên BA-Shapes (20 node dev, 3,473 cạnh):

- `max_diff` giữa 2 lần chạy cùng `seed = 42`: **`0.885080`** (`mean_diff = 0.415227`).
- `max_diff` giữa lượt chạy 1 và `notebooks/01_midterm/edge_scores.csv`: **`0.323309`**.
- **Nguyên nhân gốc:** `PGExplainerModel.__init__` khởi tạo trọng số ngẫu nhiên của `PGExplainerMLP` (`src/sna/explainers/pgexplainer.py:99`) **trước khi** `train_on_graph` gọi `torch.manual_seed(seed)` (`pgexplainer.py:132`).
- **Hướng xử lý:** Ghi nhận đây là phát hiện minh bạch trong Bảng thay đổi v1 -> v2 (Mục 6.8); trong pipeline cuối kỳ, lỗi này được khắc phục bằng F1 (Thiện) và được bảo vệ bổ sung bằng `set_global_seed(explainer_seed)` trước khi khởi tạo `PGExplainerModel` trong `scripts/run_explainers.py`.

---

## 5. Kết luận GATE 0

Toàn bộ 4 tiêu chí GATE 0 (Mục 6.7 và Mục 12) đều đã được xác minh bằng số liệu thực 100%:
1. SHA-256 tài sản gốc đã ghi nhận đầy đủ, `pred` khớp 100% (700/700 BA-Shapes, 2,708/2,708 Cora).
2. Bộ đánh giá T06 khớp tuyệt đối (`max_diff <= 1.58e-08`, vượt xa ngưỡng 4 chữ số thập phân).
3. Saliency và GNNExplainer tái lập chính xác tuyệt đối (`max_diff_6dp = 0.0`, `pre_vs_post_F8 = 0.0`).
4. Tính không tái lập của PGExplainer giữa kỳ (lỗi F1) đã được đo lường định lượng và ghi nhận trung thực (`max_diff = 0.885080`).
