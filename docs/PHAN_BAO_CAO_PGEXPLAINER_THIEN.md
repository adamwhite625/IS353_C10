# PGEXPLAINER — F1, F2, F9 VÀ PHẦN BÁO CÁO (THIỆN, PN2)

> Đối chiếu: `KE_HOACH_CUOI_KY_FINAL.md` (Mục 5, 6.8, 11). Nhánh `feat/pn2-thien-pgexplainer`, tách từ `feat/pn2-khoi-explainers`.
> Môi trường đo: Windows, Python 3.12.10, torch 2.15.0+cpu, torch-geometric 2.8.1 (khớp `requirements-lock.txt` của PN1), CPU 2 luồng.
> Dữ liệu đo: bộ giữa kỳ gốc `data/{bashapes,cora}/data.pt` + `gcn_ckpt.pt` (SHA-256 khớp manifest T06). Đây là số đo kiểm chứng kỹ thuật, **chưa phải kết quả lưới 5×5 chính thức**.

## 1. Việc đã làm

| Mã | Thay đổi | File | Bằng chứng |
|:-:|---|---|---|
| F1 | `PGExplainerModel(..., seed=...)` gọi `set_global_seed` **trước** khi tạo MLP. `PGExplainer.explain` (đường tự huấn luyện qua `registry.explain_edges`) và `batch_runner` đều truyền seed. `src/sna/core/seed.py` giống hệt bản của PN1 nên merge không conflict | `src/sna/explainers/pgexplainer.py`, `src/sna/pipelines/batch_runner.py`, `src/sna/core/seed.py` | Test: cùng seed → trọng số và điểm trùng tuyệt đối dù trạng thái RNG trước đó khác; khác seed → khác. Thứ tự gọi của `run_explainers.py` (seed rồi mới tạo model) cho cùng trọng số |
| F1b | Huấn luyện PG chạy trong `torch.use_deterministic_algorithms(True)`, xong thì khôi phục trạng thái cũ | `pgexplainer.py` | Batch 3-hop đủ lớn để phép scatter trên CPU chạy song song: **hai lượt cùng seed lệch 0,30** trọng số MLP trên BA-Shapes. Sau khi sửa: lệch **0,0**, thời gian không đổi (12,1 s so với 12,2 s) |
| F9 | Khi huấn luyện, forward trên đồ thị con **3-hop** (L+1); mask chỉ đặt trên cạnh ứng viên 2-hop, cạnh khác giữ trọng số 1. Tham số `forward_hops` (mặc định 3; `2` tái tạo cách giữa kỳ) | `pgexplainer.py` | Bảng 2 |
| F2 | `batch_runner` xuất đúng 17 cột schema v2 (giống `run_explainers.py`), tính sha256 từ `checkpoint_path`, có dòng `empty_candidates`/`failed`, bỏ tính Fidelity (`df_evals` trả rỗng để giữ chữ ký cũ) | `src/sna/pipelines/batch_runner.py`, `src/sna/pipelines/schema.py` | Test đối chiếu danh sách cột với `scripts/run_explainers.py` |

Không sửa file nào của Khôi (`run_explainers.py`, `check_scores.py`, `gnnexplainer.py`, `saliency.py` và các test của Khôi). `run_explainers.py` tự dùng F9 vì gọi `train_on_graph` với giá trị mặc định.

Test: `tests/test_pgexplainer_f1_f2_f9.py` (15 test). Toàn bộ `pytest tests/`: **78 passed**.

## 2. Kiểm chứng F9 trên dữ liệu thật

Logit của node đích khi forward trên đồ thị con, so với forward trên đồ thị đầy đủ (mọi node train, trọng số = 1):

| Dataset | Node train | Lệch logit lớn nhất, 2-hop (giữa kỳ) | Node đổi dự đoán, 2-hop | Lệch logit lớn nhất, 3-hop (F9) |
|---|---:|---:|---:|---:|
| BA-Shapes | 420 | 23,98 | **63 / 420** | 0,0 |
| Cora | 140 | 0,86 | 1 / 140 | 7,2e-7 |

Ý nghĩa: ở giữa kỳ, PGExplainer được huấn luyện để giữ một dự đoán mà với 15% node train BA-Shapes **không phải** dự đoán thật của GCN. Lý do: GCN chuẩn hóa theo bậc, và trong đồ thị con 2-hop các nút ở rìa bị thiếu cạnh nối ra hop thứ 3.

Kiểm tra thêm (20 node dev, 40 epoch):
- `forward_hops=2` với code mới cho điểm **giống hệt** code giữa kỳ (lệch 0,0).
- So với điểm 2-hop cũ, điểm 3-hop lệch tối đa 0,98 (BA-Shapes) và 0,53 (Cora).

## 3. Pilot qua `run_explainers.py` → `check_scores.py`

Chạy 20 node dev, `--split val`, g = 1000 (checkpoint giữa kỳ), e = 0 và 1:

| Dataset | Train PG / seed | Suy luận / node | `check_scores.py` | Tỷ lệ điểm hòa (trung vị/node) | Node có điểm thứ 5 = thứ 6 |
|---|---:|---:|---|---:|---:|
| BA-Shapes | ~12 s | < 1 ms | **PASS** (6.946 dòng) | 0,57–0,64 | 5–12 / 20 |
| Cora | ~4,7 s | ~4 ms | **PASS** (1.406 dòng) | 0,00–0,22 | 1–2 / 18 |

Ước lượng lưới chính thức: 25 lần train × (12 s + 5 s) ≈ **7–8 phút** cho cả hai dataset, cộng phần suy luận không đáng kể.

**Ghi chú cho error analysis (T4):** sau khi sửa F1/F9, PGExplainer trên BA-Shapes **vẫn cho rất nhiều điểm hòa** (đa số cạnh có điểm sát 0). Giữa kỳ con số này là 40,7%. Vì vậy top-5 của PG trên BA-Shapes vẫn phụ thuộc nhiều vào phá hòa; PN3 cần đo lại trên tập test chính thức. Đây là quan sát trên 20 node dev, chưa phải kết luận.

## 4. Đoạn cho báo cáo — Mục 3 (PGExplainer)

> **PGExplainer** (Luo et al., 2020) học một MLP dùng chung, nhận vector `[z_u ‖ z_v ‖ z_t]` (embedding tầng 1 của GCN) và cho ra logit tồn tại của cạnh (u, v) khi giải thích node t. Logit được đối xứng hóa giữa hai chiều cạnh. MLP được huấn luyện trên các node train với hàm mất mát gồm cross-entropy giữ lớp dự đoán gốc, phạt kích thước (tổng mask của mỗi node) và phạt entropy, dùng Gumbel-Sigmoid với nhiệt độ giảm dần từ 5,0 xuống 0,5 trong 40 epoch (lr 0,005). Mask chỉ đặt trên cạnh ứng viên 2-hop; khi huấn luyện, mô hình được forward trên đồ thị con 3-hop để chuẩn hóa bậc của GCN khớp đồ thị đầy đủ. Mọi nguồn ngẫu nhiên (khởi tạo MLP, thứ tự batch, nhiễu Gumbel) được khóa bằng seed explainer; huấn luyện chạy ở chế độ tất định của PyTorch trên CPU. Khi giải thích, điểm cạnh là `sigmoid` của logit, tính trên embedding của đồ thị đầy đủ.

## 5. Đoạn cho báo cáo — Bảng thay đổi v1 → v2 (phần PGExplainer)

| Thay đổi | Lý do | Ảnh hưởng tới số liệu giữa kỳ |
|---|---|---|
| Seed đặt trước khi khởi tạo MLP (F1) | Khởi tạo phụ thuộc trạng thái RNG trước đó | PG giữa kỳ không tái lập chính xác được (Khôi đo: hai lượt cùng seed 42 lệch tới 0,885) |
| Huấn luyện ở chế độ tất định | Scatter song song trên CPU làm hai lượt cùng seed lệch 0,30 | Bắt buộc để tách nhiễu seed khỏi nhiễu số học trong FB-2 |
| Forward 3-hop khi huấn luyện (F9) | 2-hop làm sai dự đoán của 63/420 node train BA-Shapes | Điểm PG thay đổi (lệch tối đa 0,98 trên 20 node dev BA-Shapes) |
| CSV schema v2 trong `batch_runner` (F2) | Khớp bộ đánh giá và `check_scores.py` | Không đổi số; chỉ đổi định dạng |

## 6. Việc tiếp theo và ghi chú cho nhóm

- **Chạy lưới PG chính thức** khi Vũ phát hành gói dữ liệu và checkpoint chính thức:
  ```bash
  python scripts/run_explainers.py --dataset bashapes --split test --gnn-seeds 1000 --explainer-seeds 0 1 2 3 4 --methods pgexplainer --device cpu --resume
  python scripts/run_explainers.py --dataset bashapes --split test --gnn-seeds 1001 1002 1003 1004 --explainer-seeds 0 1 2 3 4 --methods pgexplainer --device cpu --resume
  python scripts/check_scores.py results/scores/bashapes
  ```
  Chạy tương tự với `--dataset cora`.
- **Cho Khôi (refactor sau, không gấp):** `run_log.json` chưa ghi `forward_hops` trong `hyperparams` của PG. Nên thêm để manifest thể hiện F9. Hiện mặc định luôn là 3.
- `registry.explain_edges` vẫn huấn luyện lại PG mỗi lần gọi. Kết quả đã tái lập được (F1), chỉ chậm. Pipeline chính thức không dùng đường này.
