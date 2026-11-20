# BIÊN BẢN F10 - ĐỐI CHIẾU GNNEXPLAINER TỰ VIẾT VỚI PYG VÀ KIỂM TOÁN SET_MASKS

> **Đề tài:** C.10 - Making Graph Machine Learning Models Explainable (Nhóm 09)
> **Hạng mục:** F10 (Mục 5 & Mục 6.8 Kế hoạch chốt Nhóm 09, Ngày 2) - Phụ trách: Nguyễn Trọng Khôi (PN2)
> **Cấu hình thực nghiệm:** 20 dev nodes đầu (`val_nodes.json`) trên `bashapes` (3,473 cạnh; 10/20 node hợp lệ AUROC/AP) và `cora` (703 cạnh; 18/20 node hợp lệ `kappa@5` do 2 node có `m <= 5`: node 528 có `m = 2` và node 233 có `m = 4`), cùng midterm checkpoint `gcn_ckpt.pt`, CPU, `seed = 42`, `epochs = 200`, `lr = 0.01`, `budget k = 5`.
> **Lệnh tái lập:** `python scripts/check_gnnexplainer_ref.py --datasets bashapes cora --n-nodes 20 --seed 42 --epochs 200 --lr 0.01 --budget 5` (thời gian thực tế: `184.842 s`, exit code `0`, SHA-256 JSON log: `b0b004344d4cec2d...`).

---

## 1. Bối cảnh vấn đề và 5 biến thể đối chiếu

Trong CSV giữa kỳ, mask của `GNNExplainer` tự viết bị dồn chặt quanh `0.95` (BA-Shapes: trung vị `0.9481`, `98.42%` cạnh `> 0.9`; Cora: trung vị `0.9075`, `51.92%` cạnh `> 0.9`). Nguyên nhân toán học: `mask_logits` khởi tạo quanh `1.0` (`sigmoid ~ 0.73`), trong khi hạng tử phạt kích thước dùng `loss_size = 0.005 * mean(m)` chia cho tổng số cạnh ứng viên `m` (gradient phạt chỉ `~2.9e-5`/cạnh trên BA-Shapes), yếu hơn nhiều so với hạng tử entropy `0.1 * mean(H(m))` kéo giá trị `0.73` lên sát `1.0`.

Để tách biệt tác động của (i) `loss_size` (`mean` vs `sum`), (ii) `coeff_ent` (`0.1` vs `1.0`), và (iii) cơ chế áp mask (`edge_weight` trước chuẩn hóa vs `set_masks` sau chuẩn hóa của PyG), F10 đánh giá 5 biến thể trên cùng checkpoint và `seed = 42`:
1. `custom_mean`: Bản giữa kỳ (`loss_size = 0.005 * mean`, `coeff_ent = 0.1`, `edge_weight`, mask vô hướng chung cho `(u, v)` và `(v, u)`).
2. `custom_sum`: Đổi duy nhất `loss_size = 0.005 * sum` như PyG (giữ `coeff_ent = 0.1` và `edge_weight`).
3. `custom_sum_ent1`: Đổi cả `loss_size = 0.005 * sum` và `coeff_ent = 1.0` như PyG (giữ `edge_weight`).
4. `pyg_official`: `torch_geometric.explain.GNNExplainer` (`edge_reduction='sum'`, `edge_size=0.005`, `edge_ent=1.0`, `set_masks`), gộp cạnh vô hướng bằng `mean(m_uv, m_vu)`.
5. `pyg_official_max`: Cùng lượt chạy `pyg_official`, gộp cạnh vô hướng bằng `max(m_uv, m_vu)`.

---

## 2. Kết quả định lượng: Phân bố mask và Chất lượng tại k = 5

### 2.1. BA-Shapes (20 dev nodes, 3,473 cạnh; AUROC/AP tính trên 10/20 node có đủ 2 lớp cạnh)

| Biến thể | Median | Mean | `> 0.9` | `< 0.1` | Med gap 5-6 | Fid- (Drop) | Fid+ (Inv) | Motif AUROC | Motif AP | Prec@5 | Runtime (s/node) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `custom_mean` | 0.9481 | 0.9412 | 98.42% | 0.00% | 0.0223 | `0.3088 +/- 0.3771` | `0.2111 +/- 0.4259` | `0.7392 +/- 0.2078` | `0.7570 +/- 0.4003` | `0.5400 +/- 0.4639` | `0.7968 +/- 0.2636` |
| `custom_sum` | 0.2563 | 0.2655 | 0.78% | 0.00% | 0.0337 | `0.3095 +/- 0.3765` | `0.1311 +/- 0.3898` | `0.7350 +/- 0.1777` | `0.7509 +/- 0.3980` | `0.5500 +/- 0.4674` | `0.9704 +/- 0.5318` |
| `custom_sum_ent1` | 0.2081 | 0.2523 | 6.51% | 0.00% | 0.0015 | `0.2216 +/- 0.3217` | `0.2062 +/- 0.3835` | `0.5472 +/- 0.3695` | `0.6811 +/- 0.4091` | `0.5200 +/- 0.4873` | `1.0075 +/- 0.5253` |
| `pyg_official` | 0.0000 | 0.0782 | 0.03% | 84.42% | 0.0223 | `0.2098 +/- 0.3220` | `0.0941 +/- 0.2975` | `0.7964 +/- 0.1364` | `0.7699 +/- 0.4015` | `0.5700 +/- 0.4824` | `0.8997 +/- 0.4266` |
| `pyg_official_max` | 0.0000 | 0.1401 | 0.92% | 67.00% | 0.0058 | `0.2069 +/- 0.3151` | `0.2832 +/- 0.3887` | `0.7607 +/- 0.1660` | `0.7444 +/- 0.4011` | `0.5500 +/- 0.4807` | `0.8997 +/- 0.4266` |

### 2.2. Cora (20 dev nodes, 703 cạnh; không có ground truth motif)

| Biến thể | Median | Mean | `> 0.9` | `< 0.1` | Med gap 5-6 | Fid- (Drop) | Fid+ (Inv) | Runtime (s/node) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `custom_mean` | 0.9075 | 0.6129 | 51.92% | 0.00% | 0.0034 | `0.0316 +/- 0.0810` | `-0.0504 +/- 0.2645` | `1.4019 +/- 0.4431` |
| `custom_sum` | 0.2338 | 0.3573 | 13.66% | 0.00% | 0.0144 | `0.0883 +/- 0.1571` | `-0.2022 +/- 0.1959` | `1.3152 +/- 0.2099` |
| `custom_sum_ent1` | 0.9509 | 0.8362 | 84.92% | 0.00% | 0.0007 | `-0.0422 +/- 0.0423` | `0.0333 +/- 0.2470` | `1.3326 +/- 0.3613` |
| `pyg_official` | 0.0476 | 0.2506 | 1.28% | 56.61% | 0.0014 | `0.0344 +/- 0.2652` | `-0.0351 +/- 0.1412` | `1.2597 +/- 0.1625` |
| `pyg_official_max` | 0.0950 | 0.4154 | 26.03% | 54.62% | 0.0021 | `-0.0275 +/- 0.0791` | `0.0748 +/- 0.2277` | `1.2597 +/- 0.1625` |

---

## 3. Độ đồng thuận giữa các biến thể (`kappa@5`, `Jaccard@5`, Kendall `tau_b`)

| Cặp so sánh | Dataset | `kappa@5` (mean +/- std) | `Jaccard@5` (mean +/- std) | Kendall `tau_b` (mean +/- std) | Node hợp lệ (`kappa`) |
|---|---|---:|---:|---:|---:|
| `custom_mean` vs `custom_sum` | BA-Shapes | `0.9076 +/- 0.2115` | `0.9125 +/- 0.1977` | `0.9048 +/- 0.1675` | 20/20 |
| `custom_mean` vs `pyg_official` | BA-Shapes | `0.0826 +/- 0.3412` | `0.3929 +/- 0.3465` | `0.1400 +/- 0.3167` | 20/20 |
| `custom_sum` vs `pyg_official` | BA-Shapes | `0.1279 +/- 0.3947` | `0.4151 +/- 0.3620` | `0.1785 +/- 0.3386` | 20/20 |
| `pyg_official` vs `pyg_official_max` | BA-Shapes | `0.5728 +/- 0.3962` | `0.6339 +/- 0.3345` | `0.8434 +/- 0.1934` | 20/20 |
| `custom_mean` vs `custom_sum` | Cora | `0.2080 +/- 0.4719` | `0.3819 +/- 0.4021` | `0.4454 +/- 0.3800` | 18/20 |
| `custom_mean` vs `pyg_official` | Cora | `-0.0824 +/- 0.2561` | `0.2333 +/- 0.3314` | `-0.1735 +/- 0.2998` | 18/20 |
| `custom_sum` vs `pyg_official` | Cora | `0.0944 +/- 0.2904` | `0.3179 +/- 0.2978` | `0.0574 +/- 0.3561` | 18/20 |
| `pyg_official` vs `pyg_official_max` | Cora | `0.4078 +/- 0.3666` | `0.4815 +/- 0.3676` | `0.7254 +/- 0.2950` | 18/20 |

---

## 4. Bằng chứng kiểm toán `set_masks` của PyG so với `edge_weight` và Xóa cạnh thật

Kiểm thử xóa top-5 cạnh trên 20 dev nodes bằng 3 cơ chế: (a) xóa cạnh thật khỏi `edge_index` (chuẩn đo Fidelity của nhóm), (b) truyền `edge_weight = 0` vào `GCN.forward`, và (c) dùng `set_masks(model, w, edge_index)` của PyG:

| Mô hình / Dataset | Cơ chế so với Xóa cạnh thật | Max logit diff | Mean logit diff | Max prob diff | Mean prob diff | Đổi dự đoán |
|---|---|---:|---:|---:|---:|---:|
| GCN / BA-Shapes | `edge_weight = 0` (nhóm dùng) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0 / 20` |
| GCN / BA-Shapes | PyG `set_masks = 0` | `11.7885` | `8.9301` | `0.4998` | `0.0924` | `6 / 20 (30%)` |
| GCN / Cora | `edge_weight = 0` (nhóm dùng) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0 / 20` |
| GCN / Cora | PyG `set_masks = 0` | `4.8255` | `2.0134` | `0.7508` | `0.3147` | `10 / 20 (50%)` |
| GAT 2-lớp (seed=0, chưa train) / BA-Shapes | PyG `set_masks = 0` | `0.6902` | `0.2843` | `0.2616` | `0.0970` | - |

**Kết luận kỹ thuật:** `edge_weight = 0` trong `GCNConv` tác động **trước** bước chuẩn hóa bậc `D^{-1/2} A D^{-1/2}`, nên tương đương tuyệt đối (`diff = 0.0`) với xóa cạnh thật. Ngược lại, `set_masks` của PyG nhân mask vào message **sau** khi chuẩn hóa bậc (GCN) hoặc **sau** softmax attention (GAT), làm sai lệch nặng dự đoán (đổi nhãn ở 30% node BA-Shapes và 50% node Cora). Đây là nguyên nhân chính khiến `pyg_official` có `Fid-` thấp hơn rõ trên BA-Shapes (`0.2098` vs `0.3095`) và bất đồng rất mạnh với bản dùng `edge_weight` (`kappa@5 = 0.0826` BA-Shapes, `-0.0824` Cora). Thêm nữa, PyG tối ưu mask độc lập cho từng chiều cạnh có hướng, nên chỉ đổi quy tắc gộp `mean` sang `max` đã làm `kappa@5` giảm xuống `0.5728` (BA-Shapes) và `0.4078` (Cora).

---

## 5. Khuyến nghị chốt tại GATE 1 và ghi vào `CHANGELOG_PROTOCOL.md`

1. **KHÔNG thay bằng `torch_geometric.explain.GNNExplainer`:** vì `set_masks` không tương đương xóa cạnh thật (sai số xác suất tới `0.7508`, đổi nhãn 30-50% node) và phá vỡ tính nhất quán giữa hàm mục tiêu tối ưu với metric Fidelity của nhóm.
2. **KHÔNG tăng `coeff_ent` lên `1.0` (`custom_sum_ent1`):** với khởi tạo logit quanh `1.0`, phạt entropy mạnh gây sụp đổ AUROC trên BA-Shapes (`0.7392 -> 0.5472`) và đẩy `Fid-` Cora xuống âm (`-0.0422`).
3. **ĐỀ XUẤT CHỐT TẠI GATE 1 (cần Vũ & Nhóm 09 phê duyệt):** Chuyển đợt chạy chính cuối kỳ (lưới 5x5 tập test) sang **`custom_sum` (`--gnn-size-loss-mode sum`, giữ `coeff_size = 0.005`, `coeff_ent = 0.1`, `epochs = 200`, `lr = 0.01`, dùng `edge_weight`)**, vì:
   - Giải quyết triệt để hiện tượng mask dồn quanh `0.95` (BA-Shapes: tỷ lệ `> 0.9` giảm từ `98.42%` xuống `0.78%`, median `0.2563`; Cora: `51.92%` xuống `13.66%`, median `0.2338`).
   - Mở rộng rõ rệt khoảng phân tách top-5 vs top-6 (`gap_5_6`: `+51.5%` trên BA-Shapes từ `0.0223` lên `0.0337`, và `+321.3%` trên Cora từ `0.0034` lên `0.0144`), giảm rủi ro hòa điểm.
   - Bảo toàn chất lượng BA-Shapes (`Fid-` `0.3095` vs `0.3088`, `Fid+` cải thiện từ `0.2111` xuống `0.1311`, `AUROC` `0.7350` vs `0.7392`, đồng thuận cao với giữa kỳ `kappa@5 = 0.9076`) và tăng mạnh `Fid-` trên Cora (`0.0316 -> 0.0883`; ghi nhận trung thực `Fid+` Cora thành `-0.2022`, tức giữ 5 cạnh mạnh làm tăng xác suất lớp dự đoán so với đồ thị gốc).
   - Khớp trực tiếp tiêu chí F10 của Kế hoạch chốt (Mục 5): *"nếu hai bản khác nhau nhiều, đợt chạy chính dùng công thức size-loss dạng `sum` như PyG; ghi vào CHANGELOG"*.
4. **Bảo toàn tương thích ngược:** Giá trị mặc định trong `src/sna/explainers/gnnexplainer.py` vẫn giữ `size_loss_mode = 'mean'` để GATE 0 tái lập giữa kỳ giữ `max_diff_6dp = 0.0`. Khi Nhóm 09 phê duyệt tại GATE 1, cấu hình `size_loss_mode: sum` (hoặc giữ `mean` nếu chọn phương án bảo thủ) sẽ được khóa trong `configs/protocol_final.yaml`, truyền qua `--gnn-size-loss-mode` trong `scripts/run_explainers.py`, và ghi nhận vào `docs/CHANGELOG_PROTOCOL.md` cùng Bảng thay đổi v1 -> v2 (Mục 6.8).
