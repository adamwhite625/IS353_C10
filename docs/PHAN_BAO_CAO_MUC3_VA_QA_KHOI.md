# BẢN NHÁP MỤC 3 BÁO CÁO (PHẦN CỦA KHÔI) VÀ Q&A PHỤ LỤC B

> **Đề tài:** C.10 - Making Graph Machine Learning Models Explainable (Nhóm 09)
> **Người soạn:** Nguyễn Trọng Khôi (PN2) - theo phân công Mục 11, Mục 15 và Phụ lục B của `KE_HOACH_CUOI_KY_FINAL.md`
> **Phạm vi:** (1) Phần viết trong Mục 3 Báo cáo (*Saliency*, *GNNExplainer*, *Cách xử lý `edge_weight` so với `set_masks`*); (2) Thiết kế dự phòng Mục 8 cho Tầng B2 (*GAT có trọng số*, chỉ kích hoạt sau GATE 2); (3) Bộ trả lời phản biện Phụ lục B cho các câu hỏi thuộc trách nhiệm của Khôi.
> **Nguyên tắc số liệu:** 100% con số trích dẫn đều có nguồn thực nghiệm thật từ `docs/GATE0_TAI_LAP_GIUA_KY.md` và `docs/BIEN_BAN_F10_GNNEXPLAINER_PYG.md` (Zero-Mockup).

---

## PHẦN A. BẢN NHÁP MỤC 3 BÁO CÁO - PHƯƠNG PHÁP (PHẦN PHỤ TRÁCH CỦA KHÔI)

### 3.x. Quy ước không gian cạnh ứng viên và giao diện mô hình

Với mỗi đỉnh đích $v \in V$ cần giải thích và mô hình GNN $f_\theta$ có độ sâu $L = 2$ đã được khóa ở chế độ suy luận (`model.eval()`, không cập nhật tham số), chúng tôi trích xuất đồ thị con lân cận $2$-hop quanh $v$, loại bỏ toàn bộ self-loop ($u = w$), và ánh xạ mỗi cặp cạnh có hướng $(u, w), (w, u)$ về một cạnh vô hướng chuẩn hóa $e_i = (\min(u, w), \max(u, w))$. Tập cạnh ứng viên có thứ tự cố định $E_v = \{e_0, e_1, \dots, e_{m-1}\}$ (với `edge_id` từ $0$ đến $m-1$) được dùng chung cho mọi phương pháp giải thích và mọi hạt giống trên cùng $(g, v)$, đảm bảo so sánh ghép cặp chính xác ở mức từng cạnh.

Mọi phương pháp đều giải thích cho **lớp mô hình thực sự dự đoán** trên đồ thị gốc $\hat{y}_v = \arg\max_c f_\theta(X, A)_{v,c}$ thay vì nhãn thật $y_v$, nhằm đo lường trung thực cơ chế ra quyết định của mô hình. Nếu đỉnh đích là đỉnh cô lập hoặc không có cạnh ứng viên ($m = 0$), các explainer trả về trạng thái tường minh `status = 'empty_candidates'` (thay vì gây lỗi `NaN` khi tính trung bình trên tensor rỗng). Toàn bộ lời gọi lan truyền xuôi sử dụng giao diện tổng quát `model(x, edge_index, edge_weight=w)` (bản sửa F8), tách rời hoàn toàn logic explainer khỏi tên tầng nội bộ (`c1`, `c2`) của mô hình.

### 3.x.1. Edge Saliency (Gradient bậc nhất theo trọng số cạnh)

**Ý tưởng và công thức:** Edge Saliency lượng hóa độ nhạy cục bộ bậc nhất của logit dự đoán theo từng cạnh trong đồ thị. Chúng tôi khởi tạo vector trọng số đơn vị $\mathbf{w} = \mathbf{1}_{|E|} \in \mathbb{R}^{|E|}$ có bật theo dõi đạo hàm (`requires_grad=True`), thực hiện một lượt lan truyền xuôi $Z = f_\theta(X, \text{edge\_index}, \text{edge\_weight}=\mathbf{w})$, rồi tính gradient của logit lớp dự đoán $Z_{v, \hat{y}_v}$ theo $\mathbf{w}$ thông qua `torch.autograd.grad(Z[v, y_hat], w)`:

$$g_{u \to w} = \left| \frac{\partial Z_{v, \hat{y}_v}}{\partial w_{u \to w}} \right|_{\mathbf{w} = \mathbf{1}}$$

Sử dụng `torch.autograd.grad` thay cho `.backward()` đảm bảo không phát sinh hoặc ghi đè gradient lên các tham số $\theta$ của mô hình gốc. Để thu được điểm tầm quan trọng cho mỗi cạnh vô hướng $e_i = (u, w)$ ($u < w$) trong $E_v$, chúng tôi gộp hai chiều bằng phép lấy cực đại:

$$s_{\text{Sal}}(e_i) = \max\left( g_{u \to w},\; g_{w \to u} \right)$$

**Đặc tính thực nghiệm:** Saliency không có tham số khởi tạo ngẫu nhiên và chỉ cần một lượt xuôi cộng một lượt ngược cho mỗi đỉnh. Khi chạy trên CPU, phương pháp hoàn toàn tất định theo `explainer_seed` (kiểm chứng `max_diff = 0.0` giữa `seed = 0` và `seed = 1`), do đó đóng vai trò **mốc tham chiếu không nhiễu seed** ($I_{\text{within}} = 1.0$) trong phân tích FB-2 để tách bạch độ nhạy theo trọng số GNN ($I_{\text{gnn}}$).

### 3.x.2. GNNExplainer (Tối ưu mặt nạ cạnh liên tục theo từng đỉnh)

**Ý tưởng và công thức:** Kế thừa khung tối ưu thông tin tương hỗ của Ying et al. (NeurIPS 2019), GNNExplainer tìm một mặt nạ cạnh liên tục tối giản cho từng đỉnh đích $v$ sao cho phân bố dự đoán trên đồ thị đã che gần nhất với dự đoán gốc. Với $m = |E_v| > 0$ cạnh ứng viên vô hướng, chúng tôi khởi tạo vector tham số $\mathbf{M} \in \mathbb{R}^m$ theo phân phối $M_i \sim \mathcal{N}(1.0, 0.1^2)$ (cố định bởi `explainer_seed`) và ánh xạ sang xác suất cạnh $\mathbf{p} = \sigma(\mathbf{M}) \in (0, 1)^m$.

Khác với cách gán mặt nạ độc lập cho từng cạnh có hướng trong bản PyG mặc định, cài đặt của nhóm gán **chung một xác suất $p_i$** cho cả hai chiều $(u, w)$ và $(w, u)$ của cạnh vô hướng $e_i \in E_v$, còn các cạnh nằm ngoài đồ thị con $2$-hop giữ nguyên trọng số $1.0$ trong vector `edge_weight` toàn đồ thị. Trong suốt vòng lặp tối ưu, tham số $\theta$ của GNN được tạm khóa `requires_grad = False` (và khôi phục nguyên trạng sau khi chạy xong) nhằm giữ vệ sinh gradient của mô hình. Vector $\mathbf{M}$ được tối ưu trong $T = 200$ bước bằng Adam (`lr = 0.01`) theo hàm mất mát:

$$\mathcal{L}(\mathbf{M}) = \underbrace{-\log P_\theta(\hat{y}_v \mid X, \text{edge\_index}, \mathbf{w}(\mathbf{p}))}_{\mathcal{L}_{\text{pred}}} + \underbrace{\lambda_{\text{size}} \cdot \text{Agg}(\mathbf{p})}_{\mathcal{L}_{\text{size}}} + \underbrace{\lambda_{\text{ent}} \cdot \frac{1}{m} \sum_{i=1}^{m} H(p_i)}_{\mathcal{L}_{\text{ent}}}$$

trong đó $H(p) = -p \log(p + \varepsilon) - (1 - p)\log(1 - p + \varepsilon)$ với $\varepsilon = 10^{-12}$, hệ số phạt $\lambda_{\text{size}} = 0.005$ và $\lambda_{\text{ent}} = 0.1$. Điểm quan trọng cuối cùng của cạnh $e_i$ là $s_{\text{GNN}}(e_i) = \sigma(M_i^{(T)})$.

**Lưu ý về toán tử tổng hợp kích thước $\text{Agg}(\mathbf{p})$ (kết quả đối chiếu F10):**
- Ở phiên bản giữa kỳ (`v1`), $\text{Agg}(\mathbf{p}) = \frac{1}{m}\sum_{i=1}^m p_i$ (`mean`). Do chia cho số cạnh ứng viên $m$, lực phạt kích thước trên mỗi cạnh giảm tỷ lệ nghịch với $m$ và bị lấn át bởi hạng tử entropy kéo xác suất khởi tạo $\sigma(1.0) \approx 0.73$ lên sát $1.0$, dẫn đến phân bố điểm dồn quanh $0.95$ (BA-Shapes: trung vị $0.9481$, $98.42\%$ cạnh $> 0.9$; Cora: trung vị $0.9075$, $51.92\%$ cạnh $> 0.9$).
- Trong kiểm định F10 trên 20 dev nodes (`val_nodes.json`, tuyệt đối không sử dụng tập test), khi chuyển riêng $\text{Agg}(\mathbf{p}) = \sum_{i=1}^m p_i$ (`sum`, tương tự quy ước `edge_reduction='sum'` của PyG nhưng vẫn giữ `edge_weight` và $\lambda_{\text{ent}} = 0.1$), hiện tượng dồn điểm quanh $0.95$ được khắc phục triệt để (trung vị giảm còn $0.2563$ trên BA-Shapes và $0.2338$ trên Cora; khoảng phân tách hạng 5-6 tăng $+51.5\%$ trên BA-Shapes và $+321.3\%$ trên Cora; `Fid-` Cora tăng từ $0.0316$ lên $0.0883$ trong khi chất lượng BA-Shapes giữ ổn định với `Fid-` $0.3095$ vs $0.3088$ và `Fid+` cải thiện từ $0.2111$ xuống $0.1311$). Quyết định dùng `sum` hay giữ `mean` cho đợt chạy cuối kỳ được chốt chính thức tại GATE 1 và ghi trong `docs/CHANGELOG_PROTOCOL.md` cùng Bảng thay đổi `v1 -> v2`.

### 3.x.3. Cách xử lý `edge_weight` và lý do không sử dụng `set_masks` của PyG

Một quyết định kỹ thuật trọng yếu của hệ thống là truyền trọng số cạnh trực tiếp qua `edge_weight` của `GCNConv` thay vì sử dụng cơ chế `set_masks` trong `torch_geometric.explain`. Sự khác biệt nằm ở **vị trí nhân mặt nạ** so với bước chuẩn hóa đồ thị:

- **Truyền `edge_weight` (nhóm sử dụng):** Trong `GCNConv` (Kipf & Welling, 2017), ma trận lan truyền được chuẩn hóa phụ thuộc trực tiếp vào $\mathbf{w}$:
  $$H^{(l+1)} = \sigma\!\left(\tilde{D}(\mathbf{w})^{-1/2}\, \tilde{A}(\mathbf{w})\, \tilde{D}(\mathbf{w})^{-1/2} H^{(l)} W^{(l)}\right), \quad \tilde{D}_{ii}(\mathbf{w}) = 1 + \sum_{j \in \mathcal{N}(i)} w_{ij}$$
  Khi đặt $w_{ij} = 0$, cả trọng số cạnh $\tilde{A}_{ij}(\mathbf{w})$ lẫn bậc chuẩn hóa $\tilde{D}_{ii}(\mathbf{w})$ đều cập nhật chính xác như khi loại bỏ hoàn toàn cạnh $(i, j)$ khỏi `edge_index`.
- **Cơ chế `set_masks` của PyG:** `MessagePassing` tính toán hệ số chuẩn hóa `norm` $= \tilde{D}(\mathbf{1})_{ii}^{-1/2}\tilde{D}(\mathbf{1})_{jj}^{-1/2}$ trên đồ thị gốc trước, rồi mới nhân `edge_mask` vào thông điệp sau chuẩn hóa. Do $\tilde{D}(\mathbf{1})$ giữ nguyên dù $m_{ij} = 0$, đặt mask bằng $0$ **không tương đương** với phép xóa cạnh thật dùng để đo `Fid-` và `Fid+`. Với `GATConv`, `set_masks` thậm chí nhân mask **sau** phép `softmax` attention, làm tổng trọng số attention của các láng giềng còn lại nhỏ hơn $1$.

Kiểm toán thực nghiệm trên 20 dev nodes (xóa top-5 cạnh bằng Saliency, chi tiết tại `docs/BIEN_BAN_F10_GNNEXPLAINER_PYG.md`) khẳng định:

| Mô hình / Dataset | Phương thức so với Xóa cạnh thật | Max logit diff | Max prob diff | Số node đổi nhãn |
|---|---|---:|---:|---:|
| GCN / BA-Shapes | `edge_weight = 0` | `0.0000` | `0.0000` | `0 / 20 (0%)` |
| GCN / BA-Shapes | PyG `set_masks = 0` | `11.7885` | `0.4998` | `6 / 20 (30%)` |
| GCN / Cora | `edge_weight = 0` | `0.0000` | `0.0000` | `0 / 20 (0%)` |
| GCN / Cora | PyG `set_masks = 0` | `4.8255` | `0.7508` | `10 / 20 (50%)` |
| GAT 2-lớp (seed=0, chưa train) / BA-Shapes | PyG `set_masks = 0` | `0.6902` | `0.2616` | - |

Nhờ dùng `edge_weight`, cả Saliency và GNNExplainer đều tối ưu hóa trên **đúng hàm phản hồi** mà bộ đánh giá Fidelity đo lường, loại bỏ hoàn toàn độ lệch cơ học giữa bước giải thích và bước chấm điểm.

### [Ghi chú thiết kế chờ GATE 2 - Mục 8 nếu kích hoạt Tầng B2: GAT 2 lớp]

> *Lưu ý: Hạng mục Tầng B2 (COULD) chỉ được triển khai sau GATE 2 nếu Nhóm trưởng phê duyệt theo Mục 9 Kế hoạch chốt. Không đưa kết quả chưa chạy vào báo cáo.*

Nếu Tầng B2 được kích hoạt, lớp `WeightedGATConv` (kế thừa từ `GATConv`) sẽ nhận tham số `edge_weight` $w_{ij} \in [0, 1]$ và cộng hạng tử $\log(w_{ij} + 10^{-12})$ trực tiếp vào logit attention $e_{ij}$ **trước** bước `softmax`:

$$\alpha_{ij}(\mathbf{w}) = \frac{\exp\!\left(\text{LeakyReLU}(e_{ij}) + \log(w_{ij} + 10^{-12})\right)}{\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp\!\left(\text{LeakyReLU}(e_{ik}) + \log(w_{ik} + 10^{-12})\right)}$$

Khi $w_{ij} = 0$, $\exp(\log 10^{-12}) = 10^{-12} \approx 0$ loại bỏ hoàn toàn cạnh $(i, j)$ trước chuẩn hóa `softmax`, đảm bảo kiểm định bắt buộc: đầu ra với $w_{ij} = 0$ khớp xóa cạnh thật trong sai số `< 1e-5`, đồng thời vẫn khả vi trơn theo $w_{ij}$ cho cả Saliency và GNNExplainer mà không cần sửa code explainer (nhờ F8).

---

## PHẦN B. CẨM NANG TRẢ LỜI PHẢN BIỆN (PHỤ LỤC B - PHẦN CỦA KHÔI)

### Câu hỏi 1 (Phụ lục B - Khôi): *"Vì sao không dùng `set_masks` của PyG?"*

- **Trả lời ngắn gọn (30 giây):**
  1. **Mâu thuẫn toán học với phép đo Fidelity:** Metric `Fid-` và `Fid+` của nhóm đo bằng cách **xóa cạnh thật** khỏi `edge_index`. Trong `GCNConv`, truyền `edge_weight = 0` làm cập nhật cả ma trận kề lẫn bậc chuẩn hóa $\tilde{D}(\mathbf{w})^{-1/2}$, nên tương đương **tuyệt đối** với xóa cạnh thật. Ngược lại, `set_masks` của PyG nhân mask **sau** bước chuẩn hóa bậc (ở GCN) hoặc **sau** bước `softmax` attention (ở GAT), nên đặt mask $= 0$ **không tương đương** với xóa cạnh.
  2. **Bằng chứng định lượng (biên bản F10):** Nhóm đã đo trực tiếp trên 20 dev nodes khi xóa top-5 cạnh: `edge_weight = 0` cho sai số logit = `0.0000`, sai số xác suất = `0.0000`, `0/20` đổi dự đoán trên cả BA-Shapes và Cora. Trong khi đó, `set_masks = 0` làm lệch xác suất tới `0.4998` (`6/20 = 30%` node đổi dự đoán) trên BA-Shapes và tới `0.7508` (`10/20 = 50%` node đổi dự đoán) trên Cora; kiểm thử trên GAT 2 lớp cũng cho sai lệch xác suất `0.2616`.
  3. **Hệ quả thực tế:** Khi chạy `torch_geometric.explain.GNNExplainer` (dùng `set_masks`), `Fid-` trên BA-Shapes tụt từ `0.3095` xuống `0.2098` và độ đồng thuận `kappa@5` chỉ còn `0.0826` (BA-Shapes) và `-0.0824` (Cora) vì PyG đang tối ưu một hàm mục tiêu lệch với định nghĩa xóa cạnh. Ngoài ra PyG học mask riêng theo hai chiều có hướng, gây nhạy cảm với cách gộp vô hướng (`mean` vs `max` chỉ đồng thuận `kappa@5 = 0.5728` trên BA-Shapes và `0.4078` trên Cora).

### Câu hỏi 2 (Phụ lục B - Khôi, Thiện): *"Siêu tham số explainer chọn thế nào?"*

- **Trả lời ngắn gọn (30 giây):**
  1. **Nguyên tắc chống rò rỉ dữ liệu:** Tuyệt đối **không** tinh chỉnh siêu tham số của bất kỳ explainer nào trên tập test (`test_nodes.json`). Toàn bộ siêu tham số được khóa trước trong `configs/protocol_final.yaml` tại GATE 1.
  2. **Chi tiết từng phương pháp:**
     - **Saliency:** Không có siêu tham số tối ưu; hoàn toàn tất định theo trọng số mô hình.
     - **GNNExplainer:** Giữ nguyên các siêu tham số cốt lõi từ bản giữa kỳ (`epochs = 200`, `lr = 0.01`, `coeff_size = 0.005`, `coeff_ent = 0.1`) theo tài liệu gốc và kinh nghiệm triển khai trên đồ thị cỡ nhỏ-vừa. Riêng hiện tượng điểm dồn quanh `0.95` phát hiện trong bản giữa kỳ đã được khảo sát có kiểm soát ở **F10** trên đúng **20 dev nodes** (`val_nodes.json`), tách biệt riêng `loss_size` (`mean` vs `sum`) và `coeff_ent` (`0.1` vs `1.0`). Kết quả cho thấy tăng `coeff_ent` lên `1.0` làm hỏng AUROC (`0.7392 -> 0.5472`), còn giữ `coeff_ent = 0.1` và đổi `loss_size` sang `sum` giải quyết sạch hiện tượng bão hòa mask (`> 0.9` giảm từ `98.42%` xuống `0.78%`) và tăng khoảng phân tách top-5/top-6. Mọi thay đổi so với giữa kỳ đều được ghi rõ trong `docs/CHANGELOG_PROTOCOL.md` và Bảng thay đổi `v1 -> v2`.
     - **PGExplainer (phần Thiện bổ sung):** Giữ nguyên `epochs = 40`, `lr = 0.005`, lịch ủ nhiệt độ `5.0 -> 0.5` từ giữa kỳ, chỉ khắc phục lỗi đặt seed trước khi khởi tạo MLP (F1) và đồ thị con 3-hop (F9).

### Câu hỏi dự phòng 3 (Khôi): *"Làm sao đảm bảo file điểm của PN2 bàn giao cho PN3 không có lỗi ngầm hay lệch checkpoint?"*

- **Trả lời ngắn gọn:**
  - Mỗi lượt chạy từ `scripts/run_explainers.py` tự động ghi đủ 17 cột chuẩn Schema v2, bao gồm `checkpoint_sha256` (so khớp trực tiếp với file `.pt` và `.json` của PN1), `runtime_sec` từng node, và `status` (`success` / `empty_candidates` / `failed`), kèm `run_log.json` lưu mã băm SHA-256 của `data.pt`, checkpoint, danh sách node và từng CSV.
  - Trước khi bàn giao H2 sang PN3, bắt buộc chạy `scripts/check_scores.py` kiểm tra toàn diện hợp đồng Schema v2 và chống sửa tay/rò rỉ (đủ 17 cột, không có `NaN/Inf` hay khoảng trắng thừa, chuẩn hóa `u < v`, kiểm tra định dạng dòng placeholder `empty_candidates`/`failed`, không trùng dòng, đủ danh sách node chính thức từ `test_nodes.json`/`val_nodes.json`, kiểm tra `split_mask` chống rò rỉ `train_mask`, khớp tập cạnh 2-hop với `data.pt` trên mọi `gnn_seed` và đồng bộ giữa các phương pháp, khớp `checkpoint_sha256` và `csv_sha256` chống chỉnh tay CSV so với `run_log.json`, xác minh Saliency tất định `max_diff = 0.0` giữa `e = 0` và `e = 1`). Chỉ bàn giao khi script in **PASS** (`exit code 0`).
