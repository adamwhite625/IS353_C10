## 7. Error Analysis (trả lời FB-3)

Câu hỏi của thầy: vì sao Fidelity và Motif AUROC không thống nhất? Phạm vi: BA-Shapes, GCN, 140 nút test, 5 GNN seed × 5 seed explainer = 25 cấu hình, bốn phương pháp (Saliency, GNNExplainer, PGExplainer, Nearest-First). Fidelity là Drop@5 (đặc trưng cố định). Một "cặp nút–GNN" là một nút test trên một checkpoint; các cặp này không độc lập nên các tỷ lệ dưới đây là số đếm mô tả. Mọi bảng và lệnh sinh lại ở `docs/PHAN_TICH_LOI_ERROR_ANALYSIS.md`; số lấy từ `results/analysis/a3_final/`.

### 7.1. Xác minh lại mâu thuẫn (Bảng A3-0)

| Phương pháp | Drop@5 (140 nút), TB qua 25 cấu hình (min–max) | AUROC (71/140 nút), TB (min–max) |
|---|---:|---:|
| Saliency | 0,248 (0,225–0,285) | 0,527 (0,511–0,545) |
| GNNExplainer | 0,270 (0,247–0,303) | 0,631 (0,554–0,698) |
| PGExplainer | 0,185 (0,128–0,226) | 0,731 (0,488–0,944) |
| Nearest-First | 0,291 (0,286–0,299) | 0,668 (hằng số) |

(Nguồn: `a3_0_metrics.csv`; Drop@5 khớp `results/analysis/main/main_by_method_budget.csv`.) Ở **17/25 cấu hình (68%)**, phương pháp đứng đầu Drop@5 khác phương pháp đứng đầu AUROC. Drop@5 do Nearest-First dẫn ở 20 cấu hình và GNNExplainer ở 5; AUROC do PGExplainer dẫn ở 17 và Nearest-First ở 8. Kết quả giống hệt với cả 5 seed phá hòa (`a3_0_conflicts.csv`). Số cấu hình mâu thuẫn theo GNN seed là 2, 3, 3, 4, 5 trên 5; AUROC của PGExplainer dao động mạnh giữa các cấu hình (độ lệch chuẩn 0,109), và ở 8 cấu hình không mâu thuẫn nó không vượt 0,668 của Nearest-First. Mâu thuẫn vì vậy **không biến mất** khi tăng mẫu: PGExplainer có AUROC cao nhất nhưng Drop@5 thấp nhất.

### 7.2. Quét nút và bốn góc phần tư (Bảng A3-1)

Ngưỡng khóa trước: Drop@5 ≥ 0,10 và AUROC ≥ 0,75. Số cặp nút–GNN trên 355 cặp có AUROC (5 GNN seed × 71 nút):

| Phương pháp | Q1 (cả hai cao) | Q2 (Fid cao, AUROC thấp) | Q3 (Fid thấp, AUROC cao) | Q4 (cả hai thấp) |
|---|---:|---:|---:|---:|
| Saliency | 56 | 169 | 66 | 64 |
| GNNExplainer | 128 | 156 | 0 | 71 |
| PGExplainer | 212 | 19 | 43 | 81 |
| Nearest-First | 115 | 125 | 20 | 95 |

Thứ tự định tính (PGExplainer chủ yếu ở Q1; Saliency và GNNExplainer chủ yếu ở Q2; GNNExplainer không có Q3) giữ nguyên với hai cặp ngưỡng còn lại (0,05; 0,70) và (0,20; 0,80). Ngoại lệ là Nearest-First: Q1 giảm từ 32,4% xuống 1,4% ở (0,20; 0,80), nên vị trí góc phần tư của phương pháp này nhạy với ngưỡng. **Mâu thuẫn theo nút** (hai tập đồng hạng tốt nhất rời nhau) xảy ra ở **208/355 cặp nút–GNN (58,6%)**. Spearman giữa Drop@5 và AUROC trên các nút (riêng từng phương pháp, trung bình 5 GNN seed): GNNExplainer 0,20; PGExplainer 0,31; Nearest-First 0,07; Saliency **−0,53** (từ −0,65 đến −0,37 qua 5 GNN seed). (Nguồn: `a3_1_counts.csv`, `a3_1_node_conflicts.csv`, `a3_2_mechanisms.csv`.)

### 7.3. Kiểm định cơ chế T1–T5 (Bảng A3-2)

Mẫu số là 208 cặp nút–GNN mâu thuẫn (trừ khi ghi khác).

| Mã | Kết quả | Kết luận |
|:-:|---|---|
| **T2** (mô hình không dựa vào motif) | Xóa toàn bộ cạnh motif trong 2-hop cho Drop < 0,01 ở **95/208 (45,7%)** nút mâu thuẫn, so với 30/147 (20,4%) nút có AUROC nhưng không mâu thuẫn. Drop trung bình 0,281 so với 0,546 | **Ủng hộ** |
| **T3** (đặc trưng bậc tĩnh) | Tính lại bậc sau khi xóa: Drop của xóa 1-hop tăng 0,274 → 0,521 (+0,248), xóa motif 0,281 → 0,303 (+0,021). Hiệu ứng +0,25 đến +0,36 có ở mọi nhóm nút | Ủng hộ như tính chất của metric, **không đặc thù cho mâu thuẫn** |
| **T1** (cạnh 1-hop) | Tỷ lệ cạnh 1-hop trong top-5 của phương pháp thắng Drop@5 là 0,531, của phương pháp thắng AUROC là 0,537 (77 nút lớn hơn, 59 bằng, 72 nhỏ hơn) | **Không ủng hộ** là nguyên nhân mâu thuẫn |
| **T4** (phạm vi metric, điểm hòa) | PGExplainer có khoảng cách điểm 5–6 bằng 0 ở 126/208 (61%) và Nearest-First ở 208/208; biên độ Drop@5 qua 5 seed phá hòa > 0,10 ở 199/1.040 lượt PGExplainer (19%), lớn nhất 0,999. Nhưng người thắng cấp cấu hình không đổi qua 5 seed phá hòa | Ủng hộ một phần: nhiễu từng nút, không tạo mâu thuẫn cấp cấu hình |
| **T5** (Fidelity âm) | Lượt có Drop@5 < 0 ở PGExplainer: 1.849/5.200 (35,6%) trong nhóm mâu thuẫn và 6.727/17.500 (38,4%) trên toàn bộ; Saliency 25,0%, Nearest-First 25,7%, GNNExplainer 6,7% (nhóm mâu thuẫn) | Ủng hộ một phần |

Chi tiết:

- **T2** là cơ chế mạnh nhất. 95 nút có Drop < 0,01 gồm 45 nút lớp dự đoán 0 (xác suất gốc ≥ 0,99, bão hòa) và 50 nút lớp dự đoán 2, ở đó xóa motif làm **tăng** xác suất (Drop trung bình −0,132); không nút lớp 1 hay 3 nào có Drop < 0,01. Trong các nút mâu thuẫn mà phương pháp thắng AUROC có Drop@5 < 0,10, **89/90** thuộc nhóm 95 nút này. Phần còn lại, 113/208 nút, mô hình có dựa vào motif và phương pháp thắng AUROC hầu như cũng có Drop@5 ≥ 0,10 (chỉ 1 ngoại lệ); ở đó hai metric chỉ chọn người thắng khác nhau trong số các phương pháp đều khá, và **không cơ chế T1–T5 nào giải thích**.
- **T3**: xóa toàn bộ cạnh 1-hop cô lập nút, nhưng đặc trưng `[deg, √deg]` giữ nguyên. Trên 140 nút, Drop của can thiệp 1-hop tăng từ 0,29–0,38 lên 0,65–0,67 khi tính lại bậc (+0,26 đến +0,39 theo GNN seed), còn can thiệp motif gần như không đổi (từ −0,006 đến +0,014) (`a3_t2_t3/t2_t3_summary.csv`, `a3_2_t2_t3.csv`). T3 chỉ là phân tích độ nhạy, không đổi metric chính.
- **T1**: có liên quan *bên trong* một phương pháp (Spearman giữa tỷ lệ 1-hop và Drop@5 là 0,59 ở PGExplainer và 0,53 ở GNNExplainer, nhưng 0,10 và 0,13 ở Saliency và Nearest-First); nhưng không giải thích việc hai metric chọn người thắng khác nhau.
- **T5**: đếm dùng ngưỡng `< 0` nên có lẫn nhiễu số học; bảng cạnh bị xóa (`t5_deleted_edges.csv`) không có trong repo nên chưa báo cáo lớp của nút ở hai đầu cạnh.

### 7.4. Case study

Chọn theo quy tắc định trước: GNN seed 1000, góc phần tư gán theo PGExplainer ở (0,10; 0,75), lấy nút gần trung vị của mỗi góc nhất (khoảng cách Euclid trong mặt phẳng Drop–AUROC, không chọn nút đẹp nhất), bổ sung để phủ các nhãn 1–3: 665 (Q1), 596 (Q2), 342 (Q3), 106 (Q4), thêm 389, 535, 528 ở phụ lục (`manifest.json`). Node 692 của hình giữa kỳ thuộc tập validation nên ở `docs/PHAN_TICH_LOI_ERROR_ANALYSIS.md`. Hình dùng explainer seed 0 và phá hòa 0.

| Nút | Bậc, xác suất gốc | Góc | Drop@5: Sal / GNNExp / PG / NF | AUROC: Sal / GNNExp / PG / NF |
|---|---|:-:|---|---|
| 665 | 4, 0,703 | Q1 | 0,212 / 0,212 / **0,480** / 0,212 | 0,951 / 0,963 / 0,862 / 0,806 |
| 596 | 3, 0,615 | Q2 | 0,529 / 0,529 / 0,525 / 0,529 | 0,000 / 0,633 / 0,633 / 0,750 |
| 342 | 2, 0,928 | Q3 | −0,072 / **0,928** / −0,033 / −0,072 | 0,500 / 0,467 / **0,867** / 0,583 |
| 106 | 9, 1,000 | Q4 | 0,000 / 0,000 / −0,000 / 0,002 | 0,756 / 0,476 / 0,005 / 0,629 |

![Node 665 (Q1): xác suất giảm rồi tăng lại khi top-5 cô lập nút](figures/final/a3_node_665.png)

**Node 665.** Top-5 của Saliency, GNNExplainer và Nearest-First chứa cả 4 cạnh trực tiếp (cô lập nút): xác suất xuống 0,000–0,087 rồi **tăng lại 0,491 ở k = 4**, nên Drop@5 chỉ 0,212. PGExplainer chừa lại một cạnh trực tiếp, nên Drop@5 là 0,480. T2/T3 riêng của nút: xóa 1-hop cho Drop 0,212 (đặc trưng cố định) và 0,703 (tính lại bậc). Đường Fid@k không đơn điệu theo k, nên so sánh Drop@5 giữa các phương pháp phụ thuộc vào việc top-5 có cô lập nút hay không.

![Node 342 (Q3): PGExplainer chọn đúng motif nhưng Drop âm](figures/final/a3_node_342.png)

**Node 342.** PGExplainer chọn 5 cạnh motif (precision@5 0,96, AUROC 0,867) nhưng Drop@5 = −0,033. Xóa toàn bộ 6 cạnh motif cho Drop −0,072: xác suất lớp 2 tăng từ 0,928 lên 1,000. GNNExplainer có Drop@5 = 0,928 dù AUROC chỉ 0,467; đó là kết quả của một tổ hợp cạnh cụ thể, không phải bằng chứng nó tìm ra motif. Nút lớp 2 này minh họa T2 và T5; node 528 (phụ lục) cùng kiểu, và node 692 giữa kỳ (lớp 2, Drop −0,0966) cũng vậy.

![Node 596 (Q2): Drop gần bằng nhau, AUROC khác nhau](figures/final/a3_node_596.png)

**Node 596.** Bốn phương pháp có Drop@5 trong 0,525–0,529 nhưng AUROC từ 0,000 đến 0,750. Nút có bậc 3 nên k = 5 phủ cả cạnh trực tiếp, kết quả bằng xóa toàn bộ 1-hop (0,529): Fidelity không phân biệt được phương pháp, còn AUROC thì có. Đây là quan sát bổ sung ngoài T1–T5; k = 5 ≥ bậc xảy ra ở 94/140 nút test và tỷ lệ này tương đương ở nhóm mâu thuẫn (163/208) và không mâu thuẫn (117/147), nên không đặc thù cho mâu thuẫn.

![Node 106 (Q4): xác suất bão hòa ở 1,000](figures/final/a3_node_106.png)

**Node 106.** Xác suất gốc 1,000; xóa top-5 của mọi phương pháp, kể cả toàn bộ motif, không đổi xác suất. AUROC từ 0,005 đến 0,756 mà không phương pháp nào ảnh hưởng dự đoán; đây là kiểu 45 nút lớp 0 bão hòa trong T2.

### 7.5. Kết luận

1. Mâu thuẫn là thật và bền: 17/25 cấu hình (68%), không đổi với 5 seed phá hòa; ở cấp nút 208/355 (58,6%).
2. Trên 208 nút mâu thuẫn, **T2 giải thích 95 nút (45,7%)** (mô hình không dựa vào motif hoặc bão hòa) và 89/90 nút mà phương pháp thắng AUROC có Drop@5 thấp. **T1 không giải thích** (hai nhóm thắng có cùng tỷ lệ 1-hop). T4 và T5 làm nhiễu từng nút (PGExplainer: 19% lượt có biên độ phá hòa > 0,10, 35,6% lượt Drop âm) nhưng không đổi kết quả cấp cấu hình. T3 cho thấy Fidelity với đặc trưng cố định đánh giá thấp tác động của cạnh 1-hop khoảng gấp đôi, ở mọi nhóm nút. **113/208 nút mâu thuẫn chưa có cơ chế nào trong T1–T5 giải thích.**
3. **Khuyến nghị: không chọn explainer theo một metric duy nhất.** Fidelity đo tác động lên mô hình (kể cả khi mô hình bão hòa hoặc dựa vào shortcut), còn AUROC đo độ khớp với motif do người thiết kế; số liệu cho thấy hai thước đo tách nhau đúng ở những nút mà mô hình không dựa vào motif. Trong nhóm mâu thuẫn, phương pháp thắng AUROC có AUROC trung bình 0,907 nhưng Drop@5 chỉ 0,232; phương pháp thắng Drop@5 có Drop@5 0,523 nhưng AUROC 0,540.

### 7.6. Giới hạn

Chỉ BA-Shapes có ground truth (Cora không có). Chỉ GCN 2 lớp. Chỉ 71/140 nút (355/700 cặp nút–GNN) có AUROC xác định, nên mâu thuẫn theo nút chỉ nói về nhóm này. Kết quả đúng với 5 GNN seed cụ thể (1000–1004) và 5 seed explainer. Các con số T2 theo nhóm mâu thuẫn, T1 theo người thắng và 89/90 được tính thêm từ các CSV (đoạn mã ở cuối `PHAN_TICH_LOI_ERROR_ANALYSIS.md`) và là quan sát suy ra từ CSV sau khi xem số, không phải phép thử định trước. T2 và T3 dùng can thiệp có số cạnh khác nhau, và xóa toàn bộ cạnh 1-hop cô lập nút.
