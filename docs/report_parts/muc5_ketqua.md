## 5. Kết quả chính

Mục này báo cáo kết quả trên tập test cho 3 phương pháp giải thích và 2 baseline. Mỗi ô là trung bình theo nút qua mọi cấu hình (g, e), kèm khoảng tin cậy 95% bootstrap trên nút (2.000 lần, seed 0; nguồn: `results/analysis/main/main_by_method_budget.csv`). BA-Shapes có 140 nút test; Cora có 100 nút test. Bảng chính dùng ngân sách k = 5 (Mục 3.7). Mục này chỉ mô tả số liệu; nguyên nhân của các khác biệt thuộc Mục 7.

**Quy ước dấu.** Với p là xác suất lớp dự đoán ŷ:

- **Drop = p(ŷ | G) − p(ŷ | G∖top-k).** Cao hơn là tốt hơn: các cạnh được chọn càng cần cho dự đoán.
- **Inv = p(ŷ | G) − p(ŷ | chỉ giữ top-k).** Thấp hơn là tốt hơn: gần 0 nghĩa là top-k đủ để giữ dự đoán, âm nghĩa là chỉ giữ top-k còn làm xác suất tăng lên.
- AUROC, AP, Precision@5: cao hơn là tốt hơn (chỉ BA-Shapes).

GraphFramEx dùng nhãn +/− ngược với chúng tôi, nên bảng ghi Drop và Inv (Mục 3.7; Phụ lục A của kế hoạch).

### 5.1. Bảng chính trên tập test (k = 5)

**Bảng 5.1a. BA-Shapes, 140 nút test.** AUROC và AP chỉ xác định ở 71/140 nút (nút có cả cạnh motif lẫn cạnh nền); các cột còn lại dùng đủ 140 nút. Mỗi ô là `mean [CI 95%]`.

| Phương pháp | Drop@5 (cao = tốt) | Inv@5 (thấp = tốt) | AUROC | AP | Precision@5 |
|---|---|---|---|---|---|
| Saliency | 0,248 [0,198; 0,298] | 0,059 [0,031; 0,087] | 0,527 [0,456; 0,598] | 0,648 [0,573; 0,724] | 0,479 [0,410; 0,547] |
| GNNExplainer | 0,270 [0,219; 0,326] | 0,058 [0,022; 0,102] | 0,631 [0,580; 0,683] | 0,687 [0,602; 0,769] | 0,497 [0,426; 0,568] |
| PGExplainer | 0,185 [0,141; 0,228] | 0,265 [0,217; 0,317] | 0,731 [0,652; 0,808] | 0,767 [0,674; 0,860] | 0,549 [0,471; 0,627] |
| Nearest-First | 0,291 [0,232; 0,353] | 0,091 [0,062; 0,124] | 0,668 [0,649; 0,689] | 0,645 [0,562; 0,728] | 0,491 [0,421; 0,562] |
| Random | 0,156 [0,121; 0,192] | 0,293 [0,252; 0,340] | 0,500 [0,500; 0,500] | 0,578 [0,495; 0,658] | 0,463 [0,392; 0,531] |

**Bảng 5.1b. Cora, 100 nút test** (không có ground truth nên chỉ có Fidelity).

| Phương pháp | Drop@5 (cao = tốt) | Inv@5 (thấp = tốt) |
|---|---|---|
| Saliency | 0,074 [0,040; 0,108] | −0,079 [−0,098; −0,061] |
| GNNExplainer | 0,080 [0,055; 0,106] | −0,129 [−0,152; −0,106] |
| PGExplainer | 0,009 [−0,012; 0,031] | 0,022 [−0,003; 0,048] |
| Nearest-First | 0,039 [0,005; 0,074] | −0,078 [−0,096; −0,061] |
| Random | 0,003 [−0,015; 0,024] | 0,022 [−0,002; 0,047] |

*Ghi chú.* (i) Mỗi phương pháp trung bình qua 25 cấu hình (g, e), cân bằng theo GNN; riêng Saliency chạy 6 lượt (5 GNN với e = 0, thêm e = 1 trên GNN 1000), nên trung bình được lấy theo seed explainer trong từng GNN rồi qua 5 GNN để mỗi GNN có một trọng số (nguồn: `main_by_method_budget.csv`, `scripts/summarize_main_results.py`). (ii) Random không có điểm thô để xếp hạng nên AUROC bằng đúng 0,5 theo định nghĩa. (iii) Ở Cora, 14/100 nút có m ≤ 5 cạnh ứng viên; với các nút này top-5 là toàn bộ cạnh nên mọi phương pháp, kể cả Random, cho cùng một giá trị Fidelity (nguồn: `results/eval/cora_test/metrics.csv`).

### 5.2. Fidelity theo ngân sách k ∈ {1, 3, 5, 10}

**Bảng 5.2.** Giá trị trung bình (CI 95% ở `main_by_method_budget.csv`). Drop: cao = tốt; Inv: thấp = tốt. Giá trị tốt nhất mỗi cột k in đậm.

| BA-Shapes | Drop k=1 | k=3 | k=5 | k=10 | Inv k=1 | k=3 | k=5 | k=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Saliency | 0,125 | 0,179 | 0,248 | **0,341** | 0,310 | 0,168 | 0,059 | 0,072 |
| GNNExplainer | 0,109 | **0,232** | 0,270 | 0,190 | 0,248 | 0,130 | **0,058** | 0,099 |
| PGExplainer | 0,103 | 0,203 | 0,185 | 0,178 | 0,335 | 0,308 | 0,265 | 0,182 |
| Nearest-First | **0,151** | 0,210 | **0,291** | 0,340 | **0,246** | **0,070** | 0,091 | **0,042** |
| Random | 0,071 | 0,130 | 0,156 | 0,188 | 0,343 | 0,321 | 0,293 | 0,192 |

| Cora | Drop k=1 | k=3 | k=5 | k=10 | Inv k=1 | k=3 | k=5 | k=10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Saliency | **0,058** | **0,082** | 0,074 | 0,064 | **−0,071** | **−0,079** | −0,079 | −0,068 |
| GNNExplainer | 0,013 | 0,058 | **0,080** | **0,090** | 0,013 | −0,078 | **−0,129** | **−0,144** |
| PGExplainer | −0,002 | −0,0003 | 0,009 | 0,016 | 0,036 | 0,033 | 0,022 | 0,014 |
| Nearest-First | 0,019 | 0,046 | 0,039 | 0,041 | −0,0007 | −0,069 | −0,078 | −0,075 |
| Random | −0,002 | −0,002 | 0,003 | 0,008 | 0,039 | 0,030 | 0,022 | 0,012 |

Trên BA-Shapes, 65/140 nút có m ≤ 10 cạnh ứng viên, nên ở k = 10 top-k là toàn bộ cạnh của các nút đó với mọi phương pháp; các đường cong vì vậy xích lại gần nhau ở k = 10 và Drop của GNNExplainer, PGExplainer và Random chênh nhau ít (0,190; 0,178; 0,188). Trên Cora, 29/100 nút có m ≤ 10. Bảng 5.2 do đó không nên đọc như một đường cong đơn điệu theo k.

### 5.3. Độ đồng thuận giữa các phương pháp

Bảng dưới tóm tắt κ@5 (nguồn: `results/eval/{bashapes,cora}_test/agreement_summary.csv`; κ@5 chỉ xác định khi m > 5, tức 140/140 nút BA-Shapes và 86/100 nút Cora). Phân tích đầy đủ, kèm kiểm định so với mức tự lệch của từng phương pháp, nằm ở Mục 6 (`results/analysis/a2/a2_2_between.csv`).

| κ@5 | BA-Shapes | Cora |
|---|---:|---:|
| Saliency – GNNExplainer | 0,143 | 0,342 |
| Saliency – PGExplainer | 0,102 | −0,010 |
| GNNExplainer – PGExplainer | 0,048 | −0,017 |
| Saliency – Nearest-First | 0,425 | 0,507 |
| GNNExplainer – Nearest-First | 0,146 | 0,283 |
| PGExplainer – Nearest-First | 0,066 | −0,013 |
| Random – Random (mốc ngẫu nhiên) | −0,004 | −0,001 |

Các κ@5 giữa hai phương pháp khác nhau đều thấp: không giá trị nào vượt 0,35, và năm trong sáu giá trị nằm trong khoảng ±0,15 quanh mốc Random (≈ 0); cao nhất là Saliency – GNNExplainer trên Cora (0,342). Saliency – Nearest-First cao hơn hẳn (0,425 và 0,507). Để so sánh, Jaccard@5 của cặp Random – Random là 0,316 (BA-Shapes) và 0,275 (Cora), tức Jaccard thô cao hơn nhiều so với mốc κ ≈ 0 (lý do dùng κ, Mục 3.7).

### 5.4. So sánh với kết quả giữa kỳ

Số giữa kỳ (`results/analysis/main/midterm_reference.csv`, từ bảng T06) chỉ để đối chiếu, không phải so sánh trực tiếp, vì bốn điều kiện đã đổi: (i) nút đánh giá: 20 nút validation, trong đó chỉ 10/20 nút BA-Shapes có AUROC, thay cho 140/100 nút test; (ii) checkpoint: một GCN giữa kỳ thay cho 5 GCN huấn luyện lại; (iii) PGExplainer: bản giữa kỳ không tái lập được và huấn luyện trên đồ thị con 2-hop, bản cuối kỳ đã sửa (Mục 4.5); (iv) GNNExplainer: size-loss `mean` thay bằng `sum`. Thứ hạng (k = 5):

| Metric | Giữa kỳ (20 nút val) | Cuối kỳ (nút test) |
|---|---|---|
| BA-Shapes Drop@5 | NF 0,373 > Sal 0,309 ≈ GNNE 0,309 > PG 0,220 > Random 0,167 | NF 0,291 > GNNE 0,270 > Sal 0,248 > PG 0,185 > Random 0,156 |
| BA-Shapes AUROC | PG 0,800 > GNNE 0,739 > NF 0,653 > Random 0,500 > Sal 0,421 | PG 0,731 > NF 0,668 > GNNE 0,631 > Sal 0,527 > Random 0,500 |
| BA-Shapes Inv@5 (thấp = tốt) | Sal 0,090 < NF 0,151 < GNNE 0,211 < PG 0,325 < Random 0,398 | GNNE 0,058 < Sal 0,059 < NF 0,091 < PG 0,265 < Random 0,293 |
| Cora Drop@5 | Sal 0,044 > GNNE 0,032 > NF 0,011 > PG −0,008 > Random −0,013 | GNNE 0,080 > Sal 0,074 > NF 0,039 > PG 0,009 > Random 0,003 |

Quan sát: (a) Nearest-First đứng đầu Drop@5 trên BA-Shapes ở cả hai thời điểm; PGExplainer đứng đầu AUROC ở cả hai thời điểm và đứng thứ tư về Drop@5; Random đứng cuối Drop@5. (b) Thứ tự giữa Saliency và GNNExplainer (Drop@5), giữa NF và GNNExplainer (AUROC) và giữa Saliency và GNNExplainer (Cora) đảo chỗ; các khác biệt này nhỏ so với CI (Mục 5.5). (c) Ở giữa kỳ, độ lệch chuẩn theo nút của Drop@5 BA-Shapes là 0,22–0,41 trên chỉ 20 nút (`midterm_reference.csv`), nên các thứ hạng giữa kỳ không có CI để so. (d) Giá trị AUROC của PGExplainer giảm từ 0,800 xuống 0,731 và, theo từng cấu hình (g, e), dao động từ 0,488 đến 0,944 (`results/analysis/a3_final/a3_0_metrics.csv`).

### 5.5. Đọc kết quả

1. **BA-Shapes, Fidelity.** Về Drop@5, Nearest-First (0,291), GNNExplainer (0,270) và Saliency (0,248) có CI chồng lấn nhau gần như hoàn toàn. Hiệu cặp theo nút (tính lại từ `results/eval/bashapes_test/metrics.csv`, bootstrap trên nút, chưa hiệu chỉnh Holm): Nearest-First − GNNExplainer = +0,021 [−0,048; 0,086], còn Nearest-First − Saliency = +0,043 [0,016; 0,073]. Cả ba cao hơn Random (0,156) rõ ràng (GNNExplainer − Random = +0,114 [0,082; 0,152]). PGExplainer (0,185) thấp hơn GNNExplainer 0,085 [0,048; 0,128] và chỉ cao hơn Random 0,029 [0,013; 0,046]. Về Inv@5, GNNExplainer (0,058) và Saliency (0,059) thấp nhất; Nearest-First 0,091; PGExplainer (0,265) và Random (0,293) có CI chồng lấn và đều cách xa ba phương pháp còn lại.
2. **BA-Shapes, khớp motif.** PGExplainer cao nhất ở AUROC (0,731), AP (0,767) và Precision@5 (0,549). Nhưng CI của PGExplainer rộng (AUROC [0,652; 0,808]) và chồng lấn với Nearest-First và GNNExplainer; chỉ AUROC của Saliency (CI trên 0,598) và của Random (0,5) nằm hẳn dưới CI của PGExplainer. Precision@5 của mọi phương pháp, kể cả Random (0,463), nằm trong khoảng 0,46–0,55 với CI chồng lấn. Nearest-First, không dùng mô hình, đạt AUROC 0,668 cao hơn GNNExplainer (0,631) và Saliency (0,527).
3. **Cora.** GNNExplainer (Drop 0,080; Inv −0,129) và Saliency (0,074; −0,079) dẫn đầu, hiệu của chúng +0,006 [−0,017; 0,030] không phân biệt được. Nearest-First (0,039) thấp hơn cả hai (so GNNExplainer: −0,041 [−0,065; −0,017]). **PGExplainer có Drop@5 = 0,009 [−0,012; 0,031] và Inv@5 = 0,022 [−0,003; 0,048], không khác Random (0,003; 0,022):** hiệu PGExplainer − Random về Drop là +0,006 [−0,002; 0,014] và về Inv là −0,001 [−0,012; 0,011]. Inv âm của GNNExplainer, Saliency và Nearest-First cho thấy chỉ giữ top-5 cạnh còn làm xác suất lớp dự đoán tăng so với đồ thị đầy đủ.
4. **Biến thiên theo GNN.** SD giữa 5 checkpoint GNN của Drop@5 (`main_sd_over_gnn_seeds.csv`): BA-Shapes 0,006 (PG), 0,006 (NF), 0,011 (Random), 0,019 (GNNE), 0,024 (Saliency); Cora 0,006–0,024. Chênh lệch Nearest-First – GNNExplainer (0,021) cỡ bằng SD giữa GNN của GNNExplainer (0,019), nên thứ hạng giữa hai phương pháp này không vững khi đổi checkpoint. AUROC của PGExplainer có SD giữa GNN lớn nhất (0,075; từ 0,659 đến 0,839 theo checkpoint). CI bootstrap trên nút của Mục 5.1 không bao hàm nguồn biến thiên này (Mục 9.1).
5. **Hai metric không chọn cùng một phương pháp tốt nhất.** Ở BA-Shapes, Nearest-First cao nhất về Drop@5 còn PGExplainer cao nhất về AUROC, AP và Precision@5. Mục 7 kiểm tra mâu thuẫn này trên 25 cấu hình; mục này không kết luận nguyên nhân.
