# BẢNG SỐ LIỆU FB-2: INSTABILITY VS DISAGREEMENT

> Mọi số trong tài liệu này lấy từ các file trong repo; tên file ghi cạnh số hoặc ở đầu mỗi bảng. Số thập phân dùng dấu phẩy. Agr = κ@5 (budget 5) trừ khi ghi khác. CI 95% là bootstrap lấy mẫu lại **nút** (2.000 lần, seed 0). Nguồn gốc: `results/analysis/a2/` (sinh bởi `scripts/analyze_variance.py`, `manifest.json` ghi commit `b786475`, `git_dirty: false`) và `results/eval/{bashapes,cora}_test/`.

## (a) Dữ liệu và lưới chạy

| Hạng mục | BA-Shapes | Cora |
|---|---|---|
| Nút test | 140 | 100 |
| Nút có κ@5 xác định | 140 | **86** (14 nút có m ≤ 5 cạnh ứng viên nên $k_{\rm eff}=m$, κ là NaN) |
| Checkpoint GCN | 5 (seed 1000–1004) | 5 (seed 1000–1004) |
| Seed explainer | 0–4 | 0–4 |
| GNNExplainer, PGExplainer | 25 lượt (g, e) | 25 lượt (g, e) |
| Saliency | e0 trên mọi g; thêm e1 trên g = 1000 | như BA-Shapes |

Hệ quả cho Saliency: chỉ có **một** cặp seed explainer (e0, e1, trên g = 1000), nên $I_{\rm within}$ (5 GNN) của Saliency trùng $I_{\rm within}$ (g = 1000) trong Bảng A2-1; đây không phải hai phép đo độc lập. Tỷ lệ nút đổi dự đoán lấy trên 10 cặp GNN (`pred_change.csv`).

## (b) Bảng A2-1 và A2-2

### Bảng A2-1. Bất ổn nội tại và độ nhạy theo GNN (κ@5, trung bình theo nút [CI 95%])

Nguồn: `results/analysis/a2/a2_1_within_cross_gnn.csv`. "I_gnn, cùng e" là định nghĩa của kế hoạch (Mục 7.3): ghép $(g,e)$ với $(g',e)$. "I_gnn, khác e" là phép đối chứng `cross_gnn_diff_e`: ghép $(g,e)$ với $(g',e')$, $e'$ là seed kế tiếp theo vòng tròn (Saliency không có, vì chỉ có e0 trên các GNN khác). Cột cuối là Agr với Nearest-First (`ref_nearest_first`). I_gnn chỉ tính trên nút mà hai GNN dự đoán cùng lớp.

**BA-Shapes (140 nút test)**

| Phương pháp | I_within, g=1000 (Bước 1) | I_within, 5 GNN | I_gnn, cùng e | I_gnn, khác e (đối chứng) | Agr với Nearest-First | n |
|---|---|---|---|---|---|---|
| Saliency | 0,968 [0,940; 0,990] | 0,968 [0,940; 0,990] | 0,855 [0,821; 0,885] | — | 0,425 [0,372; 0,480] | 140 |
| GNNExplainer | 0,488 [0,424; 0,552] | 0,499 [0,440; 0,558] | 0,825 [0,794; 0,855] | 0,403 [0,351; 0,457] | 0,146 [0,101; 0,190] | 140 |
| PGExplainer | 0,640 [0,601; 0,675] | 0,650 [0,620; 0,680] | 0,611 [0,580; 0,642] | 0,574 [0,538; 0,609] | 0,066 [0,035; 0,099] | 140 |

**Cora (100 nút test, 86 nút có κ@5)**

| Phương pháp | I_within, g=1000 (Bước 1) | I_within, 5 GNN | I_gnn, cùng e | I_gnn, khác e (đối chứng) | Agr với Nearest-First | n |
|---|---|---|---|---|---|---|
| Saliency | 1,000 [1,000; 1,000] | 1,000 [1,000; 1,000] | 0,821 [0,791; 0,849] | — | 0,507 [0,464; 0,551] | 86 |
| GNNExplainer | 0,788 [0,749; 0,824] | 0,798 [0,771; 0,823] | 0,711 [0,677; 0,744] | 0,628 [0,590; 0,668] | 0,283 [0,243; 0,326] | 86 |
| PGExplainer | 0,691 [0,654; 0,726] | 0,752 [0,730; 0,772] | 0,308 [0,276; 0,340] | 0,296 [0,263; 0,329] | -0,013 [-0,039; 0,011] | 86 |

Mốc Random–Random (`agreement_summary.csv`, `ref_random`): κ@5 = -0,004 (BA-Shapes), -0,001 (Cora). Tỷ lệ nút đổi dự đoán giữa hai GNN (`pred_change.csv`, 10 cặp GNN): BA-Shapes trung bình 1,6% (0,0–3,6%, tức tối đa 5/140 nút); Cora trung bình 4,5% (3–6%).

### Bảng A2-2. Bất đồng giữa các phương pháp (κ@5)

Nguồn: `results/analysis/a2/a2_2_between.csv`. Δ_A = I_within(A) − D_between, Δ_B tương tự, Δ_min = min(I_within(A), I_within(B)) − D_between (min lấy theo từng nút). I_within ở đây là trung bình trên 5 GNN. Cột "D_avg − D_between" so $D_{\rm between\_avg}$ (xếp hạng trung bình qua 5 seed) với $D_{\rm between}$. p là $p_{\rm holm}$ của Wilcoxon cặp, cách đọc ở (e)6.

**BA-Shapes (n = 140)**

| Cặp | D_between | D_between_avg | Δ_A [CI 95%] | Δ_B [CI 95%] | Δ_min [CI 95%] | p_holm (Δ_A, Δ_B, Δ_min) | D_avg − D_between [CI 95%] | p_holm |
|---|---|---|---|---|---|---|---|---|
| Saliency – GNNExplainer | 0,143 | 0,180 | 0,825 [0,767; 0,882] | 0,356 [0,287; 0,426] | 0,329 [0,263; 0,401] | < 0,001 cả ba | 0,037 [0,022; 0,053] | 0,002 |
| Saliency – PGExplainer | 0,102 | 0,098 | 0,865 [0,815; 0,915] | 0,548 [0,491; 0,605] | 0,538 [0,482; 0,594] | < 0,001 cả ba | -0,005 [-0,019; 0,008] | 0,361 |
| GNNExplainer – PGExplainer | 0,048 | 0,044 | 0,451 [0,389; 0,513] | 0,603 [0,558; 0,646] | 0,313 [0,265; 0,364] | < 0,001 cả ba | -0,004 [-0,019; 0,013] | 0,095 |

**Cora (n = 86)**

| Cặp | D_between | D_between_avg | Δ_A [CI 95%] | Δ_B [CI 95%] | Δ_min [CI 95%] | p_holm (Δ_A, Δ_B, Δ_min) | D_avg − D_between [CI 95%] | p_holm |
|---|---|---|---|---|---|---|---|---|
| Saliency – GNNExplainer | 0,342 | 0,345 | 0,658 [0,608; 0,711] | 0,456 [0,408; 0,505] | 0,456 [0,408; 0,505] | < 0,001 cả ba | 0,003 [-0,010; 0,015] | 0,477 |
| Saliency – PGExplainer | -0,010 | -0,005 | 1,010 [0,982; 1,038] | 0,762 [0,725; 0,797] | 0,762 [0,725; 0,797] | < 0,001 cả ba | 0,005 [-0,002; 0,012] | 0,477 |
| GNNExplainer – PGExplainer | -0,017 | -0,021 | 0,815 [0,783; 0,847] | 0,769 [0,735; 0,801] | 0,722 [0,691; 0,754] | < 0,001 cả ba | -0,004 [-0,016; 0,008] | 0,477 |

Số phụ ở budget 5 (`agreement_summary.csv`, cho D_between theo Jaccard@5 và τ_b): BA-Shapes Jaccard@5 0,348 (GNNExplainer–PG), 0,404 (Saliency–GNNExplainer), 0,371 (Saliency–PG); Cora 0,269, 0,467, 0,275. Mốc Random–Random Jaccard@5 là 0,316 (BA-Shapes) và 0,275 (Cora), nên Jaccard chưa hiệu chỉnh cho trùng ngẫu nhiên và không dùng làm kết luận chính.

### Bảng A2-3. SD của Fid-@5 (Drop) theo seed explainer so với chênh lệch giữa phương pháp

Nguồn: `results/analysis/a2/a2_3_fid_sd_vs_method_diff.csv`. SD (ddof = 1) qua 5 seed explainer trong từng (g, nút), rồi trung bình theo nút qua g. Chênh lệch giữa phương pháp = |Drop@5 trung bình của A − của B| tính theo nút. Cora dùng đủ 100 nút (Fid không cần κ).

| Dataset | SD trong nút: Saliency / GNNExp / PG | |chênh| Sal–GNNExp | |chênh| Sal–PG | |chênh| GNNExp–PG | Tỷ lệ nút SD(PG) > chênh GNNExp–PG | Tỷ lệ nút SD(GNNExp) > chênh GNNExp–PG |
|---|---|---|---|---|---|---|
| BA-Shapes | 0 / 0,011 / 0,045 | 0,183 | 0,125 | 0,099 | 42,1% | 26,4% |
| Cora | 0 / 0,019 / 0,013 | 0,080 | 0,109 | 0,083 | 8,0% | 13,0% |

Tỷ lệ nút mà SD của Saliency vượt chênh lệch là 0% (SD bằng 0). Với Saliency–GNNExp trên BA-Shapes, SD(GNNExp) vượt chênh ở 0,7% nút; với Saliency–PG, SD(PG) vượt chênh ở 33,6% nút (BA-Shapes) và 10% (Cora).

## (c) Trả lời ba câu của Mục 7.1

**1. Một phương pháp có tự lệch với chính nó không?** Có, và mức độ phụ thuộc phương pháp. Với g = 1000 (Bước 1), $I_{\rm within}$ của GNNExplainer là 0,488 trên BA-Shapes và 0,788 trên Cora; của PGExplainer là 0,640 và 0,691; của Saliency là 0,968 và 1,000. Qua 5 GNN (Bước 2) các số gần như không đổi trên BA-Shapes (0,499 và 0,650) và tăng ở PGExplainer trên Cora (0,752). Thứ tự bất ổn khác nhau giữa hai dataset: trên BA-Shapes GNNExplainer bất ổn nhất (0,499 so với 0,650 của PG; CI không giao nhau), trên Cora PGExplainer thấp nhất nhưng CI của hai phương pháp ngẫu nhiên gần chạm nhau (0,752 [0,730; 0,772] và 0,798 [0,771; 0,823]). SD của Fid-@5 theo seed explainer (Bảng A2-3) là 0,011 (GNNExplainer) và 0,045 (PGExplainer) trên BA-Shapes, 0,019 và 0,013 trên Cora; Saliency bằng 0.

**2. Đổi seed train GNN thì giải thích đổi bao nhiêu?** Tính trên nút mà hai GNN dự đoán cùng lớp (tỷ lệ nút đổi dự đoán trung bình 1,6% trên BA-Shapes, 4,5% trên Cora). Theo định nghĩa của kế hoạch (cùng e): Saliency 0,855 (BA-Shapes) và 0,821 (Cora), thấp hơn rõ so với $I_{\rm within}$ (0,968 và 1,000); GNNExplainer 0,825 và 0,711; PGExplainer 0,611 và 0,308. Số cùng e của GNNExplainer bị thổi phồng bởi khởi tạo mask chung seed (xem (e)2); số đối chứng (khác e) là 0,403 (BA-Shapes) và 0,628 (Cora). PGExplainer trên Cora là trường hợp nhạy GNN mạnh nhất: 0,308 cùng e và 0,296 khác e, so với $I_{\rm within}$ 0,752. Trên BA-Shapes, PGExplainer ít nhạy hơn (0,611 so với 0,650, CI chồng nhau).

**3. Hai phương pháp lệch nhau có nhiều hơn mức chúng tự lệch không?** Có, ở cả sáu cặp (3 cặp × 2 dataset): $D_{\rm between}$ thấp hơn $I_{\rm within}$ của cả hai phương pháp, CI 95% của Δ_A và Δ_B đều dương và $p_{\rm holm} < 0{,}001$. $D_{\rm between}$ trên BA-Shapes là 0,143 (Saliency–GNNExplainer), 0,102 (Saliency–PG), 0,048 (GNNExplainer–PG); trên Cora là 0,342, -0,010, -0,017. Trên Cora, mọi cặp có PGExplainer có $D_{\rm between}$ quanh 0, tức mức Random–Random (-0,001); PGExplainer cũng không khác Nearest-First (-0,013 [-0,039; 0,011]). Gộp seed không đổi kết luận: $D_{\rm between\_avg}$ chỉ cao hơn $D_{\rm between}$ đáng kể ở một cặp (Saliency–GNNExplainer, BA-Shapes: +0,037 [0,022; 0,053]).

## (d) Áp dụng bảng quy tắc diễn giải (Mục 7.4)

Quy tắc không định lượng "thấp hơn rõ" ở dòng 4; ở đây người viết hiểu là CI 95% của hai đại lượng không giao nhau (so bằng CI riêng lẻ, không phải kiểm định cặp). Dòng 1 dùng đúng tiêu chí của quy tắc (CI của Δ > 0).

| Dòng quy tắc | Quan sát | Thỏa? |
|---|---|---|
| 1. $D_{\rm between}$ thấp hơn rõ cả $I_{\rm within}(A)$ và $I_{\rm within}(B)$ (CI của Δ_A và Δ_B đều > 0) | Cả 6 cặp (BA-Shapes và Cora): CI của Δ_A, Δ_B và Δ_min đều nằm trên 0; $p_{\rm holm} < 0{,}001$ | **Thỏa** → bất đồng thực sự, không giải thích được bằng nhiễu seed |
| 2. $I_{\rm within}(M)$ thấp và gần bằng $D_{\rm between}$ | Không có cặp nào: nhỏ nhất là Δ_min = 0,313 (GNNExplainer–PG, BA-Shapes), tức $I_{\rm within}$ vẫn cao hơn $D_{\rm between}$ ít nhất 0,26 (CI dưới) | **Không thỏa** |
| 3. $D_{\rm between\_avg}$ cao hơn hẳn $D_{\rm between}$ | Chỉ Saliency–GNNExplainer trên BA-Shapes: +0,037 [0,022; 0,053], $p_{\rm holm}$ = 0,002 (0,143 → 0,180). Năm cặp còn lại CI chứa 0 | **Chỉ thỏa yếu, một cặp**; không đủ để nói "hơn hẳn" |
| 4. $I_{\rm gnn}$ thấp hơn rõ $I_{\rm within}$ | CI không giao nhau ở 4/6 ô (dataset × phương pháp): Saliency (cả hai dataset), GNNExplainer Cora (0,711 [0,677; 0,744] so với 0,798 [0,771; 0,823]), PGExplainer Cora (0,308 [0,276; 0,340] so với 0,752 [0,730; 0,772]). Không thỏa: PGExplainer BA-Shapes (0,611 so với 0,650, CI chồng nhau); GNNExplainer BA-Shapes (số cùng e 0,825 cao hơn $I_{\rm within}$ 0,499 vì khởi tạo chung; số khác e 0,403 thấp hơn nhưng CI chồng nhau) | **Thỏa ở 4/6 ô** → giải thích nhạy với việc huấn luyện lại GNN, rõ nhất ở PGExplainer trên Cora |
| 5. Saliency có $I_{\rm within}=1$ nhưng $I_{\rm gnn}$ thấp | Cora: 1,000 so với 0,821 [0,791; 0,849]. BA-Shapes: 0,968 so với 0,855 [0,821; 0,885] ($I_{\rm within}$ không đúng bằng 1, xem (e)1) | **Thỏa** |


## (e) Lưu ý khi đọc số

1. **Saliency tất định.** Hai lượt e0 và e1 trên g = 1000 cho điểm thô giống nhau: τ_b giữa hai lượt là 1,000 trên cả hai dataset (độ lệch chuẩn 7×10⁻¹⁷ và 1×10⁻¹⁶, tức sai số làm tròn; `agreement_summary.csv`, `kind = within`), và `scripts/check_scores.py` dừng với lỗi nếu `max_diff` của Saliency giữa các seed lớn hơn 0 (dòng 2149–2164 của file). Các file điểm thô và `max_diff` không nằm trong repo (`run_log.json` không ghi `max_diff`), nên bằng chứng trong repo là τ_b và mã kiểm tra. κ@5 within của Saliency trên BA-Shapes là 0,968 chứ không phải 1, và Jaccard@5 là 0,983, trong khi τ_b bằng 1. Chênh lệch này đến từ cách chọn top-5 khi nhiều cạnh cùng điểm (phá hòa ngẫu nhiên có seed, Mục 3.8); đó là hệ quả của điểm hòa, không phải bất ổn của thuật toán. Trên Cora κ@5 bằng 1,000.
2. **Cross-GNN cùng e bị thổi phồng với GNNExplainer.** Theo Mục 7.3, cross-GNN ghép $(g,e)$ với $(g',e)$ cùng e. GNNExplainer khởi tạo mask từ seed, nên hai lượt cùng e dùng chung khởi tạo (`src/sna/evaluation/runner_v2.py`, chú thích tại `cross_gnn_rows`). Số liệu: BA-Shapes 0,825 (cùng e) so với 0,403 (khác e); Cora 0,711 so với 0,628. Với PGExplainer chênh lệch nhỏ (0,611 so với 0,574; 0,308 so với 0,296). Khuyến nghị: khi diễn giải "nhạy theo GNN", dùng số đối chứng `cross_gnn_diff_e`. Số này vẫn lẫn hai nguồn thay đổi (đổi cả GNN lẫn seed explainer), nên là cận trên của độ nhạy theo GNN cộng với bất ổn seed, không phải độ nhạy GNN thuần.
3. **PGExplainer nhạy với nhiễu số học.** So hai lượt chạy lưới PG trên BA-Shapes cùng seed nhưng khác số luồng CPU (lượt 22/11 và lượt cuối kỳ với `OMP_NUM_THREADS=2`; `results/analysis/a2/pg_thread_sensitivity.csv`, 25 cấu hình): κ@5 trung bình giữa hai lượt là 0,801 (nhỏ nhất 0,396, lớn nhất 0,970); |AUROC motif cũ − mới| trung bình 0,089, lớn nhất 0,474. Cấu hình (g = 1000, e = 3) có AUROC 0,158 ở lượt cũ và 0,632 ở lượt cuối kỳ. Ý nghĩa: PGExplainer chỉ tái lập chính xác khi giữ nguyên số luồng (2, theo `configs/protocol_final.yaml`); mọi phân tích cuối kỳ dùng lượt mới (`docs/CHANGELOG_PROTOCOL.md`, 23/11). Việc chạy lại một ô cho điểm trùng tuyệt đối (lệch 0,0) ở cả hai dataset được ghi trong `docs/PHAN_BAO_CAO_PGEXPLAINER_THIEN.md` (mục chạy lưới, ô g = 1003, e = 2; đo trên lượt 22/11, chưa kiểm lại trên lượt cuối kỳ). Phép so này chỉ có cho BA-Shapes; chưa đo trên Cora. Số này cho thấy PG nhạy với nhiễu số học; nó không đo $I_{\rm within}$ và không làm PG thành phương pháp bất ổn nhất theo $I_{\rm within}$ (trên BA-Shapes GNNExplainer thấp hơn).
4. **Tỷ lệ nút đổi dự đoán.** Giữa hai GNN: BA-Shapes trung bình 1,6% (từ 0,0% đến 3,6%); Cora trung bình 4,5% (từ 3,0% đến 6,0%) trên 10 cặp (`pred_change.csv`, cột `pred_change_rate`). Cross-GNN chỉ tính trên nút hai GNN dự đoán cùng lớp, nên I_gnn không đo độ nhạy trên nút đã đổi dự đoán.
5. **Bảng A2-3.** SD của Fid-@5 theo seed explainer nhỏ so với chênh lệch giữa phương pháp ở mức trung bình (BA-Shapes: SD 0,011 và 0,045 so với chênh 0,099 đến 0,183; Cora: SD 0,013 và 0,019 so với chênh 0,080 đến 0,109), nhưng theo từng nút không phải lúc nào cũng vậy: SD của PGExplainer vượt chênh GNNExplainer–PG ở 42,1% nút BA-Shapes. Do đó thứ hạng phương pháp theo Fid-@5 ở mức nút có thể đảo theo seed. Tỷ lệ này chưa quy ra mức ý nghĩa thống kê.
6. **Thống kê.** Mọi so sánh dùng bootstrap theo nút (2.000 lần), kiểm định Wilcoxon cặp trên nút, hiệu chỉnh Holm trong từng dataset; họ kiểm định gồm mọi phép so sánh trong `a2_2_between.csv` của dataset đó (3 cặp × {Δ_A, Δ_B, Δ_min, D_avg − D_between} = 12 phép mỗi dataset). Cột `p_holm` là $p$ đã hiệu chỉnh. CSV lưu giá trị thật (nhỏ nhất ở mức 10⁻²⁴ trước hiệu chỉnh); bảng in "< 0,001" khi $p_{\rm holm} < 0{,}001$ (mọi dòng Δ_A, Δ_B, Δ_min đều như vậy). Với n = 140 hoặc 86 và hiệu ứng lớn, p rất nhỏ không bổ sung thông tin ngoài CI; kích thước hiệu ứng là CI của Δ. Lưu ý: GNNExplainer–PG BA-Shapes ở dòng $D_{\rm avg} - D_{\rm between}$ có $p_{\rm raw}$ = 0,047 nhưng $p_{\rm holm}$ = 0,095.
7. **Giới hạn.** Mọi suy luận dựa trên 5 GNN × 5 seed explainer và 140 nút (BA-Shapes) hoặc 100 nút (Cora, 86 nút có κ@5); chỉ kiến trúc GCN (GAT không làm). 5 GNN chỉ khác nhau ở seed huấn luyện, nên "độ nhạy theo GNN" là độ nhạy theo seed huấn luyện, không phải theo kiến trúc. Hai dataset khác bản chất (đồ thị tổng hợp có motif và đồ thị trích dẫn), không nên gộp. Quy tắc ở (d) được ghi trong kế hoạch trước khi có số liệu lưới; cách hiểu "thấp hơn rõ" ở dòng 4 là của người viết báo cáo.
