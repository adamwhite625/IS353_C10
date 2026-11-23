## 9. Hạn chế và kết luận

### 9.1. Hạn chế

**Phạm vi.**

1. **Chỉ GCN 2 lớp.** Mini-Proposal dự kiến GCN và GAT; báo cáo chỉ có GCN. Cách truyền `edge_weight` (Mục 3.2) đúng với `GCNConv`; GAT cần cách xử lý khác (cộng log trọng số vào logit attention trước softmax) mà nhóm mới có thiết kế, chưa cài đặt.
2. **Chỉ hai dataset.** BA-Shapes (đồ thị tổng hợp, có ground truth cạnh) và Cora (mạng trích dẫn, không có ground truth). Không suy rộng được cho đồ thị thuộc loại khác.
3. **Tầng B không làm** (B1 đặc trưng hằng với GCN 3 lớp, B2 GAT, B3 Amazon Photo): theo FB-4, nhóm làm kiểm chứng cốt lõi trước (Mục 8.2). Hệ quả: giả thuyết đặc trưng bậc (T3) chỉ có phép tính lại đặc trưng bậc sau khi xóa cạnh, không có thí nghiệm nhân quả bằng đặc trưng hằng.
4. **Error analysis (A3) chỉ chạy trên BA-Shapes**, không có bản tương ứng trên Cora (`results/analysis/a3_final/manifest.json` chỉ gồm nút BA-Shapes). Ở Cora không có ground truth nên không có góc phần tư theo AUROC; những gì nói về Cora chỉ dựa trên Fidelity và độ đồng thuận.

**Cỡ mẫu và thống kê.**

5. **Nút đánh giá.** BA-Shapes dùng toàn bộ 140 nút test, nhưng AUROC và AP chỉ xác định ở 71/140 nút (69 nút còn lại không có đồng thời cạnh motif và cạnh nền). Cora dùng 100 nút lấy mẫu phân tầng, không phải 1.000 nút test của Planetoid; lựa chọn này ghi ở giao thức (`configs/protocol_final.yaml`). Kế hoạch (Mục 7.2) chỉ ước tính GNNExplainer trên Cora mất 1,5–3 giờ cho 100 nút × 25 lượt; không tài liệu nào nêu riêng lý do chọn 100 nút. Ở Cora, κ@5 không xác định ở 14/100 nút vì có m ≤ 5 cạnh ứng viên (còn 86 nút). Ở k = 10, 65/140 nút BA-Shapes và 29/100 nút Cora có m ≤ 10, nên top-k là toàn bộ cạnh.
6. **Độc lập của mẫu.** Các cấu hình (g, e) dùng chung nút và dữ liệu, nên không độc lập. CI bootstrap trên nút (Mục 5) chỉ phản ánh biến thiên giữa các nút, không phản ánh biến thiên giữa 5 GNN: SD của Drop@5 giữa các GNN là 0,006–0,024 (`main_sd_over_gnn_seeds.csv`), cỡ bằng chênh lệch Nearest-First – GNNExplainer (0,021) trên BA-Shapes. Năm seed GNN và năm seed explainer đủ để mô tả bằng trung bình và CI, không đủ để suy diễn về phân phối.
7. **Kiểm định.** Các hiệu cặp ở Mục 5.5 được tính lại bằng bootstrap trên nút, chưa hiệu chỉnh Holm (khác với bảng A2).

**Phương pháp giải thích.**

8. **Siêu tham số explainer không tinh chỉnh.** Giữ như giữa kỳ (GNNExplainer 200 bước, lr 0,01, λ_size = 0,005, λ_ent = 0,1; PGExplainer 40 epoch, lr 0,005), không chỉnh trên tập test. PGExplainer chưa được tinh chỉnh và trên Cora không khác Random về Drop@5 (Mục 5.5); báo cáo không kết luận gì về mức tốt nhất có thể của PGExplainer.
9. **Size-loss của GNNExplainer.** Lượt chạy chính dùng `size_loss_mode = sum` theo biên bản F10 (đo trên 20 nút validation; `run_log.json` ghi `sum` ở mọi lượt), ghi vào `docs/CHANGELOG_PROTOCOL.md` ngày 23/11 và đang chờ nhóm trưởng xác nhận. Nếu đổi về `mean`, phải chạy lại GNNExplainer: κ@5 giữa hai công thức là 0,908 (BA-Shapes) và 0,208 (Cora) trên 20 nút validation (`docs/BIEN_BAN_F10_GNNEXPLAINER_PYG.md`), nên kết quả Cora sẽ đổi nhiều hơn BA-Shapes.
10. **PGExplainer rất nhạy nhiễu số học.** Huấn luyện PGExplainer chỉ tái lập chính xác khi giữ nguyên cấu hình CPU, kể cả số luồng (2 luồng, `OMP_NUM_THREADS=2`). Một ô (g = 1003, e = 2) chạy lại trên cùng cấu hình cho điểm trùng tuyệt đối (`docs/PHAN_BAO_CAO_PGEXPLAINER_THIEN.md`, Mục 3b). So hai lượt chạy khác số luồng theo `docs/CHANGELOG_PROTOCOL.md` (22/11 và 23/11, BA-Shapes, 25 cấu hình), κ@5 trung bình giữa hai lượt chỉ 0,80 (nhỏ nhất 0,40, ở g = 1000, e = 0) và AUROC của cùng một ô có thể lệch tới 0,47 (g = 1000, e = 3: 0,158 so với 0,632) (`results/analysis/a2/pg_thread_sensitivity.csv`). CSV không ghi số luồng của lượt 22/11. Mọi kết quả cuối kỳ dùng lượt 23/11. Kết luận về PGExplainer vì vậy gắn với cấu hình này.
11. **Điểm hòa.** Ở PGExplainer, điểm thứ 5 bằng điểm thứ 6 ở 661/3.500 cặp (nút, cấu hình) trên BA-Shapes, tức 18,9% (`t4_tie_sensitivity.csv`; lượt 22/11 là 19,3%). Nearest-First hòa ở 75,7% cặp vì chỉ có hai mức (cạnh 1-hop và 2-hop). Saliency tất định theo seed explainer nhưng hòa ở 11,9% cặp, nên κ@5 giữa hai lượt Saliency trên BA-Shapes là 0,968 chứ không phải 1 (trên Cora là 1,0), trong khi Drop@5 của hai lượt trùng nhau ở cả 140 nút (tính lại từ `metrics.csv`). Chúng tôi đo độ nhạy bằng 5 seed phá hòa, không loại bỏ nguồn bất định này.

**Đánh giá và tái lập.**

12. **Fidelity trên đồ thị con 3-hop.** Fidelity được tính trên đồ thị con 3-hop quanh nút đích. Với GCN 2 lớp, kết quả tương đương toàn đồ thị với sai khác khoảng 10⁻⁷ (ghi chú trong `src/sna/evaluation/core.py` và `manifest.json` của bộ đánh giá; với PGExplainer, độ lệch logit đã đo là 0,0 trên BA-Shapes và 7,2·10⁻⁷ trên Cora, Mục 3.5). Báo cáo không đo lại sai khác này trên từng nút test.
13. **Xóa cạnh có thể ra ngoài phân phối.** Ở k = 5, Drop < 0 chiếm 19–20% số lượt (nút, seed explainer, seed phá hòa) của Saliency, GNNExplainer, Nearest-First và 38,4% của PGExplainer trên BA-Shapes (`a3_2_mechanisms.csv`, tổng `t5_negative_runs` chia `t5_total_runs` trên 5 GNN); báo cáo giữ nguyên giá trị âm. Motif AUROC đo độ khớp với motif do người thiết kế, không bảo đảm mô hình dựa vào motif.
14. **Không dùng GPU.** Protocol khóa CPU để kết quả tất định (kế hoạch Mục 4, 14). Báo cáo không có số liệu GPU, nên không biết kết quả có đổi trên GPU hay không.
15. **Tái lập giữa các máy.** Train lại checkpoint trên máy khác không cho cùng SHA-256; mốc tái lập là hash của file `.pt` (mười checkpoint đã được đưa vào Git). Nhóm không kiểm tra tính tất định giữa các hệ điều hành hoặc phiên bản thư viện: checkpoint do PN1 tạo trên Python 3.12.14, torch 2.15.0+cpu; lượt PGExplainer đầu tiên chạy trên Windows, Python 3.12.10, torch 2.15.0+cpu; các manifest cuối kỳ ghi Python 3.12.14.
16. **D1.** Kết quả Fid/AUROC của bản gộp seed xem Mục 8.1; nếu không cải thiện, đó là kết quả được ghi nhận.

### 9.2. Kết luận

Các ý dưới đây gắn với câu hỏi nghiên cứu (Mục 1.3) và phản hồi của thầy (Mục 1.2). Con số lấy từ `results/analysis/a2/*.csv`, `results/analysis/a3_final/*.csv` và Mục 5. Các ý (1)–(3) cần đối chiếu lại với văn bản Mục 6; ý (4) với Mục 7.

1. **RQ1 (bất ổn định nội tại, FB-2 bước 1).** Mức tự lệch khác nhau rõ giữa các phương pháp. Với 5 GNN, κ@5 tự lệch của Saliency là 0,968 (BA-Shapes) và 1,000 (Cora); của GNNExplainer là 0,499 và 0,798; của PGExplainer là 0,650 và 0,752 (`a2_1_within_cross_gnn.csv`, cột `I_within_5gnn`). GNNExplainer trên BA-Shapes tự lệch nhiều nhất: hai lượt cùng GNN, khác seed, chỉ trùng ở mức κ@5 ≈ 0,5.
2. **RQ2 (nhạy theo GNN).** Đổi checkpoint GNN làm giải thích lệch rõ, kể cả khi dự đoán không đổi (chỉ tính cặp hai GNN cùng dự đoán; tỷ lệ nút đổi dự đoán trung bình là 1,6% ở BA-Shapes và 4,5% ở Cora). Trên Cora, κ@5 giữa hai GNN, cùng seed explainer, là 0,821 (Saliency), 0,711 (GNNExplainer) và 0,308 (PGExplainer), thấp hơn mức tự lệch (1,000; 0,798; 0,752), nên PGExplainer ở đó phụ thuộc mạnh vào việc huấn luyện lại GNN. Trên BA-Shapes, giá trị tương ứng là 0,855; 0,825; 0,611; riêng GNNExplainer cao hơn mức tự lệch vì hai lượt cùng seed dùng chung khởi tạo mask (khi khác seed là 0,403), nên với GNNExplainer chưa tách được tác động của GNN khỏi nhiễu seed.
3. **RQ3 (bất đồng thực sự).** Với cả ba cặp phương pháp trên cả hai dataset, κ@5 giữa hai phương pháp thấp hơn mức tự lệch của cả hai phương pháp, với CI 95% của hiệu đều nằm trên 0 (`a2_2_between.csv`, các dòng Delta_A và Delta_B). Theo quy tắc ghi trước ở Mục 6.3, đây là **bất đồng thực sự**, không giải thích được bằng nhiễu seed. Mức bất đồng (κ@5 giữa hai phương pháp) nằm trong khoảng −0,017 đến 0,342 và năm trong sáu giá trị không quá 0,15; hai cặp có PGExplainer trên Cora (−0,010 và −0,017) xấp xỉ mốc ngẫu nhiên. Gộp 5 seed (D1) chỉ làm tăng đồng thuận có ý nghĩa ở một cặp (Saliency – GNNExplainer trên BA-Shapes: +0,037; Mục 8.1).
4. **FB-3 (mâu thuẫn giữa metric).** Mâu thuẫn giữa kỳ vẫn còn trên tập test: trong 25 cấu hình (g, e) của BA-Shapes, phương pháp tốt nhất theo Drop@5 khác phương pháp tốt nhất theo AUROC ở 17/25 cấu hình (`a3_0_conflicts.csv`, dòng `average`). Nearest-First đứng đầu Drop@5 ở 20/25 cấu hình (GNNExplainer ở 5/25); PGExplainer đứng đầu AUROC ở 17/25 (Nearest-First ở 8/25). Do đó không nên chọn explainer theo một metric duy nhất. Cơ chế (T1–T5) và các nút minh họa là nội dung Mục 7; **cần đối chiếu câu kết luận về nguyên nhân với Mục 7 trước khi chốt**, vì Mục này không tự suy ra nguyên nhân từ số liệu.
5. **FB-1 và FB-4.** Quy trình tái lập có kiểm tra băm (dữ liệu, 10 checkpoint, CSV điểm; GATE 0 tái lập số giữa kỳ của Saliency và GNNExplainer, và chỉ ra PGExplainer giữa kỳ không tái lập chính xác được; Mục 4). Phạm vi bị giới hạn ở GCN và hai dataset theo FB-4; mọi kết luận ở trên không suy rộng ra GAT hay dataset khác. Ngoài ra, thứ hạng giữa các phương pháp theo Fidelity trên BA-Shapes không vững qua các checkpoint GNN khi các khác biệt nhỏ (Mục 5.5), và thứ hạng PGExplainer gắn với cấu hình số luồng CPU (Mục 9.1, ý 10).

### 9.3. Hướng tiếp theo

1. **GAT với trọng số cạnh** đưa vào attention trước softmax (cộng log trọng số vào logit), với kiểm tra w = 0 tương đương xóa cạnh (< 10⁻⁵) như đã làm cho GCN.
2. **Dataset khác** (ví dụ Amazon Photo) và toàn bộ 1.000 nút test của Cora; error analysis A3 trên Cora.
3. **Tinh chỉnh PGExplainer** trên tập validation (số epoch, nhiệt độ, hệ số phạt), kèm kiểm tra độ nhạy với số luồng và thiết bị; xem Cora có còn ngang Random hay không.
4. **Seed-ensemble** với ablation S ∈ {1, 3, 5} (thêm seed 5–9) và thử gộp điểm thay vì gộp thứ hạng.
5. **Thí nghiệm nhân quả cho giả thuyết đặc trưng bậc**: BA-Shapes đặc trưng hằng với GCN 3 lớp (B1).
