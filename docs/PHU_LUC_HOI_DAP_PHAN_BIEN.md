# PHỤ LỤC: HỎI – ĐÁP PHẢN BIỆN (NHÓM 09, C.10)

> Mở rộng Phụ lục B của `docs/KE_HOACH_CUOI_KY_FINAL.md` thành bản trả lời. Mỗi câu có người trả lời chính theo bảng phân công Mục 11 của kế hoạch (người trả lời là đề xuất, nhóm trưởng có thể đổi), số liệu và file nguồn.
> Quy ước: Drop = p(ŷ|G) − p(ŷ|G∖top-k), cao là tốt; Inv = p(ŷ|G) − p(ŷ|chỉ giữ top-k), thấp hoặc âm là tốt. Số lấy từ lượt chạy cuối (23/11/2026, CPU, 2 luồng). "CI" là khoảng tin cậy 95% bootstrap trên nút.
> Một số hiệu cặp dưới đây được tính lại từ `results/eval/*/metrics.csv` (bootstrap trên nút, 2.000 lần, chưa hiệu chỉnh Holm); chúng được đánh dấu "(tính lại)".

## Phần A. Các câu trong Phụ lục B của kế hoạch

**A1. Làm sao chắc số này tính đúng checkpoint?** — *Vũ, Đăng*
Mỗi dòng CSV điểm có cột `checkpoint_sha256`, mỗi `run_log.json` ghi `csv_sha256`, `data_sha256` và `node_list_sha256`. Bộ đánh giá (`src/sna/evaluation/runner_v2.py`) kiểm tra hash và dừng với `ProvenanceError` nếu lệch. Mỗi `manifest.json` ghi hash dữ liệu (BA-Shapes `8fbf94a7ce58…`, Cora `f2846a82cdd1…`) và hash giao thức; `results/analysis/a3_final/t4_tie_sensitivity.csv` mang lại hash checkpoint từng GNN (ví dụ g = 1000: `26465138c327f1a2…`). Mười checkpoint nằm trong Git; hash của file `.pt` là mốc tái lập (train lại trên máy khác không cho cùng hash). Nguồn: `results/checkpoints/checkpoint_table.csv`, `docs/CHANGELOG_PROTOCOL.md`.

**A2. Hai phương pháp bất đồng hay chỉ là nhiễu seed?** — *Đức*
Bất đồng thực sự. Với 5 GNN, κ@5 tự lệch (cùng GNN, khác seed) của Saliency, GNNExplainer, PGExplainer là 0,968; 0,499; 0,650 trên BA-Shapes và 1,000; 0,798; 0,752 trên Cora, trong khi κ@5 giữa hai phương pháp chỉ −0,017 đến 0,342. Hiệu Δ (tự lệch − giữa hai phương pháp) có CI 95% nằm trên 0 cho cả hai phương pháp của cả ba cặp, trên cả hai dataset (ví dụ BA-Shapes, GNNExplainer – PGExplainer: Δ so với mức tự lệch thấp nhất = 0,313 [0,265; 0,364]; Cora, Saliency – GNNExplainer: 0,456 [0,408; 0,505]). Gộp 5 seed (D1) không kéo hai phương pháp lại gần nhau, trừ cặp Saliency – GNNExplainer trên BA-Shapes (+0,037). Nguồn: `results/analysis/a2/a2_1_within_cross_gnn.csv`, `a2_2_between.csv`; Mục 6.3 của báo cáo.

**A3. Đổi seed GNN mà dự đoán đổi thì sao?** — *Đức*
Khi tính độ nhạy theo GNN, chỉ giữ các nút mà hai GNN dự đoán cùng lớp, và báo kèm tỷ lệ nút đổi dự đoán. Tỷ lệ trung bình giữa các cặp GNN là 1,6% ở BA-Shapes (lớn nhất 3,6%) và 4,5% ở Cora (lớn nhất 6,0%), nên việc loại các nút này ảnh hưởng ít. Nguồn: `a2_1_within_cross_gnn.csv`, cột `pred_change_rate_mean_over_gnn_pairs` và `pred_change_rate_max_over_gnn_pairs`; `results/eval/*_test/pred_change.csv`.

**A4. Vì sao PGExplainer tốt theo motif mà Fid thấp?** (xem thêm B1) — *Kha, Đăng*
Hai điều này đúng đồng thời trên tập test: PGExplainer có AUROC 0,731 [0,652; 0,808], AP 0,767 và Precision@5 0,549 cao nhất, nhưng Drop@5 chỉ 0,185 [0,141; 0,228], thấp hơn GNNExplainer 0,085 (tính lại: [0,048; 0,128]) và chỉ cao hơn Random 0,029 (tính lại: [0,013; 0,046]). Mâu thuẫn tồn tại ở 17/25 cấu hình (g, e) (Nearest-First đứng đầu Drop@5 ở 20/25; PGExplainer đứng đầu AUROC ở 17/25). Cơ chế được kiểm định ở Mục 7 (T1–T5); câu trả lời cuối phải khớp với văn bản Mục 7. Nguồn: `results/analysis/a3_final/a3_0_conflicts.csv`, `a3_2_mechanisms.csv`, `results/analysis/main/main_by_method_budget.csv`.

**A5. Fidelity âm nghĩa là gì?** — *Kha*
Drop < 0 nghĩa là sau khi xóa top-k cạnh, xác suất của lớp dự đoán tăng chứ không giảm: cạnh bị coi là "quan trọng" lại đang cản dự đoán, hoặc việc xóa tạo ra đồ thị khác phân phối huấn luyện. Ở k = 5 trên BA-Shapes, Drop < 0 ở 19–20% số lượt của Saliency, GNNExplainer, Nearest-First và 38,4% của PGExplainer (6.727/17.500). Inv âm là hiện tượng khác: chỉ giữ top-k làm xác suất tăng (Cora: GNNExplainer −0,129, Saliency −0,079). Nhóm giữ nguyên giá trị âm. Nguồn: `results/analysis/a3_final/a3_2_mechanisms.csv` (`t5_negative_runs`), Mục 5.1b; nguyên nhân cụ thể: Mục 7 (T5).

**A6. Vì sao dùng κ@k?** — *Đức*
Khi số cạnh ứng viên m nhỏ, hai tập top-k trùng nhau ngẫu nhiên khá nhiều, nên Jaccard bị thổi phồng. Trên tập test, Jaccard@5 của cặp Random – Random là 0,316 (BA-Shapes) và 0,275 (Cora), còn κ@5 là −0,004 và −0,001. κ hiệu chỉnh theo kỳ vọng ngẫu nhiên k²/m: 0 là mức ngẫu nhiên, 1 là trùng hoàn toàn, âm là kém ngẫu nhiên. κ không xác định khi k_eff = m (Cora: 14/100 nút có m ≤ 5, còn 86 nút). Nguồn: `results/eval/*_test/agreement_summary.csv`; kế hoạch, Phụ lục A.

**A7. Vì sao không dùng `set_masks` của PyG?** — *Khôi*
`set_masks` nhân mask sau bước chuẩn hóa bậc của GCN, nên mask = 0 không tương đương xóa cạnh và không khớp với cách đo Fidelity (xóa cạnh thật). Biên bản F10, 20 nút validation: `edge_weight = 0` lệch xác suất 0,0000 so với xóa cạnh thật (0/20 nút đổi dự đoán, cả hai dataset); `set_masks = 0` lệch tới 0,4998 (BA-Shapes, 6/20 nút đổi dự đoán) và 0,7508 (Cora, 10/20). Nguồn: `docs/BIEN_BAN_F10_GNNEXPLAINER_PYG.md`, Mục 4; báo cáo, Mục 3.2.

**A8. Kết quả giữa kỳ còn đúng không?** — *Thiện*
Một phần. GATE 0: nạp lại checkpoint giữa kỳ cho dự đoán khớp 700/700 (BA-Shapes) và 2.708/2.708 (Cora); chạy lại bộ đánh giá T06 cho 2.080 dòng metric, 0 lỗi, mean và SD khớp 4 chữ số; Saliency và GNNExplainer tái lập đến cỡ 10⁻⁷ (sau khi làm tròn cùng cách thì khớp tuyệt đối). PGExplainer giữa kỳ **không tái lập chính xác**: hai lượt cùng seed lệch tới 0,885 do lỗi thứ tự đặt seed. Số cuối kỳ khác số giữa kỳ vì tập nút, checkpoint, PG và size-loss GNNExplainer đã đổi; thứ hạng ở hai thời điểm xem Mục 5.4 của báo cáo. Nguồn: `docs/GATE0_TAI_LAP_GIUA_KY.md`, `results/gate0/verification.json`.

**A9. Sao không làm nhiều dataset?** — *Vũ*
Theo FB-4, nhóm làm kiểm chứng cốt lõi (tái lập, tách bất ổn định khỏi bất đồng, error analysis) trước rồi mới xét mở rộng. Tầng B (GAT, BA-Shapes đặc trưng hằng, Amazon Photo) không được thực hiện và được ghi ở Mục 9.1 là hạn chế; báo cáo chỉ có GCN, BA-Shapes và Cora, và kết luận không suy rộng cho kiến trúc hoặc dataset khác. Nguồn: kế hoạch, Mục 9; `docs/report_parts/muc8_d1_tangB.md`.

**A10. Siêu tham số explainer chọn thế nào?** — *Khôi, Thiện*
Giữ nguyên như giữa kỳ, không chỉnh trên tập test: GNNExplainer 200 bước Adam, lr 0,01, λ_size = 0,005, λ_ent = 0,1; PGExplainer 40 epoch, lr 0,005, nhiệt độ 5,0 → 0,5. Ngoại lệ có kiểm soát là F10 (size-loss `mean` → `sum`), đo trên 20 nút validation. Hệ quả là PGExplainer chưa được tinh chỉnh (Mục 9.1). Nguồn: `configs/datasets/*.yaml`, `docs/BIEN_BAN_F10_GNNEXPLAINER_PYG.md`.

## Phần B. Câu hỏi bổ sung

**B1. Vì sao PGExplainer kém Fid- (Drop) nhưng AUROC cao?** — *Kha, Đăng*
(Bổ sung cho A4.) Hai metric đo hai việc khác nhau: AUROC xếp hạng toàn bộ m cạnh theo ground truth motif, còn Drop chỉ nhìn top-5 và xem xóa chúng có làm đổi dự đoán không. Số liệu thô: tỷ lệ cạnh 1-hop trong top-5 (T1) là 0,24 ở PGExplainer, so với 0,33 (GNNExplainer), 0,67 (Saliency), 0,72 (Nearest-First). Với PGExplainer, điểm thứ 5 bằng điểm thứ 6 ở 661/3.500 (18,9%) cặp (nút, cấu hình), nên top-5 phụ thuộc phá hòa ở khoảng 1/5 trường hợp; AUROC của PGExplainer còn dao động rất mạnh theo cấu hình (0,488 đến 0,944; SD giữa GNN 0,075). Ở 71 nút xác định AUROC, PGExplainer rơi vào góc "cả hai cùng tốt" (Drop ≥ 0,10, AUROC ≥ 0,75) trung bình 42,4 nút và góc "đúng motif nhưng xóa không ảnh hưởng" 8,6 nút (trung bình 5 GNN). Nguyên nhân do T1–T5 giải thích: xem Mục 7. Nguồn: `results/analysis/a3_final/a3_2_mechanisms.csv`, `a3_1_counts.csv`, `a3_0_metrics.csv`, `t4_tie_sensitivity.csv`, `main_sd_over_gnn_seeds.csv`.

**B2. Vì sao Nearest-First thắng Fidelity?** — *Kha, Đức*
Nearest-First, baseline chỉ chọn cạnh gần nút đích, đứng đầu Drop@5 trên BA-Shapes (0,291 [0,232; 0,353]) và chiếm đầu bảng ở 20/25 cấu hình. Nhưng "thắng" cần đọc thận trọng: hiệu so với GNNExplainer là +0,021 (tính lại: [−0,048; 0,086]), không phân biệt được, và cỡ bằng SD giữa GNN của GNNExplainer (0,019); chỉ hơn Saliency +0,037 [0,012; 0,065] và hơn PGExplainer +0,106 [0,059; 0,159] (tính lại). Số liệu thô gợi ý lý do: 72,4% cạnh trong top-5 của Nearest-First là cạnh 1-hop, và xóa toàn bộ cạnh 1-hop làm Drop trung bình 0,36 trên các nút không mâu thuẫn (`a3_2_t2_t3.csv`, trung bình 5 GNN); cơ chế đầy đủ ở Mục 7. Trên Cora Nearest-First (0,039) thấp hơn GNNExplainer (0,080) và Saliency (0,075). Điều này cho thấy Nearest-First là mốc mạnh, không phải bằng chứng explainer vô dụng. Nguồn: Mục 5.1, 5.5; `a3_2_mechanisms.csv`.

**B3. Vì sao không dùng GPU?** — *Vũ, Thiện*
Protocol khóa CPU để kết quả tất định (kế hoạch, Mục 4 và rủi ro ở Mục 14: GPU không tất định, Saliency hai lần chạy có thể khác nhau). Quy mô thí nghiệm nhỏ (PGExplainer ~11 s mỗi lượt huấn luyện trên BA-Shapes, ~4 s trên Cora) nên CPU đủ nhanh. Ngay cả trên CPU, PGExplainer đã nhạy với số luồng (xem B7); nhóm không đo trên GPU nên không khẳng định kết quả GPU. Nguồn: `configs/protocol_final.yaml`, `docs/PHAN_BAO_CAO_PGEXPLAINER_THIEN.md`, Mục 3b.

**B4. Vì sao chọn size_loss `sum` cho GNNExplainer?** — *Khôi*
Với `mean` (giữa kỳ), mask dồn quanh 0,95 (BA-Shapes, 20 nút validation: trung vị 0,9481; 98,42% cạnh > 0,9), nên điểm cạnh gần như không phân biệt. Với `sum` (như PyG), trung vị 0,2563 và 0,78% cạnh > 0,9, trong khi Drop@5 gần như không đổi (0,3095 so với 0,3088) và AUROC gần nhau (0,7350 so với 0,7392). Hai công thức chọn các cạnh khác nhau (κ@5 giữa hai công thức: 0,908 trên BA-Shapes, 0,208 trên Cora). Quyết định ghi trong CHANGELOG ngày 23/11 và **đang chờ nhóm trưởng xác nhận**; nếu đổi phải chạy lại GNNExplainer. Mọi `run_log.json` của lượt chính ghi `size_loss_mode = sum`. Nguồn: `docs/BIEN_BAN_F10_GNNEXPLAINER_PYG.md`, Mục 2–3; `docs/CHANGELOG_PROTOCOL.md`.

**B5. Vì sao tập test là 140 (BA-Shapes) và 100 (Cora) nút?** — *Đăng, Vũ*
BA-Shapes: toàn bộ 140 nút test của split phân tầng 420/140/140 (seed 0). Cora: 100 nút lấy mẫu phân tầng theo nhãn thật từ 1.000 nút test Planetoid (seed 0), không phải toàn bộ. Cả hai danh sách tạo bằng script và commit trước khi chạy explainer (`data/*/test_nodes.json`, kèm SHA-256 dữ liệu). Giữa kỳ chỉ có 20 nút validation, và chỉ 10/20 nút BA-Shapes có AUROC; trên test, AUROC xác định ở 71/140 nút. Kế hoạch ước tính GNNExplainer trên Cora mất 1,5–3 giờ cho 100 nút × 25 lượt; không tài liệu nào nêu lý do riêng cho con số 100. Hệ quả: CI rộng và 14/100 nút Cora không có κ@5 (m ≤ 5). Nguồn: `configs/protocol_final.yaml`, kế hoạch Mục 4 và 7.2.

**B6. Vì sao PGExplainer trên Cora gần như bằng Random về Fid-?** — *Thiện*
Trên Cora, Drop@5 của PGExplainer là 0,009 [−0,012; 0,031] và Random là 0,003; hiệu (tính lại) +0,006 [−0,002; 0,014]; Inv cũng không khác (0,022 và 0,022). κ@5 giữa PGExplainer và Nearest-First là −0,013, và giữa PGExplainer và hai phương pháp kia xấp xỉ 0. Nhóm chưa tinh chỉnh PGExplainer (Mục 9.1) và không kết luận nguyên nhân; kế hoạch có ghi điểm PG giữa kỳ trên Cora trải hẹp (trung vị độ trải 0,025). Nguồn: Mục 5.5; `agreement_summary.csv` của Cora; kế hoạch, Mục 14.

**B7. Điều gì xảy ra nếu đổi số luồng CPU?** — *Thiện*
Điểm PGExplainer thay đổi. Hai lượt chạy khác số luồng (22/11 và 23/11) cho κ@5 trung bình giữa hai lượt 0,80 trên 25 cấu hình BA-Shapes, nhỏ nhất 0,40 (g = 1000, e = 0); AUROC của cùng một ô lệch tuyệt đối trung bình 0,089 và tối đa 0,474 (g = 1000, e = 3: 0,158 so với 0,632). Khi giữ nguyên cấu hình (2 luồng), chạy lại một ô cho điểm trùng tuyệt đối. Vì vậy mọi kết luận về PGExplainer gắn với `OMP_NUM_THREADS=2`; CSV không ghi số luồng của lượt 22/11. Nhóm chưa đo độ nhạy tương ứng cho Saliency và GNNExplainer (GATE 0 chỉ cho thấy hai phương pháp này tái lập đến cỡ 10⁻⁷ so với số giữa kỳ). Nguồn: `results/analysis/a2/pg_thread_sensitivity.csv`, `docs/CHANGELOG_PROTOCOL.md` (23/11).

**B8. Saliency tất định nhưng κ < 1, vì sao?** — *Khôi, Kha*
Điểm Saliency giống hệt giữa hai seed (không có thành phần ngẫu nhiên); chỉ hoán vị phá hòa của bộ đánh giá phụ thuộc seed. Trên BA-Shapes, κ@5 giữa hai lượt Saliency (GNN 1000, e = 0 và 1) là 0,968 [0,940; 0,990] chứ không phải 1; trên Cora là 1,000. Ở g = 1000, 15/140 nút BA-Shapes có điểm thứ 5 bằng điểm thứ 6, và Drop@5 của hai lượt trùng nhau ở cả 140 nút (tính lại từ `metrics.csv`), nên chênh lệch nằm ở việc chọn cạnh nào trong nhóm hòa, không ở Fidelity. Cách giải thích này phù hợp với số liệu nhưng chưa được kiểm tra riêng từng nút. Nguồn: `results/analysis/a2/a2_1_within_cross_gnn.csv`, `results/analysis/a3_final/t4_tie_sensitivity.csv`.

## Phần C. Điểm cần xác nhận trước buổi bảo vệ

1. `size_loss_mode = sum` chưa có xác nhận của nhóm trưởng (B4).
2. Các câu trả lời về cơ chế (A4, A5, B1, B2) phải khớp với văn bản Mục 7 sau khi Mục 7 hoàn thiện.
3. Số liệu D1 (bản gộp seed) chưa có (Mục 8.1); không trả lời "D1 cải thiện bao nhiêu" cho tới khi có số.
4. Tài liệu của Thiện ghi lượt PG 22/11 chạy "CPU 2 luồng", còn CHANGELOG 23/11 ghi hai lượt khác số luồng; cần thống nhất số luồng của lượt 22/11 trước khi trả lời B7.
