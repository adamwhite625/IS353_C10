# Ghi chú biên tập báo cáo cuối kỳ

Tài liệu nội bộ, không thuộc bản nộp. Bản nộp: `docs/BAO_CAO_CUOI_KY.md` và `docs/BAO_CAO_CUOI_KY.docx` (sinh bằng `scripts/build_report_docx.js`).

## 1. Đối chiếu số liệu (tính lại bằng pandas từ CSV nguồn)

Đã tính lại và khớp báo cáo: Bảng 5.1a/5.1b và A.2 (từ `results/eval/*_test/metrics.csv` và `main_by_method_budget.csv`), số nút m ≤ 5 và m ≤ 10 (14/100, 65/140, 29/100), 71/140 nút có AUROC, Bảng 5.2 và 6.1–6.2 (`agreement_summary.csv`, `a2_1`, `a2_2`), Bảng A.3, độ nhạy luồng PG (`pg_thread_sensitivity.csv`), Bảng 7.1–7.3 (17/25 cấu hình mâu thuẫn, người thắng, 208/355, các góc phần tư, Spearman, T2 95/208 và 30/147, 45 nút lớp 0 + 50 nút lớp 2, T3, T4, T5, 94/140 và 163/208 và 117/147), Bảng 7.4 (`case_metrics.csv`), D1 (Bảng 8.1, độ ổn định, 3/10, 0/4, 0/14, 2/14), bảng checkpoint (A.1).

Các số đã sửa so với bản nháp:

| Vị trí | Bản nháp | Đã sửa thành | Lý do |
|---|---|---|---|
| Mục 3.8 / 9.1: điểm hòa PGExplainer | 676/3.500 (19,3%), lượt 22/11 | 661/3.500 (18,9%) | Số của lượt cuối kỳ, `t4_tie_sensitivity.csv` |
| Mục 5.5 (nay 5.2), SD giữa GNN của Drop@5, BA-Shapes | NF 0,006; Random 0,011; GNNExplainer 0,019 | NF 0,005; Random 0,010; GNNExplainer 0,018 | `main_sd_over_gnn_seeds.csv` |
| Mục 5.5, hiệu cặp bootstrap (BA-Shapes) | NF−Sal [0,016; 0,073]; GNNE−PG [0,048; 0,128]; GNNE−Random [0,082; 0,152]; PG−Random [0,013; 0,046] | [0,017; 0,072]; [0,047; 0,126]; [0,081; 0,149]; [0,012; 0,046] | Tính lại bằng bootstrap 2.000 lần, seed 0; giá trị trung bình không đổi. CI lệch tối đa 0,003 do khác số ngẫu nhiên |
| Mục 5.5, hiệu cặp (Cora) | NF−GNNE [−0,065; −0,017]; PG−Random Inv [−0,012; 0,011] | [−0,066; −0,017]; [−0,011; 0,010] | Như trên |
| Mục 7.3, T3: "+0,25 đến +0,36 có ở mọi nhóm nút" | | Đã bỏ khoảng này | Theo `t2_t3_by_node.csv`, mức tăng 1-hop theo nhóm và GNN seed nằm trong 0,23–0,49, không phải 0,25–0,36. Báo cáo chỉ giữ các số tổng hợp đã kiểm tra |

Chưa tính lại được từ repo (số lấy nguyên từ `docs/PHAN_TICH_LOI_ERROR_ANALYSIS.md`, cần điểm thô theo nút không có trong Git): T1 (0,531 so với 0,537; 77/59/72), 89/90 nút, và các số "phương pháp thắng AUROC có AUROC 0,907 nhưng Drop@5 0,232; phương pháp thắng Drop@5 có Drop@5 0,523 nhưng AUROC 0,540". Các số này được nêu ở Mục 7 như quan sát suy ra từ CSV sau khi xem số.

## 2. Nội dung cũ đã xử lý

- Đã xóa banner "BẢN NHÁP", mọi chú thích `<!-- -->`, "CẦN XÁC NHẬN", `[CẦN KIỂM TRA]`, các phần "Danh sách trích dẫn cần kiểm tra" và "Ghi chú cho người biên tập" (chuyển sang file này).
- GNNExplainer `size_loss = sum` được viết như quyết định đã chốt (nhóm trưởng xác nhận 23/11). **Việc còn lại cho người phụ trách:** dòng 23/11 của `docs/CHANGELOG_PROTOCOL.md` vẫn ghi "Chờ Vũ xác nhận"; nên sửa thành "đã xác nhận" (người biên tập không được sửa file này theo phạm vi giao việc). `configs/protocol_final.yaml` chưa có trường `size_loss_mode` (giá trị `sum` truyền bằng `--gnn-size-loss-mode sum` và ghi trong `run_log.json`).
- Mục 3.8 và mọi chỗ còn trích số sơ bộ (676/3.500; 40,7%) đã thay bằng số lượt cuối kỳ hoặc bỏ. Số "10/20 nút hợp lệ" và 0,8000 ở Mục 1.1 là số giữa kỳ, giữ vì được dùng để mô tả mâu thuẫn giữa kỳ (nguồn T06).
- Mục 4: viết như hiện trạng (các script đã có); lệnh lấy từ README, mục "Kết quả cuối kỳ và cách chạy lại"; nêu `OMP_NUM_THREADS=2`, `git_dirty: false`, mốc tái lập là SHA-256 file `.pt`, PG chỉ tái lập chính xác với cùng số luồng. Lượt 22/11 dùng 6 luồng mặc định của torch là thông tin do người giao việc cung cấp; CSV không ghi số luồng và repo không có chứng cứ độc lập, nên báo cáo ghi "theo nhóm chạy lưới".
- Mục 1.4 và 9.2 viết lại cho khớp Mục 5–8. Câu RQ1–RQ3 trả lời đúng số liệu: RQ2 chỉ khẳng định nhạy theo GNN ở 4/6 ô (CI không giao nhau); riêng GNNExplainer trên BA-Shapes ghi rõ là chưa tách được tác động GNN khỏi khởi tạo chung.

## 3. Trích dẫn

Đã tra web (tìm kiếm và trang arXiv) trong phiên biên tập:

| Mục | Kết quả | Quyết định |
|---|---|---|
| [4] Boon et al. | Xác minh: báo cáo tái lập "[Re] Parameterized Explainer for Graph Neural Network", ML Reproducibility Challenge 2020; tác giả theo thứ tự Holdijk, Boon, Henckens, de Jong (bản nháp ghi thứ tự khác). Tóm tắt trang: nhiều điểm không nhất quán giữa bài báo và mã gốc | Giữ, sửa thứ tự tác giả, ghi năm 2021; không ghi số trang hay DOI vì chưa xác minh |
| [7] Krishna et al. | arXiv ghi "Published in TMLR", không ghi năm; không nói rõ dữ liệu bảng hay không (chỉ "four real-world datasets, six explanation methods, six predictive models") | Giữ, bỏ năm; mô tả theo abstract, không khẳng định phạm vi dữ liệu; bỏ câu "chưa thấy bản tương ứng cho GNN" |
| [8] Faber et al. | Tên bài, tác giả, KDD 2021 khớp qua tìm kiếm; luận điểm được các bài khác (Agarwal et al.) tóm tắt giống vậy. Chưa mở trang ACM/DOI | Giữ, mô tả thận trọng một luận điểm; không ghi trang/DOI |
| [10] GNNX-Bench | Xác minh arXiv: ICLR 2024; benchmark các explainer dựa trên nhiễu loạn (counterfactual, ổn định giữa các kiến trúc, nhiễu, tính ngẫu nhiên). Abstract không nói đo ổn định giữa các seed | Bỏ nhận định "đo độ ổn định giữa các seed trong graph classification" (từ Mini-Proposal) |
| [11] GInX-Eval | Xác minh arXiv 2309.16223, tác giả Amara, El-Assady, Ying; luận điểm: metric dựa trên xóa cạnh có thể ngoài phân phối | Giữ và dùng ở Mục 2 và 9.1 |
| [12] Platonov et al. (heterophily) | Bài có thật (ICLR 2023) nhưng không liên quan tới nội dung báo cáo và không được trích trong thân bài | Bỏ khỏi danh sách |
| [9] GraphFramEx: dấu +/− | Chưa đối chiếu bài gốc | Bỏ khẳng định về quy ước của GraphFramEx; chỉ nói các tài liệu đặt dấu khác nhau nên dùng Drop/Inv |
| [2], [5], [6] | Đã đối chiếu từ phiên trước (arXiv, CVF/dblp); NeurIPS 2019 của [2] lấy từ Mini-Proposal | Giữ; chưa ghi số trang cho [2] |

Không trích: Wilcoxon, Holm, bootstrap, Kendall, Cohen kappa (không có trong tài liệu tham khảo); các bài khác của Mini-Proposal (GraphXAI, Longa, Nerini, ...) không đưa vào vì chưa xác minh.

## 4. Độ dài và bố cục

- Bản nộp: thân bài (Mục 1–9 và tài liệu tham khảo) khoảng 15 trang A4, Times New Roman 11 pt, lề 2,2 cm; thêm trang bìa, mục lục và hai trang phụ lục. Hình A2 xếp 2 cột; case study 665 và 342 nằm ở thân bài, 596 và 106 chuyển sang Phụ lục A (Hình A.1, A.2) để giữ độ rộng đọc được.
- Chuyển sang Phụ lục A: bảng checkpoint đầy đủ, Fidelity theo k, bảng SD Fidelity theo nút (A2-3), can thiệp T2/T3 theo GNN seed, case study phụ. Phụ lục B (hỏi–đáp) chỉ ghi đường dẫn `docs/PHU_LUC_HOI_DAP_PHAN_BIEN.md` vì hết chỗ.
- Mã nguồn chuyển đổi: `scripts/build_report_docx.js` (cần `npm install docx`, đặt `NODE_PATH`; chạy từ gốc repo). Số trang đã đo bằng Word (COM) khi xuất PDF; công thức dùng ký hiệu Unicode (không phải OMML).

## 5. Mâu thuẫn tài liệu còn lại (từ bản nháp, chưa thuộc phạm vi sửa)

1. README còn mô tả cũ (thư mục `experiments/exp01…08`, phạm vi GAT, "kiểm thử 2/2 passed").
2. `data/bashapes/config.json` (giữa kỳ) ghi weight decay 0 và 400 epoch, khác YAML cuối kỳ (1e-4, 600 epoch); báo cáo lấy YAML làm nguồn.
3. Hai bản số giữa kỳ T05/T06 khác nhau (ví dụ Drop@5 của PGExplainer 0,2475 và 0,2195); báo cáo dùng T06.
4. Hạn nộp: kế hoạch ghi 26/11, Mini-Proposal ghi 25/11/2026; trang bìa dùng 26/11/2026 theo giao việc.
5. Môi trường: PN1 Python 3.12.14 / torch 2.15.0+cpu; GATE 0 chạy Python 3.11.6 / torch 2.14.0+cpu trên Windows; lượt PGExplainer đầu trên Windows Python 3.12.10. Mục 9.1 chỉ nêu các phiên bản đã kiểm chứng trong manifest.
