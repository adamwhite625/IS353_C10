# Slide cuối kỳ nhóm 09 (C.10): nguồn số liệu

Tái tạo: `python docs/slides_cuoi_ky/verify_numbers.py` (tính lại mọi số từ CSV, ghi `numbers.json`), rồi
`NODE_PATH=<thư mục có pptxgenjs> node docs/slides_cuoi_ky/build_deck.js`. Mọi số trên slide được `build_deck.js` đọc từ
`numbers.json`, không gõ tay. Các số đã đối chiếu với `docs/report_parts/muc5..muc9` và khớp.

| Slide | Nội dung | File nguồn | Giá trị chính |
|---|---|---|---|
| 1 | Bìa | (không có số) | |
| 2 | Bài toán, RQ1-3 | `docs/BAO_CAO_CUOI_KY.md` Mục 1 | (không có số) |
| 3 | FB-1..FB-4 | `docs/BAO_CAO_CUOI_KY.md` Mục 1.2 | 25 cấu hình, 355 cặp nút-GNN, T1-T5 |
| 4 | Thiết lập, tái lập | `results/checkpoints/checkpoint_table.csv`; Mục 3.1, 4.5 của báo cáo | 10 checkpoint (seed 1000-1004), 140 / 100 nút test, 25 cấu hình; F1: lệch 0,885; F10: mask quanh 0,95 (lấy từ báo cáo, không tính lại) |
| 5 | Bảng k = 5 | `results/analysis/main/main_by_method_budget.csv` | BA-Shapes Drop@5: NF 0,291 [0,232; 0,353], GNNE 0,270, Sal 0,248, PG 0,185, Random 0,156; AUROC: PG 0,731 [0,652; 0,808]; Cora Drop@5: GNNE 0,080, PG 0,009, Random 0,003 |
| 6 | Hình A2 | `docs/figures/final/a2_*.png`; `results/analysis/a2/a2_1_within_cross_gnn.csv`, `a2_2_between.csv` | within BA 0,499 / 0,650 / 0,968; between BA 0,048-0,143; Cora within 0,752-1,000, between -0,017-0,342 |
| 7 | Bảng A2-2, lưu ý | `a2_2_between.csv`, `a2_1_within_cross_gnn.csv`, `pg_thread_sensitivity.csv` | Δ_A, Δ_B của 6 cặp, mọi CI > 0 và p_holm < 0,001 (script dừng nếu sai); GNNE cross-GNN 0,825 (cùng e) vs 0,403 (khác e); PG hai lượt khác luồng: κ@5 TB 0,801, min 0,396, AUROC lệch tối đa 0,474 (25 cấu hình) |
| 8 | A3-0 | `results/analysis/a3_final/a3_0_conflicts.csv` (tie_seed = average), `a3_0_metrics.csv`, `a3_1_node_conflicts.csv` | 17/25 (68,0%); Drop dẫn: NF 20, GNNE 5; AUROC dẫn: PG 17, NF 8; mâu thuẫn cấp nút 208/355 (58,6%) |
| 9 | Góc phần tư, T1-T5 | `a3_1_by_node.csv` (ngưỡng 0,10 / 0,75), `t2_t3_conflict_cohort.csv`, `a3_1_node_conflicts.csv`, `a3_2_mechanisms.csv` | Q1 PG 212, Q2 Sal 169, GNNE Q3 = 0; T2 95/208 (45,7%) vs 30/147 (20,4%); T1 0,531 vs 0,537; T3 0,274 -> 0,521; T5 PG 1849/5200 (35,6%); 113/208 chưa giải thích |
| 10 | Node 665 | `docs/figures/final/a3_node_665.png`, `case_metrics.csv`, `case_deletion_curves.csv` | Drop@5 0,212 (Sal, GNNE, NF) vs 0,480 (PG); p(k=4) = 0,491 |
| 11 | D1 | `results/analysis/d1/d1_stability.csv`, `d1_metrics.csv`, `d1_vs_single_seed.csv` | κ@5 1 seed -> gộp (2v3): BA GNNE 0,499 -> 0,604, BA PG 0,650 -> 0,697, Cora GNNE 0,798 -> 0,839, Cora PG 0,752 -> 0,798; BA GNNE AUROC 0,631 -> 0,633, PG AUROC 0,731 -> 0,737; có ý nghĩa (Holm) 3/10 BA, 0/4 Cora; chênh lệch tối đa 0,021 |
| 12 | Kết luận, hạn chế | tổng hợp các slide trên; Mục 9 | Δ nhỏ nhất 0,313; PG Cora cross-GNN 0,308 |
| B1 | κ@k | `results/eval/{bashapes,cora}_test/agreement_summary.csv` | κ@k các cặp và Random-Random (k = 1, 3, 5, 10); k = 10: 75 nút BA, 71 nút Cora |
| B2 | Hash | `results/checkpoints/checkpoint_table.csv` | SHA-256 (16 ký tự đầu), test acc 0,921-0,957 (BA), 0,815-0,838 (Cora) |
| B3 | Node 342 | `a3_node_342.png`, `case_metrics.csv` | PG AUROC 0,867, Drop@5 -0,033; GNNE Drop@5 0,928, AUROC 0,467 |
| B4 | Hỏi-đáp | `docs/PHU_LUC_HOI_DAP_PHAN_BIEN.md` + CSV như trên | PG Drop âm 6727/17500 (38,4%) |

## Chỗ cần lưu ý
- Slide 11: "2v3" là hai bản gộp có kích cỡ khác nhau (2 seed và 3 seed), vì chỉ có 5 seed; là bằng chứng gián tiếp. Ablation S = 5 chưa chạy.
- Slide 4: các số 0,885 (F1) và 0,95 (F10) lấy từ báo cáo/biên bản, không có CSV trong `results/analysis/`.
- Slide 12 và ghi chú: size-loss `sum` của GNNExplainer (F10) đã được nhóm trưởng chốt (đã chốt; báo cáo Mục 9.1, ý 9 cần đồng bộ).
- Bìa ghi 6 thành viên và MSSV theo README.md.
- `validate.py` của skill không chạy được trên máy này (thiếu file schema); file đã được mở và xuất ảnh thành công bằng PowerPoint và đọc bằng python-pptx.
