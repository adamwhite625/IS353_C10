# Kiểm tra bàn giao H2 (lưới cuối kỳ, 23/11/2026)

- Lệnh chạy lưới: xem `README.md` (mục "Kết quả cuối kỳ và cách chạy lại"), `OMP_NUM_THREADS=2`, CPU, commit `7d6e9c4`, cây sạch (`git_dirty: false` trong mọi `run_log.json`).
- `python scripts/check_scores.py results/scores/{bashapes,cora}`: **PASS** cả hai dataset
  (báo cáo máy đọc: `check_scores_bashapes.json`, `check_scores_cora.json`). Mỗi dataset 56 CSV
  (Saliency 6, GNNExplainer 25, PGExplainer 25), 0 node lỗi. Kiểm tra tất định Saliency (g = 1000, e = 0 và 1): `max_diff = 0.000e+00`.
- Tái lập: chạy lại riêng ô (g = 1003, e = 2) của PGExplainer trên BA-Shapes và Cora với cùng số luồng cho điểm trùng tuyệt đối
  (`max|Δ| = 0.0`; chỉ cột `runtime_sec` khác).
- GNNExplainer chạy với `--gnn-size-loss-mode sum` (ghi trong `hyperparams` của `run_log.json`).
- CSV điểm thô nằm ngoài Git: `PN2_final_results_20261123.zip` trên Drive; hash từng CSV nằm trong `run_log.json` (`csv_sha256`).
