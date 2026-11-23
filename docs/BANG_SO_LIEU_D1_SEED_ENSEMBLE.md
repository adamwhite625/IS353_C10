# Bảng số liệu D1 Seed-Ensemble

Sinh tự động bởi `scripts/analyze_ensemble.py`; số liệu trong `results/analysis/d1/`. Giải thích gộp = −(thứ hạng trung bình qua 5 seed explainer), cùng tập cạnh ứng viên, phá hòa bằng hoán vị có seed (khóa `{method}:avgrank`, trùng `D_between_avg` của A2). Fidelity tính trên đồ thị con 3-hop như bộ đánh giá mặc định. Trung bình theo node (trước đó trung bình qua 5 GNN seed); CI 95% bootstrap theo node (B=2000). Fid- Drop và AP/AUROC/Precision: cao hơn tốt hơn; Fid+ Inv: thấp hơn (gần 0) tốt hơn.

Kiểm tra chéo với bộ đánh giá: 12000 cặp (g, phương pháp, seed, node) ở k=5, 0 lệch seed phá hòa; sai khác lớn nhất precision 0.00e+00, AUROC 0.00e+00; Fid-@5 (seed e0, 2400 cặp) 2.38e-07.

## 1. Metric tại k = 5: bản gộp so với một seed (trung bình 5 seed)

### bashapes

| Phương pháp | Metric | Gộp (D1) | Một seed (TB 5 seed) | Gộp − TB | CI 95% hiệu | p_Holm | Gộp − tốt nhất | Gộp − tệ nhất |
|---|---|---|---|---|---|---|---|---|
| GNNExplainer | Fid-@k Drop | 0.272 [0.220; 0.328] | 0.270 [0.219; 0.326] | +0.002 | [-0.002; +0.006] | 0.093 | -0.002 (p_Holm 1.000) | +0.005 (p_Holm 0.018) |
| GNNExplainer | Fid+@k Inv | 0.037 [0.002; 0.079] | 0.058 [0.022; 0.102] | -0.021 | [-0.035; -0.009] | 0.006* | -0.007 (p_Holm 1.000) | -0.035 (p_Holm 0.001) |
| GNNExplainer | AUROC | 0.633 [0.578; 0.688] | 0.631 [0.580; 0.683] | +0.002 | [-0.009; +0.012] | 1.000 | -0.028 (p_Holm 0.479) | +0.048 (p_Holm 0.001) |
| GNNExplainer | AP | 0.691 [0.606; 0.774] | 0.687 [0.602; 0.769] | +0.004 | [-0.001; +0.011] | 1.000 | -0.000 (p_Holm 1.000) | +0.018 (p_Holm 0.001) |
| GNNExplainer | Precision@k | 0.495 [0.425; 0.567] | 0.497 [0.426; 0.568] | -0.002 | [-0.004; +0.000] | 1.000 | -0.007 (p_Holm 0.157) | +0.002 (p_Holm 1.000) |
| PGExplainer | Fid-@k Drop | 0.192 [0.146; 0.238] | 0.185 [0.141; 0.228] | +0.007 | [+0.001; +0.013] | 1.000 | -0.004 (p_Holm 1.000) | +0.013 (p_Holm 0.059) |
| PGExplainer | Fid+@k Inv | 0.254 [0.203; 0.309] | 0.265 [0.217; 0.317] | -0.011 | [-0.023; +0.001] | 1.000 | +0.010 (p_Holm 1.000) | -0.028 (p_Holm 0.015) |
| PGExplainer | AUROC | 0.737 [0.650; 0.820] | 0.731 [0.652; 0.808] | +0.006 | [-0.003; +0.015] | 0.460 | -0.047 (p_Holm 0.000) | +0.088 (p_Holm 0.000) |
| PGExplainer | AP | 0.772 [0.678; 0.865] | 0.767 [0.674; 0.860] | +0.004 | [+0.003; +0.006] | 0.000* | -0.009 (p_Holm 0.000) | +0.027 (p_Holm 0.000) |
| PGExplainer | Precision@k | 0.558 [0.479; 0.637] | 0.549 [0.471; 0.627] | +0.009 | [+0.006; +0.012] | 0.000* | -0.004 (p_Holm 0.149) | +0.030 (p_Holm 0.000) |

### cora

| Phương pháp | Metric | Gộp (D1) | Một seed (TB 5 seed) | Gộp − TB | CI 95% hiệu | p_Holm | Gộp − tốt nhất | Gộp − tệ nhất |
|---|---|---|---|---|---|---|---|---|
| GNNExplainer | Fid-@k Drop | 0.079 [0.054; 0.106] | 0.080 [0.055; 0.106] | -0.001 | [-0.005; +0.003] | 1.000 | -0.005 (p_Holm 1.000) | +0.002 (p_Holm 1.000) |
| GNNExplainer | Fid+@k Inv | -0.131 [-0.154; -0.107] | -0.129 [-0.152; -0.106] | -0.002 | [-0.005; +0.001] | 0.619 | +0.001 (p_Holm 1.000) | -0.004 (p_Holm 1.000) |
| PGExplainer | Fid-@k Drop | 0.011 [-0.011; 0.033] | 0.009 [-0.012; 0.031] | +0.001 | [-0.001; +0.003] | 1.000 | -0.001 (p_Holm 1.000) | +0.004 (p_Holm 0.891) |
| PGExplainer | Fid+@k Inv | 0.020 [-0.005; 0.047] | 0.022 [-0.003; 0.048] | -0.002 | [-0.004; +0.001] | 1.000 | +0.002 (p_Holm 1.000) | -0.004 (p_Holm 1.000) |

`*` = p_Holm < 0,05 (Holm trong từng dataset × budget). "Tốt nhất/tệ nhất" chọn hậu nghiệm theo trung bình toàn dataset nên so với "tốt nhất" thiên về bất lợi cho D1.

## 2. Các budget khác (Gộp − TB một seed)

| Dataset | Phương pháp | Metric | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|---|
| bashapes | GNNExplainer | Fid-@k Drop | +0.008 | -0.002 | +0.002 | +0.005 |
| bashapes | GNNExplainer | Fid+@k Inv | -0.012 | -0.022 | -0.021* | -0.014* |
| bashapes | GNNExplainer | AUROC | +0.002 | +0.002 | +0.002 | +0.002 |
| bashapes | GNNExplainer | AP | +0.004 | +0.004 | +0.004 | +0.004 |
| bashapes | GNNExplainer | Precision@k | +0.013 | +0.004 | -0.002 | -0.000 |
| bashapes | PGExplainer | Fid-@k Drop | +0.009 | +0.001 | +0.007 | -0.000 |
| bashapes | PGExplainer | Fid+@k Inv | +0.004 | +0.001 | -0.011 | +0.001 |
| bashapes | PGExplainer | AUROC | +0.006 | +0.006 | +0.006 | +0.006 |
| bashapes | PGExplainer | AP | +0.004* | +0.004* | +0.004* | +0.004* |
| bashapes | PGExplainer | Precision@k | +0.002 | +0.004* | +0.009* | +0.000 |
| cora | GNNExplainer | Fid-@k Drop | +0.000 | -0.000 | -0.001 | -0.002 |
| cora | GNNExplainer | Fid+@k Inv | +0.000 | -0.007* | -0.002 | +0.001 |
| cora | PGExplainer | Fid-@k Drop | +0.000 | +0.000 | +0.001 | -0.000 |
| cora | PGExplainer | Fid+@k Inv | +0.002 | -0.001 | -0.002 | +0.003 |

## 3. Độ ổn định: κ@5 giữa hai bản gộp rời nhau (trung bình 5 GNN)

| Dataset | Phương pháp | 1v1 (= I_within) | I_within (a2_1) | 2v2 | 2v3 | 2v2 − 1v1 (p_Holm) | 2v3 − 1v1 (p_Holm) |
|---|---|---|---|---|---|---|---|
| bashapes | GNNExplainer | 0.499 [0.440; 0.558] | 0.499 | 0.573 [0.509; 0.635] | 0.604 [0.543; 0.663] | +0.075 (0.000) | +0.106 (0.000) |
| bashapes | PGExplainer | 0.650 [0.620; 0.680] | 0.650 | 0.680 [0.649; 0.708] | 0.697 [0.666; 0.726] | +0.029 (0.000) | +0.047 (0.000) |
| cora | GNNExplainer | 0.798 [0.771; 0.823] | 0.798 | 0.830 [0.808; 0.852] | 0.839 [0.818; 0.861] | +0.032 (0.000) | +0.041 (0.000) |
| cora | PGExplainer | 0.752 [0.730; 0.772] | 0.752 | 0.785 [0.767; 0.804] | 0.798 [0.779; 0.816] | +0.034 (0.000) | +0.046 (0.000) |

Chỉ có 5 seed nên không có hai bản gộp 5 seed rời nhau: 2v2 dùng 4 seed, 2v3 dùng đủ 5 seed (nhưng hai nhóm khác kích cỡ). Ablation S=5 chưa chạy (cần seed 5–9). Bảng đầy đủ các k và g=1000: `d1_stability.csv`.

## 4. Bằng chứng bổ sung D_between_avg (A2, κ@5, từ `a2_2_between.csv`)

| Dataset | Cặp phương pháp | D_between (từng seed) | D_between_avg (sau gộp seed) |
|---|---|---|---|
| bashapes | saliency vs gnnexplainer | 0.143 | 0.180 |
| bashapes | saliency vs pgexplainer | 0.102 | 0.098 |
| bashapes | gnnexplainer vs pgexplainer | 0.048 | 0.044 |
| cora | saliency vs gnnexplainer | 0.342 | 0.345 |
| cora | saliency vs pgexplainer | -0.010 | -0.005 |
| cora | gnnexplainer vs pgexplainer | -0.017 | -0.021 |

## 5. Diễn giải

- bashapes, k=5, so với trung bình một seed: 3/10 so sánh (phương pháp × metric) có D1 tốt hơn có ý nghĩa (Holm<0,05), 0 kém hơn có ý nghĩa, 7 không phân biệt được. Tốt hơn: GNNExplainer Fid+@k Inv (-0.021), PGExplainer AP (+0.004), PGExplainer Precision@k (+0.009)
- cora, k=5, so với trung bình một seed: 0/4 so sánh (phương pháp × metric) có D1 tốt hơn có ý nghĩa (Holm<0,05), 0 kém hơn có ý nghĩa, 4 không phân biệt được.
- Độ lớn: mọi hiệu có ý nghĩa ở k=5 đều nhỏ (|Gộp − TB| lớn nhất = 0.021 trên thang metric 0–1); ý nghĩa thống kê không đồng nghĩa lợi ích thực tế đáng kể. Các hiệu còn lại đều trong khoảng ±0.011.
- So với seed đơn tốt nhất (chọn hậu nghiệm, k=5): D1 thắng có ý nghĩa 0/14, thua có ý nghĩa 2/14; phần còn lại không phân biệt được.
- Độ ổn định (bashapes): GNNExplainer κ@5 1v1 0.499 → 2v2 0.573 → 2v3 0.604; PGExplainer κ@5 1v1 0.650 → 2v2 0.680 → 2v3 0.697.
- Độ ổn định (cora): GNNExplainer κ@5 1v1 0.798 → 2v2 0.830 → 2v3 0.839; PGExplainer κ@5 1v1 0.752 → 2v2 0.785 → 2v3 0.798.
- κ tăng theo kích thước nhóm gộp là kết quả cơ học của việc lấy trung bình (giảm phương sai seed), nên chỉ chứng tỏ bản gộp ổn định hơn một seed, KHÔNG chứng tỏ giải thích đúng hơn: các metric ở bảng 1 (cùng k) gần như không đổi.
- D_between_avg so với D_between (GNNExplainer vs PGExplainer): bashapes 0.048 → 0.044; cora -0.017 → -0.021 (chưa có CI theo node cho D_between_avg trong A2). Gộp seed không làm hai phương pháp xích lại gần nhau; bất đồng giữa GNNExplainer và PGExplainer không phải do nhiễu seed.
- Giới hạn: chỉ 5 seed explainer (S=5 không có phần ablation); Saliency tất định nên không đưa vào D1; chọn seed "tốt nhất/tệ nhất" là hậu nghiệm; phân tích theo node coi các node là độc lập dù chúng dùng chung GNN; Cora chỉ có metric fidelity (không ground truth).

Lệnh: `PYTHONPATH=src python scripts/analyze_ensemble.py --scores-root <final_scores> --eval-root <final_eval> --workers 4`.
