## 8. D1 (Seed-Ensemble) và Tầng B

### 8.1. D1: Seed-Ensemble

**Phương pháp.** Với mỗi cặp (g, nút v), D1 lấy trung bình *thứ hạng* của cạnh qua 5 seed explainer (e = 0…4) để được một giải thích gộp, rồi cắt top-k như thường lệ. D1 không cần chạy thêm explainer: nó dùng lại 25 lượt đã có. Giả thuyết cần kiểm tra: gộp seed làm giảm nhiễu seed của explainer, nên giải thích gộp ổn định hơn và có thể có Fidelity hoặc AUROC tốt hơn bản một seed. Nếu D1 không cải thiện, báo cáo ghi đúng như vậy (kế hoạch, Mục 10).

**Bằng chứng gián tiếp có sẵn từ FB-2.** Mục 6 đã tính đại lượng $D_{\rm between\_avg}$, tức κ@5 giữa hai phương pháp sau khi mỗi phương pháp được gộp 5 seed. Hiệu $D_{\rm between\_avg}-D_{\rm between}$ cho biết gộp seed làm hai phương pháp đồng thuận thêm bao nhiêu (dòng `avg_minus_between` trong `results/analysis/a2/a2_2_between.csv`; CI 95% bootstrap trên nút, p Wilcoxon đã hiệu chỉnh Holm).

| Dataset (n nút) | Cặp | $D_{\rm between}$ | $D_{\rm between\_avg}$ | Hiệu [CI 95%] | p (Holm) |
|---|---|---:|---:|---|---:|
| BA-Shapes (140) | Saliency – GNNExplainer | 0,143 | 0,180 | +0,037 [0,022; 0,053] | 0,0016 |
| BA-Shapes (140) | Saliency – PGExplainer | 0,102 | 0,098 | −0,005 [−0,019; 0,008] | 0,36 |
| BA-Shapes (140) | GNNExplainer – PGExplainer | 0,048 | 0,044 | −0,004 [−0,019; 0,013] | 0,095 |
| Cora (86) | Saliency – GNNExplainer | 0,342 | 0,345 | +0,003 [−0,010; 0,015] | 0,48 |
| Cora (86) | Saliency – PGExplainer | −0,010 | −0,005 | +0,005 [−0,002; 0,012] | 0,48 |
| Cora (86) | GNNExplainer – PGExplainer | −0,017 | −0,021 | −0,004 [−0,016; 0,008] | 0,48 |

Theo quy tắc diễn giải đặt trước (Mục 6.3), "D_between_avg cao hơn hẳn D_between" mới cho phép kết luận gộp seed giúp ích. Số liệu chỉ có một trường hợp thỏa điều kiện thống kê (Saliency – GNNExplainer trên BA-Shapes, tăng 0,037). Năm cặp còn lại có CI chứa 0, và mọi $D_{\rm between\_avg}$ của cặp có PGExplainer đều ≤ 0,098 (mốc Random – Random: −0,004 trên BA-Shapes và −0,001 trên Cora, `agreement_summary.csv`). Vì vậy, ngoài một cặp trên BA-Shapes, gộp seed không kéo hai phương pháp lại gần nhau. Đây là bằng chứng gián tiếp về đồng thuận giữa phương pháp, chưa phải Fidelity hay AUROC của bản gộp.

<!-- ĐIỀN KẾT QUẢ D1 -->
*(Chờ số liệu: Drop@5, Inv@5 và AUROC (BA-Shapes) của giải thích gộp D1 so với giải thích một seed, cùng 140/100 nút test và 5 GNN; nếu có ablation S ∈ {1, 3, 5} thì thêm bảng.)*

### 8.2. Tầng B: không thực hiện

Tầng B gồm B1 (BA-Shapes đặc trưng hằng với GCN 3 lớp), B2 (GAT 2 lớp) và B3 (Amazon Photo). Nhóm **không làm** mục nào. Lý do: kế hoạch (Mục 9) chỉ cho mở Tầng B sau GATE 2, và theo FB-4 của thầy, nhóm ưu tiên hoàn thành các kiểm chứng cốt lõi (tái lập, tách bất ổn định khỏi bất đồng, error analysis) trước khi mở rộng. Hệ quả được ghi ở Mục 9.1: báo cáo chỉ có GCN, chỉ có BA-Shapes và Cora, và giả thuyết đặc trưng bậc (T3) chỉ được kiểm tra bằng phép tính lại đặc trưng bậc sau khi xóa cạnh, không phải bằng thí nghiệm huấn luyện trên đặc trưng hằng (B1).
