## 6. Instability vs Disagreement (trả lời FB-2)

### 6.1. Thiết kế

Lưới đầy đủ gồm 5 seed GNN (1000–1004) × 5 seed explainer (0–4) × {GNNExplainer, PGExplainer}; Saliency chạy seed 0 trên mọi GNN và thêm seed 1 trên GNN 1000 để kiểm tra tính tất định. Phân tích theo thứ tự thầy dặn: **Bước 1** giữ GNN 1000 và chỉ đổi seed explainer; **Bước 2** mới đổi seed GNN (nguồn: `docs/KE_HOACH_CUOI_KY_FINAL.md`, Mục 7.2; `configs/protocol_final.yaml`). Tập đánh giá là các nút test: BA-Shapes 140 nút; Cora 100 nút, trong đó chỉ 86 nút có $\kappa@5$ xác định (14 nút có $m\le5$ cạnh ứng viên, Mục 3.7). Thiết kế dữ liệu và explainer ở Mục 3; cách đo và thống kê ở Mục 3.7–3.8. Vì chỉ có một cặp seed explainer cho Saliency, hai cột $I_{\rm within}$ của Saliency trong Bảng A2-1 là cùng một phép đo.

### 6.2. Đại lượng

Với nút $v$, seed GNN $g$, seed explainer $e$, phương pháp $M$; Agr = $\kappa@5$ (chính), Jaccard@5 và $\tau_b$ là phụ. Mọi đại lượng tính cho từng nút rồi mới lấy trung bình; CI 95% bằng bootstrap lấy mẫu lại nút.

| Đại lượng | Định nghĩa | Ý nghĩa |
|---|---|---|
| $I_{\rm within}(M)$ | Trung bình Agr giữa $(g,e)$ và $(g,e')$ trên 10 cặp $e<e'$. Bước 1: chỉ $g=1000$. Bước 2: trung bình trên 5 $g$ | Phương pháp tự lệch bao nhiêu |
| $I_{\rm gnn}(M)$ | Trung bình Agr giữa $(g,e)$ và $(g',e)$, chỉ trên nút có $\mathrm{pred}_g(v)=\mathrm{pred}_{g'}(v)$; báo kèm tỷ lệ nút đổi dự đoán | Nhạy theo GNN |
| $I_{\rm gnn}^{\rm diff\,e}(M)$ | Như $I_{\rm gnn}$ nhưng ghép $(g,e)$ với $(g',e')$, $e'\ne e$ (phép đối chứng, mục 6.6) | Nhạy theo GNN khi không dùng chung seed |
| $D_{\rm between}(A,B)$ | Trung bình trên $(g,e)$ của Agr$(A_{g,e},B_{g,e})$ | Hai phương pháp lệch nhau bao nhiêu |
| $D_{\rm between\_avg}(A,B)$ | Agr giữa $\bar A_g$ và $\bar B_g$, với $\bar A_g$ là trung bình thứ hạng qua 5 seed explainer (D1) | Bất đồng còn lại sau khi khử nhiễu seed |
| Mốc tham chiếu | Agr(explainer, Nearest-First); Agr(Random, Random) ($\kappa\approx0$) | Explainer có chỉ đang chọn cạnh gần nút đích không |

### 6.3. Quy tắc diễn giải (ghi trước khi xem số)

Quy tắc dưới đây được nhóm đặt ra trong kế hoạch (`docs/KE_HOACH_CUOI_KY_FINAL.md`, Mục 7.4) trước khi có số liệu lưới; báo cáo giữ nguyên văn. Đặt $\Delta_A(v)=I_{\rm within}(A,v)-D_{\rm between}(A,B,v)$, CI 95% bằng bootstrap trên nút.

| Quan sát | Kết luận được phép viết |
|---|---|
| $D_{\rm between}$ thấp hơn rõ cả $I_{\rm within}(A)$ và $I_{\rm within}(B)$ (CI của $\Delta_A$ và $\Delta_B$ đều > 0) | **Bất đồng thực sự**, không giải thích được bằng nhiễu seed |
| $I_{\rm within}(M)$ thấp và gần bằng $D_{\rm between}$ | Bất đồng quan sát được **phần lớn đến từ bất ổn định** của $M$ |
| $D_{\rm between\_avg}$ cao hơn hẳn $D_{\rm between}$ | Một phần bất đồng là nhiễu seed; gộp seed (D1) giúp ích |
| $I_{\rm gnn}$ thấp hơn rõ $I_{\rm within}$ | Giải thích **nhạy với việc huấn luyện lại GNN**, kể cả khi dự đoán không đổi |
| Saliency có $I_{\rm within}=1$ nhưng $I_{\rm gnn}$ thấp | Saliency tất định theo seed explainer nhưng vẫn phụ thuộc trọng số GNN |

Kế hoạch đã bỏ ý tưởng "trần tin cậy" $\sqrt{\mathrm{rel}_A\,\mathrm{rel}_B}$ của Mini-Proposal, vì công thức Spearman–Brown không áp dụng cho Jaccard và $\kappa$ (xem `docs/KE_HOACH_CUOI_KY_FINAL.md`, Mục 2.1, dòng 13). Quy tắc không định lượng "thấp hơn rõ" ở dòng 4; khi đọc số, nhóm hiểu là CI 95% của hai đại lượng không giao nhau (so CI riêng lẻ, không phải kiểm định cặp).

### 6.4. Kết quả

**Bảng A2-1** ($\kappa@5$, trung bình theo nút [CI 95%]; nguồn: `results/analysis/a2/a2_1_within_cross_gnn.csv`). "Cùng $e$" là $I_{\rm gnn}$ theo kế hoạch; "khác $e$" là đối chứng.

| Dataset | Phương pháp | $I_{\rm within}$, $g=1000$ | $I_{\rm within}$, 5 GNN | $I_{\rm gnn}$, cùng $e$ | $I_{\rm gnn}$, khác $e$ | $n$ |
|---|---|---|---|---|---|---|
| BA-Shapes | Saliency | 0,968 [0,940; 0,990] | 0,968 [0,940; 0,990] | 0,855 [0,821; 0,885] | — | 140 |
| | GNNExplainer | 0,488 [0,424; 0,552] | 0,499 [0,440; 0,558] | 0,825 [0,794; 0,855] | 0,403 [0,351; 0,457] | 140 |
| | PGExplainer | 0,640 [0,601; 0,675] | 0,650 [0,620; 0,680] | 0,611 [0,580; 0,642] | 0,574 [0,538; 0,609] | 140 |
| Cora | Saliency | 1,000 [1,000; 1,000] | 1,000 [1,000; 1,000] | 0,821 [0,791; 0,849] | — | 86 |
| | GNNExplainer | 0,788 [0,749; 0,824] | 0,798 [0,771; 0,823] | 0,711 [0,677; 0,744] | 0,628 [0,590; 0,668] | 86 |
| | PGExplainer | 0,691 [0,654; 0,726] | 0,752 [0,730; 0,772] | 0,308 [0,276; 0,340] | 0,296 [0,263; 0,329] | 86 |

Tỷ lệ nút đổi dự đoán giữa hai GNN, trung bình trên 10 cặp: BA-Shapes 1,6% (tối đa 3,6%); Cora 4,5% (tối đa 6,0%) (nguồn: `results/eval/*_test/pred_change.csv`). Mốc Random–Random: $\kappa@5=-0{,}004$ (BA-Shapes), $-0{,}001$ (Cora) (nguồn: `agreement_summary.csv`).

**Bảng A2-2** ($\kappa@5$; nguồn: `results/analysis/a2/a2_2_between.csv`). $\Delta_A$, $\Delta_B$ so với $I_{\rm within}$ (5 GNN) của từng phương pháp; $p_{\rm holm}<0{,}001$ cho mọi $\Delta$ (Wilcoxon cặp, Holm trong từng dataset).

| Dataset | Cặp (A – B) | $D_{\rm between}$ | $D_{\rm between\_avg}$ | $\Delta_A$ [CI 95%] | $\Delta_B$ [CI 95%] |
|---|---|---|---|---|---|
| BA-Shapes | Saliency – GNNExplainer | 0,143 | 0,180 | 0,825 [0,767; 0,882] | 0,356 [0,287; 0,426] |
| | Saliency – PGExplainer | 0,102 | 0,098 | 0,865 [0,815; 0,915] | 0,548 [0,491; 0,605] |
| | GNNExplainer – PGExplainer | 0,048 | 0,044 | 0,451 [0,389; 0,513] | 0,603 [0,558; 0,646] |
| Cora | Saliency – GNNExplainer | 0,342 | 0,345 | 0,658 [0,608; 0,711] | 0,456 [0,408; 0,505] |
| | Saliency – PGExplainer | -0,010 | -0,005 | 1,010 [0,982; 1,038] | 0,762 [0,725; 0,797] |
| | GNNExplainer – PGExplainer | -0,017 | -0,021 | 0,815 [0,783; 0,847] | 0,769 [0,735; 0,801] |

![Hình A2 BA-Shapes: within, cross-GNN (cùng e và khác e), between theo từng phương pháp](figures/final/a2_bashapes.png)

![Hình A2 Cora: within, cross-GNN (cùng e và khác e), between theo từng phương pháp](figures/final/a2_cora.png)

*Hình A2 (nguồn: `docs/figures/final/a2_bashapes.png`, `a2_cora.png`, sinh bởi `scripts/analyze_variance.py`). Thanh "between" của mỗi phương pháp là trung bình $D_{\rm between}$ của hai cặp có phương pháp đó; thanh lỗi là CI 95% bootstrap trên nút; đường nét đứt là mức Random–Random.*

**Bảng A2-3** (nguồn: `results/analysis/a2/a2_3_fid_sd_vs_method_diff.csv`). SD của Fid-@5 (Drop) theo seed explainer trong từng nút, so với chênh lệch giữa phương pháp:

| Dataset | SD: Saliency / GNNExp / PG | chênh Sal–GNNExp | chênh Sal–PG | chênh GNNExp–PG | Nút có SD(PG) > chênh GNNExp–PG |
|---|---|---|---|---|---|
| BA-Shapes (140) | 0 / 0,011 / 0,045 | 0,183 | 0,125 | 0,099 | 42,1% |
| Cora (100) | 0 / 0,019 / 0,013 | 0,080 | 0,109 | 0,083 | 8,0% |

### 6.5. Trả lời RQ1–RQ3

**RQ1: một phương pháp có tự lệch với chính nó không?** Có, trừ Saliency. Ở Bước 1 ($g=1000$), $I_{\rm within}$ của GNNExplainer là 0,488 (BA-Shapes) và 0,788 (Cora), của PGExplainer là 0,640 và 0,691. Saliency là 0,968 và 1,000. Qua 5 GNN, các số gần như không đổi trên BA-Shapes (0,499; 0,650) và PGExplainer trên Cora tăng lên 0,752. Thứ tự khác nhau giữa hai dataset: trên BA-Shapes GNNExplainer bất ổn nhất (CI không giao nhau với PGExplainer), trên Cora PGExplainer thấp hơn GNNExplainer nhưng CI gần chạm nhau (0,752 [0,730; 0,772] so với 0,798 [0,771; 0,823]).

**RQ2: đổi seed train GNN thì giải thích đổi bao nhiêu?** Tính trên nút hai GNN dự đoán cùng lớp (tỷ lệ nút đổi dự đoán 1,6% và 4,5%). Saliency tụt từ 0,968 và 1,000 xuống 0,855 và 0,821. PGExplainer trên Cora nhạy nhất: $I_{\rm gnn}$ = 0,308 (khác $e$: 0,296) so với $I_{\rm within}$ = 0,752. GNNExplainer trên Cora 0,711 (khác $e$: 0,628) so với 0,798. Trên BA-Shapes, độ nhạy của PGExplainer nhỏ (0,611 so với 0,650, CI chồng nhau); số $I_{\rm gnn}$ cùng $e$ của GNNExplainer (0,825) cao hơn $I_{\rm within}$ vì khởi tạo chung seed (mục 6.6); số đối chứng là 0,403. Theo cách đọc CI không giao nhau, dòng 4 của quy tắc 6.3 thỏa ở 4 trong 6 ô (dataset × phương pháp).

**RQ3: hai phương pháp lệch nhau có nhiều hơn mức chúng tự lệch không?** Có, ở cả 6 cặp. $D_{\rm between}$ thấp hơn $I_{\rm within}$ của cả hai phương pháp (CI của $\Delta_A$ và $\Delta_B$ đều dương, $p_{\rm holm}<0{,}001$), nên dòng 1 của quy tắc 6.3 thỏa: **bất đồng thực sự**, không giải thích được bằng nhiễu seed. Dòng 2 không thỏa ở cặp nào ($\Delta_{\min}$ nhỏ nhất là 0,313, GNNExplainer–PGExplainer trên BA-Shapes). Dòng 3 chỉ thỏa yếu ở một cặp: Saliency–GNNExplainer trên BA-Shapes, $D_{\rm between\_avg}-D_{\rm between}=+0{,}037$ [0,022; 0,053]; năm cặp còn lại CI chứa 0. Dòng 5 thỏa (Cora 1,000 so với 0,821; BA-Shapes 0,968 so với 0,855). Trên Cora, mọi cặp có PGExplainer có $D_{\rm between}\approx0$ (-0,010; -0,017), tức mức Random–Random (-0,001), và PGExplainer cũng không khác Nearest-First (-0,013 [-0,039; 0,011]).

### 6.6. Lưu ý và hạn chế phương pháp luận

1. **Saliency tất định.** τ_b giữa hai lượt (seed 0 và 1) bằng 1,000 trên cả hai dataset; `scripts/check_scores.py` dừng nếu `max_diff` giữa các seed lớn hơn 0 (các file điểm thô không có trong repo nên báo cáo chỉ dựa vào τ_b và mã kiểm tra). $\kappa@5$ within của Saliency trên BA-Shapes là 0,968 chứ không phải 1 vì phá hòa ngẫu nhiên có seed giữa các cạnh cùng điểm (Mục 3.8); đó là hệ quả của điểm hòa, không phải bất ổn của thuật toán. Trên Cora bằng 1,000.
2. **Cross-GNN cùng $e$ bị thổi phồng với GNNExplainer.** Theo định nghĩa của kế hoạch, cross-GNN ghép $(g,e)$ với $(g',e)$; GNNExplainer khởi tạo mask từ seed nên hai lượt cùng $e$ dùng chung khởi tạo: 0,825 (cùng $e$) so với 0,403 (khác $e$) trên BA-Shapes, 0,711 so với 0,628 trên Cora. Với PGExplainer chênh lệch nhỏ (0,611 và 0,574; 0,308 và 0,296). Khi diễn giải "nhạy theo GNN", nên dùng số đối chứng `cross_gnn_diff_e`. Số này vẫn lẫn hai nguồn thay đổi (GNN và seed explainer), nên không phải độ nhạy GNN thuần (nguồn: `src/sna/evaluation/runner_v2.py`).
3. **PGExplainer nhạy với nhiễu số học.** So hai lượt chạy lưới PG trên BA-Shapes cùng seed nhưng khác số luồng CPU (lượt 22/11 và lượt cuối kỳ): $\kappa@5$ trung bình giữa hai lượt chỉ 0,801 (nhỏ nhất 0,396), AUROC motif lệch trung bình 0,089 (tối đa 0,474); một cấu hình từng có AUROC 0,158 ở lượt cũ nhưng 0,632 ở lượt cuối kỳ (nguồn: `results/analysis/a2/pg_thread_sensitivity.csv`, 25 cấu hình; chưa đo trên Cora). PGExplainer chỉ tái lập chính xác khi giữ nguyên số luồng (2, theo `configs/protocol_final.yaml`); chạy lại một ô cho điểm trùng tuyệt đối ở cả hai dataset (`docs/PHAN_BAO_CAO_PGEXPLAINER_THIEN.md`, đo trên lượt 22/11). Phép so này cho thấy độ nhạy số học, không đo $I_{\rm within}$.
4. **Nút đổi dự đoán.** $I_{\rm gnn}$ chỉ tính trên nút hai GNN dự đoán cùng lớp, nên không nói gì về nút đã đổi dự đoán (1,6% và 4,5% nút trung bình).
5. **Bảng A2-3.** SD của Fid-@5 theo seed nhỏ so với chênh lệch giữa phương pháp ở mức trung bình, nhưng theo từng nút thì SD của PGExplainer vượt chênh GNNExplainer–PGExplainer ở 42,1% nút BA-Shapes; thứ hạng phương pháp theo Fid-@5 ở mức nút có thể đảo theo seed. Chưa có kiểm định thống kê cho điểm này.
6. **Thống kê.** Bootstrap theo nút (2.000 lần), Wilcoxon cặp, Holm trong từng dataset (họ: mọi phép so sánh của dataset, 12 phép mỗi dataset). Cột `p_holm` là $p$ đã hiệu chỉnh; bảng ghi "< 0,001" khi $p_{\rm holm}<0{,}001$ (mọi $\Delta_A$, $\Delta_B$, $\Delta_{\min}$ đều như vậy). Kích thước hiệu ứng là CI của $\Delta$.
7. **Giới hạn.** Mọi suy luận dựa trên 5 GNN × 5 seed explainer, 140 nút (BA-Shapes) và 100 nút (Cora; 86 nút có $\kappa@5$), chỉ kiến trúc GCN (GAT không làm). 5 GNN chỉ khác seed huấn luyện, nên không suy ra được cho kiến trúc khác. Bảng đầy đủ và cách đọc: `docs/BANG_SO_LIEU_FB2_INSTABILITY.md`.
