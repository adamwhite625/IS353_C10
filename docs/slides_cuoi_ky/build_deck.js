// Build the final-term slide deck (Group 09, C.10).
// Every number is read from numbers.json, which verify_numbers.py recomputes from the result CSVs.
// Usage: NODE_PATH=<dir with pptxgenjs> node build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8").replace(/\bNaN\b/g, "null"));
const FIG = path.join(__dirname, "..", "figures", "final");

// ---------- formatting helpers (Vietnamese decimal comma, true minus sign)
const f = (x, d = 3) => (x < 0 && Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : x).toFixed(d).replace(".", ",").replace("-", "−");
const pct = (a, b) => f((100 * a) / b, 1) + "%";
const ci = (a) => `${f(a[0])} [${f(a[1])}; ${f(a[2])}]`;

// ---------- theme
const THEME = {
  name: "SNA G09 C10",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2933", lt1: "FFFFFF", dk2: "0F2A4A", lt2: "F3F6F9",
    accent1: "0E7C86", accent2: "D9701A", accent3: "7B5EA7", accent4: "9AA5B1",
    accent5: "B42318", accent6: "2E7D32", hlink: "0E7C86", folHlink: "7B5EA7",
  },
};
const NAVY = "0F2A4A", TEAL = "0E7C86", ORANGE = "D9701A", PURPLE = "7B5EA7", GRAY = "9AA5B1";
const RED = "B42318", INK = "1F2933", MUTED = "5B6773", CARD = "F3F6F9", LINE = "D5DCE3", WHITE = "FFFFFF";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Making Graph ML Models Explainable - Bao cao cuoi ky - Nhom 09";
pres.author = "Nhom 09";

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9.0, h: 0.85, fontFace: "Cambria", fontSize: 24, bold: true, color: NAVY, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.1, y: 5.3, w: 0.5, h: 0.25, fontFace: "Calibri", fontSize: 10, color: MUTED, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: NAVY } });

let section = null;
function newSection(title) { pres.addSection({ title }); section = title; }

function content(title, source, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (source) s.addText("Nguồn: " + source, { x: 0.5, y: 5.3, w: 8.4, h: 0.25, fontSize: 10, color: MUTED, margin: 0, valign: "middle", isTextBox: true });
  if (notes) s.addNotes(notes);
  return s;
}

function card(s, x, y, w, h, fill = CARD) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: LINE, width: 0.75 }, rectRadius: 0.08 });
}

function table(s, rows, opts) {
  // rows: array of arrays of strings or {text, options}
  const body = rows.map((r, ri) => r.map((c, ci_) => {
    const cell = typeof c === "string" ? { text: c, options: {} } : { text: c.text, options: { ...(c.options || {}) } };
    const o = cell.options;
    o.fontSize = o.fontSize || opts.fontSize || 11;
    o.fontFace = "Calibri";
    o.valign = "middle";
    o.margin = [0.03, 0.07, 0.03, 0.07];
    o.border = { type: "solid", pt: 0.5, color: LINE };
    if (ri === 0) { o.bold = true; o.color = WHITE; o.fill = { color: NAVY }; }
    else { o.color = o.color || INK; if (!o.fill) o.fill = { color: ri % 2 === 0 ? CARD : WHITE }; }
    if (ci_ > 0 && !o.align) o.align = opts.align || "center";
    return cell;
  }));
  s.addTable(body, { x: opts.x, y: opts.y, w: opts.w, colW: opts.colW, rowH: opts.rowH || 0.3, autoPage: false });
}
const B = (t, extra = {}) => ({ text: t, options: { bold: true, color: TEAL, ...extra } });

const METH = { saliency: "Saliency", gnnexplainer: "GNNExplainer", pgexplainer: "PGExplainer", nearest_first: "Nearest-First", random: "Random" };
const M5 = ["saliency", "gnnexplainer", "pgexplainer", "nearest_first", "random"];

// =====================================================================
// 1. Cover (dark)
// =====================================================================
{
  newSection("Mở đầu");
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
  s.addText("ĐỀ TÀI C.10  |  BÁO CÁO CUỐI KỲ", { x: 0.7, y: 0.6, w: 8.6, h: 0.35, fontSize: 14, bold: true, color: "7FD1D9", charSpacing: 3, margin: 0, isTextBox: true });
  s.addText("Making Graph ML Models Explainable", { x: 0.7, y: 1.2, w: 8.6, h: 0.9, fontFace: "Cambria", fontSize: 38, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Bất ổn định, bất đồng và mâu thuẫn metric trong giải thích GNN", { x: 0.7, y: 2.2, w: 8.6, h: 0.6, fontFace: "Cambria", fontSize: 22, color: "CADCFC", margin: 0, isTextBox: true });
  s.addText("Nhóm 09 — GVHD: TS. Trần Hưng Nghiệp", { x: 0.7, y: 3.15, w: 7.3, h: 0.35, fontSize: 16, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText([
    { text: "Nguyễn Trường Vũ (23521812)  ·  Nguyễn Lê Bảo Đăng (23520230)", options: { breakLine: true } },
    { text: "Nguyễn Hải Thiện (23521481)  ·  Nguyễn Trọng Khôi (23520783)", options: { breakLine: true } },
    { text: "Vòng Việt Đức (23520324)  ·  Nguyễn Hoàng Kha (23520667)", options: { breakLine: true } },
    { text: "Phân tích mạng xã hội (SNA)  ·  UIT  ·  Tháng 11/2026" },
  ], { x: 0.7, y: 3.55, w: 7.3, h: 1.5, fontSize: 14, color: "CADCFC", margin: 0, paraSpaceAfter: 3, valign: "top", isTextBox: true });
  // motif: three small linked nodes at right
  const nodes = [[8.2, 3.9], [8.9, 4.4], [8.0, 4.75], [9.2, 3.6]];
  const edges = [[0, 1], [0, 2], [1, 2], [0, 3]];
  edges.forEach(([a, b]) => s.addShape(pres.shapes.LINE, { x: Math.min(nodes[a][0], nodes[b][0]), y: Math.min(nodes[a][1], nodes[b][1]), w: Math.abs(nodes[a][0] - nodes[b][0]) || 0.001, h: Math.abs(nodes[a][1] - nodes[b][1]) || 0.001, flipV: (nodes[a][0] - nodes[b][0]) * (nodes[a][1] - nodes[b][1]) < 0, line: { color: "7FD1D9", width: 1.5 } }));
  nodes.forEach(([x, y], i) => s.addShape(pres.shapes.OVAL, { x: x - 0.1, y: y - 0.1, w: 0.2, h: 0.2, fill: { color: i === 0 ? ORANGE : "7FD1D9" }, line: { color: NAVY, width: 0.5 } }));
  s.addNotes("Xin chào thầy và các bạn. Nhóm 09 trình bày đề tài C.10 về khả năng giải thích của mô hình đồ thị. Báo cáo cuối kỳ trả lời bốn phản hồi của thầy sau giữa kỳ bằng một lưới thực nghiệm 5 checkpoint GCN × 5 seed explainer trên BA-Shapes và Cora. Nhóm không đề xuất thuật toán mới mà làm nghiên cứu thực nghiệm có kiểm soát.");
}

// =====================================================================
// 2. Problem + RQs
// =====================================================================
{
  const s = content("Ba explainer chọn ba tập cạnh khác nhau: bất đồng thật, hay chỉ là nhiễu?",
    "docs/BAO_CAO_CUOI_KY.md, Mục 1.1 và 1.3",
    "Một explainer cho mỗi nút đích một điểm trên từng cạnh trong lân cận 2-hop, và giải thích là top-k cạnh điểm cao nhất. Ba phương pháp phổ biến là Saliency, GNNExplainer, PGExplainer thường chọn tập cạnh khác nhau. Câu hỏi của thầy là sự khác nhau đó có thật hay chỉ do mỗi phương pháp tự dao động theo seed. Vì vậy nhóm đặt ba câu hỏi RQ1 đến RQ3, đi theo đúng thứ tự thầy dặn: đổi seed explainer trước, đổi seed GNN sau, rồi so giữa các phương pháp. Thêm một câu hỏi phụ về mâu thuẫn Fidelity và AUROC từ giữa kỳ.");
  s.addText([
    { text: "Giải thích = top-k cạnh quan trọng nhất quanh một nút đích (GCN 2 lớp).", options: { bullet: true, breakLine: true } },
    { text: "Saliency, GNNExplainer, PGExplainer thường chọn các tập cạnh khác nhau.", options: { bullet: true } },
  ], { x: 0.5, y: 1.2, w: 9.0, h: 0.85, fontSize: 16, color: INK, paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
  const rq = [
    ["RQ1", "Bất ổn định nội tại", "Giữ nguyên GNN, chỉ đổi seed explainer: một phương pháp tự lệch với chính nó bao nhiêu?"],
    ["RQ2", "Nhạy theo mô hình", "Đổi seed huấn luyện GNN: lời giải thích đổi bao nhiêu, kể cả khi dự đoán không đổi?"],
    ["RQ3", "Bất đồng thực sự", "Hai phương pháp lệch nhau nhiều hơn mức mỗi phương pháp tự lệch không?"],
  ];
  rq.forEach((r, i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 2.2, 2.9, 2.2);
    s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: 2.35, w: 0.7, h: 0.7, fill: { color: [TEAL, ORANGE, PURPLE][i] }, line: { color: WHITE, width: 0 } });
    s.addText(r[0], { x: x + 0.15, y: 2.35, w: 0.7, h: 0.7, fontSize: 15, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[1], { x: x + 0.95, y: 2.35, w: 1.85, h: 0.7, fontFace: "Cambria", fontSize: 15, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
    s.addText(r[2], { x: x + 0.15, y: 3.15, w: 2.6, h: 1.2, fontSize: 14, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
  card(s, 0.5, 4.55, 9.0, 0.6, "FFF4E8");
  s.addText([{ text: "Câu hỏi phụ (FB-3): ", options: { bold: true, color: ORANGE } }, { text: "vì sao Fidelity và Motif AUROC không chọn cùng một phương pháp tốt nhất?" }],
    { x: 0.65, y: 4.55, w: 8.7, h: 0.6, fontSize: 14, color: INK, valign: "middle", margin: 0, isTextBox: true });
}

// =====================================================================
// 3. Feedback -> answers
// =====================================================================
{
  const s = content("Bốn phản hồi của thầy được trả lời bằng bốn khối thực nghiệm",
    "docs/BAO_CAO_CUOI_KY.md, Mục 1.2; docs/KE_HOACH_CUOI_KY_FINAL.md, Mục 1",
    "Sau giữa kỳ ngày 29/10, thầy nêu bốn điểm. FB-1 yêu cầu quy trình tái lập rõ ràng, nhóm trả lời bằng mã SHA-256 cho dữ liệu, checkpoint và CSV điểm. FB-2 yêu cầu tách bất đồng khỏi bất ổn định, nhóm trả lời bằng lưới 5 seed GNN × 5 seed explainer. FB-3 về mâu thuẫn metric, nhóm xác minh trên 25 cấu hình và quét 355 cặp nút–GNN. FB-4 là làm lõi trước khi mở rộng, nên các phần như GAT hay Amazon Photo được ghi vào hạn chế.");
  const rows = [
    ["Mã", "Thầy nhắc", "Nhóm trả lời bằng", "Slide"],
    [B("FB-1"), { text: "Viết rõ quy trình train GNN → explainer → metric, trên đúng data và checkpoint", options: { align: "left" } }, { text: "SHA-256 cho dữ liệu, checkpoint, CSV điểm; bộ kiểm tra hợp đồng dữ liệu", options: { align: "left" } }, "4"],
    [B("FB-2"), { text: "Tách bất đồng giữa phương pháp khỏi bất ổn định của từng phương pháp; seed explainer trước, seed GNN sau", options: { align: "left" } }, { text: "Lưới 5 GNN × 5 seed explainer; quy tắc diễn giải ghi trước khi xem số", options: { align: "left" } }, "5–7, 11"],
    [B("FB-3"), { text: "Các metric mâu thuẫn: xác minh lại rồi phân tích một số nút", options: { align: "left" } }, { text: "25 cấu hình, 355 cặp nút–GNN, 5 cơ chế T1–T5, case study", options: { align: "left" } }, "8–10"],
    [B("FB-4"), { text: "Làm kiểm chứng cốt lõi trước, rồi mới mở rộng nhiều dataset", options: { align: "left" } }, { text: "Tầng B (GAT, Amazon Photo) không làm, ghi ở hạn chế", options: { align: "left" } }, "12"],
  ];
  table(s, rows, { x: 0.5, y: 1.25, w: 9.0, colW: [0.8, 3.6, 3.7, 0.9], rowH: [0.35, 0.85, 0.85, 0.75, 0.75], fontSize: 13 });
}

// =====================================================================
// 4. Setup and reproducibility
// =====================================================================
{
  newSection("Thiết lập và kết quả chính");
  const ck = N.ckpt;
  const s = content("Mọi con số bám vào hash: dữ liệu, 10 checkpoint và 25 cấu hình mỗi dataset",
    "docs/BAO_CAO_CUOI_KY.md, Mục 3.1, 4.3, 4.5; results/checkpoints/checkpoint_table.csv; configs/protocol_final.yaml",
    `Chuỗi chạy là dữ liệu cố định bằng SHA-256, năm checkpoint GCN seed 1000 đến 1004 cho mỗi dataset, ba explainer cùng hai baseline, rồi bộ đánh giá. Tổng cộng ${ck.length} checkpoint đã vào Git, hash của file .pt là mốc tái lập vì train lại trên máy khác không cho cùng hash. Đánh giá trên 140 nút test của BA-Shapes và 100 nút test của Cora, mỗi phương pháp có tối đa 25 cấu hình (g, e). Bảng thay đổi v1 sang v2 liệt kê các lỗi đã sửa so với giữa kỳ: thứ tự đặt seed của PGExplainer, forward 3-hop, size-loss của GNNExplainer và cỡ mẫu đánh giá.`);
  const chips = [["Dữ liệu", "SHA-256 cố định"], ["5 GCN / dataset", "seed 1000–1004"], ["3 explainer + 2 baseline", "Sal · GNNE · PG · NF · Rand"], ["Metric", "Fidelity · AUROC · κ@k"]];
  chips.forEach((c, i) => {
    const x = 0.5 + i * 2.28;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.2, w: 2.0, h: 0.8, fill: { color: NAVY }, line: { color: NAVY, width: 0 }, rectRadius: 0.08 });
    s.addText([{ text: c[0], options: { bold: true, fontSize: 13, color: WHITE, breakLine: true } }, { text: c[1], options: { fontSize: 11, color: "CADCFC" } }], { x, y: 1.2, w: 2.0, h: 0.8, align: "center", valign: "middle", margin: 0.03, isTextBox: true });
    if (i < 3) s.addText("→", { x: x + 2.0, y: 1.2, w: 0.28, h: 0.8, fontSize: 18, bold: true, color: ORANGE, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  // stat cards
  const stats = [["140", "nút test BA-Shapes"], ["100", "nút test Cora"], ["25", "cấu hình (g, e) mỗi dataset"]];
  stats.forEach((st, i) => {
    const y = 2.2 + i * 0.85;
    card(s, 0.5, y, 2.6, 0.75);
    s.addText(st[0], { x: 0.6, y, w: 0.95, h: 0.75, fontFace: "Cambria", fontSize: 30, bold: true, color: TEAL, valign: "middle", margin: 0, isTextBox: true });
    s.addText(st[1], { x: 1.55, y, w: 1.5, h: 0.75, fontSize: 13, color: INK, valign: "middle", margin: 0, isTextBox: true });
  });
  const rows = [
    ["Thay đổi v1 → v2", "Lý do"],
    [{ text: "F1: đặt seed trước khi khởi tạo MLP của PGExplainer", options: { align: "left" } }, { text: "Hai lượt cùng seed từng lệch tới 0,885", options: { align: "left" } }],
    [{ text: "F9: PGExplainer forward trên đồ thị con 3-hop", options: { align: "left" } }, { text: "Logit đồ thị con phải bằng đồ thị đầy đủ", options: { align: "left" } }],
    [{ text: "F10: GNNExplainer size-loss sum thay mean", options: { align: "left" } }, { text: "Mask dồn quanh 0,95 ở bản mean", options: { align: "left" } }],
    [{ text: "20 nút validation → 140 / 100 nút test", options: { align: "left" } }, { text: "Cỡ mẫu: giữa kỳ chỉ 10/20 nút có AUROC", options: { align: "left" } }],
    [{ text: "Checkpoint, split cố định bằng SHA-256", options: { align: "left" } }, { text: "FB-1, FB-2", options: { align: "left" } }],
  ];
  table(s, rows, { x: 3.3, y: 2.2, w: 6.2, colW: [3.5, 2.7], rowH: [0.32, 0.42, 0.42, 0.42, 0.42, 0.42], fontSize: 11 });
  s.addText([{ text: "Mốc tái lập = SHA-256 của file .pt ", options: { bold: true, color: NAVY } }, { text: "(10 checkpoint trong Git). PGExplainer chỉ tái lập chính xác khi giữ 2 luồng CPU." }],
    { x: 0.5, y: 4.82, w: 9.0, h: 0.4, fontSize: 13, color: INK, valign: "middle", margin: 0, isTextBox: true });
}

// =====================================================================
// 5. Main results
// =====================================================================
{
  const bd = (m, k) => N[`main.bashapes.${m}.${k}`];
  const cr = (m, k) => N[`main.cora.${m}.${k}`];
  const s = content("Không phương pháp nào thắng mọi tiêu chí: Nearest-First dẫn Drop@5, PGExplainer dẫn AUROC",
    "results/analysis/main/main_by_method_budget.csv (k = 5, TB qua 25 cấu hình, CI 95% bootstrap theo nút)",
    `Bảng này là trung bình theo nút qua 25 cấu hình, kèm CI 95% bootstrap. Trên BA-Shapes, Nearest-First, một baseline chỉ chọn cạnh gần, có Drop@5 ${f(bd("nearest_first", "fid_drop")[0])}, cao nhất, còn PGExplainer có AUROC ${f(bd("pgexplainer", "roc_auc")[0])} cao nhất nhưng Drop@5 chỉ ${f(bd("pgexplainer", "fid_drop")[0])}. Các CI của Nearest-First, GNNExplainer và Saliency chồng lấn gần như hoàn toàn nên thứ hạng Drop giữa chúng không vững. Trên Cora, PGExplainer có Drop@5 ${f(cr("pgexplainer", "fid_drop")[0])}, không khác Random ${f(cr("random", "fid_drop")[0])}. Inv âm trên Cora nghĩa là chỉ giữ top-5 cạnh còn làm xác suất tăng.`);
  const best = { drop: "nearest_first", inv: "gnnexplainer", auc: "pgexplainer" };
  const rowsB = [["BA-Shapes (140 nút)", "Drop@5 (cao = tốt)", "Inv@5 (thấp = tốt)", "AUROC (71 nút)"]];
  M5.forEach((m) => {
    const d = bd(m, "fid_drop"), iv = bd(m, "fid_inv"), au = bd(m, "roc_auc");
    const c = (arr, isBest) => isBest ? B(ci(arr), { color: TEAL }) : ci(arr);
    rowsB.push([{ text: METH[m], options: { align: "left", bold: true } }, c(d, m === best.drop), c(iv, m === best.inv), c(au, m === best.auc)]);
  });
  table(s, rowsB, { x: 0.5, y: 1.2, w: 9.0, colW: [2.1, 2.3, 2.3, 2.3], rowH: 0.27, fontSize: 12 });
  const rowsC = [["Cora (100 nút)", "Drop@5 (cao = tốt)", "Inv@5 (thấp = tốt)"]];
  M5.forEach((m) => {
    const d = cr(m, "fid_drop"), iv = cr(m, "fid_inv");
    const g = m === "gnnexplainer";
    rowsC.push([{ text: METH[m], options: { align: "left", bold: true } }, g ? B(ci(d)) : ci(d), g ? B(ci(iv)) : ci(iv)]);
  });
  table(s, rowsC, { x: 0.5, y: 3.15, w: 6.3, colW: [1.5, 2.4, 2.4], rowH: 0.27, fontSize: 12 });
  card(s, 7.0, 3.15, 2.5, 1.62, "FFF4E8");
  s.addText([
    { text: `PGExplainer ≈ Random trên Cora: Drop ${f(cr("pgexplainer", "fid_drop")[0])} vs ${f(cr("random", "fid_drop")[0])}`, options: { breakLine: true, bullet: true } },
    { text: "Inv âm: chỉ giữ top-5 còn làm xác suất tăng", options: { bullet: true } },
  ], { x: 7.1, y: 3.2, w: 2.3, h: 1.52, fontSize: 12, color: INK, valign: "middle", paraSpaceAfter: 4, margin: 0, isTextBox: true });
}

// =====================================================================
// 6. Instability vs Disagreement (Figure A2)
// =====================================================================
{
  newSection("FB-2: bất ổn định và bất đồng");
  const a = (ds, m) => N[`a21.${ds}.${m}`];
  const dB = ["saliency vs gnnexplainer", "saliency vs pgexplainer", "gnnexplainer vs pgexplainer"].map((p) => N[`a22.bashapes.${p}.D`][0]);
  const dC = ["saliency vs gnnexplainer", "saliency vs pgexplainer", "gnnexplainer vs pgexplainer"].map((p) => N[`a22.cora.${p}.D`][0]);
  const s = content("Mỗi explainer tự lệch ít hơn mức hai explainer lệch nhau (cột xanh lục thấp nhất)",
    "docs/figures/final/a2_bashapes.png, a2_cora.png; results/analysis/a2/a2_1_within_cross_gnn.csv, a2_2_between.csv",
    `Hình A2 so ba đại lượng theo κ@5. Cột xanh dương là within: giữ nguyên GNN, đổi seed explainer. Saliency tự lệch gần như bằng 0 với κ ${f(a("bashapes", "saliency").within)} trên BA-Shapes, GNNExplainer chỉ ${f(a("bashapes", "gnnexplainer").within)} và PGExplainer ${f(a("bashapes", "pgexplainer").within)}. Cột xanh lục là between giữa hai phương pháp, chỉ từ ${f(Math.min(...dB))} đến ${f(Math.max(...dB))} trên BA-Shapes, sát mức Random–Random gần 0. Trên Cora between nằm từ ${f(Math.min(...dC))} đến ${f(Math.max(...dC))}. Nghĩa là sự khác nhau giữa các phương pháp lớn hơn nhiều mức dao động seed của từng phương pháp.`);
  s.addImage({ path: path.join(FIG, "a2_bashapes.png"), x: 0.5, y: 1.15, w: 4.4, h: 2.933, altText: "Hình A2 BA-Shapes: within, cross-GNN, between theo phương pháp" });
  s.addImage({ path: path.join(FIG, "a2_cora.png"), x: 5.1, y: 1.15, w: 4.4, h: 2.933, altText: "Hình A2 Cora: within, cross-GNN, between theo phương pháp" });
  card(s, 0.5, 4.25, 9.0, 0.95);
  s.addText([
    { text: "BA-Shapes: ", options: { bold: true, color: TEAL } }, { text: `tự lệch (within) ${f(a("bashapes", "gnnexplainer").within)} – ${f(a("bashapes", "saliency").within)}; giữa hai phương pháp chỉ ${f(Math.min(...dB))} – ${f(Math.max(...dB))}`, options: { breakLine: true } },
    { text: "Cora: ", options: { bold: true, color: TEAL } }, { text: `tự lệch ${f(a("cora", "pgexplainer").within)} – ${f(a("cora", "saliency").within)}; giữa hai phương pháp ${f(Math.min(...dC))} – ${f(Math.max(...dC))}` },
  ], { x: 0.65, y: 4.25, w: 8.7, h: 0.95, fontSize: 14, color: INK, valign: "middle", paraSpaceAfter: 3, margin: 0, isTextBox: true });
}

// =====================================================================
// 7. A2-2 + caveats
// =====================================================================
{
  const pairs = [["saliency vs gnnexplainer", "Saliency – GNNExplainer"], ["saliency vs pgexplainer", "Saliency – PGExplainer"], ["gnnexplainer vs pgexplainer", "GNNExplainer – PGExplainer"]];
  const allPos = Object.keys(N).filter((k) => /^a22\..*\.Delta_[AB]$/.test(k)).every((k) => N[k][1] > 0 && N[k][3] < 0.001);
  if (!allPos) throw new Error("A2-2 claim (all Delta CI > 0, p_holm < 0.001) not supported by data");
  const pg = N.pgthread;
  const s = content("Cả 6 cặp đều là bất đồng thực sự, không giải thích được bằng nhiễu seed",
    "results/analysis/a2/a2_2_between.csv, a2_1_within_cross_gnn.csv, pg_thread_sensitivity.csv",
    `Quy tắc ghi trước khi xem số: nếu κ giữa hai phương pháp thấp hơn rõ mức tự lệch của cả hai, và CI của hiệu Δ_A và Δ_B đều dương, thì đó là bất đồng thực sự. Cả 6 cặp trên hai dataset thỏa quy tắc, với p Holm dưới 0,001; Δ nhỏ nhất là ${f(Math.min(...N.delta_min))}. Hai lưu ý phương pháp luận. Thứ nhất, cross-GNN cùng seed explainer bị thổi phồng với GNNExplainer vì dùng chung khởi tạo mask: ${f(N["a21.bashapes.gnnexplainer"].gnn_same)} so với ${f(N["a21.bashapes.gnnexplainer"].gnn_diff)} khi khác seed. Thứ hai, PGExplainer nhạy nhiễu số học: hai lượt chạy khác số luồng CPU cho κ@5 trung bình ${f(pg.kappa_mean)}, nhỏ nhất ${f(pg.kappa_min)}, và AUROC của cùng một ô có thể lệch tới ${f(pg.auc_diff_max)}.`);
  const rows = [["Dataset", "Cặp (A – B)", "κ@5 giữa A, B", "Δ_A = within(A) − between [CI 95%]", "Δ_B [CI 95%]"]];
  ["bashapes", "cora"].forEach((ds) => pairs.forEach(([k, label], i) => {
    const D = N[`a22.${ds}.${k}.D`][0];
    const da = N[`a22.${ds}.${k}.Delta_A`], db = N[`a22.${ds}.${k}.Delta_B`];
    rows.push([i === 0 ? { text: ds === "bashapes" ? "BA-Shapes" : "Cora", options: { bold: true, align: "left" } } : { text: "", options: {} }, { text: label, options: { align: "left" } }, f(D), `${f(da[0])} [${f(da[1])}; ${f(da[2])}]`, `${f(db[0])} [${f(db[1])}; ${f(db[2])}]`]);
  }));
  table(s, rows, { x: 0.5, y: 1.15, w: 9.0, colW: [1.1, 2.4, 1.3, 2.4, 1.8], rowH: 0.29, fontSize: 11 });
  card(s, 0.5, 3.55, 4.4, 1.5, "FFF4E8");
  s.addText([{ text: "Cross-GNN cùng seed bị thổi phồng", options: { bold: true, color: ORANGE, breakLine: true } },
    { text: `GNNExplainer (BA-Shapes): ${f(N["a21.bashapes.gnnexplainer"].gnn_same)} cùng e, nhưng ${f(N["a21.bashapes.gnnexplainer"].gnn_diff)} khác e. Dùng số khác e.` }],
    { x: 0.65, y: 3.6, w: 4.1, h: 1.4, fontSize: 14, color: INK, valign: "top", paraSpaceAfter: 4, margin: 0, isTextBox: true });
  card(s, 5.1, 3.55, 4.4, 1.5, "FFF4E8");
  s.addText([{ text: "PGExplainer nhạy số luồng CPU", options: { bold: true, color: ORANGE, breakLine: true } },
    { text: `Hai lượt khác luồng, ${pg.n} cấu hình BA-Shapes: κ@5 TB ${f(pg.kappa_mean)} (min ${f(pg.kappa_min)}); AUROC lệch tới ${f(pg.auc_diff_max)}.` }],
    { x: 5.25, y: 3.6, w: 4.1, h: 1.4, fontSize: 14, color: INK, valign: "top", paraSpaceAfter: 4, margin: 0, isTextBox: true });
}

// =====================================================================
// 8. A3-0 contradiction
// =====================================================================
{
  newSection("FB-3: error analysis");
  const [nc, nt] = N["a30.conflicts"];
  const fw = N["a30.fid_win"], aw = N["a30.auc_win"];
  const s = content(`Mâu thuẫn Fidelity–AUROC là thật: ${nc}/${nt} cấu hình có hai người thắng khác nhau`,
    "results/analysis/a3_final/a3_0_conflicts.csv (tie_seed = average), a3_0_metrics.csv; BA-Shapes, 25 cấu hình (g, e)",
    `Với mỗi trong 25 cấu hình (g, e) của BA-Shapes, nhóm so phương pháp dẫn đầu theo Drop@5 với phương pháp dẫn đầu theo AUROC. Hai người dẫn đầu khác nhau ở ${nc}/${nt} cấu hình, tức ${pct(nc, nt)}. Nearest-First dẫn Drop@5 ở ${fw.nearest_first} cấu hình và GNNExplainer ở ${fw.gnnexplainer}; PGExplainer dẫn AUROC ở ${aw.pgexplainer} cấu hình và Nearest-First ở ${aw.nearest_first}. Kết quả giống hệt với 5 seed phá hòa khác nhau. PGExplainer có AUROC trung bình cao nhất nhưng Drop@5 thấp nhất, và AUROC của nó dao động rất mạnh từ ${f(N["a30.pgexplainer.roc_auc"][1])} đến ${f(N["a30.pgexplainer.roc_auc"][2])} giữa các cấu hình.`);
  card(s, 0.5, 1.2, 4.2, 1.15, "E6F4F5");
  s.addText(`${nc}/${nt}`, { x: 0.6, y: 1.2, w: 1.9, h: 1.15, fontFace: "Cambria", fontSize: 40, bold: true, color: TEAL, valign: "middle", margin: 0, isTextBox: true });
  s.addText(`cấu hình mâu thuẫn (${pct(nc, nt)})`, { x: 2.5, y: 1.2, w: 2.1, h: 1.15, fontSize: 15, color: INK, valign: "middle", margin: 0, isTextBox: true });
  s.addChart(pres.charts.BAR, [
    { name: "Dẫn đầu Drop@5", labels: ["Nearest-First", "GNNExplainer", "PGExplainer"], values: [fw.nearest_first || 0, fw.gnnexplainer || 0, fw.pgexplainer || 0] },
    { name: "Dẫn đầu AUROC", labels: ["Nearest-First", "GNNExplainer", "PGExplainer"], values: [aw.nearest_first || 0, aw.gnnexplainer || 0, aw.pgexplainer || 0] },
  ], {
    x: 0.5, y: 2.5, w: 4.2, h: 2.7, barDir: "col", barGrouping: "clustered", chartColors: [ORANGE, TEAL],
    showTitle: true, title: "Số cấu hình dẫn đầu (trên 25)", titleFontSize: 12, titleColor: INK, titleFontFace: "Calibri",
    showValue: true, dataLabelFontSize: 11, dataLabelColor: INK, dataLabelPosition: "outEnd", dataLabelFontFace: "Calibri",
    catAxisLabelFontSize: 11, catAxisLabelColor: INK, catAxisLabelFontFace: "Calibri", valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "Calibri",
    valAxisMaxVal: 25, valAxisMajorUnit: 5, valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: INK, legendFontFace: "Calibri",
  });
  const mk = (m, met) => { const v = N[`a30.${m}.${met}`]; return v[1] === v[2] ? `${f(v[0])} (hằng số)` : `${f(v[0])} (${f(v[1])}–${f(v[2])})`; };
  const rows = [["Phương pháp", "Drop@5, TB (min–max)", "AUROC, TB (min–max)"]];
  ["saliency", "gnnexplainer", "pgexplainer", "nearest_first"].forEach((m) => rows.push([{ text: METH[m], options: { align: "left", bold: true } }, mk(m, "fid_drop"), mk(m, "roc_auc")]));
  table(s, rows, { x: 4.95, y: 1.2, w: 4.55, colW: [1.35, 1.6, 1.6], rowH: [0.45, 0.45, 0.45, 0.45, 0.45], fontSize: 11 });
  s.addText([
    { text: "PGExplainer: AUROC cao nhất nhưng Drop@5 thấp nhất.", options: { bullet: true, breakLine: true } },
    { text: "Kết quả không đổi với 5 seed phá hòa.", options: { bullet: true, breakLine: true } },
    { text: `Cấp nút: ${N["a31.node_conflict"][0]}/${N["a31.node_conflict"][1]} cặp nút–GNN (${pct(...N["a31.node_conflict"])}).`, options: { bullet: true } },
  ], { x: 4.95, y: 3.65, w: 4.55, h: 1.5, fontSize: 14, color: INK, valign: "top", paraSpaceAfter: 5, margin: 0, isTextBox: true });
}

// =====================================================================
// 9. Quadrants + T1-T5
// =====================================================================
{
  const q = N["a31.q"], np = N["a31.pairs"];
  const t2c = N["t2.conflict"], t2o = N["t2.other"], t1 = N["t1.means"], t3 = N["t3.one_hop"], t5 = N["t5.pg.conflict"];
  const s = content(`Cơ chế T2 giải thích ${pct(...t2c)} nút mâu thuẫn; T1 (cạnh trực tiếp) thì không`,
    "results/analysis/a3_final/ (a3_1_*.csv, t2_t3_conflict_cohort.csv, a3_2_mechanisms.csv); ngưỡng 0,10 / 0,75",
    `Nhóm quét toàn bộ ${np} cặp nút–GNN có AUROC và chia bốn góc phần tư theo ngưỡng khóa trước Drop ≥ 0,10 và AUROC ≥ 0,75. PGExplainer chủ yếu nằm ở Q1 với ${q.pgexplainer.Q1} cặp, còn Saliency và GNNExplainer chủ yếu ở Q2, tức Fidelity cao nhưng AUROC thấp. Trong ${t2c[1]} cặp mâu thuẫn, T2 được ủng hộ: xóa toàn bộ cạnh motif cho Drop dưới 0,01 ở ${t2c[0]}/${t2c[1]} nút (${pct(...t2c)}), so với ${t2o[0]}/${t2o[1]} nút (${pct(...t2o)}) ở nhóm không mâu thuẫn. T1 không ủng hộ: tỷ lệ cạnh 1-hop trong top-5 của bên thắng Drop là ${f(t1[0])} và của bên thắng AUROC là ${f(t1[1])}. Còn ${t2c[1] - t2c[0]}/${t2c[1]} nút mâu thuẫn chưa có cơ chế nào trong T1 đến T5 giải thích; đây là phần nhóm nói thẳng là chưa giải thích được.`);
  const ms = ["pgexplainer", "nearest_first", "gnnexplainer", "saliency"];
  const labels = ms.map((m) => METH[m]);
  s.addChart(pres.charts.BAR, ["Q1", "Q2", "Q3", "Q4"].map((qq, i) => ({
    name: ["Q1 cả hai cao", "Q2 Fid cao, AUROC thấp", "Q3 Fid thấp, AUROC cao", "Q4 cả hai thấp"][i], labels, values: ms.map((m) => q[m][qq]),
  })), {
    x: 0.5, y: 1.15, w: 4.5, h: 4.05, barDir: "bar", barGrouping: "stacked", chartColors: [TEAL, ORANGE, PURPLE, GRAY],
    showTitle: true, title: `Số cặp nút–GNN theo góc phần tư (tổng ${np})`, titleFontSize: 12, titleColor: INK, titleFontFace: "Calibri",
    showValue: true, dataLabelPosition: "ctr", dataLabelFontSize: 10, dataLabelColor: WHITE, dataLabelFontFace: "Calibri",
    catAxisLabelFontSize: 11, catAxisLabelColor: INK, catAxisLabelFontFace: "Calibri", catAxisOrientation: "maxMin", valAxisHidden: true,
    valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "Calibri", valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: INK, legendFontFace: "Calibri",
  });
  const items = [
    ["T2", "ỦNG HỘ", TEAL, `Drop < 0,01 khi xóa motif: ${t2c[0]}/${t2c[1]} nút mâu thuẫn (${pct(...t2c)}) vs ${t2o[0]}/${t2o[1]} (${pct(...t2o)})`],
    ["T1", "KHÔNG", RED, `Tỷ lệ 1-hop trong top-5: ${f(t1[0])} (thắng Drop) vs ${f(t1[1])} (thắng AUROC)`],
    ["T3", "TÍNH CHẤT METRIC", PURPLE, `Tính lại bậc: Drop xóa 1-hop ${f(t3[0])} → ${f(t3[1])}, ở mọi nhóm nút`],
    ["T4–T5", "MỘT PHẦN", ORANGE, `Nhiễu từng nút (PG: ${pct(...t5)} lượt Drop < 0), không đổi người thắng`],
  ];
  items.forEach((it, i) => {
    const y = 1.2 + i * 0.78;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: y + 0.04, w: 0.75, h: 0.6, fill: { color: it[2] }, line: { color: it[2], width: 0 }, rectRadius: 0.06 });
    s.addText(it[0], { x: 5.2, y: y + 0.04, w: 0.75, h: 0.6, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: it[1] + "  ", options: { bold: true, color: it[2], fontSize: 11, breakLine: true } }, { text: it[3], options: { fontSize: 12 } }],
      { x: 6.05, y, w: 3.45, h: 0.72, color: INK, valign: "middle", margin: 0, isTextBox: true });
  });
  card(s, 5.2, 4.4, 4.3, 0.75, "FFF4E8");
  s.addText(`${t2c[1] - t2c[0]}/${t2c[1]} nút mâu thuẫn chưa có cơ chế nào giải thích`, { x: 5.3, y: 4.4, w: 4.1, h: 0.75, fontSize: 14, bold: true, color: ORANGE, valign: "middle", margin: 0, isTextBox: true });
}

// =====================================================================
// 10. Case study 665
// =====================================================================
{
  const c = (m) => N[`case.665.${m}`];
  const cv = N["curve665.saliency"];
  const s = content("Node 665: Drop@5 phụ thuộc vào việc top-5 có cô lập nút hay không",
    "docs/figures/final/a3_node_665.png; a3_final/case_metrics.csv, case_deletion_curves.csv (GNN 1000, seed 0)",
    `Node 665 thuộc góc Q1 và được chọn theo quy tắc định trước: gần trung vị của góc phần tư, không chọn nút đẹp nhất. Top-5 của Saliency, GNNExplainer và Nearest-First chứa cả bốn cạnh trực tiếp của nút, nên nút bị cô lập: xác suất lớp gốc từ ${f(cv[0])} tụt xuống rất thấp rồi tăng lại ${f(cv[4])} ở k = 4, nên Drop@5 chỉ ${f(c("saliency")[0])}. PGExplainer chừa lại một cạnh trực tiếp nên Drop@5 là ${f(c("pgexplainer")[0])}. Đường Fidelity theo k không đơn điệu, nên so Drop@5 giữa các phương pháp phụ thuộc việc top-5 có cô lập nút hay không.`);
  s.addImage({ path: path.join(FIG, "a3_node_665.png"), x: 0.5, y: 1.2, w: 6.5, h: 6.5 * 1305 / 2175, altText: "Node 665: bốn giải thích top-5 và đường xác suất khi xóa cạnh" });
  card(s, 7.15, 1.2, 2.35, 1.3);
  s.addText([{ text: "Drop@5", options: { bold: true, color: MUTED, fontSize: 12, breakLine: true } },
    { text: `${f(c("saliency")[0])}`, options: { bold: true, color: ORANGE, fontSize: 26, breakLine: false } }, { text: " Sal·GNNE·NF", options: { fontSize: 11, color: INK, breakLine: true } },
    { text: `${f(c("pgexplainer")[0])}`, options: { bold: true, color: TEAL, fontSize: 26 } }, { text: " PGExplainer", options: { fontSize: 11, color: INK } }],
    { x: 7.25, y: 1.2, w: 2.2, h: 1.3, valign: "middle", margin: 0, isTextBox: true });
  s.addText([
    { text: "Top-5 chứa cả 4 cạnh trực tiếp: nút bị cô lập.", options: { bullet: true, breakLine: true } },
    { text: `Xác suất tụt rồi tăng lại ${f(cv[4])} ở k = 4.`, options: { bullet: true, breakLine: true } },
    { text: "PGExplainer chừa 1 cạnh trực tiếp.", options: { bullet: true } },
  ], { x: 7.15, y: 2.7, w: 2.35, h: 2.5, fontSize: 14, color: INK, valign: "top", paraSpaceAfter: 6, margin: 0, isTextBox: true });
}

// =====================================================================
// 11. D1
// =====================================================================
{
  newSection("Đề xuất D1 và kết luận");
  const st = (ds, m, d) => N[`d1stab.${ds}.${m}.${d}`];
  const dm = (ds, m, met, v) => N[`d1.${ds}.${m}.${met}.${v}`][0];
  const maxAbs = Math.max(...["bashapes", "cora"].flatMap((ds) => ["gnnexplainer", "pgexplainer"].flatMap((m) => ["fid_drop", "fid_inv"].map((met) => Math.abs(dm(ds, m, met, "ensemble") - dm(ds, m, met, "single_seed_mean"))))));
  const sig = N["d1.sig"];
  const s = content("Seed-Ensemble làm giải thích ổn định hơn, nhưng chất lượng gần như không đổi",
    "results/analysis/d1/d1_stability.csv, d1_metrics.csv, d1_vs_single_seed.csv (k = 5, TB 5 GNN)",
    `D1 lấy trung bình thứ hạng cạnh qua 5 seed explainer rồi cắt top-k, không cần chạy thêm explainer. Về ổn định, κ@5 giữa hai bản gộp rời nhau (2 seed so với 3 seed) cao hơn κ@5 giữa hai lượt đơn ở cả 4 ô, ví dụ GNNExplainer BA-Shapes từ ${f(st("bashapes", "gnnexplainer", "1v1"))} lên ${f(st("bashapes", "gnnexplainer", "2v3"))}. Về chất lượng, AUROC của GNNExplainer là ${f(dm("bashapes", "gnnexplainer", "roc_auc", "single_seed_mean"))} lúc một seed và ${f(dm("bashapes", "gnnexplainer", "roc_auc", "ensemble"))} lúc gộp. Chỉ ${sig.bashapes[0]}/${sig.bashapes[1]} so sánh trên BA-Shapes và ${sig.cora[0]}/${sig.cora[1]} trên Cora khác biệt có ý nghĩa sau hiệu chỉnh Holm, và mọi khác biệt không quá ${f(maxAbs)}. Hai nhóm so sánh có kích cỡ khác nhau nên đây là bằng chứng gián tiếp; ablation S = 5 chưa chạy.`);
  const cats = ["BA · GNNExp", "BA · PGExp", "Cora · GNNExp", "Cora · PGExp"];
  const keys = [["bashapes", "gnnexplainer"], ["bashapes", "pgexplainer"], ["cora", "gnnexplainer"], ["cora", "pgexplainer"]];
  s.addChart(pres.charts.BAR, [
    { name: "Hai lượt đơn (1 seed)", labels: cats, values: keys.map(([d, m]) => +st(d, m, "1v1").toFixed(3)) },
    { name: "Hai bản gộp (2 vs 3 seed)", labels: cats, values: keys.map(([d, m]) => +st(d, m, "2v3").toFixed(3)) },
  ], {
    x: 0.5, y: 1.15, w: 4.7, h: 4.05, barDir: "col", barGrouping: "clustered", chartColors: [GRAY, TEAL],
    showTitle: true, title: "Ổn định: κ@5 giữa hai giải thích", titleFontSize: 12, titleColor: INK, titleFontFace: "Calibri",
    showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFontFace: "Calibri",
    catAxisLabelFontSize: 10, catAxisLabelColor: INK, catAxisLabelFontFace: "Calibri", valAxisLabelFontSize: 10, valAxisLabelColor: MUTED, valAxisLabelFontFace: "Calibri",
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: INK, legendFontFace: "Calibri",
  });
  const rows = [["BA-Shapes, k = 5", "Một seed", "Gộp D1"]];
  [["gnnexplainer", "roc_auc", "GNNExp · AUROC"], ["gnnexplainer", "fid_drop", "GNNExp · Drop"], ["pgexplainer", "roc_auc", "PGExp · AUROC"], ["pgexplainer", "fid_drop", "PGExp · Drop"]].forEach(([m, met, lab]) =>
    rows.push([{ text: lab, options: { align: "left", bold: true } }, f(dm("bashapes", m, met, "single_seed_mean")), f(dm("bashapes", m, met, "ensemble"))]));
  table(s, rows, { x: 5.4, y: 1.2, w: 4.1, colW: [1.9, 1.1, 1.1], rowH: 0.34, fontSize: 12 });
  card(s, 5.4, 3.1, 4.1, 2.05, "E6F4F5");
  s.addText([
    { text: "Chất lượng không đổi đáng kể", options: { bold: true, color: TEAL, breakLine: true } },
    { text: `Khác biệt có ý nghĩa: ${sig.bashapes[0]}/${sig.bashapes[1]} (BA-Shapes), ${sig.cora[0]}/${sig.cora[1]} (Cora)`, options: { bullet: true, breakLine: true } },
    { text: `Mọi chênh lệch ≤ ${f(maxAbs)}`, options: { bullet: true } },
  ], { x: 5.55, y: 3.15, w: 3.8, h: 1.95, fontSize: 14, color: INK, valign: "middle", paraSpaceAfter: 5, margin: 0, isTextBox: true });
}

// =====================================================================
// 12. Conclusion (dark)
// =====================================================================
{
  const sig = N["d1.sig"];
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
  s.addText("Kết luận: không chọn explainer theo một metric duy nhất, và đừng đọc bất đồng như nhiễu", { x: 0.5, y: 0.3, w: 9.0, h: 0.9, fontFace: "Cambria", fontSize: 24, bold: true, color: WHITE, valign: "middle", margin: 0, isTextBox: true });
  const dm = N["delta_min"];
  const left = [
    `RQ3: bất đồng thực sự ở 6/6 cặp (Δ nhỏ nhất ${f(Math.min(...dm))}).`,
    `RQ1–2: GNNExplainer tự lệch κ@5 ${f(N["a21.bashapes.gnnexplainer"].within)} (BA-Shapes); PGExplainer trên Cora chỉ ${f(N["a21.cora.pgexplainer"].gnn_same)} giữa hai GNN.`,
    `FB-3: ${N["a30.conflicts"][0]}/${N["a30.conflicts"][1]} cấu hình mâu thuẫn; T2 giải thích ${pct(...N["t2.conflict"])} nút.`,
    `D1: ổn định hơn nhưng chất lượng gần như không đổi.`,
  ];
  const right = [
    "Chỉ GCN 2 lớp; chỉ BA-Shapes và Cora (Tầng B không làm).",
    "5 GNN × 5 seed explainer; CI theo nút không bao phủ biến thiên giữa GNN.",
    `A3 chỉ trên BA-Shapes; ${N["t2.conflict"][1] - N["t2.conflict"][0]}/${N["t2.conflict"][1]} nút mâu thuẫn chưa giải thích được.`,
    "PGExplainer nhạy số luồng CPU; siêu tham số chưa tinh chỉnh.",
  ];
  card(s, 0.5, 1.35, 4.4, 3.75, "163A63");
  s.addText("Điều nhóm kết luận được", { x: 0.7, y: 1.4, w: 4.0, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: "7FD1D9", margin: 0, isTextBox: true });
  s.addText(left.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < left.length - 1 } })), { x: 0.7, y: 1.85, w: 4.0, h: 3.2, fontSize: 16, color: WHITE, valign: "top", paraSpaceAfter: 6, margin: 0, isTextBox: true });
  card(s, 5.1, 1.35, 4.4, 3.75, "163A63");
  s.addText("Hạn chế (FB-4)", { x: 5.3, y: 1.4, w: 4.0, h: 0.4, fontFace: "Cambria", fontSize: 16, bold: true, color: "F6B26B", margin: 0, isTextBox: true });
  s.addText(right.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < right.length - 1 } })), { x: 5.3, y: 1.85, w: 4.0, h: 3.2, fontSize: 16, color: WHITE, valign: "top", paraSpaceAfter: 6, margin: 0, isTextBox: true });
  s.addText("Nguồn: docs/BAO_CAO_CUOI_KY.md, Mục 9; results/analysis/", { x: 0.5, y: 5.3, w: 8.4, h: 0.25, fontSize: 10, color: "CADCFC", margin: 0, isTextBox: true });
  s.addNotes(`Tóm lại ba ý. Một, bất đồng giữa các explainer là thật ở cả 6 cặp, không phải nhiễu seed. Hai, mâu thuẫn Fidelity và AUROC xuất hiện ở ${N["a30.conflicts"][0]}/${N["a30.conflicts"][1]} cấu hình và cơ chế T2 giải thích được ${pct(...N["t2.conflict"])} nút mâu thuẫn, nên không nên chọn explainer theo một metric duy nhất. Ba, gộp seed (D1) giúp ổn định hơn nhưng không cải thiện chất lượng đáng kể: ${sig.bashapes[0]}/${sig.bashapes[1]} so sánh trên BA-Shapes và ${sig.cora[0]}/${sig.cora[1]} trên Cora có ý nghĩa. Hạn chế: chỉ GCN, chỉ hai dataset, Tầng B không làm theo FB-4, và PGExplainer nhạy số học nên kết luận về nó gắn với cấu hình 2 luồng CPU. Xin cảm ơn thầy và các bạn đã lắng nghe.`);
}

// =====================================================================
// Backup slides
// =====================================================================
newSection("Dự phòng");
// B1: formulas + kappa@k
{
  const s = content("B1. κ@k hiệu chỉnh theo mức ngẫu nhiên: Random–Random ≈ 0 ở mọi ngân sách k",
    "docs/BAO_CAO_CUOI_KY.md, Mục 3.7; results/eval/{bashapes,cora}_test/agreement_summary.csv (κ giữa hai phương pháp, TB theo nút)",
    "Dự phòng cho câu hỏi vì sao dùng κ thay vì Jaccard. Khi số cạnh ứng viên m nhỏ, hai tập top-k trùng nhau ngẫu nhiên khá nhiều nên Jaccard bị thổi phồng; κ trừ kỳ vọng ngẫu nhiên k bình phương chia m. Bảng cho thấy cặp Random–Random có κ gần 0 ở mọi k, trong khi các cặp phương pháp thật cũng thấp. Ở k = 10, chỉ những nút có m lớn hơn 10 mới có κ xác định (75 nút BA-Shapes), nên các số ở k = 10 không so thẳng với k nhỏ. Fidelity: Drop là p(G) trừ p(G bỏ S_k), Inv là p(G) trừ p(G chỉ giữ S_k).");
  card(s, 0.5, 1.15, 9.0, 1.2);
  s.addText([
    { text: "Drop = p(G) − p(G∖S_k)   cao = tốt            Inv = p(G) − p(G[S_k])   thấp (gần 0, âm) = tốt", options: { breakLine: true, bold: true, color: NAVY } },
    { text: "κ@k = (|A∩B| − k²/m) / (k − k²/m):  0 = ngẫu nhiên, 1 = trùng hoàn toàn, có thể âm; NaN khi k_eff = m", options: { color: INK } },
  ], { x: 0.65, y: 1.15, w: 8.7, h: 1.2, fontSize: 14, valign: "middle", paraSpaceAfter: 6, margin: 0, isTextBox: true });
  const ks = [1, 3, 5, 10];
  const rows = [["κ@k giữa hai phương pháp", ...ks.map((k) => `BA k=${k}`), ...ks.map((k) => `Cora k=${k}`)]];
  [["saliency-gnnexplainer", "Saliency – GNNExp"], ["saliency-pgexplainer", "Saliency – PGExp"], ["gnnexplainer-pgexplainer", "GNNExp – PGExp"], ["random", "Random – Random"]].forEach(([key, lab]) => {
    const get = (ds, k) => (key === "random" ? N[`b1.${ds}.random.${k}`] : N[`b1.${ds}.${key}.${k}`])[0];
    rows.push([{ text: lab, options: { align: "left", bold: true } }, ...["bashapes", "cora"].flatMap((ds) => ks.map((k) => f(get(ds, k))))]);
  });
  table(s, rows, { x: 0.5, y: 2.55, w: 9.0, colW: [2.2, ...Array(8).fill(0.85)], rowH: 0.42, fontSize: 12 });
  s.addText(`Ở k = 10 chỉ các nút có m > 10 cạnh ứng viên mới có κ (BA-Shapes: ${N["b1.bashapes.random.10"][1]} nút; Cora: ${N["b1.cora.random.10"][1]} nút).`,
    { x: 0.5, y: 4.85, w: 9.0, h: 0.35, fontSize: 12, color: MUTED, margin: 0, valign: "middle", isTextBox: true });
}

// B2: hash table
{
  const ck = N.ckpt;
  const s = content("B2. Mười checkpoint GCN và hai file dữ liệu đều được khóa bằng SHA-256",
    "results/checkpoints/checkpoint_table.csv",
    `Mười checkpoint GCN, năm cho mỗi dataset, seed từ 1000 đến 1004, đều nằm trong Git. Hash file .pt là mốc tái lập; huấn luyện lại trên máy khác không cho cùng hash. Dữ liệu BA-Shapes có hash bắt đầu bằng ${ck[0].data_sha256.slice(0, 12)} và Cora bắt đầu bằng ${ck[5].data_sha256.slice(0, 12)}. Độ chính xác test của BA-Shapes từ ${f(Math.min(...ck.filter((c) => c.dataset === "bashapes").map((c) => c.test_acc)))} đến ${f(Math.max(...ck.filter((c) => c.dataset === "bashapes").map((c) => c.test_acc)))}, của Cora từ ${f(Math.min(...ck.filter((c) => c.dataset === "cora").map((c) => c.test_acc)))} đến ${f(Math.max(...ck.filter((c) => c.dataset === "cora").map((c) => c.test_acc)))}.`);
  const rows = [["Dataset", "Seed GNN", "SHA-256 checkpoint (16 ký tự đầu)", "Test acc"]];
  ck.forEach((c) => rows.push([c.dataset === "bashapes" ? "BA-Shapes" : "Cora", String(c.gnn_seed), { text: c.checkpoint_sha256.slice(0, 16) + "…", options: { fontFace: "Courier New" } }, f(c.test_acc)]));
  table(s, rows, { x: 0.5, y: 1.15, w: 6.2, colW: [1.2, 1.0, 3.0, 1.0], rowH: 0.3, fontSize: 11 });
  card(s, 6.95, 1.15, 2.55, 3.3);
  s.addText([
    { text: "SHA-256 dữ liệu", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "BA-Shapes", options: { color: MUTED, fontSize: 12, breakLine: true } },
    { text: ck[0].data_sha256.slice(0, 12) + "…", options: { fontFace: "Courier New", fontSize: 12, breakLine: true } },
    { text: "Cora", options: { color: MUTED, fontSize: 12, breakLine: true } },
    { text: ck[5].data_sha256.slice(0, 12) + "…", options: { fontFace: "Courier New", fontSize: 12, breakLine: true } },
    { text: "Hash file .pt là mốc tái lập; train lại trên máy khác không cho cùng hash.", options: { fontSize: 12, color: INK } },
  ], { x: 7.05, y: 1.2, w: 2.35, h: 3.2, fontSize: 14, valign: "top", paraSpaceAfter: 4, margin: 0, isTextBox: true });
}

// B3: case 342
{
  const c = (m) => N[`case.342.${m}`];
  const s = content("B3. Node 342: PGExplainer chọn đúng motif nhưng Drop@5 vẫn âm",
    "docs/figures/final/a3_node_342.png; results/analysis/a3_final/case_metrics.csv (GNN seed 1000, explainer seed 0)",
    `Node 342 ở góc Q3: Fidelity thấp nhưng AUROC cao. PGExplainer có AUROC ${f(c("pgexplainer")[1])} vì chọn các cạnh motif, nhưng Drop@5 là ${f(c("pgexplainer")[0])}, âm, tức xóa các cạnh đó còn làm xác suất lớp dự đoán tăng. Ngược lại GNNExplainer có Drop@5 ${f(c("gnnexplainer")[0])} dù AUROC chỉ ${f(c("gnnexplainer")[1])}; đó là kết quả của một tổ hợp cạnh cụ thể chứ không phải bằng chứng nó tìm ra motif. Nút này minh họa cơ chế T2: mô hình không dựa vào motif cho nút lớp 2 này.`);
  s.addImage({ path: path.join(FIG, "a3_node_342.png"), x: 0.5, y: 1.2, w: 6.5, h: 6.5 * 1305 / 2175, altText: "Node 342: bốn giải thích top-5 và đường xác suất khi xóa cạnh" });
  const rows = [["Phương pháp", "Drop@5", "AUROC"]];
  ["saliency", "gnnexplainer", "pgexplainer", "nearest_first"].forEach((m) => rows.push([{ text: METH[m], options: { align: "left", bold: true } }, f(c(m)[0]), f(c(m)[1])]));
  table(s, rows, { x: 7.15, y: 1.2, w: 2.35, colW: [1.15, 0.6, 0.6], rowH: 0.34, fontSize: 10 });
  s.addText([
    { text: `PGExplainer: AUROC ${f(c("pgexplainer")[1])}, Drop@5 ${f(c("pgexplainer")[0])}.`, options: { bullet: true, breakLine: true } },
    { text: `GNNExplainer: Drop@5 ${f(c("gnnexplainer")[0])} dù AUROC ${f(c("gnnexplainer")[1])}.`, options: { bullet: true } },
  ], { x: 7.15, y: 3.1, w: 2.35, h: 2.1, fontSize: 14, color: INK, valign: "top", paraSpaceAfter: 6, margin: 0, isTextBox: true });
}

// B4: Q&A
{
  const pg = N.pgthread, t5a = N["t5.pg.all"];
  const mm = (m, k) => N[`main.bashapes.${m}.${k}`][0];
  const s = content("B4. Bốn câu hỏi phản biện dự kiến và câu trả lời ngắn",
    "docs/PHU_LUC_HOI_DAP_PHAN_BIEN.md; số liệu tính lại từ results/analysis/",
    "Bốn câu hỏi dự kiến. Một, bất đồng hay nhiễu seed: bất đồng thực sự ở cả 6 cặp. Hai, vì sao PGExplainer tốt motif mà Fidelity thấp: hai metric đo hai thứ khác nhau và tách nhau đúng ở các nút mà mô hình không dựa vào motif. Ba, Fidelity âm nghĩa là xóa top-k làm xác suất tăng; PGExplainer có tỷ lệ này cao nhất. Bốn, kết quả có tái lập không: có, nếu giữ nguyên hash và số luồng CPU.");
  const qa = [
    ["Bất đồng hay nhiễu seed?", `Bất đồng thực sự: 6/6 cặp có CI của Δ_A và Δ_B > 0, Δ nhỏ nhất ${f(Math.min(...N.delta_min))}.`],
    ["PG tốt motif mà Fidelity thấp?", `AUROC ${f(mm("pgexplainer", "roc_auc"))} nhưng Drop@5 ${f(mm("pgexplainer", "fid_drop"))}; hai metric tách nhau ở nút mô hình không dựa motif (T2).`],
    ["Fidelity âm nghĩa là gì?", `Xóa top-5 làm xác suất tăng: PGExplainer ${pct(...t5a)} lượt (${t5a[0].toLocaleString("en-US").replace(",", ".")}/${t5a[1].toLocaleString("en-US").replace(",", ".")}). Giữ nguyên giá trị âm.`],
    ["Có tái lập được không?", `Có, khi giữ hash và 2 luồng CPU; khác luồng thì κ@5 PG chỉ ${f(pg.kappa_mean)} giữa hai lượt.`],
  ];
  qa.forEach((q, i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.2 + Math.floor(i / 2) * 2.0;
    card(s, x, y, 4.4, 1.85);
    s.addText(q[0], { x: x + 0.15, y: y + 0.08, w: 4.1, h: 0.5, fontFace: "Cambria", fontSize: 15, bold: true, color: NAVY, valign: "middle", margin: 0, isTextBox: true });
    s.addText(q[1], { x: x + 0.15, y: y + 0.62, w: 4.1, h: 1.15, fontSize: 14, color: INK, valign: "top", margin: 0, isTextBox: true });
  });
}

(async () => {
  const out = path.join(__dirname, "SNA_G09_C10_Cuoi_Ky.pptx");
  await pres.writeFile({ fileName: out });
  const skill = process.env.PPTX_SKILL_DIR;
  if (skill) { const { applyTheme } = require(path.join(skill, "scripts", "apply_theme.js")); await applyTheme(out, THEME); }
  console.log("written", out);
})();
