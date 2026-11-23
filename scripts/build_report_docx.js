#!/usr/bin/env node
/*
 * Build docs/BAO_CAO_CUOI_KY.docx from docs/BAO_CAO_CUOI_KY.md.
 *
 * Usage (from the repo root):
 *   npm install docx            # once, in any folder; then set NODE_PATH to its node_modules
 *   NODE_PATH=<path>/node_modules node scripts/build_report_docx.js
 *
 * Supports the Markdown subset used by the report: headings (##, ###), paragraphs,
 * bullet/numbered lists, pipe tables, fenced code, images (adjacent image lines form
 * one row), inline **bold**, *italic*, `code` and $LaTeX$ math (converted to Unicode).
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Footer,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, LevelFormat,
  PageNumber, PageBreak, VerticalAlign,
} = D;

const ROOT = path.resolve(__dirname, "..");
const MD = path.join(ROOT, "docs", "BAO_CAO_CUOI_KY.md");
const OUT = path.join(ROOT, "docs", "BAO_CAO_CUOI_KY.docx");
const FONT = "Times New Roman";
const BODY = 22; // half-points = 11pt
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1247; // A4, 2.2 cm
const CONTENT_W = PAGE_W - 2 * MARGIN;

// ---------- LaTeX -> runs ----------
const SYM = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", theta: "θ", kappa: "κ", lambda: "λ", sigma: "σ",
  tau: "τ", Delta: "Δ", sum: "Σ", partial: "∂", cdot: "·", le: "≤", leq: "≤", ge: "≥", geq: "≥",
  ne: "≠", sim: "∼", approx: "≈", to: "→", in: "∈", cap: "∩", cup: "∪", setminus: "∖", mid: "|", times: "×",
  ",": " ", ";": " ", "!": "", " ": " ", quad: "  ", qquad: "    ", "|": "‖", "{": "{", "}": "}",
  left: "", right: "", Big: "", big: "", rm: "", ldots: "…", infty: "∞",
};
const UPRIGHT_FN = new Set(["arg", "max", "min", "log", "exp"]);
const COMB = { hat: "̂", tilde: "̃", bar: "̄" };
const BLACKBOARD = { R: "ℝ" }, CAL = { L: "ℒ", N: "𝒩" };

function texRuns(src) {
  let i = 0, upAll = false;
  const out = [];
  function readGroup() {
    // returns substring of the next argument (group or single token)
    while (src[i] === " ") i++;
    if (src[i] === "{") {
      let depth = 0, j = i;
      for (; j < src.length; j++) {
        if (src[j] === "{") depth++;
        else if (src[j] === "}") { depth--; if (depth === 0) break; }
      }
      const s = src.slice(i + 1, j);
      i = j + 1;
      return s;
    }
    if (src[i] === "\\") {
      let j = i + 1;
      while (j < src.length && /[A-Za-z]/.test(src[j])) j++;
      if (j === i + 1) j++;
      const s = src.slice(i, j);
      i = j;
      return s;
    }
    return src[i++];
  }
  function sub(s, extra) {
    return texRuns(s).map((r) => ({ ...r, ...extra }));
  }
  while (i < src.length) {
    const c = src[i];
    if (c === "\\") {
      let j = i + 1;
      while (j < src.length && /[A-Za-z]/.test(src[j])) j++;
      let name = src.slice(i + 1, j);
      if (name === "") { name = src[i + 1]; j = i + 2; }
      i = j;
      if (name === "rm") { upAll = true; continue; }
      if (name === "mathrm" || name === "text") { out.push(...sub(readGroup(), { up: true })); continue; }
      if (name === "mathbf") { out.push(...sub(readGroup(), { bold: true })); continue; }
      if (name === "mathcal") { const g = readGroup(); out.push({ t: CAL[g] || g, up: true }); continue; }
      if (name === "mathbb") { const g = readGroup(); out.push({ t: BLACKBOARD[g] || g, up: true }); continue; }
      if (name === "frac") {
        const a = readGroup(), b = readGroup();
        const wrap = (s) => (/^[A-Za-z0-9]+$/.test(s) ? [{ t: s }] : [{ t: "(", up: true }, ...sub(s, {}), { t: ")", up: true }]);
        const ra = a.length > 1 && !/^[A-Za-z0-9]+$/.test(a) ? wrap(a) : sub(a, {});
        out.push(...ra, { t: "/", up: true }, ...(b.length > 1 && !/^[A-Za-z0-9]+$/.test(b) ? wrap(b) : sub(b, {})));
        continue;
      }
      if (COMB[name]) {
        const g = readGroup();
        const base = sub(g, {});
        if (base.length && base[0].t) base[0] = { ...base[0], t: base[0].t[0] + COMB[name] + base[0].t.slice(1) };
        out.push(...base);
        continue;
      }
      if (UPRIGHT_FN.has(name)) { out.push({ t: name, up: true }); continue; }
      if (name in SYM) { out.push({ t: SYM[name], up: true }); continue; }
      out.push({ t: name, up: true });
    } else if (c === "_" || c === "^") {
      i++;
      const g = readGroup();
      const flag = c === "_" ? { subs: true } : { sups: true };
      out.push(...sub(g, flag));
    } else if (c === "{" || c === "}") {
      i++;
    } else {
      out.push({ t: c, up: upAll || !/[A-Za-z]/.test(c) });
      i++;
    }
  }
  return out;
}

function mathToTextRuns(src, base) {
  return texRuns(src).map((r) => new TextRun({
    text: r.t, font: FONT, size: base.size || BODY,
    italics: !r.up || false, bold: r.bold || base.bold,
    subScript: !!r.subs, superScript: !!r.sups,
  }));
}

// ---------- inline markdown ----------
function inlineRuns(text, base = {}) {
  const size = base.size || BODY;
  const runs = [];
  const re = /(\$[^$]+\$)|(\*\*[^*]+\*\*)|(`[^`]+`)|(\*[^*\s][^*]*?\*)/g;
  let last = 0, m;
  const push = (t, o = {}) => { if (t) runs.push(new TextRun({ text: t, font: FONT, size, bold: base.bold, italics: base.italics, ...o })); };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok[0] === "$") runs.push(...mathToTextRuns(tok.slice(1, -1), { size, bold: base.bold }));
    else if (tok.startsWith("**")) runs.push(...inlineRuns(tok.slice(2, -2), { ...base, bold: true }));
    else if (tok[0] === "`") push(tok.slice(1, -1), { font: "Consolas", size: size - 2 });
    else runs.push(...inlineRuns(tok.slice(1, -1), { ...base, italics: true }));
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return runs;
}

function plainLen(s) {
  return s.replace(/\$([^$]+)\$/g, (_, x) => x.replace(/\\[a-zA-Z]+|[{}_^\\]/g, "")).replace(/[*`]/g, "").length;
}

// ---------- parsing ----------
const lines = fs.readFileSync(MD, "utf8").replace(/\r\n/g, "\n").split("\n");
const blocks = [];
let i0 = 0;
while (i0 < lines.length) {
  const L = lines[i0];
  if (/^```/.test(L)) {
    const code = [];
    i0++;
    while (i0 < lines.length && !/^```/.test(lines[i0])) code.push(lines[i0++]);
    i0++;
    blocks.push({ type: "code", lines: code });
  } else if (/^# /.test(L)) { blocks.push({ type: "title", text: L.slice(2) }); i0++; }
  else if (/^## /.test(L)) { blocks.push({ type: "h1", text: L.slice(3) }); i0++; }
  else if (/^### /.test(L)) { blocks.push({ type: "h2", text: L.slice(4) }); i0++; }
  else if (/^\|/.test(L)) {
    const rows = [];
    while (i0 < lines.length && /^\|/.test(lines[i0])) rows.push(lines[i0++]);
    blocks.push({ type: "table", rows });
  } else if (/^!\[/.test(L)) {
    const imgs = [];
    while (i0 < lines.length && /^!\[/.test(lines[i0])) {
      const mm = lines[i0++].match(/^!\[(.*?)\]\((.*?)\)/);
      imgs.push({ alt: mm[1], src: mm[2] });
    }
    blocks.push({ type: "images", imgs });
  } else if (/^\$\$/.test(L)) {
    let s = L.replace(/^\$\$/, "");
    if (!/\$\$$/.test(s)) { i0++; while (i0 < lines.length && !/\$\$$/.test(lines[i0])) s += " " + lines[i0++]; s += " " + lines[i0].replace(/\$\$$/, ""); }
    else s = s.replace(/\$\$$/, "");
    i0++;
    blocks.push({ type: "math", text: s.trim() });
  } else if (/^- /.test(L)) {
    const items = [];
    while (i0 < lines.length && /^- /.test(lines[i0])) items.push(lines[i0++].slice(2));
    blocks.push({ type: "bullets", items });
  } else if (/^\d+\. /.test(L)) {
    const items = [];
    while (i0 < lines.length && /^\d+\. /.test(lines[i0])) items.push(lines[i0++].replace(/^\d+\. /, ""));
    blocks.push({ type: "numbers", items });
  } else if (L.trim() === "") { i0++; }
  else {
    let s = L;
    i0++;
    while (i0 < lines.length && lines[i0].trim() !== "" && !/^(#|\||!\[|```|- |\d+\. |\$\$)/.test(lines[i0])) s += " " + lines[i0++];
    blocks.push({ type: "p", text: s });
  }
}

// ---------- builders ----------
const thin = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: thin, bottom: thin, left: thin, right: thin };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: none, bottom: none, left: none, right: none };
const numberingConfigs = [{
  reference: "bul",
  levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }],
}];
let numCount = 0;

function para(text, opts = {}) {
  return new Paragraph({
    children: inlineRuns(text, { size: opts.size, italics: opts.italics }),
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 90, before: opts.before ?? 0, line: 264 },
    keepNext: opts.keepNext, keepLines: opts.keepLines,
    indent: opts.indent,
  });
}

function buildTable(rows) {
  const cells = rows.map((r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  const sep = cells[1];
  const aligns = sep.map((s) => (/^:-+:$/.test(s) ? AlignmentType.CENTER : /-:$/.test(s) ? AlignmentType.RIGHT : AlignmentType.LEFT));
  const data = [cells[0], ...cells.slice(2)];
  const n = cells[0].length;
  const wts = [];
  for (let j = 0; j < n; j++) {
    const lens = data.map((r) => plainLen(r[j] || ""));
    const mx = Math.max(...lens), avg = lens.reduce((a, b) => a + b, 0) / lens.length;
    wts.push(Math.min(Math.max((mx + avg) / 2, 5), 48));
  }
  const tot = wts.reduce((a, b) => a + b, 0);
  let widths = wts.map((w) => Math.max(Math.round((w / tot) * CONTENT_W), 520));
  const diff = CONTENT_W - widths.reduce((a, b) => a + b, 0);
  widths[widths.indexOf(Math.max(...widths))] += diff;
  const fs_ = n >= 7 ? 16 : 18;
  const trs = data.map((r, ri) => new TableRow({
    tableHeader: ri === 0, cantSplit: true,
    children: r.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA }, borders,
      verticalAlign: VerticalAlign.CENTER,
      shading: ri === 0 ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 30, bottom: 30, left: 70, right: 70 },
      children: [new Paragraph({
        alignment: ri === 0 ? AlignmentType.CENTER : aligns[j] || AlignmentType.LEFT,
        spacing: { after: 0, line: 240 },
        children: inlineRuns(c, { size: fs_, bold: ri === 0 }),
      })],
    })),
  }));
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: trs }),
    new Paragraph({ spacing: { after: 60 }, children: [] })];
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function buildImages(imgs) {
  const mdDir = path.dirname(MD);
  const cols = imgs.length === 1 ? 1 : 2;
  const colW = Math.floor(CONTENT_W / cols);
  const pxW = Math.floor(((colW - 120) / 20) * (96 / 72));
  const rowsOut = [];
  for (let k = 0; k < imgs.length; k += cols) {
    const slice = imgs.slice(k, k + cols);
    while (slice.length < cols) slice.push(null);
    rowsOut.push(new TableRow({
      cantSplit: true,
      children: slice.map((im) => {
        if (!im) return new TableCell({ width: { size: colW, type: WidthType.DXA }, borders: noBorders, children: [new Paragraph({ children: [] })] });
        const file = path.resolve(mdDir, im.src);
        const { w, h } = pngSize(file);
        const width = cols === 1 ? Math.min(pxW, 520) : pxW;
        return new TableCell({
          width: { size: colW, type: WidthType.DXA }, borders: noBorders,
          children: [new Paragraph({
            alignment: AlignmentType.CENTER, spacing: { after: 0 }, keepNext: true,
            children: [new ImageRun({
              type: "png", data: fs.readFileSync(file),
              transformation: { width, height: Math.round((width * h) / w) },
              altText: { title: im.alt, description: im.alt, name: path.basename(file) },
            })],
          })],
        });
      }),
    }));
  }
  return [new Table({ width: { size: colW * cols, type: WidthType.DXA }, columnWidths: Array(cols).fill(colW), rows: rowsOut, borders: { ...noBorders, insideHorizontal: none, insideVertical: none } })];
}

// ---------- cover ----------
function center(text, o = {}) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: o.after ?? 120, before: o.before ?? 0 },
    children: [new TextRun({ text, font: FONT, size: o.size || 26, bold: o.bold, italics: o.italics })],
  });
}
const students = [
  ["Nguyễn Trường Vũ", "23521812"], ["Nguyễn Lê Bảo Đăng", "23520230"], ["Nguyễn Hải Thiện", "23521481"],
  ["Nguyễn Trọng Khôi", "23520783"], ["Vòng Việt Đức", "23520324"], ["Nguyễn Hoàng Kha", "23520667"],
];
const titleBlock = blocks.find((b) => b.type === "title");
const cover = [
  center("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN – ĐHQG-HCM", { bold: true, size: 26, before: 600 }),
  center("Môn học: Phân tích mạng xã hội (SNA)", { size: 24, after: 1400 }),
  center("BÁO CÁO CUỐI KỲ", { bold: true, size: 30, after: 300 }),
  center(titleBlock.text, { bold: true, size: 36, after: 300 }),
  center("Đề tài C.10 – Making Graph Machine Learning Models Explainable", { italics: true, size: 26, after: 1200 }),
  center("Nhóm 09", { bold: true, size: 28, after: 120 }),
  center("Giảng viên hướng dẫn: TS. Trần Hưng Nghiệp", { size: 26, after: 360 }),
  new Table({
    width: { size: 6200, type: WidthType.DXA }, columnWidths: [4400, 1800], alignment: AlignmentType.CENTER,
    rows: [["Sinh viên", "MSSV"], ...students].map((r, ri) => new TableRow({
      children: r.map((c, j) => new TableCell({
        width: { size: j ? 1800 : 4400, type: WidthType.DXA }, borders,
        shading: ri === 0 ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 50, bottom: 50, left: 100, right: 100 },
        children: [new Paragraph({ alignment: j ? AlignmentType.CENTER : AlignmentType.LEFT, children: [new TextRun({ text: c, font: FONT, size: 24, bold: ri === 0 })] })],
      })),
    })),
  }),
  center("Ngày nộp: 26/11/2026", { size: 24, before: 600 }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------- body ----------
const body = [];
const tocEntries = blocks.filter((b) => b.type === "h1").map((b) => b.text);
body.push(new Paragraph({
  heading: HeadingLevel.HEADING_1, pageBreakBefore: false,
  children: [new TextRun({ text: "Mục lục", font: FONT, size: 28, bold: true })],
}));
tocEntries.forEach((t) => body.push(new Paragraph({ spacing: { after: 40 }, indent: { left: 300 }, children: [new TextRun({ text: t, font: FONT, size: BODY })] })));
body.push(new Paragraph({ children: [new PageBreak()] }));

for (let b = 0; b < blocks.length; b++) {
  const bl = blocks[b];
  if (bl.type === "title") continue;
  if (bl.type === "h1") {
    const appendix = /^Phụ lục/.test(bl.text);
    body.push(new Paragraph({
      heading: HeadingLevel.HEADING_1, pageBreakBefore: appendix && /Phụ lục A/.test(bl.text),
      children: [new TextRun({ text: bl.text })],
    }));
  } else if (bl.type === "h2") {
    body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: bl.text })] }));
  } else if (bl.type === "p") {
    const t = bl.text;
    const small = /^\*[^*].*\*$/.test(t) && !/^\*\*/.test(t);
    const isCap = /^\*\*(Bảng|Hình)/.test(t);
    body.push(para(small ? t.slice(1, -1) : t, { size: small ? 18 : undefined, italics: small, keepNext: isCap, align: small ? AlignmentType.LEFT : AlignmentType.JUSTIFIED }));
  } else if (bl.type === "bullets") {
    bl.items.forEach((it) => body.push(new Paragraph({
      numbering: { reference: "bul", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 264 },
      children: inlineRuns(it),
    })));
  } else if (bl.type === "numbers") {
    const ref = "num" + numCount++;
    numberingConfigs.push({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 320 } } } }] });
    bl.items.forEach((it) => body.push(new Paragraph({
      numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 264 },
      children: inlineRuns(it),
    })));
  } else if (bl.type === "table") {
    body.push(...buildTable(bl.rows));
  } else if (bl.type === "images") {
    body.push(...buildImages(bl.imgs));
  } else if (bl.type === "math") {
    body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 100 }, children: mathToTextRuns(bl.text, {}) }));
  } else if (bl.type === "code") {
    bl.lines.forEach((l, k) => body.push(new Paragraph({
      spacing: { after: 0, line: 220 }, shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
      keepLines: true, keepNext: k < bl.lines.length - 1,
      children: [new TextRun({ text: l === "" ? " " : l, font: "Consolas", size: 16 })],
    })));
    body.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
  }
}

const doc = new Document({
  creator: "Nhóm 09", title: titleBlock.text,
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 260, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: numberingConfigs },
  sections: [{
    properties: { titlePage: true, page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: {
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }),
    },
    children: [...cover, ...body],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, buf.length, "bytes");
});
