/**
 * generate_eval_report.js
 * ========================
 * Generates a professional .docx evaluation report from eval_metrics.py output.
 *
 * Usage (called automatically by eval_metrics.py — do not run manually):
 *   node generate_eval_report.js <input_json_path> <output_docx_path>
 *
 * Install once:
 *   npm install -g docx
 */

"use strict";

const fs   = require("fs");
const path = require("path");

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumberElement, LevelFormat, TabStopType,
} = require("docx");

// ── Args ──────────────────────────────────────────────────────────────────
const [,, jsonPath, docxPath] = process.argv;
if (!jsonPath || !docxPath) {
  console.error("Usage: node generate_eval_report.js <input.json> <output.docx>");
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(jsonPath, "utf8"));

// ── Design tokens ─────────────────────────────────────────────────────────
const BLUE_DARK  = "1B3A6B";
const BLUE_MED   = "2E5FA3";
const GREY_LIGHT = "F4F6FB";
const WHITE      = "FFFFFF";
const BLACK      = "1A1A1A";
const GREEN      = "1A6E3C";
const YELLOW     = "7A5F00";
const RED        = "9B1C1C";
const GREY_TEXT  = "6B7280";

const STATUS_COLOR = { ok: GREEN, warn: YELLOW, fail: RED, skip: GREY_TEXT };
const STATUS_ICON  = { ok: "✓", warn: "!", fail: "✗", skip: "–" };

// ── Section label map — trimmed to match new eval_metrics.py (6 layers) ──
const SECTION_LABELS = {
  planner:        "PHASE 2     Planner Agent",
  retrieval:      "PHASE 3/4   Hybrid Retrieval",
  rag:            "PHASE 6     RAG Pipeline",
  summarizer:     "PHASE 8     Summarization Agent",
  report_quality: "PHASE 11    Report Quality",
  performance:    "SYSTEM      Performance & Cost",
};

// ── Helpers ───────────────────────────────────────────────────────────────
const border1 = { style: BorderStyle.SINGLE, size: 1, color: "C5D3E8" };
const borders  = { top: border1, bottom: border1, left: border1, right: border1 };
const noBorder = {
  top:    { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left:   { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right:  { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
};

function cell(children, opts = {}) {
  return new TableCell({
    borders:       opts.noBorder ? noBorder : borders,
    width:         { size: opts.width || 2340, type: WidthType.DXA },
    shading:       { fill: opts.fill || WHITE, type: ShadingType.CLEAR },
    margins:       { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children,
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    spacing:   { before: opts.spaceBefore || 0, after: opts.spaceAfter || 0 },
    children: [
      new TextRun({
        text:    String(text),
        bold:    opts.bold    || false,
        italics: opts.italic  || false,
        color:   opts.color   || BLACK,
        size:    opts.size    || 20,
        font:    "Arial",
      }),
    ],
  });
}

function spacer(lines = 1) {
  return new Paragraph({
    spacing: { before: 0, after: lines * 60 },
    children: [new TextRun({ text: "", size: 18 })],
  });
}

function sectionHeading(label) {
  return new Paragraph({
    spacing: { before: 280, after: 100 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE_MED, space: 4 },
    },
    children: [
      new TextRun({ text: label, bold: true, color: BLUE_MED, size: 22, font: "Arial" }),
    ],
  });
}

// ── Score badge ───────────────────────────────────────────────────────────
function scoreColor(score) {
  if (score >= 75) return GREEN;
  if (score >= 50) return YELLOW;
  return RED;
}

function scoreBadge(score) {
  const color = scoreColor(score);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 40, after: 80 },
    children: [
      new TextRun({ text: `${score}`, bold: true, color, size: 72, font: "Arial" }),
      new TextRun({ text: " / 100", bold: false, color: GREY_TEXT, size: 36, font: "Arial" }),
    ],
  });
}

// ── Summary counts table ──────────────────────────────────────────────────
function summaryCountsTable(allMetrics) {
  const ok   = allMetrics.filter(m => m.status === "ok").length;
  const warn = allMetrics.filter(m => m.status === "warn").length;
  const fail = allMetrics.filter(m => m.status === "fail").length;
  const skip = allMetrics.filter(m => m.status === "skip").length;

  const mkCell = (label, count, color, fill) =>
    cell([
      para(label,         { bold: true, color, size: 18, align: AlignmentType.CENTER }),
      para(String(count), { bold: true, color, size: 36, align: AlignmentType.CENTER }),
    ], { width: 2340, fill });

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2340, 2340, 2340, 2340],
    rows: [
      new TableRow({ children: [
        mkCell("PASSED",  ok,   GREEN,     "E6F4EC"),
        mkCell("WARN",    warn, YELLOW,    "FFFBEB"),
        mkCell("FAILED",  fail, RED,       "FEE2E2"),
        mkCell("SKIPPED", skip, GREY_TEXT, "F3F4F6"),
      ]}),
    ],
  });
}

// ── Per-layer stat bar ────────────────────────────────────────────────────
function layerStatBar(allMetrics) {
  // One row: layer name | pass | warn | fail | skip
  const rows = [];

  // Header
  rows.push(new TableRow({
    tableHeader: true,
    children: [
      cell([para("Layer",   { bold: true, color: WHITE, size: 17 })], { width: 3200, fill: BLUE_DARK }),
      cell([para("Passed",  { bold: true, color: WHITE, size: 17, align: AlignmentType.CENTER })], { width: 1540, fill: BLUE_DARK }),
      cell([para("Warn",    { bold: true, color: WHITE, size: 17, align: AlignmentType.CENTER })], { width: 1540, fill: BLUE_DARK }),
      cell([para("Failed",  { bold: true, color: WHITE, size: 17, align: AlignmentType.CENTER })], { width: 1540, fill: BLUE_DARK }),
      cell([para("Skipped", { bold: true, color: WHITE, size: 17, align: AlignmentType.CENTER })], { width: 1540, fill: BLUE_DARK }),
    ],
  }));

  Object.entries(SECTION_LABELS).forEach(([key, label], idx) => {
    const metrics = (allMetrics[key] || []);
    if (!metrics.length) return;
    const ok   = metrics.filter(m => m.status === "ok").length;
    const warn = metrics.filter(m => m.status === "warn").length;
    const fail = metrics.filter(m => m.status === "fail").length;
    const skip = metrics.filter(m => m.status === "skip").length;
    const fill = idx % 2 === 0 ? WHITE : GREY_LIGHT;

    rows.push(new TableRow({ children: [
      cell([para(label, { bold: true, size: 16 })],                                         { width: 3200, fill }),
      cell([para(String(ok),   { color: GREEN,     size: 17, align: AlignmentType.CENTER })], { width: 1540, fill }),
      cell([para(String(warn), { color: YELLOW,    size: 17, align: AlignmentType.CENTER })], { width: 1540, fill }),
      cell([para(String(fail), { color: RED,       size: 17, align: AlignmentType.CENTER })], { width: 1540, fill }),
      cell([para(String(skip), { color: GREY_TEXT, size: 17, align: AlignmentType.CENTER })], { width: 1540, fill }),
    ]}));
  });

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3200, 1540, 1540, 1540, 1540],
    rows,
  });
}

// ── Metrics table for one section ─────────────────────────────────────────
function metricsTable(metrics) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: [
      cell([para("Metric", { bold: true, color: WHITE, size: 18 })], { width: 2800, fill: BLUE_DARK }),
      cell([para("Value",  { bold: true, color: WHITE, size: 18 })], { width: 4000, fill: BLUE_DARK }),
      cell([para("Status", { bold: true, color: WHITE, size: 18 })], { width: 700,  fill: BLUE_DARK }),
      cell([para("Note",   { bold: true, color: WHITE, size: 18 })], { width: 1860, fill: BLUE_DARK }),
    ],
  });

  const dataRows = metrics.map((m, idx) => {
    const fill   = idx % 2 === 0 ? WHITE : GREY_LIGHT;
    const color  = STATUS_COLOR[m.status] || BLACK;
    const icon   = STATUS_ICON[m.status]  || "?";
    const valStr = m.value !== null && m.value !== undefined ? String(m.value) : "—";
    const fullVal = m.unit ? `${valStr}  ${m.unit}` : valStr;

    return new TableRow({ children: [
      cell([para(m.name,   { bold: true, color: BLACK,     size: 17 })],                                    { width: 2800, fill }),
      cell([para(fullVal,  { color: BLACK,     size: 17 })],                                                { width: 4000, fill }),
      cell([para(icon,     { bold: true, color, size: 18, align: AlignmentType.CENTER })],                  { width: 700,  fill }),
      cell([para(m.note || "", { color: GREY_TEXT, size: 16, italic: true })],                              { width: 1860, fill }),
    ]});
  });

  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2800, 4000, 700, 1860],
    rows: [headerRow, ...dataRows],
  });
}

// ── Cover page ────────────────────────────────────────────────────────────
function coverPage(data) {
  const score = data.overall_score;
  const items = [];

  // Top accent bar
  items.push(new Paragraph({
    spacing: { before: 0, after: 200 },
    border: { top: { style: BorderStyle.SINGLE, size: 24, color: BLUE_MED } },
    children: [],
  }));

  // Title block
  items.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 600, after: 80 },
    children: [
      new TextRun({ text: "Agentic Research Assistant", bold: true, color: BLUE_DARK, size: 52, font: "Arial" }),
    ],
  }));
  items.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 400 },
    children: [
      new TextRun({ text: "Pipeline Evaluation Report", bold: false, color: BLUE_MED, size: 36, font: "Arial" }),
    ],
  }));

  // Score
  items.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 20 },
    children: [
      new TextRun({ text: "Overall Health Score", color: GREY_TEXT, size: 22, font: "Arial" }),
    ],
  }));
  items.push(scoreBadge(score));
  items.push(spacer(1));

  // Meta info table
  const metaTable = new Table({
    width: { size: 6000, type: WidthType.DXA },
    columnWidths: [2200, 3800],
    rows: [
      new TableRow({ children: [
        cell([para("Session ID", { bold: true, color: WHITE, size: 18 })], { width: 2200, fill: BLUE_DARK }),
        cell([para(data.session_id,  { size: 18 })],                        { width: 3800 }),
      ]}),
      new TableRow({ children: [
        cell([para("Query",     { bold: true, color: WHITE, size: 18 })], { width: 2200, fill: BLUE_DARK }),
        cell([para(data.query,       { size: 18 })],                        { width: 3800 }),
      ]}),
      new TableRow({ children: [
        cell([para("Generated", { bold: true, color: WHITE, size: 18 })], { width: 2200, fill: BLUE_DARK }),
        cell([para(data.timestamp,   { size: 18 })],                        { width: 3800 }),
      ]}),
    ],
  });
  items.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 20 }, children: [] }));
  items.push(metaTable);
  items.push(spacer(2));

  // Summary counts
  const allMetrics = Object.values(data.sections || {}).flat();
  items.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new TextRun({ text: "Metric Summary", bold: true, color: BLUE_DARK, size: 22, font: "Arial" }),
    ],
  }));
  items.push(summaryCountsTable(allMetrics));
  items.push(spacer(1.5));

  // Layer-by-layer stat bar
  items.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 60 },
    children: [
      new TextRun({ text: "Results by Layer", bold: true, color: BLUE_DARK, size: 22, font: "Arial" }),
    ],
  }));
  items.push(layerStatBar(data.sections || {}));

  // Bottom bar
  items.push(new Paragraph({
    spacing: { before: 400, after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 24, color: BLUE_MED } },
    children: [],
  }));

  return items;
}

// ── Build document ────────────────────────────────────────────────────────
function buildDocument(data) {
  const children = [];

  children.push(...coverPage(data));

  // Page break
  children.push(new Paragraph({
    children: [new TextRun({ text: "", break: 1 })],
    pageBreakBefore: true,
  }));

  // Detailed results heading
  children.push(new Paragraph({
    spacing: { before: 200, after: 160 },
    children: [
      new TextRun({ text: "Detailed Evaluation Results", bold: true, color: BLUE_DARK, size: 32, font: "Arial" }),
    ],
  }));

  // One section per layer — only renders layers that exist in the data
  for (const [key, label] of Object.entries(SECTION_LABELS)) {
    const metrics = (data.sections || {})[key];
    if (!metrics || metrics.length === 0) continue;

    children.push(sectionHeading(label));
    children.push(spacer(0.3));
    children.push(metricsTable(metrics));
    children.push(spacer(1));
  }

  // Key findings page
  children.push(new Paragraph({
    pageBreakBefore: true,
    spacing: { before: 200, after: 160 },
    children: [
      new TextRun({ text: "Key Findings & Action Items", bold: true, color: BLUE_DARK, size: 32, font: "Arial" }),
    ],
  }));
  children.push(...keyFindingsSection(data));

  return children;
}

// ── Key findings ──────────────────────────────────────────────────────────
function keyFindingsSection(data) {
  const items      = [];
  const allMetrics = Object.values(data.sections || {}).flat();
  const fails      = allMetrics.filter(m => m.status === "fail");
  const warns      = allMetrics.filter(m => m.status === "warn");

  if (fails.length > 0) {
    items.push(sectionHeading("Critical Issues (must fix)"));
    items.push(spacer(0.3));
    items.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3200, 6160],
      rows: [
        new TableRow({ children: [
          cell([para("Metric",          { bold: true, color: WHITE, size: 18 })], { width: 3200, fill: BLUE_DARK }),
          cell([para("Action Required", { bold: true, color: WHITE, size: 18 })], { width: 6160, fill: BLUE_DARK }),
        ]}),
        ...fails.map((m, i) => new TableRow({ children: [
          cell([para(m.name, { bold: true, size: 17 })],
            { width: 3200, fill: i % 2 === 0 ? "FEE2E2" : "FEF2F2" }),
          cell([para(m.note || `${m.name} scored below acceptable threshold (value: ${m.value})`, { size: 17 })],
            { width: 6160, fill: i % 2 === 0 ? "FEE2E2" : "FEF2F2" }),
        ]})),
      ],
    }));
    items.push(spacer(1));
  }

  if (warns.length > 0) {
    items.push(sectionHeading("Warnings (should fix)"));
    items.push(spacer(0.3));
    items.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [3200, 6160],
      rows: [
        new TableRow({ children: [
          cell([para("Metric",         { bold: true, color: WHITE, size: 18 })], { width: 3200, fill: BLUE_DARK }),
          cell([para("Recommendation", { bold: true, color: WHITE, size: 18 })], { width: 6160, fill: BLUE_DARK }),
        ]}),
        ...warns.map((m, i) => new TableRow({ children: [
          cell([para(m.name, { bold: true, size: 17 })],
            { width: 3200, fill: i % 2 === 0 ? "FFFBEB" : "FEF9C3" }),
          cell([para(m.note || `${m.name} is below recommended threshold (value: ${m.value})`, { size: 17 })],
            { width: 6160, fill: i % 2 === 0 ? "FFFBEB" : "FEF9C3" }),
        ]})),
      ],
    }));
    items.push(spacer(1));
  }

  if (fails.length === 0 && warns.length === 0) {
    items.push(new Paragraph({
      spacing: { before: 80, after: 80 },
      children: [
        new TextRun({
          text: "✓  All metrics passed. No critical issues or warnings detected.",
          bold: true, color: GREEN, size: 22, font: "Arial",
        }),
      ],
    }));
  }

  return items;
}

// ── Header / Footer ───────────────────────────────────────────────────────
function makeHeader() {
  return new Header({
    children: [
      new Paragraph({
        alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: "C5D3E8" } },
        spacing: { before: 0, after: 120 },
        children: [
          new TextRun({
            text: "Agentic Research Assistant — Evaluation Report",
            color: GREY_TEXT, size: 16, font: "Arial",
          }),
        ],
      }),
    ],
  });
}

function makeFooter(data) {
  return new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.LEFT,
        border: { top: { style: BorderStyle.SINGLE, size: 2, color: "C5D3E8" } },
        spacing: { before: 80, after: 0 },
        tabStops: [{ type: TabStopType.RIGHT, position: 9360 }],
        children: [
          new TextRun({
            text: `Score: ${data.overall_score}/100  |  ${data.timestamp}`,
            color: GREY_TEXT, size: 16, font: "Arial",
          }),
          new TextRun({ text: "\tPage ", color: GREY_TEXT, size: 16, font: "Arial" }),
          new PageNumberElement(),
        ],
      }),
    ],
  });
}

// ── Main ──────────────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 20, color: BLACK } },
    },
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size:   { width: 12240, height: 15840 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
      },
      headers: { default: makeHeader() },
      footers: { default: makeFooter(data) },
      children: buildDocument(data),
    },
  ],
});

Packer.toBuffer(doc).then(buffer => {
  fs.mkdirSync(path.dirname(docxPath), { recursive: true });
  fs.writeFileSync(docxPath, buffer);
  console.log(`OK: ${docxPath}`);
}).catch(err => {
  console.error("DOCX generation failed:", err.message);
  process.exit(1);
});
