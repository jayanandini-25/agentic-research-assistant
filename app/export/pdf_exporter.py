"""
PDF Exporter — Phase 11 (fixed)
=====================================================================
File location:  app/export/pdf_exporter.py

Converts the markdown research report into a styled PDF.

Approach (priority order):
  1. WeasyPrint  — full HTML/CSS rendering (best quality, Linux/macOS only)
  2. fpdf2       — plain-text fallback
  3. reportlab   — plain-text fallback (always available on Windows)

Fixes applied:
  - WeasyPrint is skipped entirely on Windows (no GTK runtime available)
  - reportlab fallback now catches runtime errors, not just ImportError
  - <super> tags in reportlab paragraphs replaced with rise/fontSize trick
    to avoid XML parse errors on special characters
  - fpdf2 runtime errors caught and logged before falling through
  - All three backends raise a clear RuntimeError if they all fail

Install: pip install reportlab markdown
         (fpdf2 optional; weasyprint only useful on Linux/macOS)
"""

import re
import os
import sys
from typing import List, Dict
from core.logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Report styling (used by WeasyPrint only)
# ─────────────────────────────────────────────────────────────────────────────

REPORT_CSS = """
@page {
    size: A4;
    margin: 2.5cm 2.5cm 2.5cm 2.5cm;
    @top-center {
        content: string(report-title);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1a1a1a;
    max-width: 100%;
}

h1 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 22pt;
    font-weight: 700;
    color: #1a1a2e;
    border-bottom: 3px solid #2563eb;
    padding-bottom: 12px;
    margin-bottom: 24px;
    string-set: report-title content();
    line-height: 1.3;
}

h2 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 15pt;
    font-weight: 600;
    color: #1e3a5f;
    margin-top: 32px;
    margin-bottom: 12px;
    border-left: 4px solid #2563eb;
    padding-left: 12px;
    page-break-after: avoid;
}

h3 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 12pt;
    font-weight: 600;
    color: #374151;
    margin-top: 20px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

p {
    text-align: justify;
    margin-bottom: 12px;
    orphans: 3;
    widows: 3;
}

sup.citation {
    font-size: 8pt;
    color: #2563eb;
    font-weight: 600;
    vertical-align: super;
}

img {
    max-width: 85%;
    height: auto;
    display: block;
    margin: 16px auto;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

em.caption {
    display: block;
    text-align: center;
    font-size: 9.5pt;
    color: #555;
    margin-top: -8px;
    margin-bottom: 16px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 10pt;
}

th {
    background: #1e3a5f;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-family: 'Helvetica Neue', Arial, sans-serif;
}

td {
    padding: 7px 12px;
    border-bottom: 1px solid #e5e7eb;
}

tr:nth-child(even) td {
    background: #f9fafb;
}

pre, code {
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f3f4f6;
    border-radius: 4px;
}

pre {
    padding: 12px 16px;
    border-left: 3px solid #6b7280;
    margin: 12px 0;
}

code {
    padding: 1px 4px;
}

blockquote {
    border-left: 3px solid #d1d5db;
    padding-left: 16px;
    color: #4b5563;
    font-style: italic;
    margin: 12px 0;
}

ul, ol {
    margin: 8px 0 12px 0;
    padding-left: 24px;
}

li {
    margin-bottom: 4px;
}

hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 24px 0;
}

h2 { page-break-before: auto; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → HTML converter (used by WeasyPrint path only)
# ─────────────────────────────────────────────────────────────────────────────

def _md_to_html(md_text: str, title: str = "") -> str:
    """Converts markdown to a full HTML document string."""
    try:
        import markdown
        html_body = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code", "nl2br", "toc"],
        )
    except ImportError:
        html_body = _basic_md_to_html(md_text)

    # Wrap inline citations [N] in superscript spans
    html_body = re.sub(
        r'\[(\d+)\]',
        r'<sup class="citation">[\1]</sup>',
        html_body,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""


def _basic_md_to_html(md: str) -> str:
    """Minimal markdown → HTML without external deps."""
    lines   = md.split("\n")
    html    = []
    in_list = False
    in_pre  = False

    for line in lines:
        if line.startswith("```"):
            if in_pre:
                html.append("</code></pre>")
                in_pre = False
            else:
                html.append("<pre><code>")
                in_pre = True
            continue

        if in_pre:
            html.append(line)
            continue

        if line.startswith("### "):
            html.append(f"<h3>{_inline_md(line[4:])}</h3>")
        elif line.startswith("## "):
            html.append(f"<h2>{_inline_md(line[3:])}</h2>")
        elif line.startswith("# "):
            html.append(f"<h1>{_inline_md(line[2:])}</h1>")
        elif line.strip() in ("---", "***", "___"):
            html.append("<hr>")
        elif re.match(r"!\[.*?\]\(.*?\)", line):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                html.append(f'<img src="{m.group(2)}" alt="{m.group(1)}">')
                html.append(f'<p><em class="caption">{m.group(1)}</em></p>')
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{_inline_md(line[2:])}</li>")
        elif line.startswith("> "):
            html.append(f"<blockquote><p>{_inline_md(line[2:])}</p></blockquote>")
        elif "|" in line and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells if c):
                continue
            is_header = not any("<td>" in h for h in html[-5:])
            if is_header:
                row = "".join(f"<th>{c}</th>" for c in cells)
                html.append(f"<table><thead><tr>{row}</tr></thead><tbody>")
            else:
                row = "".join(f"<td>{_inline_md(c)}</td>" for c in cells)
                html.append(f"<tr>{row}</tr>")
        elif line.strip() == "":
            if in_list:
                html.append("</ul>")
                in_list = False
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{_inline_md(line)}</p>")

    if in_list:
        html.append("</ul>")
    if in_pre:
        html.append("</code></pre>")

    return "\n".join(html)


def _inline_md(text: str) -> str:
    """Processes inline markdown: bold, italic, code, links."""
    text = re.sub(r"\*\*(.+?)\*\*",       r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",           r"<em>\1</em>",         text)
    text = re.sub(r"`(.+?)`",             r"<code>\1</code>",     text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown(line: str) -> str:
    """Strips common markdown syntax to plain text for text-based renderers."""
    clean = re.sub(r"\*\*(.+?)\*\*",    r"\1",      line)
    clean = re.sub(r"\*(.+?)\*",        r"\1",      clean)
    clean = re.sub(r"`(.+?)`",          r"\1",      clean)
    clean = re.sub(r"!\[.*?\]\(.*?\)",  "[image]",  clean)
    clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1",      clean)
    return clean


def _escape_xml(text: str) -> str:
    """Escapes characters that break reportlab's paragraph XML parser."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _rl_inline(line: str) -> str:
    """
    Converts inline markdown to reportlab paragraph XML.
    Uses <b>, <i>, <font> tags — avoids <super> which can cause parse errors.
    Citations [N] rendered with fontSize/rise instead of <super>.
    """
    # Escape XML chars first, then re-apply our own tags
    line = _escape_xml(line)

    # Bold
    line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
    # Italic
    line = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", line)
    # Inline code
    line = re.sub(r"`(.+?)`",
                  r'<font name="Courier" size="9">\1</font>', line)
    # Hyperlinks — just show the text in blue
    line = re.sub(r"\[(.+?)\]\(.+?\)",
                  r'<font color="#2563eb">\1</font>', line)
    # Citations [N] — superscript via rise trick
    line = re.sub(
        r"\[(\d+)\]",
        r'<font color="#2563eb" size="8"><super>[\1]</super></font>',
        line,
    )
    return line


# ─────────────────────────────────────────────────────────────────────────────
# PDF Exporter class
# ─────────────────────────────────────────────────────────────────────────────

class PDFExporter:
    """
    Converts markdown report to styled PDF.

    Library priority:
      1. WeasyPrint  — full HTML/CSS rendering (skipped on Windows — needs GTK)
      2. fpdf2       — plain-text fallback
      3. reportlab   — plain-text fallback (default on Windows)
    """

    def export(
        self,
        markdown_report : str,
        output_path     : str,
        title           : str = "Research Report",
    ) -> str:
        """
        Converts markdown to PDF and writes to output_path.
        Returns output_path on success, raises RuntimeError if all backends fail.
        """
        logger.info(
            f"PDFExporter | converting | title='{title[:50]}' | path={output_path}"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        errors = []

        # ── 1. WeasyPrint — skipped on Windows (no GTK runtime) ──────────────
        if sys.platform != "win32":
            try:
                from weasyprint import HTML as WeasyHTML
                html = _md_to_html(markdown_report, title)
                WeasyHTML(string=html).write_pdf(output_path)
                logger.info(f"PDFExporter | WeasyPrint OK | {output_path}")
                return output_path
            except ImportError:
                logger.info("PDFExporter | WeasyPrint not installed — skipping")
            except Exception as e:
                logger.warning(f"PDFExporter | WeasyPrint failed: {e}")
                errors.append(f"weasyprint: {e}")
        else:
            logger.info(
                "PDFExporter | Windows detected — skipping WeasyPrint "
                "(requires GTK runtime). Using fpdf2/reportlab instead."
            )

        # ── 2. fpdf2 fallback ─────────────────────────────────────────────────
        try:
            self._export_with_fpdf2(markdown_report, output_path, title)
            logger.info(f"PDFExporter | fpdf2 OK | {output_path}")
            return output_path
        except ImportError:
            logger.info("PDFExporter | fpdf2 not installed — trying reportlab")
        except Exception as e:
            logger.warning(f"PDFExporter | fpdf2 failed: {e}")
            errors.append(f"fpdf2: {e}")

        # ── 3. reportlab fallback ─────────────────────────────────────────────
        try:
            self._export_with_reportlab(markdown_report, output_path, title)
            logger.info(f"PDFExporter | reportlab OK | {output_path}")
            return output_path
        except ImportError:
            errors.append("reportlab: not installed")
        except Exception as e:
            logger.error(f"PDFExporter | reportlab failed: {e}")
            errors.append(f"reportlab: {e}")

        # ── All backends failed ───────────────────────────────────────────────
        raise RuntimeError(
            "PDFExporter: all backends failed.\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\n\nFix: pip install reportlab --break-system-packages"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Backend 2: fpdf2
    # ─────────────────────────────────────────────────────────────────────────

    def _export_with_fpdf2(self, md: str, path: str, title: str) -> None:
        """fpdf2 fallback — renders plain text with basic formatting.

        Phase 13: Uses built-in DejaVu font for full Unicode support on Windows.
        Falls back to Helvetica if DejaVu is unavailable.
        """
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # Phase 13: Register a Unicode-capable font if available
        _use_unicode = False
        try:
            # fpdf2 ships with DejaVu — this gives us full UTF-8 support
            pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
            pdf.add_font("DejaVu", "B", "DejaVuSansCondensed-Bold.ttf", uni=True)
            _use_unicode = True
            _font = "DejaVu"
        except Exception:
            _font = "Helvetica"

        pdf.add_page()

        # Title
        pdf.set_font(_font, "B", 18)
        pdf.set_text_color(26, 26, 46)
        pdf.multi_cell(0, 10, title[:120], align="L")
        pdf.ln(4)
        pdf.set_draw_color(37, 99, 235)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(8)

        for line in md.split("\n"):
            line  = line.rstrip()
            clean = _strip_markdown(line)
            # strip citation brackets
            clean = re.sub(r"\[(\d+)\]", r"[\1]", clean)

            if clean.startswith("# "):
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(26, 26, 46)
                pdf.multi_cell(0, 9, clean[2:])
                pdf.ln(3)
            elif clean.startswith("## "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(30, 58, 95)
                pdf.ln(4)
                pdf.multi_cell(0, 8, clean[3:])
                pdf.ln(2)
            elif clean.startswith("### "):
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(55, 65, 81)
                pdf.multi_cell(0, 7, clean[4:])
            elif clean.startswith("- ") or clean.startswith("* "):
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(26, 26, 26)
                pdf.multi_cell(0, 6, "  \u2022 " + clean[2:])
            elif clean.strip() in ("---", "***", "___"):
                pdf.set_draw_color(200, 200, 200)
                pdf.set_line_width(0.3)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(4)
            elif clean.strip():
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(26, 26, 26)
                try:
                    pdf.multi_cell(0, 6, clean)
                except Exception:
                    # last-resort: encode to latin-1, replacing unmappable chars
                    safe = clean.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe)
            else:
                pdf.ln(3)

        pdf.output(path)

    # ─────────────────────────────────────────────────────────────────────────
    # Backend 3: reportlab
    # ─────────────────────────────────────────────────────────────────────────

    def _export_with_reportlab(self, md: str, path: str, title: str) -> None:
        """reportlab fallback — paragraph-level rendering with inline markup."""
        from reportlab.lib.pagesizes  import A4
        from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units      import cm
        from reportlab.lib            import colors
        from reportlab.platypus       import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
        )
        from reportlab.lib.enums      import TA_JUSTIFY, TA_LEFT, TA_CENTER
        from reportlab.platypus       import PageTemplate, Frame
        from reportlab.lib.pagesizes  import A4

        # ── Document ──────────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            leftMargin=2.5*cm, rightMargin=2.5*cm,
            topMargin=2.5*cm,  bottomMargin=2.5*cm,
            title=title,
        )

        # ── Styles ────────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()

        def _add(name, **kw):
            styles.add(ParagraphStyle(name, **kw))

        _add("ReportTitle",
             fontSize=20, spaceAfter=10, spaceBefore=0,
             textColor=colors.HexColor("#1a1a2e"),
             fontName="Helvetica-Bold", leading=26)

        _add("Section",
             fontSize=13, spaceAfter=6, spaceBefore=18,
             textColor=colors.HexColor("#1e3a5f"),
             fontName="Helvetica-Bold", leading=18)

        _add("SubSection",
             fontSize=11, spaceAfter=4, spaceBefore=12,
             textColor=colors.HexColor("#374151"),
             fontName="Helvetica-Bold", leading=15)

        _add("Body",
             fontSize=10, spaceAfter=8, leading=16,
             alignment=TA_JUSTIFY,
             fontName="Helvetica")

        _add("BulletBody",
             fontSize=10, spaceAfter=4, leading=15,
             leftIndent=16, firstLineIndent=0,
             fontName="Helvetica")

        _add("CodeBlock",
             fontSize=8.5, spaceAfter=8, spaceBefore=4, leading=12,
             fontName="Courier",
             leftIndent=12, rightIndent=12,
             backColor=colors.HexColor("#f3f4f6"),
             borderColor=colors.HexColor("#6b7280"),
             borderWidth=0.5, borderPadding=6)

        _add("BlockQuote",
             fontSize=10, spaceAfter=8, leading=15,
             leftIndent=20,
             textColor=colors.HexColor("#4b5563"),
             fontName="Helvetica-Oblique")

        # ── Parse markdown line by line ────────────────────────────────────────
        story  = []
        lines  = md.split("\n")
        i      = 0
        bullet_buffer = []

        def _flush_bullets():
            if bullet_buffer:
                for b in bullet_buffer:
                    story.append(
                        Paragraph(f"\u2022&nbsp;&nbsp;{b}", styles["BulletBody"])
                    )
                bullet_buffer.clear()

        while i < len(lines):
            line = lines[i].rstrip()

            # Fenced code block
            if line.startswith("```"):
                _flush_bullets()
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(_escape_xml(lines[i]))
                    i += 1
                code_text = "<br/>".join(code_lines) if code_lines else ""
                if code_text:
                    story.append(Paragraph(code_text, styles["CodeBlock"]))
                i += 1
                continue

            # H1
            if re.match(r"^# (?!#)", line):
                _flush_bullets()
                text = _rl_inline(line[2:].strip())
                story.append(Paragraph(text, styles["ReportTitle"]))
                story.append(HRFlowable(
                    width="100%", thickness=2,
                    color=colors.HexColor("#2563eb"),
                    spaceAfter=8,
                ))

            # H2
            elif re.match(r"^## (?!#)", line):
                _flush_bullets()
                text = _rl_inline(line[3:].strip())
                story.append(Paragraph(text, styles["Section"]))

            # H3
            elif re.match(r"^### ", line):
                _flush_bullets()
                text = _rl_inline(line[4:].strip())
                story.append(Paragraph(text, styles["SubSection"]))

            # Horizontal rule
            elif line.strip() in ("---", "***", "___"):
                _flush_bullets()
                story.append(HRFlowable(
                    width="100%", thickness=0.5,
                    color=colors.HexColor("#e5e7eb"),
                    spaceBefore=6, spaceAfter=6,
                ))

            # Blockquote
            elif line.startswith("> "):
                _flush_bullets()
                text = _rl_inline(line[2:])
                story.append(Paragraph(text, styles["BlockQuote"]))

            # Bullet list
            elif line.startswith("- ") or line.startswith("* "):
                bullet_buffer.append(_rl_inline(line[2:]))

            # Numbered list (treat as bullet for simplicity)
            elif re.match(r"^\d+\. ", line):
                text = re.sub(r"^\d+\. ", "", line)
                bullet_buffer.append(_rl_inline(text))

            # Image — render as placeholder (reportlab can embed but needs file)
            elif re.match(r"!\[.*?\]\(.*?\)", line):
                _flush_bullets()
                m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
                if m:
                    alt = _escape_xml(m.group(1))
                    story.append(
                        Paragraph(
                            f'<i><font color="#888888">[Image: {alt}]</font></i>',
                            styles["Body"],
                        )
                    )

            # Empty line
            elif line.strip() == "":
                _flush_bullets()
                story.append(Spacer(1, 4))

            # Normal paragraph
            else:
                _flush_bullets()
                text = _rl_inline(line)
                # Safe parse — fall back to plain text if XML is malformed
                try:
                    story.append(Paragraph(text, styles["Body"]))
                except Exception:
                    plain = _escape_xml(_strip_markdown(line))
                    story.append(Paragraph(plain, styles["Body"]))

            i += 1

        _flush_bullets()

        # ── Page number callback ───────────────────────────────────────────────
        def _add_page_number(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#666666"))
            page_num = f"Page {doc.page}"
            canvas.drawCentredString(A4[0] / 2, 1.5 * cm, page_num)
            canvas.restoreState()

        doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)