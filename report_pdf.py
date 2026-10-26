"""
PDF validation report: the AI agent's report plus the full coded (deterministic) validation results.

Layout
  Cover summary  - period, model, overall status, breaches, guardrail result
  Part A         - AI validation report (the agent's Markdown, rendered)
  Part B         - Guardrail verification of the AI report
  Part C         - Coded validation results (the 10 sections of the Validation page)
  Part D         - Validator sign-off (human in the loop)
  Appendix       - Agent trace

build_validation_pdf(...) returns the PDF as bytes, ready for st.download_button.
"""
import io
import re
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import config as C
import validation as V

TITLE = "Search & Match Algorithm - Validation Report"
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#5f6b7a")
RULE = colors.HexColor("#d5dae1")
HEAD_BG = colors.HexColor("#eef1f5")
ACCENT = colors.HexColor("#2f5d8a")
STATUS_COLORS = {"Green": colors.HexColor("#1e7b4a"), "Amber": colors.HexColor("#b26a00"),
                 "Red": colors.HexColor("#b42318"), "Insufficient data": colors.HexColor("#5f6b7a"),
                 "Satisfactory": colors.HexColor("#1e7b4a"), "Needs attention": colors.HexColor("#b26a00")}
SERIES_COLORS = [colors.HexColor("#2f5d8a"), colors.HexColor("#c2410c"), colors.HexColor("#7c3aed")]

# The built-in PDF fonts only cover Latin-1 / Windows-1252; replace the common characters outside it.
CHAR_MAP = {"‐": "-", "‑": "-", "‒": "-", "−": "-", "≥": ">=", "≤": "<=",
            "→": "->", "←": "<-", "≈": "~", "≠": "!=", " ": " ", " ": " ",
            " ": " ", " ": " ", "✓": "Yes", "✗": "No", "✅": "", "⚠": "!"}


def _ascii_safe(text):
    text = "".join(CHAR_MAP.get(ch, ch) for ch in str(text))
    return text.encode("cp1252", errors="replace").decode("cp1252")


# ==================================================================
# Styles
# ==================================================================

def _styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                          textColor=INK, spaceAfter=5)
    return {
        "title": ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=19, leading=24,
                                spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=11, textColor=MUTED, spaceAfter=12),
        "part": ParagraphStyle("part", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=18,
                               textColor=ACCENT, spaceBefore=4, spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=12.5, leading=16,
                             spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14,
                             spaceBefore=8, spaceAfter=3),
        "h3": ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold", fontSize=9.5, leading=13,
                             spaceBefore=6, spaceAfter=2),
        "body": body,
        "note": ParagraphStyle("note", parent=body, fontSize=8.5, leading=12, textColor=MUTED),
        "caption": ParagraphStyle("caption", parent=body, fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                                  textColor=MUTED, spaceBefore=4, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.5, leading=9.5, spaceAfter=0),
        "cell_head": ParagraphStyle("cell_head", parent=body, fontName="Helvetica-Bold", fontSize=7.5, leading=9.5,
                                    spaceAfter=0),
        "status": ParagraphStyle("status", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=13,
                                 alignment=TA_CENTER, textColor=colors.white, spaceAfter=0),
    }


S = _styles()


# ==================================================================
# Markdown (agent report) -> flowables
# ==================================================================

def _inline(text):
    """Escape for ReportLab and convert **bold**, *italic* and `code`."""
    text = escape(_ascii_safe(text))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    return text


def _md_table(lines):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # separator row
        rows.append(cells)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    data = [[Paragraph(_inline(c), S["cell_head"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    return _styled_table(data, _col_widths([[re.sub(r"[*`]", "", c) for c in r] for r in rows]))


def markdown_flowables(markdown):
    """Render the agent's Markdown: headings, paragraphs, bullet / numbered lists and pipe tables."""
    out, para, items, ordered = [], [], [], False
    lines = (markdown or "").splitlines()

    def flush_para():
        if para:
            out.append(Paragraph(_inline(" ".join(para)), S["body"]))
            para.clear()

    def flush_list():
        nonlocal ordered
        if items:
            out.append(ListFlowable([ListItem(Paragraph(_inline(t), S["body"]), leftIndent=12) for t in items],
                                    bulletType="1" if ordered else "bullet", start="1" if ordered else None,
                                    leftIndent=14, bulletFontSize=8 if not ordered else 9.5))
            items.clear()
        ordered = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_para()
            flush_list()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            table = _md_table(block)
            if table:
                out += [table, Spacer(1, 6)]
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        number = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if heading:
            flush_para()
            flush_list()
            level = len(heading.group(1))
            out.append(Paragraph(_inline(heading.group(2)), S["h1" if level <= 2 else "h2" if level == 3 else "h3"]))
        elif bullet or number:
            flush_para()
            if items and ordered != bool(number):
                flush_list()
            ordered = bool(number)
            items.append((bullet or number).group(1))
        elif not stripped or re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            flush_para()
            flush_list()
        elif items and line.startswith((" ", "\t")):
            items[-1] += " " + stripped  # continuation of a list item
        else:
            flush_list()
            para.append(stripped)
        i += 1
    flush_para()
    flush_list()
    return out


# ==================================================================
# Tables
# ==================================================================

def _styled_table(data, widths):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    table.setStyle(TableStyle(style))
    return table


def _col_widths(rows):
    """Column widths from the content: never narrower than the longest single word or value (so numbers and
    words are not split), long text columns take the remaining space. rows[0] is the header."""
    n = max(len(r) for r in rows)
    cols = [[str(r[c]) if c < len(r) else "" for r in rows] for c in range(n)]
    longest_word = [max(len(w) for cell in col for w in (cell.split() or [""])) for col in cols]
    longest_value = [max(len(cell) for cell in col[1:]) if len(col) > 1 else 0 for col in cols]
    minimum = [max(w, 4) * 4.6 + 10 for w in longest_word]                 # ~4.6 pt per character at 7.5 pt
    wanted = [max(m, min(v, 45) * 4.6 + 10) for m, v in zip(minimum, longest_value)]
    if sum(wanted) <= CONTENT_W:
        extra = CONTENT_W - sum(wanted)
        return [w + extra * w / sum(wanted) for w in wanted]
    if sum(minimum) >= CONTENT_W:
        return [CONTENT_W * m / sum(minimum) for m in minimum]
    # Shrink only the part above the minimum, proportionally.
    room = CONTENT_W - sum(minimum)
    slack = [w - m for w, m in zip(wanted, minimum)]
    return [m + room * s / sum(slack) for m, s in zip(minimum, slack)]


def _fmt(column, value):
    if value is None or (not isinstance(value, (list, dict, str)) and pd.isna(value)):
        return "-"
    if isinstance(value, (bool,)) or str(type(value)).endswith("bool_'>"):
        return "Yes" if value else "No"
    if column in V.PERCENT_COLUMNS and isinstance(value, (int, float)):
        return f"{value:.1%}"
    if isinstance(value, float):
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def dataframe_table(df):
    df = df.reset_index(drop=True)
    header = [V.REPORT_LABELS.get(c, str(c)) for c in df.columns]
    body = [[_fmt(c, v) for c, v in zip(df.columns, row)] for row in df.itertuples(index=False)]
    rows = [header] + body
    data = [[Paragraph(escape(_ascii_safe(c)), S["cell_head"]) for c in header]]
    data += [[Paragraph(escape(_ascii_safe(c)), S["cell"]) for c in r] for r in body]
    return _styled_table(data, _col_widths(rows))


def _key_value_table(pairs):
    data = [[Paragraph(f"<b>{escape(_ascii_safe(k))}</b>", S["cell"]), v if not isinstance(v, str)
             else Paragraph(escape(_ascii_safe(v)), S["cell"])] for k, v in pairs]
    table = Table(data, colWidths=[CONTENT_W * 0.3, CONTENT_W * 0.7], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return table


def _status_badge(status, width=40 * mm):
    badge = Table([[Paragraph(escape(_ascii_safe(status or "-")), S["status"])]], colWidths=[width])
    badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), STATUS_COLORS.get(status, MUTED)),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return badge


def _callout(text, color):
    box = Table([[Paragraph(text, S["body"])]], colWidths=[CONTENT_W])
    box.setStyle(TableStyle([("LINEBEFORE", (0, 0), (0, -1), 3, color), ("BACKGROUND", (0, 0), (-1, -1), HEAD_BG),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return box


def trend_chart(trend):
    """Line chart of the monthly rates (values are shares 0-1, shown as %)."""
    series = [c for c in trend.columns if c != "period"]
    drawing = Drawing(CONTENT_W, 62 * mm)
    plot = LinePlot()
    plot.x, plot.y = 30, 30
    plot.width, plot.height = CONTENT_W - 150, 62 * mm - 45
    plot.data = [[(i, float(v) * 100) for i, v in enumerate(trend[s])] for s in series]
    for i, _ in enumerate(series):
        plot.lines[i].strokeColor = SERIES_COLORS[i % len(SERIES_COLORS)]
        plot.lines[i].strokeWidth = 1.5
        plot.lines[i].symbol = makeMarker("Circle", size=3)
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = -0.3, len(trend) - 0.7
    plot.xValueAxis.valueSteps = list(range(len(trend)))
    plot.xValueAxis.labelTextFormat = lambda v: str(trend["period"].iloc[int(round(v))])
    plot.yValueAxis.valueMin = 0
    plot.yValueAxis.valueMax = 100
    plot.yValueAxis.valueStep = 20
    plot.yValueAxis.labelTextFormat = "%d%%"
    for axis in (plot.xValueAxis, plot.yValueAxis):
        axis.labels.fontName, axis.labels.fontSize = "Helvetica", 7
        axis.strokeColor = RULE
    plot.yValueAxis.visibleGrid, plot.yValueAxis.gridStrokeColor = True, RULE
    drawing.add(plot)
    legend = Legend()
    legend.x, legend.y = CONTENT_W - 110, 62 * mm - 25
    legend.fontName, legend.fontSize = "Helvetica", 7.5
    legend.alignment = "right"
    legend.dx = legend.dy = 7
    legend.deltay = 12
    legend.colorNamePairs = [(SERIES_COLORS[i % len(SERIES_COLORS)], V.REPORT_LABELS.get(s, s))
                             for i, s in enumerate(series)]
    drawing.add(legend)
    return drawing


# ==================================================================
# Report
# ==================================================================

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 10 * mm, TITLE)
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    canvas.restoreState()


def _cover(run, generated_at, sample_note):
    ver = run.get("verification") or {}
    breaches = ", ".join(f"{V.REPORT_LABELS.get(b['kpi'], b['kpi'])} ({b['status']})"
                         for b in run.get("breaches") or []) or "None"
    story = [Paragraph(TITLE, S["title"]),
             Paragraph("Independent validation with AI-assisted monitoring report", S["subtitle"]),
             _key_value_table([
                 ("Overall status (AI)", _status_badge(run.get("overall_status"))),
                 ("Assessed period", run.get("period") or "-"),
                 ("Report generated", generated_at),
                 ("AI model", run.get("model") or "-"),
                 ("Threshold breaches", breaches),
                 ("Guardrail verification", "Passed" if ver.get("passed") else "Failed"),
                 ("Self-corrections", str(run.get("revisions", 0))),
                 ("Numbers checked against evidence", str(ver.get("numbers_checked", 0))),
                 ("Validation sample", sample_note),
             ]), Spacer(1, 10)]
    if ver.get("passed"):
        story.append(_callout(
            "<b>How to read this report.</b> Part A was written by the AI agent from evidence produced by "
            "deterministic tools; every figure in it was checked automatically against that evidence (Part B). "
            "Part C contains the coded validation results. The report is advisory: it becomes final only when a "
            "validator signs Part D.", ACCENT))
    else:
        story.append(_callout(
            "<b>Warning: the AI report did not pass all guardrail checks.</b> Treat Part A with caution and rely "
            "on the coded results in Part C. The failed checks are listed in Part B.", STATUS_COLORS["Red"]))
    return story


def _guardrails(run):
    ver = run.get("verification") or {}
    story = [Paragraph("Part B - Guardrail verification of the AI report", S["part"]),
             Paragraph("Deterministic checks run on the AI report before it is shown to a validator.", S["note"])]
    checks = ver.get("checks") or {}
    if checks:
        df = pd.DataFrame([{"Check": k.replace("_", " ").capitalize().replace("Kpi", "KPI"),
                            "Result": "Pass" if v else "Fail"}
                           for k, v in checks.items()])
        story.append(dataframe_table(df))
    if ver.get("expected_status"):
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"Expected overall status from the threshold check: "
                               f"<b>{escape(ver['expected_status'])}</b>.", S["body"]))
    for problem in ver.get("problems") or []:
        story.append(Paragraph("&bull; " + escape(_ascii_safe(problem)), S["body"]))
    return story


def _coded_results(sections):
    story = [Paragraph("Part C - Coded validation results", S["part"]),
             Paragraph("Deterministic, rule-based analyses of the validation sample (no AI involved).", S["note"])]
    for section in sections:
        block = [Paragraph(escape(_ascii_safe(section["title"])), S["h1"])]
        if section.get("note"):
            block.append(Paragraph(escape(_ascii_safe(section["note"])), S["note"]))
        for n, (caption, df) in enumerate(section.get("tables", [])):
            if caption:
                block.append(Paragraph(escape(_ascii_safe(caption)), S["caption"]))
            table = dataframe_table(df)
            # Keep the heading with the first table so a section never starts at the bottom of a page.
            story.append(KeepTogether(block + [table]) if n == 0 else table)
            block = []
            story.append(Spacer(1, 4))
        if section.get("trend") is not None and len(section["trend"]):
            story.append(trend_chart(section["trend"]))
        for note in section.get("observations", []):
            color = STATUS_COLORS.get(note["Status"], MUTED).hexval()[2:]
            block.append(Paragraph(f'<font color="#{color}"><b>{escape(note["Area"])} - {escape(note["Status"])}'
                                   f'</b></font>: {escape(_ascii_safe(note["Observation"]))}', S["body"]))
        if block:
            story.append(KeepTogether(block))
    return story


def _sign_off(run):
    line = "_" * 38
    rows = [("Validator name", line), ("Decision", "[ ] Approve    [ ] Request changes    [ ] Reject"),
            ("Comments", line + "\n" + line), ("Date", line), ("Signature", line)]
    story = [Paragraph("Part D - Validator sign-off", S["part"]),
             Paragraph("The AI report is advisory. The validator reviews Parts A to C and records the decision.",
                       S["note"])]
    approval = run.get("approval")
    if approval:
        rows = [("Validator name", approval.get("reviewer") or "-"), ("Decision", approval.get("decision") or "-"),
                ("Comments", approval.get("comment") or "-"), ("Date", approval.get("run_at") or "-")]
    data = [[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(escape(_ascii_safe(v)).replace("\n", "<br/>"),
                                                            S["body"])] for k, v in rows]
    table = Table(data, colWidths=[CONTENT_W * 0.25, CONTENT_W * 0.75], hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 8)]))
    return story + [table]


def _trace(run):
    trace = run.get("trace") or []
    if not trace:
        return []
    df = pd.DataFrame([{"Step": s.get("step"), "Actor": s.get("actor", ""), "Stage": s.get("stage", s.get("type")),
                        "Tool": s.get("tool", "")} for s in trace])
    return [Paragraph("Appendix - Agent trace", S["part"]),
            Paragraph("Each workflow step the agent ran, in order.", S["note"]), dataframe_table(df)]


def build_validation_pdf(run, sections=None, generated_at=None):
    """run: result of agent.run_agent (optionally with an 'approval' dict); sections: validation.report_sections."""
    generated_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    sample_note = (f"{len(C.MONITORING_PERIODS)} monthly batches "
                   f"({len(C.MONITORING_PERIODS) * (C.MONITORING_MATCH_COUNT + C.MONITORING_NO_MATCH_COUNT)} "
                   f"{C.DATASET_NAME} records)")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                            bottomMargin=20 * mm, title=TITLE, author="Search & Match validation agent",
                            subject=f"Validation report {run.get('period', '')}")
    story = _cover(run, generated_at, sample_note)
    story += [Spacer(1, 14), Paragraph("Part A - AI validation report", S["part"])]
    story += markdown_flowables(run.get("report")) or [Paragraph("No report was submitted by the agent.", S["body"])]
    story += [PageBreak()] + _guardrails(run)
    if sections:
        story += [PageBreak()] + _coded_results(sections)
    story += [PageBreak()] + _sign_off(run)
    trace = _trace(run)
    if trace:
        story += [Spacer(1, 18)] + trace
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
