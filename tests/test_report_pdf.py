"""PDF report: builds without the LLM and renders the agent's Markdown (no API calls)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from reportlab.platypus import ListFlowable, Paragraph, Table

import report_pdf as R

REPORT = """## 1. Executive summary
Recall is **66.7%** and Type II error 28.0% ‑ both ≥ the amber limit → overall Red.

- first point
- second point

## 9. Findings and recommendations
| Finding | Severity | Evidence | Recommendation |
|---|---|---|---|
| Recall below threshold | High | Recall 66.7% | Review DOB rules |
"""

RUN = {"period": "Jun-26", "model": "openai/gpt-oss-120b", "report": REPORT, "overall_status": "Red",
       "breaches": [{"kpi": "recall", "status": "Red"}], "revisions": 1,
       "verification": {"passed": False, "numbers_checked": 3, "checks": {"numbers_traceable": False},
                        "problems": ["These numbers are not found in the evidence: [12.0]."]},
       "trace": [{"step": 1, "actor": "pipeline", "stage": "Data integrity checks", "tool": "check_data_integrity"}]}

SECTIONS = [{"title": "3. Hit rate", "note": "note",
             "tables": [("", pd.DataFrame({"period": ["Jan-26", "Feb-26"], "hit_rate": [1.0, 0.986]}))],
             "trend": pd.DataFrame({"period": ["Jan-26", "Feb-26"], "hit_rate": [1.0, 0.986],
                                    "type2_error": [0.01, 0.03]})},
            {"title": "10. Validation summary", "note": "", "tables": [],
             "observations": [{"Area": "Hit rate", "Status": "Satisfactory", "Observation": "ok"}]}]


def test_pdf_builds_with_and_without_coded_sections():
    for sections in (SECTIONS, None):
        pdf = R.build_validation_pdf(RUN, sections, generated_at="2026-10-26 12:00")
        assert pdf.startswith(b"%PDF") and len(pdf) > 3000


def test_pdf_includes_approval():
    run = {**RUN, "approval": {"run_at": "2026-10-26T12:00:00", "reviewer": "A. Validator", "decision": "Approve",
                               "comment": "Fine"}}
    assert R.build_validation_pdf(run, None).startswith(b"%PDF")


def test_markdown_rendering():
    flow = R.markdown_flowables(REPORT)
    kinds = [type(f) for f in flow]
    assert ListFlowable in kinds and Table in kinds
    headings = [f.getPlainText() for f in flow if isinstance(f, Paragraph) and f.style.name == "h1"]
    assert headings == ["1. Executive summary", "9. Findings and recommendations"]


def test_characters_outside_the_pdf_font_are_replaced():
    assert R._ascii_safe("a ≥ b → c‑d") == "a >= b -> c-d"
