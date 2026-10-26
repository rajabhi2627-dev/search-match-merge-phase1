"""Full validation report: guardrails, fallback and rendering, with a stand-in for the LLM (no API calls)."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

import data_generator as G
import matching as M
import performance as P
import report_docx as RD
import report_pdf as RP
import validation_report as R


@pytest.fixture(scope="module")
def data():
    records = G.generate_master_records()
    master = M.prepare_master(records)
    results = P.run_monitoring(master)
    production = {"Algorithm": "Production algorithm", **P.document_metrics(results)}
    benchmark = pd.DataFrame([production, {**production, "Algorithm": "Challenger (name)"}])
    return results, benchmark, pd.DataFrame(records), master


def _answer(call, evidence):
    """What a well-behaved model would submit: correct ratings and only numbers from the evidence."""
    answer = {}
    for part in R.CALL_PARTS[call]:
        if part in R.QUANT_SECTIONS:
            point = "The analysis covers every period of the validation sample."
            if "performance_by_period" in evidence:
                first = evidence["performance_by_period"][0]
                point = f"The hit rate was {first['hit_rate_pct']}% in {first['period']}."
            answer[part] = {"analysis": [point, "The results are consistent across periods.", "No anomaly was found."],
                            "observation": "The review indicates results in line with the rule-based assessment.",
                            "recommendation": "The model owner may consider continued monitoring."}
        elif part in ("documentation_review", "data_quality_rows"):
            areas = R.DOC_AREAS if part == "documentation_review" else R.DQ_AREAS
            answer[part] = {a: {"observation": "Documented in the README.", "recommendation": "Keep it current."}
                            for a in areas}
        elif part in R.QUAL_SECTIONS:
            answer[part] = {"assessment": "Reviewed.", "observation": "Documented.", "recommendation": "Keep it current."}
        elif part == "validation_summary":
            answer[part] = ["Work performed.", "Results."]
        else:
            answer[part] = "Adequate."
    return answer


def good_ask(call, evidence, check):
    answer = _answer(call, evidence)
    return answer, "test-model", 0, check(answer)


def bad_ask(call, evidence, check):
    answer = json.loads(json.dumps(_answer(call, evidence)))
    if call == "quant_a":
        answer["performance"]["observation"] = "The hit rate was 77.7%."                 # invented number
        answer["challenger"]["analysis"][1] = "Recall improved by 12.3 pp."                # invented number
    if call == "qual_a1":
        answer["documentation_review"] = dict(list(answer["documentation_review"].items())[:2])  # missing areas
    if call == "qual_b2":
        answer["governance"]["observation"] = "See Table 3."                             # table reference
    return answer, "test-model", 1, check(answer)


def test_every_request_fits_the_free_tier(data):
    """Prompt estimate + reserved answer tokens of every call must stay within 8,000 tokens."""
    import agent as A
    ev = R.collect(*data)
    rates = R.ratings(ev)
    previous = {"observations": {str(i): "x" * 400 for i in range(14)}}
    for call, _, pack, parts, _ in R.CALLS:
        evidence = R.evidence_for(pack, ev, rates, previous if pack == "summary" else None)
        tool = {"type": "function", "function": {"name": "submit", "parameters": R.call_schema(parts)}}
        user = R.call_instructions(parts) + json.dumps(A._compact(evidence), default=str, separators=(",", ":"))
        prompt = A.estimate_prompt_tokens([{"role": "system", "content": R.SYSTEM_PROMPT},
                                           {"role": "user", "content": user}], [tool])
        assert prompt + R.REPORT_PART_MAX_OUTPUT <= 8000, (call, prompt)


def test_correct_answers_pass_every_check(data):
    report = R.build_report(*data, ask=good_ask)
    assert report["meta"]["ai_parts"] == report["meta"]["parts"] == len(report["log"])


def test_mistakes_are_caught_and_marked(data):
    report = R.build_report(*data, ask=bad_ask)
    failed = {row["part"].split(" - ")[-1] for row in report["log"] if row["result"] == R.CHECKS_NOT_PASSED}
    assert failed == {"performance", "challenger", "documentation review", "governance"}
    notes = [b for b in report["blocks"] if b["type"] == "note"]
    assert len(notes) == 4                                   # every failed part is marked in the report
    ratings = [b["text"] for b in report["blocks"] if b["type"] == "rating"]
    ev = R.collect(*data)
    rates = R.ratings(ev)
    overall_badges = 2 * (R.overall_rating(ev, rates) == R.NEEDS)  # executive summary + validation summary
    assert ratings.count(R.NEEDS) == sum(v == R.NEEDS for v in rates.values()) + overall_badges  # from the rules


def test_no_answer_stops_the_report(data):
    def silent(call, evidence, check):
        raise RuntimeError("Groq did not return an answer")
    with pytest.raises(RuntimeError):
        R.build_report(*data, ask=silent)


def test_ratings_come_from_rules(data):
    ev = R.collect(*data)
    rates = R.ratings(ev)
    assert set(rates.values()) <= {R.SAT, R.NEEDS}
    assert R.overall_rating(ev, rates) in R.OVERALL_RATINGS


def test_report_renders_as_pdf_and_word(data):
    report = R.build_report(*data, ask=good_ask)
    approval = {"reviewer": "A. Validator", "decision": "Approve", "comment": "ok", "run_at": "2026-10-26T12:00"}
    assert RP.render_full_pdf(report, approval).startswith(b"%PDF")
    import docx
    document = docx.Document(io.BytesIO(RD.render_full_docx(report, approval)))
    headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
    assert "1  Background" in headings and "6  Validation Summary" in headings
    assert len(document.tables) > 20
