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
        elif part == "validation_summary":
            answer[part] = ["Work performed.", "Results."]
        elif part == "conclusion":
            red = [R.A._label(b["kpi"]) for b in evidence.get("latest_period_breaches", []) if b["status"] == "Red"]
            answer[part] = "The overall rating is driven by " + (", ".join(red) or "no breach") + "."
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
    if call == "quant_b":
        answer["errors"]["observation"] = "See Table 3."                                 # table reference
        answer["disputes"]["analysis"] = ["Only one point."]                              # incomplete analysis
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
    assert failed == {"performance", "challenger", "errors", "disputes"}
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


def test_summary_must_not_hide_breaches():
    evidence = {"assessed_period": "Jun-26", "latest_period_breaches": [{"kpi": "review_rate", "status": "Red"},
                                                                        {"kpi": "recall", "status": "Amber"}]}
    hidden = R.breach_problems("conclusion", "Needs attention, while performance meets all KPI limits.", evidence)
    assert len(hidden) == 2                                   # false claim + Red KPI not named
    named = R.breach_problems("conclusion", "Needs attention because the manual review rate is Red.", evidence)
    assert named == []
    for claim in ("The model satisfied the defined limits for each metric.", "All key indicators were met.",
                  "Every KPI is within its threshold."):
        assert R.breach_problems("quantitative_assessment", claim, evidence), claim
    for neutral in ("Not all limits are met.", "All key performance indicators were evaluated.",
                    "Most indicators met their thresholds while the review rate exceeded its limit."):
        assert R.breach_problems("quantitative_assessment", neutral, evidence) == [], neutral
    assert R.breach_problems("conclusion", "All KPI limits are met.", {"latest_period_breaches": []}) == []


def test_correct_figures_pass_first_time(data):
    """Sentences the live model wrote with correct figures must not be rejected by the KPI checks."""
    ev = R.collect(*data)
    rates = R.ratings(ev)
    summary = R.evidence_for("summary", ev, rates)
    m = summary["overall_metrics_pct"]
    text = (f"Quantitative validation shows hit rate {m['hit_rate']}%, dispute rate {m['dispute_rate']}%, "
            f"type I error {m['type1_error']}%, type II error {m['type2_error']}%, precision {m['precision']}%, "
            f"recall {m['recall']}% and manual review rate {m['review_rate']}%.")
    assert R.check_answer(["quantitative_assessment"], {"quantitative_assessment": text}, summary, rates,
                          R.NEEDS) == {}
    quant_a = R.evidence_for("quant_a", ev, rates)
    prod = quant_a["challenger_benchmark"][0]
    sentence = f"Production recall is {prod['recall_pct']}% with a type-II error of {prod['type2_error_pct']}%."
    problems = R.check_answer(["challenger"], {"challenger": {"analysis": [sentence, "Second point."],
                                                              "observation": "-", "recommendation": "-"}},
                              quant_a, rates, R.NEEDS)
    assert problems == {}, problems


def test_field_names_are_rejected():
    problems = R.check_answer(["conclusion"], {"conclusion": "The overall_rating is driven by review_rate."},
                              {}, {}, R.NEEDS)
    assert any("review_rate" in p and "overall_rating" in p for p in problems["conclusion"])
    assert R.check_answer(["conclusion"], {"conclusion": "The manual review rate drives it."}, {}, {}, R.NEEDS) == {}


def test_calibration_ignores_tiny_recall_gains():
    def table(alt_recall):
        rows = [{"MATCH threshold": 85, "Current setting": True, "recall": 0.880},
                {"MATCH threshold": 80, "Current setting": False, "recall": alt_recall}]
        return pd.DataFrame([{"Varied": "MATCH threshold", "precision": 0.99, "incorrect_merges": 5,
                              "review_rate": 0.05, **r} for r in rows])
    assert R._calibration_rating(table(0.882)) == R.SAT
    assert R._calibration_rating(table(0.880 + R.C.CALIBRATION_MIN_RECALL_GAIN)) == R.NEEDS


def test_report_renders_as_pdf_and_word(data):
    report = R.build_report(*data, ask=good_ask)
    approval = {"reviewer": "A. Validator", "decision": "Approve", "comment": "ok", "run_at": "2026-10-26T12:00"}
    assert RP.render_full_pdf(report, approval).startswith(b"%PDF")
    import docx
    document = docx.Document(io.BytesIO(RD.render_full_docx(report, approval)))
    headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
    assert "1  Background" in headings and "4  Quantitative Validation" in headings
    assert "5  Validation Summary" in headings
    assert not any("Qualitative" in h for h in headings)
    assert len(document.tables) > 15
