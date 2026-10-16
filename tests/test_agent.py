"""Deterministic tests of the agent tools, guardrails and validation scoring (no LLM calls)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

import config as C
import data_generator as G
import matching as M
import performance as P
import agent as A
import agent_tools as T
import agent_validation as AV


@pytest.fixture(scope="module")
def setup():
    master = M.prepare_master(G.generate_master_records())
    results = P.run_monitoring(master)
    return results, master, AV.build_scenarios(results, master)


def _scenario(setup, sid):
    return next(s for s in setup[2] if s["id"] == sid)


def test_scenario_expected_answers(setup):
    expected = {"S02": "Green", "S03": "Red", "S04": "Amber", "S05": "Red", "S06": "Red",
                "S09": "Insufficient data", "S10": "Insufficient data", "S11": "Insufficient data"}
    for sid, status in expected.items():
        assert T.check_thresholds(_scenario(setup, sid)["ctx"])["overall_status"] == status, sid
    assert T.check_thresholds(_scenario(setup, "S03")["ctx"])["breaches"] == [{"kpi": "dispute_rate", "status": "Red"}]


def test_integrity_flags_missing_period(setup):
    out = T.check_data_integrity(_scenario(setup, "S08")["ctx"])
    assert {"period": "Apr-26", "type": "missing period"}.items() <= out["issues"][0].items()


def test_evidence_fits_token_budget(setup):
    ctx = {**_scenario(setup, "S13")["ctx"], "proposed_change": {"review_threshold": 60}}
    evidence, outputs, steps = A.gather_evidence(ctx)
    assert [s["tool"] for s in steps] == [t for _, t, _ in A.PIPELINE]
    assert len(A.SYSTEM_PROMPT) + len(evidence) < 12000  # about 3,500 tokens of the 8,000 per minute
    assert "information only" in evidence


def _good_submission(ctx, outputs):
    th = T.check_thresholds(ctx)
    kpi = next(k for k in th["kpis"] if k["kpi"] == "recall")
    report = "\n".join(f"## {i}. {s}\ntext" for i, s in enumerate(A.REPORT_SECTIONS, 1))
    report += f"\nRecall is {kpi['value_pct']}% in {ctx['period']}, fell by 33.3 pp."
    return {"report_markdown": report, "overall_status": th["overall_status"], "breaches": th["breaches"],
            "data_issues": []}


def test_verify_report_accepts_correct_report(setup):
    ctx = _scenario(setup, "S01")["ctx"]
    _, outputs, _ = A.gather_evidence(ctx)
    ver = A.verify_report(_good_submission(ctx, outputs), outputs, ctx)
    assert ver["passed"], ver["problems"]


def test_verify_report_catches_errors(setup):
    ctx = _scenario(setup, "S01")["ctx"]
    _, outputs, _ = A.gather_evidence(ctx)
    bad = _good_submission(ctx, outputs)
    bad["report_markdown"] = bad["report_markdown"].replace("## 8. Root causes", "") + " Hit rate was 93.7%."
    bad["overall_status"] = "Green"
    bad["breaches"] = []
    ver = A.verify_report(bad, outputs, ctx)
    assert not ver["passed"]
    assert 93.7 in ver["untraceable_numbers"]
    assert not ver["checks"]["overall_status_correct"]
    assert not ver["checks"]["breaches_correct"]
    assert ver["checks"]["report_sections_present"] is False


def test_policy_retrieval():
    assert all(AV.retrieval_unit_checks())


def test_sample_size_and_approval_levels():
    assert T.required_sample_size(100) == 43
    assert T._approval_level(0.5).startswith("Minor")
    assert T._approval_level(2.0).startswith("Moderate")
    assert T._approval_level(9.1).startswith("Major")


def test_summarise_rollup():
    ok = json.dumps({t: True for t in AV.TESTS if t != "Stability"})
    rows = [{"scenario_id": "S03", "agent_signature": "x", "results": ok}] * 3
    runs = pd.DataFrame(rows)
    table, outcome = AV.summarise(runs, expert_score=4.5)
    assert outcome == "Pass"
    _, outcome = AV.summarise(runs)  # expert review pending
    assert outcome == "Pass (Conditional)"
    runs.loc[1, "agent_signature"] = "y"  # unstable
    _, outcome = AV.summarise(runs, expert_score=4.5)
    assert outcome == "Fail"
