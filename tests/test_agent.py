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


def _kpi(ctx, key):
    """Level, trend and threshold figures of one KPI, read from the evidence (no hard-coded results)."""
    level = next(k for k in T.check_thresholds(ctx)["kpis"] if k["kpi"] == key)["value_pct"]
    trend = next(t for t in T.analyse_trends(ctx)["trends"] if t["kpi"] == key)
    return level, trend


def _good_submission(ctx, outputs):
    th = T.check_thresholds(ctx)
    level, trend = _kpi(ctx, "recall")
    report = "\n".join(f"## {i}. {s}\ntext" for i, s in enumerate(A.REPORT_SECTIONS, 1))
    report += f"\nRecall is {level}% in {ctx['period']}, fell by {abs(trend['change_pp'])} pp."
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
    bad["report_markdown"] = bad["report_markdown"].replace(
        f"## {A.REPORT_SECTIONS.index('Root causes') + 1}. Root causes", "") + " Hit rate was 93.7%."
    bad["overall_status"] = "Green"
    bad["breaches"] = []
    ver = A.verify_report(bad, outputs, ctx)
    assert not ver["passed"]
    assert 93.7 in ver["untraceable_numbers"]
    assert not ver["checks"]["overall_status_correct"]
    assert not ver["checks"]["breaches_correct"]
    assert ver["checks"]["report_sections_present"] is False


def _consistency(setup, sid, sentence, change=None):
    ctx = {**_scenario(setup, sid)["ctx"], "proposed_change": change}
    _, outputs, _ = A.gather_evidence(ctx)
    return A.consistency_problems(sentence, outputs, ctx)


def test_kpi_value_pairing(setup):
    ctx = _scenario(setup, "S01")["ctx"]
    recall, _ = _kpi(ctx, "recall")
    type2, _ = _kpi(ctx, "type2_error")
    review, _ = _kpi(ctx, "review_rate")
    assert _consistency(setup, "S01", f"Recall is {recall}% and the Type\u202fII error rate is {type2}%.") == ([], [], [])
    pairing, _, _ = _consistency(setup, "S01", f"Recall is {review}% in Jun-26.")
    assert pairing  # that value is the manual review rate, not recall
    pairing, _, _ = _consistency(setup, "S01", "Recall moved by 4.4 pp.")
    assert pairing


def test_direction_matches_sign(setup):
    change = {"review_threshold": 60}
    impact = T.review_change_management(_scenario(setup, "S12")["ctx"], review_threshold=60)
    pp = impact["proposed_change"]["impact_pp"]["type2_error"]
    assert pp < 0  # lowering the review threshold reduces missed matches
    ok = f"Lowering the REVIEW threshold to 60 would reduce the Type II error by {abs(pp)} pp."
    bad = f"Lowering the REVIEW threshold to 60 is expected to increase the Type II error by {abs(pp)} pp."
    assert _consistency(setup, "S12", ok, change)[1] == []
    assert _consistency(setup, "S12", bad, change)[1]
    _, trend = _kpi(_scenario(setup, "S01")["ctx"], "recall")
    assert trend["change_pp"] < 0
    assert _consistency(setup, "S01", f"Recall dropped {abs(trend['change_pp'])} pp since Jan-26.")[1] == []
    assert _consistency(setup, "S01", f"Recall rose from {trend['first_pct']}% to {trend['last_pct']}%.")[1]


def test_percent_only_on_percentages(setup):
    top = T.get_root_causes(_scenario(setup, "S01")["ctx"])["root_causes"][0]
    assert _consistency(setup, "S01", f"{top['root_cause']} explains {top['share_pct']}% of exceptions.")[2] == []
    assert _consistency(setup, "S01", f"{top['root_cause']} explains {top['cases']}% of exceptions.")[2]


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


def test_independent_validation_tool(setup):
    results, master, _ = setup
    assert T.get_independent_validation(T.live_context(results, master))["available"] is False
    benchmark = pd.DataFrame([
        {"Algorithm": "Production algorithm", "precision": 0.99, "recall": 0.9, "type1_error": 0.01,
         "type2_error": 0.1, "incorrect_merges": 4},
        {"Algorithm": "Challenger A", "precision": 0.98, "recall": 0.92, "type1_error": 0.02, "type2_error": 0.08,
         "incorrect_merges": 4}])
    ctx = T.live_context(results, master, benchmark=benchmark)
    out = T.get_independent_validation(ctx)
    assert out["available"] and out["production"]["recall_pct"] == 90.0
    assert out["challengers_with_better_recall_and_no_more_incorrect_merges"] == ["Challenger A"]
    assert sum(r["current"] for r in out["calibration"]) == 2  # current setting in both calibration tables
    evidence, outputs, steps = A.gather_evidence(ctx)
    assert "get_independent_validation" in [s["tool"] for s in steps]
    assert len(A.SYSTEM_PROMPT) + len(evidence) < 14000  # still within the free-tier token budget


def test_main_model_preferred():
    models = [C.GROQ_MODEL] + C.GROQ_FALLBACK_MODELS
    A._cooldown.clear()
    assert A._pick_model(models, 0) == (C.GROQ_MODEL, 0.0)
    A._cooldown[C.GROQ_MODEL] = C.PRIMARY_MODEL_MAX_WAIT - 5          # short wait: keep the main model
    assert A._pick_model(models, 0)[0] == C.GROQ_MODEL
    A._cooldown[C.GROQ_MODEL] = C.PRIMARY_MODEL_MAX_WAIT + 30         # long wait: use the fallback
    assert A._pick_model(models, 0) == (models[1], 0.0)
    A._cooldown.clear()
    assert not any(m.startswith("qwen") for m in models)              # production models only
