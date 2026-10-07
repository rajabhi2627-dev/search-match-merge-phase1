import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config as C
import data_generator as G
import matching as M
import performance as P


@pytest.fixture(scope="module")
def master():
    return M.prepare_master(G.generate_master_records())


# ---------------- normalisation ----------------

def test_text_normalization():
    assert M.normalize_text("  Rahul   KUMAR. ") == "rahul kumar"
    assert M.normalize_text(None) == ""


def test_phone_normalization():
    for raw in ["+91 98765 43210", "098765-43210", "0091 9876543210", "(98765) 43210", "9876543210"]:
        assert M.normalize_phone(raw) == "9876543210"


def test_email_normalization():
    assert M.normalize_email("  Rahul.Kumar @GMAIL.com ") == "rahul.kumar@gmail.com"


def test_dob_normalization():
    assert M.normalize_dob("12/04/1998") == "1998-04-12"
    assert M.normalize_dob("12 Apr 1998") == "1998-04-12"
    assert M.normalize_dob("19980412") == "1998-04-12"
    assert M.normalize_dob("not a date") == ""


# ---------------- field comparison ----------------

def test_exact_matching():
    assert M.compare_field("email", "a.b@gmail.com", "a.b@gmail.com")["score"] == 100
    assert M.compare_field("phone", "9876543210", "9876543210")["score"] == 100


def test_name_fuzzy_matching():
    assert M.compare_name("rahul kumar", "rahul kumr")["score"] >= 90
    assert M.compare_name("rahul kumar", "sneha iyer")["score"] == 0


def test_email_fuzzy_matching():
    assert M.compare_email("rahul.kumar@gmail.com", "rahul.kumar@gmai.com")["score"] >= 90


def test_phone_digit_matching():
    assert M.compare_phone("9876543210", "9876543211")["score"] == 90
    assert M.compare_phone("9876543210", "9876543201")["score"] == 75
    assert M.compare_phone("9876543210", "9123456789")["score"] == 0


def test_address_fuzzy_matching():
    a = M.normalize_address("12 MG Road Patna")
    b = M.normalize_address("12 Mahatma Gandhi Road Patna")
    assert M.compare_address(a, b)["score"] == 100
    assert M.compare_address("12 boring road patna", "patna boring road 12")["score"] == 100


def test_company_fuzzy_matching():
    a, b = M.normalize_company("ABC Technologies Pvt Ltd"), M.normalize_company("ABC Tech")
    assert M.compare_company(a, b)["score"] >= C.FIELD_CONFIG["company"]["threshold"]


def test_dob_matching():
    assert M.compare_field("date_of_birth", "1998-04-12", "1998-04-12")["score"] == 100
    assert M.compare_field("date_of_birth", "1998-04-12", "1998-04-13")["score"] == 0
    assert M.compare_field("date_of_birth", "", "1998-04-12")["score"] is None


# ---------------- score, conflicts, decision ----------------

def test_score_calculation():
    scores = {"name": 100, "email": 100, "phone": 100, "address": 50, "date_of_birth": 100, "company": 0, "city": 100}
    overall, evidence, _ = M.weighted_score(scores)
    assert overall == pytest.approx(25 + 25 + 25 + 5 + 5 + 0 + 5)
    assert evidence == 1.0


def test_missing_field_reweighting():
    scores = {"name": 100, "email": 100, "phone": 100, "address": 100, "date_of_birth": None, "company": 100, "city": 100}
    overall, evidence, _ = M.weighted_score(scores)
    assert overall == pytest.approx(100)
    assert evidence == pytest.approx(0.95)


def test_conflict_rules():
    scores = {"name": 95, "email": 20, "phone": 100, "address": 80, "date_of_birth": 100, "company": 80, "city": 100}
    assert any("Strong identifiers" in c for c in M.candidate_conflicts(scores))
    scores = {"name": 90, "email": 100, "phone": 100, "address": 80, "date_of_birth": 0, "company": 80, "city": 100}
    assert any("Date of birth" in c for c in M.candidate_conflicts(scores))


def test_decision_rules():
    assert M.make_decision(95, 30, True, [], False)[0] == C.MATCH
    assert M.make_decision(75, 30, True, [], False)[0] == C.MANUAL_REVIEW
    assert M.make_decision(60, 30, True, [], False)[0] == C.NO_MATCH
    assert M.make_decision(95, 30, True, ["conflict"], False)[0] == C.MANUAL_REVIEW
    assert M.make_decision(95, 30, False, [], True)[0] == C.MANUAL_REVIEW


def test_score_gap_logic():
    assert M.make_decision(91, 2, True, [], False)[0] == C.MANUAL_REVIEW
    assert M.make_decision(91, 2, False, [], False)[0] == C.MATCH  # no meaningful second candidate


def test_end_to_end_master_record_matches_itself(master):
    item = master[0]
    record = {f: item["raw"][f] for f in C.FIELDS}
    result = M.match_record(record, master)
    assert result["decision"] == C.MATCH
    assert result["predicted_entity_id"] == item["entity_id"]


def test_febrl_record_mapping():
    record = G.generate_master_records()[0]
    assert set(C.FIELDS) <= set(record)
    assert record["email"] == "" and record["company"] == ""
    assert M.normalize_dob(record["date_of_birth"]) != ""


# ---------------- data and metrics ----------------

def test_test_data_generation(master):
    records = G.generate_test_records()
    master_ids = {m["entity_id"] for m in master}
    assert len(records) == C.TEST_RECORD_COUNT
    matches = [r for r in records if r["true_match_status"] == C.MATCH]
    assert len(matches) == C.TEST_MATCH_COUNT
    assert all(r["true_entity_id"] in master_ids for r in matches)
    assert all(r["true_entity_id"] is None for r in records if r["true_match_status"] == C.NO_MATCH)
    assert records == G.generate_test_records()  # reproducible


def test_monitoring_batches_do_not_overlap_test_set():
    test_ids = {r["source_id"] for r in G.generate_test_records()}
    batches = G.generate_monitoring_batches()
    assert list(batches) == C.MONITORING_PERIODS
    seen = set()
    for records in batches.values():
        assert len(records) == C.MONITORING_MATCH_COUNT + C.MONITORING_NO_MATCH_COUNT
        ids = {r["source_id"] for r in records}
        assert not ids & test_ids and not ids & seen
        seen |= ids


def test_rag_status():
    assert P.rag_status("recall", 0.97) == "Green"
    assert P.rag_status("recall", 0.92) == "Amber"
    assert P.rag_status("recall", 0.80) == "Red"
    assert P.rag_status("dispute_rate", 0.0) == "Green"
    assert P.rag_status("dispute_rate", 0.05) == "Red"


def test_performance_metrics():
    m = P.compute_metrics(["TP"] * 8 + ["FP"] * 2 + ["TN"] * 6 + ["FN"] * 4 + ["REVIEW"] * 5)
    assert m["accuracy"] == pytest.approx(14 / 20)
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(8 / 12)
    assert m["fpr"] == pytest.approx(2 / 8)
    assert m["fnr"] == pytest.approx(4 / 12)
    assert m["review_rate"] == pytest.approx(5 / 25)
    assert P.compute_metrics([])["precision"] == 0.0  # division by zero handled


def test_wrong_entity_match_is_false_positive():
    assert P.classify_outcome(C.MATCH, 10002, C.MATCH, 10001) == "FP"
    assert P.classify_outcome(C.MATCH, 10001, C.MATCH, 10001) == "TP"
    assert P.classify_outcome(C.MANUAL_REVIEW, None, C.MATCH, 10001) == "REVIEW"
