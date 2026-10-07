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
    assert M.normalize_text("  Rachael   DENT. ") == "rachael dent"
    assert M.normalize_text(None) == ""


def test_id_normalization():
    for raw in ["5304218", "530 4218", "530-4218", " 5304218 "]:
        assert M.normalize_id(raw) == "5304218"


def test_dob_normalization():
    assert M.normalize_dob("19980412") == "1998-04-12"
    assert M.normalize_dob("12/04/1998") == "1998-04-12"
    assert M.normalize_dob("1998-04-12") == "1998-04-12"
    assert M.normalize_dob("19981341") == ""  # invalid date -> missing
    assert M.normalize_dob("not a date") == ""


def test_address_normalization():
    assert M.normalize_address("12 Banjine St, 2600") == "12 banjine street 2600"


# ---------------- field comparison ----------------

def test_exact_matching():
    assert M.compare_field("soc_sec_id", "5304218", "5304218")["score"] == 100
    assert M.compare_field("city", "kellerberrin", "kellerberrin")["score"] == 100


def test_name_fuzzy_matching():
    assert M.compare_name("rachael dent", "rachael dnet")["score"] >= 90
    assert M.compare_name("rachael dent", "isabella white")["score"] == 0


def test_id_digit_matching():
    assert M.compare_id("5304218", "5304219")["score"] == 90   # one digit changed
    assert M.compare_id("5304218", "530421")["score"] == 90    # one digit missing
    assert M.compare_id("5304218", "5302418")["score"] == 75   # two digits swapped
    assert M.compare_id("5304218", "9123456")["score"] == 0    # different ID


def test_address_fuzzy_matching():
    assert M.compare_address("1 banjine street 2600", "banjine street 1 2600")["score"] == 100
    assert M.compare_address("1 banjine street 2600", "4 banjine street 2600")["score"] >= 90


def test_dob_matching():
    assert M.compare_field("date_of_birth", "1998-04-12", "1998-04-12")["score"] == 100
    assert M.compare_field("date_of_birth", "1998-04-12", "1998-04-13")["score"] == 0
    assert M.compare_field("date_of_birth", "", "1998-04-12")["score"] is None


# ---------------- score, conflicts, decision ----------------

def test_score_calculation():
    scores = {"name": 100, "soc_sec_id": 100, "address": 50, "date_of_birth": 100, "city": 100}
    overall, evidence, _ = M.weighted_score(scores)
    assert overall == pytest.approx(35 + 35 + 7.5 + 7.5 + 7.5)
    assert evidence == 1.0


def test_missing_field_reweighting():
    scores = {"name": 100, "soc_sec_id": 100, "address": 100, "date_of_birth": None, "city": 100}
    overall, evidence, _ = M.weighted_score(scores)
    assert overall == pytest.approx(100)
    assert evidence == pytest.approx(0.925)
    _, evidence, _ = M.weighted_score({**scores, "soc_sec_id": None})
    assert evidence < C.MIN_EVIDENCE_WEIGHT  # without the ID a MATCH is not allowed


def test_conflict_rules():
    scores = {"name": 95, "soc_sec_id": 0, "address": 80, "date_of_birth": 100, "city": 100}
    assert any("Strong identifiers" in c for c in M.candidate_conflicts(scores))
    scores = {"name": 90, "soc_sec_id": 100, "address": 80, "date_of_birth": 0, "city": 100}
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
    assert "email" not in record and "company" not in record
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
