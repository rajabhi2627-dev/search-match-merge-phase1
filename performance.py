"""
Performance evaluation of the matcher against the automatically known ground truth.

How each test record is classified
  TP  predicted MATCH    and the record truly belongs to that same entity
  FP  predicted MATCH    but the record is a new person OR the wrong entity was chosen (a false merge)
  TN  predicted NO MATCH and the record is a new person
  FN  predicted NO MATCH but the record truly belongs to an existing entity
  REVIEW  predicted MANUAL REVIEW - not an automatic decision, so it is neither correct nor incorrect.

Accuracy, Precision, Recall and F1 are calculated on the automatically decided records only
(TP + FP + TN + FN). The manual review rate is reported separately.

Hit rate, dispute rate, incorrect merge rate, Type I and Type II error follow the definitions of the
Search & Match validation report (see document_metrics).
"""
import random

import pandas as pd

import config as C
from data_generator import generate_monitoring_batches
from matching import match_record

TP, FP, TN, FN, REVIEW = "TP", "FP", "TN", "FN", "REVIEW"
DECISIONS = [C.MATCH, C.MANUAL_REVIEW, C.NO_MATCH]
TRUE_STATUSES = [C.MATCH, C.NO_MATCH]


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classify_outcome(decision, predicted_entity_id, true_status, true_entity_id, review_as=None):
    """Return TP / FP / TN / FN / REVIEW for one record.

    review_as=None       -> MANUAL REVIEW stays REVIEW (default, used in the dashboard)
    review_as="NO MATCH" -> MANUAL REVIEW is treated as "not automatically matched" (sensitivity view)
    """
    if decision == C.MANUAL_REVIEW:
        if review_as is None:
            return REVIEW
        decision = review_as
    if decision == C.MATCH:
        correct_entity = (true_status == C.MATCH and predicted_entity_id is not None
                          and not pd.isna(predicted_entity_id) and int(predicted_entity_id) == int(true_entity_id))
        return TP if correct_entity else FP
    return TN if true_status == C.NO_MATCH else FN


def outcome_label(outcome):
    return {TP: "Correct", TN: "Correct", FP: "Incorrect", FN: "Incorrect", REVIEW: "Manual Review"}[outcome]


def compute_metrics(outcomes):
    """outcomes: iterable of TP / FP / TN / FN / REVIEW strings."""
    outcomes = list(outcomes)
    tp, fp, tn, fn = (outcomes.count(x) for x in (TP, FP, TN, FN))
    review = outcomes.count(REVIEW)
    decided = tp + fp + tn + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "review": review,
        "total": len(outcomes),
        "auto_decided": decided,
        "accuracy": safe_div(tp + tn, decided),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "review_rate": safe_div(review, len(outcomes)),
    }


def run_test_matching(test_records, master, comparators=None):
    """Run the SAME match_record() used by the Search & Match page on every test record."""
    rows = []
    for rec in test_records.to_dict("records"):
        raw_input = {f: rec[f] for f in C.FIELDS}
        result = match_record(raw_input, master, comparators)
        best = result["best"]
        true_entity = None if pd.isna(rec["true_entity_id"]) else int(rec["true_entity_id"])
        code = classify_outcome(result["decision"], result["predicted_entity_id"],
                                rec["true_match_status"], true_entity)
        row = {
            "test_record_id": rec["test_record_id"],
            "source_id": rec.get("source_id", ""),
            **{f"input_{f}": rec[f] for f in C.FIELDS},
            "scenario": rec["scenario"],
            "true_entity_id": true_entity,
            "true_match_status": rec["true_match_status"],
            "best_candidate_id": best["entity_id"] if best else None,
            "best_candidate_name": best["name"] if best else None,
            "predicted_entity_id": result["predicted_entity_id"],
            "match_score": best["overall"] if best else 0.0,
            "second_score": result["second_score"],
            "score_gap": result["score_gap"],
            "evidence_weight": best["evidence_weight"] if best else 0.0,
            "limited_evidence": result["limited_evidence"],
            "candidates_found": result["candidates_found"],
            "search_path": "; ".join(best["search_reasons"]) if best else "",
            "decision": result["decision"],
            "outcome_code": code,
            "outcome": outcome_label(code),
            "conflicts": "; ".join(result["conflicts"]),
            "decision_reasons": "; ".join(result["decision_reasons"]),
        }
        for f in C.FIELDS:
            row[f"score_{f}"] = best["field_scores"][f] if best else None
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["true_entity_id", "predicted_entity_id", "best_candidate_id"]:
        df[col] = df[col].astype("Int64")
    return df


def metrics_for_results(results, review_as=None):
    outcomes = [
        classify_outcome(r.decision, r.predicted_entity_id, r.true_match_status, r.true_entity_id, review_as)
        for r in results.itertuples()
    ]
    return compute_metrics(outcomes)


def reviews_as_no_match(results):
    """Copy of the results in which every MANUAL REVIEW is treated as NO MATCH (sensitivity view)."""
    out = results.copy()
    out.loc[out["decision"] == C.MANUAL_REVIEW, "decision"] = C.NO_MATCH
    out["outcome_code"] = [classify_outcome(r.decision, r.predicted_entity_id, r.true_match_status,
                                            r.true_entity_id) for r in out.itertuples()]
    return out


def document_metrics(results, disputed=None):
    """Metrics exactly as defined in the Search & Match validation report.

    Matches generated    = MATCH + MANUAL REVIEW decisions (the algorithm proposes an existing person)
    Confirmed matches    = generated matches whose best candidate is the true person
    Hit rate             = confirmed matches / total matches generated
    Dispute rate         = disputed matches / total matches generated
    Incorrect merge rate = incorrect merges (FP) / total matches generated
    Type I error rate    = false positives / total matches generated
    Type II error rate   = false negatives / total actual matches (records that truly exist in the master)

    disputed: optional boolean Series; defaults to the 'disputed' column of the results.
    """
    if disputed is None:
        disputed = results["disputed"] if "disputed" in results else pd.Series(False, index=results.index)
    disputed = disputed.astype(bool)
    generated = results["decision"].isin([C.MATCH, C.MANUAL_REVIEW])
    confirmed = generated & (results["best_candidate_id"] == results["true_entity_id"]).fillna(False).astype(bool)
    codes = results["outcome_code"]
    fp, fn = int((codes == FP).sum()), int((codes == FN).sum())
    disputed_matches = int((disputed & (results["decision"] == C.MATCH)).sum())
    n_generated = int(generated.sum())
    actual_matches = int((results["true_match_status"] == C.MATCH).sum())
    m = compute_metrics(codes)
    return {
        "records": len(results),
        "matches_generated": n_generated,
        "auto_merged": int((results["decision"] == C.MATCH).sum()),
        "manual_review": int((results["decision"] == C.MANUAL_REVIEW).sum()),
        "new_subjects": int((results["decision"] == C.NO_MATCH).sum()),
        "actual_matches": actual_matches,
        "confirmed": int(confirmed.sum()),
        "disputed_matches": disputed_matches,
        "duplicate_disputes": int((disputed & (codes == FN)).sum()),
        "incorrect_merges": fp,
        "hit_rate": safe_div(int(confirmed.sum()), n_generated),
        "dispute_rate": safe_div(disputed_matches, n_generated),
        "incorrect_merge_rate": safe_div(fp, n_generated),
        "type1_error": safe_div(fp, n_generated),
        "type2_error": safe_div(fn, actual_matches),
        "precision": m["precision"],
        "recall": m["recall"],
        "review_rate": m["review_rate"],
        "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"], "review": m["review"],
    }


def binary_confusion_matrix(metrics):
    """2x2 matrix. Rows = actual (should match / should not match), columns = predicted (MATCH / NO MATCH).

    A MATCH to the wrong entity is counted as FP and shown in the bottom-left cell together with
    matches of new persons, because in both cases an incorrect merge would be made.
    """
    return [[metrics["TP"], metrics["FN"]],
            [metrics["FP"], metrics["TN"]]]


def decision_distribution(results):
    """Full 2 x 3 table: true status (rows) vs predicted decision (columns)."""
    table = pd.crosstab(results["true_match_status"], results["decision"])
    return table.reindex(index=TRUE_STATUSES, columns=DECISIONS, fill_value=0)


def scenario_summary(results):
    """Simple per-scenario breakdown of decisions."""
    summary = results.groupby("scenario").agg(
        records=("test_record_id", "count"),
        true_status=("true_match_status", "first"),
        correct=("outcome", lambda s: (s == "Correct").sum()),
        incorrect=("outcome", lambda s: (s == "Incorrect").sum()),
        manual_review=("outcome", lambda s: (s == "Manual Review").sum()),
        avg_score=("match_score", "mean"),
    )
    summary["avg_score"] = summary["avg_score"].round(1)
    return summary.sort_index().reset_index()


# ==================================================================
# WEEK 4 - PERFORMANCE MONITORING KPIs
# ==================================================================

KPI_DICTIONARY = {
    "hit_rate": {
        "label": "Hit Rate",
        "meaning": "Of the matches the algorithm proposed, how many were confirmed as the right person",
        "formula": "confirmed matches / total matches generated",
    },
    "precision": {
        "label": "Precision",
        "meaning": "Of the automatic matches, how many were the right entity",
        "formula": "TP / (TP + FP)",
    },
    "recall": {
        "label": "Recall",
        "meaning": "Of the existing people (automatically decided), how many were found",
        "formula": "TP / (TP + FN)",
    },
    "type1_error": {
        "label": "Type I Error Rate",
        "meaning": "False positives: two different people matched",
        "formula": "false positives / total matches generated",
    },
    "type2_error": {
        "label": "Type II Error Rate",
        "meaning": "False negatives: an existing person not matched",
        "formula": "false negatives / total actual matches",
    },
    "review_rate": {
        "label": "Manual Review Rate",
        "meaning": "Workload sent to human reviewers",
        "formula": "MANUAL REVIEW decisions / total records",
    },
    "incorrect_merge_rate": {
        "label": "Incorrect Merge Rate",
        "meaning": "Automatic matches that merged two different people",
        "formula": "incorrect merges / total matches generated",
    },
    "dispute_rate": {
        "label": "Dispute Rate",
        "meaning": "Matches disputed by the customer ('does not belong to me')",
        "formula": "disputed matches / total matches generated",
    },
}


def simulate_disputes(results, period=""):
    """Deterministic dispute simulation: a wrong decision becomes a customer dispute with the
    probability in DISPUTE_RAISE_RATE. Each record has its own fixed random seed."""
    raised = []
    for r in results.itertuples():
        rate = C.DISPUTE_RAISE_RATE.get(r.outcome_code, 0.0)
        raised.append(rate > 0 and random.Random(f"{period}|{r.test_record_id}").random() < rate)
    return pd.Series(raised, index=results.index, dtype=bool)


def compute_kpis(results, period=""):
    return document_metrics(results, simulate_disputes(results, period))


def rag_status(kpi, value):
    t = C.KPI_THRESHOLDS[kpi]
    if t["direction"] == "higher":
        return "Green" if value >= t["green"] else "Amber" if value >= t["amber"] else "Red"
    return "Green" if value <= t["green"] else "Amber" if value <= t["amber"] else "Red"


def run_monitoring(master, comparators=None):
    """Match one batch of 100 new records per month, using the same match_record() as everywhere else."""
    frames = []
    for period, records in generate_monitoring_batches().items():
        results = run_test_matching(pd.DataFrame(records), master, comparators)
        results.insert(0, "period", period)
        results["disputed"] = simulate_disputes(results, period)
        frames.append(results)
    return pd.concat(frames, ignore_index=True)


def monthly_kpis(monitoring_results):
    rows = []
    for period in C.MONITORING_PERIODS:
        batch = monitoring_results[monitoring_results["period"] == period]
        if len(batch):
            rows.append({"period": period, **compute_kpis(batch, period)})
    return pd.DataFrame(rows)


def trend_observations(kpis, min_change=0.02):
    """Rule-based observations: KPIs that are not Green in the latest month, or moved noticeably."""
    first, last = kpis.iloc[0], kpis.iloc[-1]
    notes = []
    for key, info in KPI_DICTIONARY.items():
        start, end = first[key], last[key]
        status = rag_status(key, end)
        change = end - start
        if status == "Green" and abs(change) < min_change:
            continue
        direction = "rose" if change > 0 else "fell" if change < 0 else "stayed"
        movement = f"{direction} from {start:.1%} to {end:.1%}" if direction != "stayed" else f"stayed at {end:.1%}"
        notes.append({"KPI": info["label"], "Status": status,
                      "Observation": f"{info['label']} {movement} between {first['period']} and {last['period']}."})
    order = {"Red": 0, "Amber": 1, "Green": 2}
    return sorted(notes, key=lambda n: order[n["Status"]])
