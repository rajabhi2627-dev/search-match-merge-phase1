"""
Independent validation of the Search & Match algorithm, following the structure of a model validation report.

Validation sample: the 6 monthly monitoring batches (600 FEBRL records with known ground truth).
Each month plays the role of a reporting period.

Analyses
  1. Validation data summary
  2. Data quality checks (completeness and validity)
  3. Hit rate and dispute rate by period, with trend
  4. Search path analysis (which search rule found the match, which attributes drove the score)
  5. Challenger algorithm benchmarking (Jaro-Winkler, Levenshtein, Damerau-Levenshtein, Soundex, Jaccard)
  6. Type I and Type II error assessment
  7. Dispute and incorrect merge analysis
  8. Root cause analysis of exceptions, with grouping by error pattern
  9. Threshold calibration (sensitivity of the results to the decision thresholds)
 10. Rule-based validation observations

Everything is deterministic and rule-based - no machine learning.
"""
import re

import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

import config as C
import matching as M
import performance as P

# All rates follow the validation report's formulas - see performance.document_metrics.
METRIC_DEFINITIONS = pd.DataFrame([
    {"Metric": "Matches generated", "Formula": "MATCH + MANUAL REVIEW decisions"},
    {"Metric": "Confirmed matches", "Formula": "generated matches whose best candidate is the true person"},
    {"Metric": "Hit rate", "Formula": "confirmed matches / total matches generated × 100"},
    {"Metric": "Dispute rate", "Formula": "disputed matches / total matches generated × 100"},
    {"Metric": "Incorrect merge rate", "Formula": "incorrect merges / total matches generated × 100"},
    {"Metric": "Type I error rate", "Formula": "false positives / total matches generated × 100"},
    {"Metric": "Type II error rate", "Formula": "false negatives / total actual matches × 100"},
])


# Display settings shared by the Validation page and the PDF report.
PERCENT_COLUMNS = ["hit_rate", "dispute_rate", "incorrect_merge_rate", "type1_error", "type2_error",
                   "precision", "recall", "review_rate", "Share of matches", "Contribution", "Configured weight",
                   "Incoming completeness", "Incoming validity", "Master completeness", "Master validity"]
REPORT_LABELS = {
    "period": "Period", "records": "Records processed", "matches_generated": "Matches generated",
    "confirmed": "Confirmed matches", "auto_merged": "Auto-merged", "manual_review": "Manual review",
    "disputed_matches": "Disputed matches", "duplicate_disputes": "Missed-duplicate disputes",
    "incorrect_merges": "Incorrect merges", "hit_rate": "Hit rate", "dispute_rate": "Dispute rate",
    "incorrect_merge_rate": "Incorrect merge rate", "type1_error": "Type I error (FP)",
    "type2_error": "Type II error (FN)", "precision": "Precision", "recall": "Recall",
    "review_rate": "Manual review rate",
}


def by_period(results):
    rows = []
    for period in C.MONITORING_PERIODS:
        batch = results[results["period"] == period]
        if len(batch):
            rows.append({"period": period, **P.document_metrics(batch)})
    return pd.DataFrame(rows)


# ==================================================================
# 1. Validation data summary
# ==================================================================

def data_summary(results, master_count):
    m = P.document_metrics(results)
    return pd.DataFrame([
        ("Master records (existing subjects)", master_count),
        ("Customer records processed", m["records"]),
        ("Potential matches generated (MATCH + MANUAL REVIEW)", m["matches_generated"]),
        ("Auto-merged records (MATCH)", m["auto_merged"]),
        ("Manual review cases", m["manual_review"]),
        ("New subjects created (NO MATCH)", m["new_subjects"]),
        ("Disputes: 'does not belong to me' (wrong merges)", m["disputed_matches"]),
        ("Disputes: 'I already have an account' (missed duplicates)", m["duplicate_disputes"]),
        ("Incorrect merge cases", m["incorrect_merges"]),
    ], columns=["Data category", "No. of records"])


# ==================================================================
# 2. Data quality checks
# ==================================================================

def _valid_id(value):
    return bool(re.fullmatch(rf"\d{{{C.ID_EXPECTED_LENGTH}}}", M.normalize_id(value)))


VALIDITY_RULES = {
    "name": ("Has first name and surname", lambda v: len(M.normalize_text(v).split()) >= 2),
    "soc_sec_id": (f"Digits only, {C.ID_EXPECTED_LENGTH} digits", _valid_id),
    "date_of_birth": ("Valid calendar date", lambda v: M.normalize_dob(v) != ""),
    "address": ("Contains a street number or postcode", lambda v: bool(re.search(r"\d", str(v)))),
    "city": ("Letters only", lambda v: bool(re.fullmatch(r"[a-z ]+", M.normalize_text(v)))),
}


def data_quality(incoming, master_df):
    """Completeness and validity of every matching attribute, for incoming and master records."""
    rows = []
    for f in C.FIELDS:
        rule, check = VALIDITY_RULES[f]
        row = {"Attribute": C.FIELD_LABELS[f], "Validity rule": rule}
        for label, values in (("Incoming", incoming[f"input_{f}"]), ("Master", master_df[f])):
            values = values.fillna("").astype(str).str.strip()
            filled = values[values != ""]
            row[f"{label} completeness"] = P.safe_div(len(filled), len(values))
            row[f"{label} validity"] = P.safe_div(sum(check(v) for v in filled), len(filled))
        rows.append(row)
    return pd.DataFrame(rows)


# ==================================================================
# 4. Search path analysis
# ==================================================================

SEARCH_RULES = {
    "Exact Soc. Sec. ID": "exact soc. sec. id",
    "Similar name": "similar name",
    "Same city + partly similar name": "same city",
}


def search_path_analysis(results):
    """For correct automatic matches: which search rule(s) found the right person."""
    tp = results[results["outcome_code"] == P.TP]
    paths = tp["search_path"].fillna("").str.lower()
    hits = {rule: paths.str.contains(key, regex=False) for rule, key in SEARCH_RULES.items()}
    rows = []
    for rule, found in hits.items():
        others = pd.concat([h for r, h in hits.items() if r != rule], axis=1).any(axis=1)
        rows.append({"Search rule": rule, "Matches found by this rule": int(found.sum()),
                     "Share of matches": P.safe_div(int(found.sum()), len(tp)),
                     "Found ONLY by this rule": int((found & ~others).sum())})
    return pd.DataFrame(rows)


def attribute_contribution(results):
    """Share of the weighted match score contributed by each attribute, over correct automatic matches."""
    tp = results[results["outcome_code"] == P.TP]
    totals = {f: 0.0 for f in C.FIELDS}
    for _, r in tp.iterrows():
        present = [f for f in C.FIELDS if pd.notna(r[f"score_{f}"])]
        weight = sum(C.FIELD_WEIGHTS[f] for f in present)
        for f in present:
            totals[f] += r[f"score_{f}"] * C.FIELD_WEIGHTS[f] / weight
    grand = sum(totals.values())
    return pd.DataFrame([{"Matching attribute": C.FIELD_LABELS[f], "Configured weight": C.FIELD_WEIGHTS[f],
                          "Contribution": P.safe_div(totals[f], grand)} for f in C.FIELDS])


# ==================================================================
# 5. Challenger algorithm benchmarking
# ==================================================================

def soundex(word):
    """American Soundex code: first letter + 3 digits for the consonant sounds ("robert" -> R163)."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
             "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    out, prev = word[0].upper(), codes.get(word[0], "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            out += code
        if ch not in "hw":
            prev = code
    return (out + "000")[:4]


def soundex_similarity(a, b):
    """Share of name words that sound the same (word order ignored)."""
    codes_a, codes_b = {soundex(w) for w in a.split()}, {soundex(w) for w in b.split()}
    return 100 * len(codes_a & codes_b) / max(len(codes_a), len(codes_b), 1)


def jaccard_similarity(a, b):
    """Shared words / all distinct words."""
    set_a, set_b = set(a.split()), set(b.split())
    return 100 * len(set_a & set_b) / max(len(set_a | set_b), 1)


def _challenger(field, name, similarity):
    cutoff = C.CHALLENGER_CUTOFFS[name]

    def compare(a, b):
        sim = 100.0 if a == b else round(float(similarity(a, b)), 1)
        agrees = sim >= cutoff
        result = M._result(field, sim if agrees else 0.0, sim, None, agrees,
                           f"{name}: similarity {sim} vs cut-off {cutoff}")
        result["method"] = name
        return result
    return compare


PRODUCTION = "Production algorithm"
CHALLENGERS = {
    "Jaro-Winkler (name)": {"name": _challenger(
        "name", "Jaro-Winkler (name)", lambda a, b: JaroWinkler.normalized_similarity(a, b) * 100)},
    "Levenshtein (name)": {"name": _challenger(
        "name", "Levenshtein (name)", lambda a, b: Levenshtein.normalized_similarity(a, b) * 100)},
    "Damerau-Levenshtein (name)": {"name": _challenger(
        "name", "Damerau-Levenshtein (name)", lambda a, b: DamerauLevenshtein.normalized_similarity(a, b) * 100)},
    "Soundex (name)": {"name": _challenger("name", "Soundex (name)", soundex_similarity)},
    "Jaccard (address)": {"address": _challenger("address", "Jaccard (address)", jaccard_similarity)},
}


def run_benchmark(master, production_results):
    """Run every challenger on the same validation sample and summarise the results."""
    rows = [{"Algorithm": PRODUCTION, **P.document_metrics(production_results)}]
    for name, comparators in CHALLENGERS.items():
        rows.append({"Algorithm": name, **P.document_metrics(P.run_monitoring(master, comparators))})
    return pd.DataFrame(rows)


# ==================================================================
# 8. Root cause analysis of exceptions
# ==================================================================

ROOT_CAUSE_ORDER = [
    "Similar customer names",
    "Missing Soc. Sec. ID",
    "Soc. Sec. ID mismatch",
    "Name replaced or very different",
    "Missing or different Date of Birth",
    "Address variation",
    "Data entry errors (minor typos)",
]


def _root_cause(r, master_norm):
    """Primary root cause and the list of affected attributes for one exception record."""
    if r["true_match_status"] == C.NO_MATCH:
        return "Similar customer names", "new person resembles an existing person"
    incoming = M.normalize_record({f: r[f"input_{f}"] for f in C.FIELDS})
    true_rec = master_norm[int(r["true_entity_id"])]
    scores = {f: M.compare_field(f, incoming[f], true_rec[f])["score"] for f in C.FIELDS}
    affected = [C.FIELD_LABELS[f] + (" (missing)" if not incoming[f] else "")
                for f in C.FIELDS if scores[f] != 100]
    wrong_candidate = (r["decision"] != C.NO_MATCH and pd.notna(r["best_candidate_id"])
                       and int(r["best_candidate_id"]) != int(r["true_entity_id"]))
    if wrong_candidate:
        cause = "Similar customer names"
    elif not incoming["soc_sec_id"]:
        cause = "Missing Soc. Sec. ID"
    elif scores["soc_sec_id"] == 0:
        cause = "Soc. Sec. ID mismatch"
    elif scores["name"] == 0:
        cause = "Name replaced or very different"
    elif scores["date_of_birth"] in (None, 0):
        cause = "Missing or different Date of Birth"
    elif scores["address"] in (None, 0):
        cause = "Address variation"
    else:
        cause = "Data entry errors (minor typos)"
    return cause, ", ".join(affected) or "none"


def root_cause_analysis(results, master):
    """Exceptions = wrong merges (FP), missed duplicates (FN) and manual reviews."""
    master_norm = {m["entity_id"]: m["norm"] for m in master}
    exceptions = results[results["outcome_code"].isin([P.FP, P.FN, P.REVIEW])].copy()
    causes = [_root_cause(r, master_norm) for _, r in exceptions.iterrows()]
    exceptions["root_cause"] = [c for c, _ in causes]
    exceptions["attributes_affected"] = [a for _, a in causes]
    return exceptions


def root_cause_summary(exceptions):
    counts = exceptions["root_cause"].value_counts().reindex(ROOT_CAUSE_ORDER, fill_value=0)
    table = pd.DataFrame({"Root cause": counts.index, "Number of cases": counts.values})
    for code, label in ((P.FN, "Missed (FN)"), (P.REVIEW, "Manual review"), (P.FP, "Wrong merge (FP)")):
        sub = exceptions[exceptions["outcome_code"] == code]["root_cause"].value_counts()
        table[label] = table["Root cause"].map(sub).fillna(0).astype(int)
    table["Disputed"] = table["Root cause"].map(
        exceptions[exceptions["disputed"]]["root_cause"].value_counts()).fillna(0).astype(int)
    table["Contribution"] = table["Number of cases"] / max(len(exceptions), 1)
    table = table[table["Number of cases"] > 0]
    return table.sort_values("Number of cases", ascending=False, kind="stable").reset_index(drop=True)


def error_patterns(exceptions):
    """Group exceptions that share exactly the same set of affected attributes (rule-based clustering)."""
    table = (exceptions.groupby("attributes_affected")
             .agg(cases=("test_record_id", "count"),
                  missed=("outcome_code", lambda s: (s == P.FN).sum()),
                  review=("outcome_code", lambda s: (s == P.REVIEW).sum()),
                  avg_score=("match_score", "mean"))
             .sort_values("cases", ascending=False).reset_index())
    table["avg_score"] = table["avg_score"].round(1)
    return table.rename(columns={"attributes_affected": "Attributes that differ from the true record",
                                 "cases": "Cases", "missed": "Missed (FN)", "review": "Manual review",
                                 "avg_score": "Avg match score"})


# ==================================================================
# 9. Threshold calibration
# ==================================================================

def redecide(results, match_threshold, review_threshold):
    """Recompute every decision with other thresholds (search, scores and conflicts are unchanged)."""
    rows = []
    for r in results.itertuples():
        if r.candidates_found == 0:
            decision = C.NO_MATCH
        else:
            second = r.second_score if pd.notna(r.second_score) else 0.0
            meaningful = r.candidates_found > 1 and second >= review_threshold
            decision, _ = M.make_decision(r.match_score, r.score_gap, meaningful, bool(r.conflicts),
                                          bool(r.limited_evidence), match_threshold, review_threshold,
                                          corroborated=bool(getattr(r, "corroborated", False)))
        predicted = r.best_candidate_id if decision == C.MATCH else None
        code = P.classify_outcome(decision, predicted, r.true_match_status, r.true_entity_id)
        rows.append((decision, predicted, code))
    out = results.copy()
    out["decision"] = [d for d, _, _ in rows]
    out["predicted_entity_id"] = pd.array([p for _, p, _ in rows], dtype="Int64")
    out["outcome_code"] = [c for _, _, c in rows]
    out["disputed"] = False
    return out


def threshold_calibration(results):
    rows = []
    for mt in C.CALIBRATION_MATCH_THRESHOLDS:
        rows.append({"Varied": "MATCH threshold", "MATCH threshold": mt, "REVIEW threshold": C.REVIEW_THRESHOLD,
                     **P.document_metrics(redecide(results, mt, C.REVIEW_THRESHOLD))})
    for rt in C.CALIBRATION_REVIEW_THRESHOLDS:
        rows.append({"Varied": "REVIEW threshold", "MATCH threshold": C.MATCH_THRESHOLD, "REVIEW threshold": rt,
                     **P.document_metrics(redecide(results, C.MATCH_THRESHOLD, rt))})
    table = pd.DataFrame(rows)
    table["Current setting"] = ((table["MATCH threshold"] == C.MATCH_THRESHOLD)
                                & (table["REVIEW threshold"] == C.REVIEW_THRESHOLD))
    return table


# ==================================================================
# 10. Validation observations (rule-based)
# ==================================================================

def observations(periods, contribution, benchmark, rca, quality):
    L = C.VALIDATION_LIMITS
    notes = []

    def add(area, ok, text):
        notes.append({"Area": area, "Status": "Satisfactory" if ok else "Needs attention", "Observation": text})

    first, last = periods.iloc[0], periods.iloc[-1]
    add("Hit rate", periods["hit_rate"].min() >= C.KPI_THRESHOLDS["hit_rate"]["amber"],
        f"Hit rate ranged from {periods['hit_rate'].min():.1%} to {periods['hit_rate'].max():.1%} "
        f"({first['period']} {first['hit_rate']:.1%} -> {last['period']} {last['hit_rate']:.1%}).")
    add("Type I error", periods["type1_error"].max() < L["type1_error"],
        f"Type I (false positive) error rate: highest {periods['type1_error'].max():.1%} across all periods "
        f"(limit {L['type1_error']:.0%}).")
    type2_amber = C.KPI_THRESHOLDS["type2_error"]["amber"]
    add("Type II error", periods["type2_error"].max() <= type2_amber,
        f"Type II (false negative) error rate moved from {first['type2_error']:.1%} to {last['type2_error']:.1%}; "
        f"highest {periods['type2_error'].max():.1%} (tolerance {type2_amber:.0%}). Missed matches rise as data "
        "quality worsens.")
    add("Disputes and incorrect merges",
        periods["dispute_rate"].max() < L["dispute_rate"]
        and periods["incorrect_merge_rate"].max() < L["incorrect_merge_rate"],
        f"Dispute rate max {periods['dispute_rate'].max():.2%} (limit {L['dispute_rate']:.0%}); incorrect merge "
        f"rate max {periods['incorrect_merge_rate'].max():.2%} (limit {L['incorrect_merge_rate']:.0%}).")
    top = contribution.sort_values("Contribution", ascending=False).iloc[0]
    add("Search path / attribute reliance", top["Contribution"] <= L["max_attribute_share"],
        f"Largest attribute contribution is {top['Matching attribute']} at {top['Contribution']:.1%}; "
        "match decisions rely on a combination of attributes.")
    prod = benchmark.iloc[0]
    others = benchmark.iloc[1:]
    better_recall = others[(others["recall"] > prod["recall"]) & (others["incorrect_merges"] <= prod["incorrect_merges"])]
    add("Challenger benchmarking", better_recall.empty,
        f"Production: precision {prod['precision']:.1%}, recall {prod['recall']:.1%}, "
        f"{int(prod['incorrect_merges'])} incorrect merges. "
        + ("No challenger finds more matches without more incorrect merges."
           if better_recall.empty else
           "Better recall without extra wrong merges: " + ", ".join(better_recall["Algorithm"]) + "."))
    if len(rca):
        top_cause = rca.sort_values("Number of cases", ascending=False).iloc[0]
        add("Root cause analysis", True,
            f"Most exceptions are caused by '{top_cause['Root cause']}' ({top_cause['Contribution']:.0%}); "
            "exceptions are driven by data quality in the incoming records rather than by a single rule.")
    worst = quality.sort_values("Incoming completeness").iloc[0]
    add("Data quality", worst["Incoming completeness"] >= 0.90,
        f"Lowest completeness among incoming records: {worst['Attribute']} at {worst['Incoming completeness']:.1%}.")
    return notes


# ==================================================================
# All sections in one structure (used by the PDF report)
# ==================================================================

def report_sections(results, benchmark, master_df, master):
    """The 10 Validation page sections as data: [{title, note, tables: [(caption, DataFrame)], ...}]."""
    periods = by_period(results)
    quality = data_quality(results, master_df)
    paths = search_path_analysis(results)
    contribution = attribute_contribution(results)
    exceptions = root_cause_analysis(results, master)
    rca = root_cause_summary(exceptions)
    calibration = threshold_calibration(results)
    cutoffs = ", ".join(f"{k} >= {v}" for k, v in C.CHALLENGER_CUTOFFS.items())
    calibration_cols = ["MATCH threshold", "REVIEW threshold", "Current setting", "precision", "recall",
                        "review_rate", "type1_error", "type2_error", "incorrect_merges"]
    return [
        {"title": "1. Validation data summary", "note": "",
         "tables": [("", data_summary(results, len(master_df)))]},
        {"title": "2. Data quality checks",
         "note": "Completeness = share of records where the attribute is filled. Validity = share of filled values "
                 "that pass the rule.",
         "tables": [("", quality)]},
        {"title": "3. Hit rate and dispute rate by period", "note": "",
         "tables": [("", periods[["period", "records", "matches_generated", "confirmed", "disputed_matches",
                                  "hit_rate", "dispute_rate"]])],
         "trend": periods[["period", "hit_rate", "dispute_rate", "type2_error"]]},
        {"title": "4. Search path analysis",
         "note": "Which search rule found the correct person (correct automatic matches only), and the share of "
                 "the weighted match score contributed by each attribute.",
         "tables": [("Search rules", paths), ("Attribute contribution", contribution)]},
        {"title": "5. Challenger algorithm benchmarking",
         "note": "Each challenger replaces one production comparator; search, weights and decision rules stay the "
                 f"same. Challenger cut-offs: {cutoffs}.",
         "tables": [("", benchmark[["Algorithm", "matches_generated", "confirmed", "hit_rate", "precision", "recall",
                                    "type1_error", "type2_error", "incorrect_merges", "dispute_rate",
                                    "review_rate"]])]},
        {"title": "6. Type I and Type II error assessment",
         "note": "Type I error (false positive): two different people matched. Type II error (false negative): "
                 "the same person not matched.",
         "tables": [("", periods[["period", "records", "TP", "FP", "TN", "FN", "type1_error", "type2_error"]])]},
        {"title": "7. Dispute and incorrect merge analysis",
         "note": "Disputed matches = 'does not belong to me' complaints about automatic matches. Missed-duplicate "
                 "disputes come from Type II errors.",
         "tables": [("", periods[["period", "matches_generated", "disputed_matches", "dispute_rate",
                                  "incorrect_merges", "incorrect_merge_rate", "duplicate_disputes"]])]},
        {"title": "8. Root cause analysis of exceptions",
         "note": f"Exceptions = missed duplicates, wrong merges and manual reviews ({len(exceptions)} records).",
         "tables": [("Primary root cause", rca), ("Grouping by error pattern", error_patterns(exceptions))]},
        {"title": "9. Threshold calibration",
         "note": "Decisions recomputed for other thresholds while everything else stays the same.",
         "tables": [(f"Varying the {v}", calibration[calibration["Varied"] == v][calibration_cols])
                    for v in ("MATCH threshold", "REVIEW threshold")]},
        {"title": "10. Validation summary", "note": "", "tables": [],
         "observations": observations(periods, contribution, benchmark, rca, quality)},
    ]
