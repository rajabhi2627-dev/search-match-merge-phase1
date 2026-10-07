"""
Phase 1 - Search, Match & Merge prototype (local Streamlit dashboard).

Run:  streamlit run app.py      ->  http://localhost:8501
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import config as C
import data_generator as G
import database as db
import matching as M
import performance as P

st.set_page_config(page_title="Search, Match & Merge - Phase 1", layout="wide")

DECISION_STYLE = {C.MATCH: st.success, C.MANUAL_REVIEW: st.warning, C.NO_MATCH: st.error}


@st.cache_resource
def get_master():
    """Create the SQLite database if needed and normalise the master records once."""
    db.init_db()
    df = db.load_master()
    return df, M.prepare_master(df.to_dict("records"))


def fmt_score(value):
    return "missing" if value is None or pd.isna(value) else f"{value:.1f}"


def candidates_table(candidates):
    rows = []
    for rank, c in enumerate(candidates, start=1):
        row = {"Rank": rank, "Entity ID": c["entity_id"], "Name": c["name"], "Overall score": c["overall"]}
        for f in C.FIELDS:
            row[f"{C.FIELD_LABELS[f]} score"] = fmt_score(c["field_scores"][f])
        row["Found by"] = ", ".join(c["search_reasons"])
        rows.append(row)
    return pd.DataFrame(rows)


def explanation_table(result):
    best = result["best"]
    rows = []
    for f in C.FIELDS:
        d = best["details"][f]
        rows.append({
            "Field": C.FIELD_LABELS[f],
            "Input (normalised)": result["input_norm"][f],
            "Master (normalised)": best["norm"][f],
            "Method": d["method"],
            "Similarity": fmt_score(d["similarity"]),
            "Edit distance": "-" if d["distance"] is None else str(d["distance"]),
            "Field score": fmt_score(d["score"]),
            "Weight": f"{C.FIELD_WEIGHTS[f]:.0%}",
            "Points contributed": round(best["contributions"][f], 2),
            "Explanation": d["note"],
        })
    return pd.DataFrame(rows)


# ==================================================================
# PAGE 1 - SEARCH & MATCH
# ==================================================================

EXAMPLE_RECORD = {
    "name": "Rahul Kumr", "email": "rahul.kumar@gmai.com", "phone": "+91 98765-43210",
    "address": "12 Mahatma Gandhi Road Patna", "date_of_birth": "12/04/1998",
    "company": "ABC Tech", "city": "Patna",
}


def load_into_form(record):
    for f in C.FIELDS:
        value = record.get(f)
        st.session_state[f"in_{f}"] = "" if value is None or pd.isna(value) else str(value)
    st.session_state.pop("search_result", None)


def page_search(master_df, master):
    st.header("Search & Match")
    st.write("Enter an incoming record. The system normalises it, searches the master database for candidates, "
             "scores every field with its own method and returns MATCH, MANUAL REVIEW or NO MATCH.")

    with st.expander("Load an example into the form"):
        options = {"Example: Rahul Kumar with typos (entity 10001)": EXAMPLE_RECORD}
        tests = db.load_test_records()
        for rec in tests.to_dict("records"):
            options[f"{rec['test_record_id']} - {rec['name']} ({rec['scenario']})"] = rec
        choice = st.selectbox("Example", list(options))
        st.button("Load into form", on_click=load_into_form, args=(options[choice],))

    with st.form("search_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Name", key="in_name")
            st.text_input("Email", key="in_email")
            st.text_input("Phone", key="in_phone")
            st.text_input("Address", key="in_address")
        with col2:
            st.text_input("Date of Birth", key="in_date_of_birth", placeholder="YYYY-MM-DD or DD/MM/YYYY")
            st.text_input("Company", key="in_company")
            st.text_input("City", key="in_city")
        submitted = st.form_submit_button("Search & Match", type="primary")

    if submitted:
        raw_input = {f: st.session_state.get(f"in_{f}", "") for f in C.FIELDS}
        if not any(v.strip() for v in raw_input.values()):
            st.warning("Please fill in at least one field.")
            return
        st.session_state["search_result"] = M.match_record(raw_input, master)

    result = st.session_state.get("search_result")
    if result:
        show_result(result)


def show_result(result):
    st.divider()
    best = result["best"]
    DECISION_STYLE[result["decision"]](f"Decision: **{result['decision']}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best matched entity", best["entity_id"] if best else "-")
    c2.metric("Overall score", f"{best['overall']:.1f}" if best else "0")
    c3.metric("Score gap (1st - 2nd)", f"{result['score_gap']:.1f}" if result["score_gap"] is not None else "-")
    c4.metric("Candidates found", result["candidates_found"])

    st.subheader("Why this decision?")
    for reason in result["decision_reasons"]:
        st.write(f"- {reason}")
    if result["conflicts"]:
        for conflict in result["conflicts"]:
            st.warning(f"Conflict: {conflict}")
    else:
        st.write("- Conflicts: **None**")

    if not best:
        return

    st.subheader(f"Explanation for best candidate: entity {best['entity_id']} ({best['name']})")
    contrib = sorted(((C.FIELD_LABELS[f], v) for f, v in best["contributions"].items()), key=lambda x: -x[1])
    top3 = ", ".join(f"{name} ({pts:.1f} pts)" for name, pts in contrib[:3] if pts > 0)
    st.write(f"Weighted score **{best['overall']:.1f}** out of 100. Fields that contributed most: **{top3 or 'none'}**.")
    if best["evidence_weight"] < 1:
        st.caption(f"Missing fields were excluded; the remaining weights ({best['evidence_weight']:.0%} of the total) "
                   "were rescaled to 100%.")

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(explanation_table(result), hide_index=True, width="stretch")
    with right:
        chart_df = pd.DataFrame(contrib, columns=["Field", "Points"])
        fig = px.bar(chart_df, x="Points", y="Field", orientation="h", title="Contribution to overall score")
        fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader(f"Top {C.TOP_N_DISPLAY} candidates")
    st.dataframe(candidates_table(result["top"]), hide_index=True, width="stretch")

    with st.expander("Pipeline trace: normalised input"):
        st.dataframe(pd.DataFrame([{"Field": C.FIELD_LABELS[f], "Raw input": result["input_raw"][f],
                                    "Normalised": result["input_norm"][f]} for f in C.FIELDS]),
                     hide_index=True, width="stretch")

    if result["decision"] != C.NO_MATCH:
        with st.expander("Merge preview (display only - the master database is not changed)"):
            st.write("Survivorship rule: keep the master value; fill empty master fields from the incoming record.")
            st.dataframe(pd.DataFrame(M.merge_preview(result["input_raw"], best["record"])),
                         hide_index=True, width="stretch")


# ==================================================================
# PAGE 2 - TEST DATA
# ==================================================================

def page_test_data(master_df, master):
    st.header("Test Data")
    st.write(f"Generates exactly {C.TEST_RECORD_COUNT} synthetic incoming records from the master data "
             f"(fixed random seed {C.TEST_SEED}, so the data is always the same). "
             "The ground truth (true entity and true status) is known automatically - no manual labelling.")

    with st.expander("Scenario distribution (configurable in config.py)"):
        st.dataframe(pd.DataFrame([{"Scenario": s, "Records": v["count"], "Ground truth": v["true_status"]}
                                   for s, v in C.TEST_SCENARIOS.items()]), hide_index=True)

    col1, col2 = st.columns(2)
    if col1.button(f"Generate {C.TEST_RECORD_COUNT} Test Records", type="primary"):
        db.save_test_records(G.generate_test_records(master_df.to_dict("records")))
        st.success(f"{C.TEST_RECORD_COUNT} test records generated and saved to the local database.")

    tests = db.load_test_records()
    if tests.empty:
        st.info("No test records yet. Click the button above.")
        return

    if col2.button("Run Matching", type="primary"):
        with st.spinner("Running the Search & Match algorithm on every test record..."):
            db.save_match_results(P.run_test_matching(tests, master))
        st.success("Matching complete. Results saved locally - see the Performance page.")

    results = db.load_match_results()
    if results.empty:
        st.subheader(f"Generated test records ({len(tests)})")
        st.dataframe(tests, hide_index=True, width="stretch")
        return

    st.subheader("Matching results")
    view = results.rename(columns={
        "test_record_id": "Test Record ID", "input_name": "Input Name", "input_email": "Input Email",
        "predicted_entity_id": "Predicted Entity", "true_entity_id": "True Entity", "match_score": "Match Score",
        "decision": "Predicted Decision", "true_match_status": "True Status", "outcome": "Correct / Incorrect",
        "best_candidate_id": "Best Candidate", "scenario": "Scenario",
    })
    st.dataframe(view[["Test Record ID", "Input Name", "Input Email", "Predicted Entity", "True Entity",
                       "Match Score", "Predicted Decision", "True Status", "Correct / Incorrect",
                       "Best Candidate", "Scenario"]], hide_index=True, width="stretch")
    st.caption("Predicted Entity is filled only for MATCH decisions. For MANUAL REVIEW, 'Best Candidate' is the "
               "entity suggested to the reviewer.")

    with st.expander("Generated test records with ground truth"):
        st.dataframe(tests, hide_index=True, width="stretch")


# ==================================================================
# PAGE 3 - PERFORMANCE
# ==================================================================

def heatmap(z, x, y, title):
    fig = px.imshow(z, x=x, y=y, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Predicted", y="Actual (ground truth)", color="Records"))
    fig.update_layout(title=title, height=330, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False)
    return fig


def page_performance():
    st.header("Performance")
    results = db.load_match_results()
    if results.empty:
        st.info("No results yet. Go to Test Data, generate the records and click Run Matching.")
        return

    m = P.metrics_for_results(results)
    counts = results["decision"].value_counts()

    c = st.columns(4)
    c[0].metric("Total records", m["total"])
    c[1].metric("Correct predictions", m["TP"] + m["TN"])
    c[2].metric("Incorrect predictions", m["FP"] + m["FN"])
    c[3].metric("Sent to manual review", m["review"])
    c = st.columns(4)
    c[0].metric("MATCH", int(counts.get(C.MATCH, 0)))
    c[1].metric("MANUAL REVIEW", int(counts.get(C.MANUAL_REVIEW, 0)))
    c[2].metric("NO MATCH", int(counts.get(C.NO_MATCH, 0)))
    c[3].metric("Manual review rate", f"{m['review_rate']:.0%}")

    st.info(
        "**How MANUAL REVIEW is treated:** a MANUAL REVIEW is not an automatic decision, so it is counted "
        "neither as correct nor as incorrect. Accuracy, Precision, Recall, F1, FPR and FNR are calculated on the "
        f"{m['auto_decided']} automatically decided records (MATCH or NO MATCH). The review rate is shown separately. "
        "A MATCH to the wrong entity counts as a False Positive (it would cause a wrong merge)."
    )

    st.subheader("Metrics (automatic decisions)")
    c = st.columns(6)
    for col, (label, key) in zip(c, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                     ("F1 Score", "f1"), ("False Positive Rate", "fpr"),
                                     ("False Negative Rate", "fnr")]):
        col.metric(label, f"{m[key]:.1%}")
    st.caption(f"TP = {m['TP']}, FP = {m['FP']}, TN = {m['TN']}, FN = {m['FN']}")

    with st.expander("Sensitivity check: what if MANUAL REVIEW were treated as NO MATCH?"):
        alt = P.metrics_for_results(results, review_as=C.NO_MATCH)
        keys = [("Records evaluated", "auto_decided"), ("TP", "TP"), ("FP", "FP"), ("TN", "TN"), ("FN", "FN"),
                ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1"),
                ("False Positive Rate", "fpr"), ("False Negative Rate", "fnr")]
        fmt = lambda d, k: f"{d[k]:.1%}" if isinstance(d[k], float) else str(d[k])
        st.dataframe(pd.DataFrame({
            "Metric": [k[0] for k in keys],
            "Review excluded (default)": [fmt(m, k[1]) for k in keys],
            "Review treated as NO MATCH": [fmt(alt, k[1]) for k in keys],
        }), hide_index=True)
        st.caption("If nobody reviews the queue, review cases behave like NO MATCH: duplicates are not linked, "
                   "so they become False Negatives.")

    st.subheader("Confusion matrix")
    left, right = st.columns(2)
    with left:
        z = P.binary_confusion_matrix(m)
        st.plotly_chart(heatmap(z, ["MATCH", "NO MATCH"], ["Should MATCH", "Should NOT MATCH"],
                                "View 1 - Binary: MATCH vs NO MATCH"), width="stretch")
        st.caption("Manual review records are excluded from this view. Top-left TP, top-right FN, "
                   "bottom-left FP (incl. wrong-entity matches), bottom-right TN.")
    with right:
        dist = P.decision_distribution(results)
        st.plotly_chart(heatmap(dist.values, list(dist.columns), ["True MATCH", "True NO MATCH"],
                                "View 2 - Full decision distribution"), width="stretch")
        st.caption("All 100 records. Shows where the MANUAL REVIEW decisions come from.")
    st.dataframe(dist.rename_axis("True status \\ Predicted").reset_index(), hide_index=True)

    st.subheader("Results by scenario")
    st.dataframe(P.scenario_summary(results), hide_index=True, width="stretch")

    st.subheader("Incorrect Predictions")
    score_cols = [f"score_{f}" for f in C.FIELDS]
    cols = (["test_record_id", "scenario"] + [f"input_{f}" for f in C.FIELDS]
            + ["best_candidate_id", "predicted_entity_id", "true_entity_id", "match_score", "score_gap", "decision",
               "true_match_status"] + score_cols + ["conflicts", "decision_reasons"])
    wrong = results[results["outcome"] == "Incorrect"]
    if wrong.empty:
        st.success("No incorrect automatic decisions.")
    else:
        st.dataframe(wrong[cols], hide_index=True, width="stretch")

    review = results[results["outcome"] == "Manual Review"]
    with st.expander(f"Records sent to manual review ({len(review)})"):
        review = review.assign(best_is_true_entity=review["best_candidate_id"] == review["true_entity_id"])
        st.dataframe(review[cols + ["best_is_true_entity"]], hide_index=True, width="stretch")
        st.caption("best_is_true_entity shows whether the candidate offered to the reviewer is the correct entity.")


# ==================================================================
# PAGE 4 - METHODOLOGY
# ==================================================================

def page_methodology():
    st.header("Methodology")
    st.write("The algorithm is fully deterministic: the same input always gives the same output. "
             "There is no machine learning, AI model or external service involved.")

    st.markdown("""
**Pipeline:** Incoming record → Normalisation → Candidate search → Exact matching → Field-specific fuzzy matching
→ Field-level scores → Weighted overall score → Conflict detection → Candidate ranking → Score gap → Final decision
""")

    st.subheader("1. Normalisation")
    st.markdown("""
The same rules are applied to master records and incoming records.
- **Text** (name, address, company, city): lowercase, trim spaces, replace punctuation with spaces, collapse multiple spaces.
- **Address**: common abbreviations are expanded (Rd → road, St → street, MG → mahatma gandhi, Sec → sector, Ngr → nagar).
- **Company**: legal suffixes are removed (Pvt, Private, Ltd, Limited, LLP, Inc).
- **City**: old names are mapped to current ones (Bangalore → bengaluru, Bombay → mumbai, Gurgaon → gurugram ...).
- **Phone**: keep digits only; remove a leading `00`, a `91` country code (12 digits) or a trunk `0` (11 digits) → 10 digits.
- **Email**: lowercase and remove all spaces.
- **Date of birth**: converted to `YYYY-MM-DD` (day-first formats such as 12/04/1998 are accepted). Unreadable → missing.
""")

    st.subheader("2. Candidate search (blocking)")
    st.markdown(f"""
A master record becomes a candidate if **at least one** rule is true:
1. Exact phone match
2. Exact email match
3. Similar name (WRatio ≥ {C.CANDIDATE_NAME_MIN})
4. Same city **and** partly similar name (WRatio ≥ {C.CANDIDATE_CITY_NAME_MIN})

Candidates with exact identifier hits come first, then by name similarity; at most {C.MAX_CANDIDATES} are fully scored.
""")

    st.subheader("3-6. Exact matching, field-specific fuzzy matching, thresholds and edit distances")
    st.write("Each field is first compared exactly (identical normalised values = 100). If not identical, the "
             "field's own method is used. A field **agrees** if its similarity ≥ threshold **or** its edit distance "
             "≤ max distance; then the field score is the similarity. Otherwise the field score is **0**.")
    st.dataframe(pd.DataFrame([{"Field": C.FIELD_LABELS[f], "Matching method": cfg["method"],
                                "Threshold": cfg["threshold"], "Max distance": cfg["max_distance"],
                                "Weight": f"{C.FIELD_WEIGHTS[f]:.0%}"}
                               for f, cfg in C.FIELD_CONFIG.items()]), hide_index=True)
    diffs = ", ".join(f"{k} digit(s) different = {v}" for k, v in C.PHONE_SCORE_BY_DIGIT_DIFF.items())
    st.markdown(f"""
- **Name / Company - WRatio:** tolerant of spelling mistakes, word order and partial strings.
- **Email - Levenshtein:** character edits; a strong identifier, so the threshold is high.
- **Phone - digit comparison (not generic fuzzy):** {diffs}; more = 0. If the lengths differ, the last
  {C.PHONE_LAST_N_DIGITS} digits identical = {C.PHONE_LAST_N_SCORE}. A phone "agrees" at ≥ {C.FIELD_CONFIG['phone']['threshold']}.
- **Address - Token Set Ratio:** ignores word order and extra or missing words.
- **Date of birth - exact:** same = 100, different = 0.
- **City - exact, then Ratio:** catches small typos such as "Ptna".
- **Missing values:** if a field is empty on either side, it is left out and the other weights are rescaled.
""")

    st.subheader("7-8. Field-level scores and weighted score")
    st.latex(r"\text{Overall} = \frac{\sum_{f \in \text{present}} \text{score}_f \times w_f}{\sum_{f \in \text{present}} w_f}")
    st.write("With all fields present this is simply 0.25×Name + 0.25×Email + 0.25×Phone + 0.10×Address + "
             "0.05×DOB + 0.05×Company + 0.05×City (always between 0 and 100).")

    st.subheader("9. Conflict detection")
    st.markdown(f"""
| Rule | Condition | Effect |
|---|---|---|
| 1. Strong identifiers disagree | one of Name / Email / Phone ≥ {C.STRONG_AGREE_SCORE} while another < {C.STRONG_DISAGREE_SCORE} | MANUAL REVIEW |
| 2. Date of birth differs | DOB present on both sides and different | MANUAL REVIEW |
| 3. Identifiers point to different people | exact phone → one entity, exact email → another entity | MANUAL REVIEW |
| 4. Too little evidence | compared fields carry < {C.MIN_EVIDENCE_WEIGHT:.0%} of the total weight | MATCH not allowed → MANUAL REVIEW |

Conflicts only matter when the score is at least {C.REVIEW_THRESHOLD}; below that the result is NO MATCH anyway.
""")

    st.subheader("10-11. Candidate ranking and score gap")
    st.write(f"Candidates are sorted by overall score; the top {C.TOP_N_DISPLAY} are shown. "
             f"Score gap = best score - second best score. If the gap is below {C.SCORE_GAP_THRESHOLD} the two "
             f"candidates are too close and the result is MANUAL REVIEW - unless there is no meaningful second "
             f"candidate (no second candidate, or its score is below {C.REVIEW_THRESHOLD}).")

    st.subheader("12. Final decision")
    st.markdown(f"""
| Decision | Rule |
|---|---|
| **NO MATCH** | best score < {C.REVIEW_THRESHOLD} (or no candidate found) |
| **MANUAL REVIEW** | score {C.REVIEW_THRESHOLD}-{C.MATCH_THRESHOLD}, or score gap < {C.SCORE_GAP_THRESHOLD}, or any conflict, or too little evidence |
| **MATCH** | score ≥ {C.MATCH_THRESHOLD}, no conflict and a clear score gap |
""")

    st.subheader("13. Performance evaluation")
    st.markdown("""
- **TP**: MATCH to the correct entity. **FP**: MATCH for a new person or to the wrong entity.
  **TN**: NO MATCH for a new person. **FN**: NO MATCH although the person exists.
- **MANUAL REVIEW** is not an automatic decision: it is excluded from the binary metrics and reported as the review rate.
- Accuracy = (TP + TN) / (TP + FP + TN + FN), Precision = TP / (TP + FP), Recall = TP / (TP + FN),
  F1 = 2 × P × R / (P + R), FPR = FP / (FP + TN), FNR = FN / (FN + TP). Division by zero returns 0.
""")
    st.caption("All parameters are in config.py. All data is stored only in data/matching.db on this computer.")


# ==================================================================
# NAVIGATION
# ==================================================================

def main():
    master_df, master = get_master()
    st.sidebar.title("Search, Match & Merge")
    st.sidebar.caption("Phase 1 prototype - deterministic matching")
    page = st.sidebar.radio("Go to", ["Search & Match", "Test Data", "Performance", "Methodology"])
    st.sidebar.divider()
    st.sidebar.caption(f"Master records: {len(master_df):,}")
    st.sidebar.caption("Runs locally. Data stays in data/matching.db.")

    if page == "Search & Match":
        page_search(master_df, master)
    elif page == "Test Data":
        page_test_data(master_df, master)
    elif page == "Performance":
        page_performance()
    else:
        page_methodology()


main()
