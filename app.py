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
import validation as V

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

def load_into_form(record):
    for f in C.FIELDS:
        value = record.get(f)
        st.session_state[f"in_{f}"] = "" if value is None or pd.isna(value) else str(value)
    st.session_state.pop("search_result", None)


def page_search(master_df, master):
    st.header("Search & Match")
    st.write("Enter an incoming record. The system normalises it, searches the master database for candidates, "
             "scores every field with its own method and returns MATCH, MANUAL REVIEW or NO MATCH.")
    st.caption(f"Master data: {len(master_df):,} people from the {C.DATASET_NAME} benchmark. "
               "The form uses exactly the FEBRL person fields.")

    with st.expander("Load a FEBRL test record into the form"):
        options = {f"{rec['test_record_id']} - {rec['name']} ({rec['source_id']}, {rec['scenario']})": rec
                   for rec in G.generate_test_records()}
        choice = st.selectbox("Example", list(options))
        st.button("Load into form", on_click=load_into_form, args=(options[choice],))

    with st.form("search_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Name", key="in_name")
            st.text_input("Soc. Sec. ID", key="in_soc_sec_id")
            st.text_input("Address", key="in_address", placeholder="street number, street, postcode")
        with col2:
            st.text_input("Date of Birth", key="in_date_of_birth", placeholder="YYYYMMDD, YYYY-MM-DD or DD/MM/YYYY")
            st.text_input("City / Suburb", key="in_city")
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
    st.write(f"Takes exactly {C.TEST_RECORD_COUNT} incoming records from the {C.DATASET_NAME} benchmark "
             "(fixed random seed, so the selection is always the same):")
    st.markdown(f"""
- **{C.TEST_MATCH_COUNT} records** are FEBRL's corrupted duplicates of people who **are** in the master database → true status **MATCH**
- **{C.TEST_NO_MATCH_COUNT} records** are duplicates of people who are **not** in the master database → true status **NO MATCH** (new person)

The ground truth comes from the benchmark itself: FEBRL record `rec-N-dup-0` is the same person as `rec-N-org`
(master entity N). Nobody labels anything by hand. "Scenario" shows how many fields FEBRL corrupted in the record.
""")

    col1, col2 = st.columns(2)
    if col1.button(f"Generate {C.TEST_RECORD_COUNT} Test Records", type="primary"):
        db.save_test_records(G.generate_test_records())
        st.success(f"{C.TEST_RECORD_COUNT} test records taken from {C.DATASET_NAME} and saved to the database.")

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
        "test_record_id": "Test Record ID", "source_id": "FEBRL Record", "input_name": "Input Name",
        "input_soc_sec_id": "Input Soc. Sec. ID", "predicted_entity_id": "Predicted Entity", "true_entity_id": "True Entity",
        "match_score": "Match Score", "decision": "Predicted Decision", "true_match_status": "True Status",
        "outcome": "Correct / Incorrect", "best_candidate_id": "Best Candidate", "scenario": "Scenario",
    })
    st.dataframe(view[["Test Record ID", "FEBRL Record", "Input Name", "Input Soc. Sec. ID", "Predicted Entity",
                       "True Entity", "Match Score", "Predicted Decision", "True Status", "Correct / Incorrect",
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
        "neither as correct nor as incorrect. Accuracy, Precision, Recall and F1 are calculated on the "
        f"{m['auto_decided']} automatically decided records (MATCH or NO MATCH). The review rate is shown separately. "
        "A MATCH to the wrong entity counts as a False Positive (it would cause a wrong merge)."
    )

    k = P.compute_kpis(results, "test")
    st.subheader("Metrics")
    c = st.columns(6)
    for col, (label, value) in zip(c, [("Accuracy", m["accuracy"]), ("Precision", m["precision"]),
                                       ("Recall", m["recall"]), ("F1 Score", m["f1"]),
                                       ("Type I Error Rate", k["type1_error"]),
                                       ("Type II Error Rate", k["type2_error"])]):
        col.metric(label, f"{value:.1%}")
    st.caption(f"TP = {m['TP']}, FP = {m['FP']}, TN = {m['TN']}, FN = {m['FN']}. "
               f"Type I = false positives / total matches generated ({k['FP']} / {k['matches_generated']}); "
               f"Type II = false negatives / total actual matches ({k['FN']} / {k['actual_matches']}), "
               "as defined in the validation report.")

    st.subheader("Monitoring KPIs for this run")
    c = st.columns(4)
    for col, key in zip(c, ["hit_rate", "review_rate", "incorrect_merge_rate", "dispute_rate"]):
        col.metric(P.KPI_DICTIONARY[key]["label"], f"{k[key]:.1%}", P.rag_status(key, k[key]),
                   delta_color="off")
    st.caption(f"Hit rate = confirmed matches / total matches generated ({k['confirmed']} / "
               f"{k['matches_generated']}). {k['disputed_matches']} disputed match(es) and "
               f"{k['duplicate_disputes']} 'I already have an account' complaint(s) (simulated). "
               "See the Monitoring page for definitions and trends.")

    with st.expander("Sensitivity check: what if MANUAL REVIEW were treated as NO MATCH?"):
        alt = P.metrics_for_results(results, review_as=C.NO_MATCH)
        m = {**m, **{key: k[key] for key in ("type1_error", "type2_error")}}
        alt_doc = P.document_metrics(P.reviews_as_no_match(results))
        alt = {**alt, **{key: alt_doc[key] for key in ("type1_error", "type2_error")}}
        keys = [("Records evaluated", "auto_decided"), ("TP", "TP"), ("FP", "FP"), ("TN", "TN"), ("FN", "FN"),
                ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1"),
                ("Type I Error Rate", "type1_error"), ("Type II Error Rate", "type2_error")]
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

    st.subheader("0. Dataset - FEBRL-4 benchmark")
    st.markdown(f"""
FEBRL (Freely Extensible Biomedical Record Linkage) is a public record-linkage benchmark from the Australian
National University (Christen, 2008). FEBRL-4 has 5,000 original person records and 5,000 corrupted duplicates
(typos, phonetic spelling changes, swapped or replaced names, missing values, changed digits), with the true links published.

| Project field | FEBRL column(s) |
|---|---|
| Name | given_name + surname |
| Soc. Sec. ID | soc_sec_id (strong numeric identifier, compared digit by digit) |
| Address | street_number + address_1 + address_2 + postcode |
| Date of birth | date_of_birth (YYYYMMDD) |
| City | suburb |

Only FEBRL's own fields are used - nothing is invented or added.
- **Master database:** {C.MASTER_RECORD_COUNT:,} FEBRL originals (fixed random sample).
- **Test set:** {C.TEST_MATCH_COUNT} duplicates of master people (true MATCH) + {C.TEST_NO_MATCH_COUNT} duplicates of
  people not in the master (true NO MATCH).
""")

    st.subheader("1. Normalisation")
    st.markdown("""
The same rules are applied to master records and incoming records.
- **Text** (name, address, city): lowercase, trim spaces, replace punctuation with spaces, collapse multiple spaces.
- **Address**: common street abbreviations are expanded (Rd → road, St → street, Ave → avenue, Cres → crescent, Pl → place ...).
- **Soc. Sec. ID**: keep digits only (spaces and dashes are removed).
- **Date of birth**: converted to `YYYY-MM-DD` (FEBRL `YYYYMMDD` and day-first formats such as 12/04/1998 are accepted). Invalid dates → missing.
""")

    st.subheader("2. Candidate search (blocking)")
    st.markdown(f"""
A master record becomes a candidate if **at least one** rule is true:
1. Exact Soc. Sec. ID match
2. Similar name (WRatio ≥ {C.CANDIDATE_NAME_MIN})
3. Same city **and** partly similar name (WRatio ≥ {C.CANDIDATE_CITY_NAME_MIN})

Blocking only decides **who is compared** - an exact ID is a quick way to find the person. The ID is then
**scored** digit by digit (section 3-6), so an ID with a typo can still count once the person is found by name.
Candidates with an exact ID hit come first, then by name similarity; at most {C.MAX_CANDIDATES} are fully scored.
""")

    st.subheader("3-6. Exact matching, field-specific fuzzy matching, thresholds and edit distances")
    st.write("Each field is first compared exactly (identical normalised values = 100). If not identical, the "
             "field's own method is used. A field **agrees** if its similarity ≥ threshold **or** its edit distance "
             "≤ max distance; then the field score is the similarity. Otherwise the field score is **0**.")
    st.dataframe(pd.DataFrame([{"Field": C.FIELD_LABELS[f], "Matching method": cfg["method"],
                                "Threshold": cfg["threshold"], "Max distance": cfg["max_distance"],
                                "Weight": f"{C.FIELD_WEIGHTS[f]:.0%}"}
                               for f, cfg in C.FIELD_CONFIG.items()]), hide_index=True)
    edits = ", ".join(f"{k} digit edit(s) = {v}" for k, v in C.ID_SCORE_BY_DIGIT_EDITS.items())
    st.markdown(f"""
- **Name - WRatio:** tolerant of spelling mistakes, word order and partial strings.
- **Soc. Sec. ID - digit comparison (not generic fuzzy):** count the digits that must be changed, added or removed:
  {edits}; more = 0 (a different ID). This tolerates a typing slip but not a different number.
- **Address - Token Set Ratio:** ignores word order and extra or missing words.
- **Date of birth - exact:** same = 100, different = 0.
- **City - exact, then Ratio:** catches small typos such as "Ptna".
- **Missing values:** if a field is empty on either side, it is left out and the other weights are rescaled.
""")

    st.subheader("7-8. Field-level scores and weighted score")
    st.latex(r"\text{Overall} = \frac{\sum_{f \in \text{present}} \text{score}_f \times w_f}{\sum_{f \in \text{present}} w_f}")
    st.write("With all fields present this is simply 0.35×Name + 0.35×Soc. Sec. ID + 0.15×Address + "
             "0.075×DOB + 0.075×City (always between 0 and 100).")

    st.subheader("9. Conflict detection")
    st.markdown(f"""
| Rule | Condition | Effect |
|---|---|---|
| 1. Strong identifiers disagree | Name or Soc. Sec. ID ≥ {C.STRONG_AGREE_SCORE} while the other < {C.STRONG_DISAGREE_SCORE} | MANUAL REVIEW |
| 2. Date of birth differs | DOB present on both sides and different | MANUAL REVIEW |
| 3. Identifiers point to different people | exact Soc. Sec. ID → one entity, best overall score → another entity | MANUAL REVIEW |
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
  F1 = 2 × P × R / (P + R). Division by zero returns 0.
- Validation-report definitions (used on every page): **matches generated** = MATCH + MANUAL REVIEW decisions;
  Hit rate = confirmed matches / total matches generated; Dispute rate = disputed matches / total matches generated;
  Incorrect merge rate = incorrect merges / total matches generated; Type I error rate = false positives / total
  matches generated; Type II error rate = false negatives / total actual matches.
""")

    st.subheader("14. Assumptions and limitations")
    st.markdown("""
**Assumptions**
- The FEBRL ground truth is correct: `rec-N-dup-0` is the same person as `rec-N-org`, and FEBRL people are all different.
- A manual reviewer always takes the right decision, so manual review cases cause no disputes.
- Disputes are simulated with fixed rates (see config.py); real complaint behaviour may differ.
- Thresholds and weights are set by expert judgement and checked with threshold calibration on the Validation page;
  they were not tuned on the test set.

**Limitations**
- FEBRL has only 5 person fields (no phone, email or gender), and the Soc. Sec. ID is rarely wrong or shared,
  which is easier than real bureau data.
- Records with a completely replaced name or a wrong ID usually fall below the review threshold and are missed (Type II errors).
- People with genuinely similar names and the same date of birth are not present in FEBRL, so look-alike risk is under-tested.
- The prototype runs in memory on 1,000 master records; large volumes would need indexed blocking.

**Mitigating controls**: conflict rules and the score gap send uncertain cases to manual review; the Monitoring and
Validation pages track error rates, disputes and root causes over time.
""")

    st.subheader("15. AI agent (monitoring and validation)")
    st.markdown(f"""
- The matching algorithm uses no AI. An LLM agent (Groq `{C.GROQ_MODEL}`) only **monitors and validates** it.
- Deterministic tools calculate every figure; the agent interprets them and writes the report.
- Guardrails reject reports with numbers not found in the evidence, a wrong status or wrong breaches, or missing
  sections; the agent then revises (up to {C.AGENT_MAX_REVISIONS} times).
- The report is advisory: a human validator records the final decision (human in the loop).
- The agent itself is validated on 13 golden scenarios (hallucination, numeric integrity, stability, guardrails,
  retrieval, workflow, adversarial tests and expert review). LLM output can still vary between runs, which is why
  stability is tested and human approval stays mandatory.
""")
    st.caption("All parameters are in config.py. Data is stored in data/matching.db.")


# ==================================================================
# PAGE 5 - MONITORING (WEEK 4)
# ==================================================================

RAG_COLOURS = {"Green": "#d4edda", "Amber": "#fff3cd", "Red": "#f8d7da"}


def kpi_dictionary_table():
    rows = []
    for key, info in P.KPI_DICTIONARY.items():
        t = C.KPI_THRESHOLDS[key]
        sign = "≥" if t["direction"] == "higher" else "≤"
        worse = "<" if t["direction"] == "higher" else ">"
        rows.append({"KPI": info["label"], "What it measures": info["meaning"], "Formula": info["formula"],
                     "Green": f"{sign} {t['green']:.0%}", "Amber": f"{sign} {t['amber']:.0%}",
                     "Red": f"{worse} {t['amber']:.0%}"})
    return pd.DataFrame(rows)


def page_monitoring(master):
    st.header("Performance Monitoring")
    st.write("Week 4: the same matching algorithm is run on a new batch of incoming records every month, "
             "and a fixed set of KPIs is tracked against Green / Amber / Red thresholds.")

    with st.expander("KPI dictionary", expanded=True):
        st.dataframe(kpi_dictionary_table(), hide_index=True, width="stretch")
        fp, fn = C.DISPUTE_RAISE_RATE["FP"], C.DISPUTE_RAISE_RATE["FN"]
        st.caption(f"Matches generated = MATCH + MANUAL REVIEW decisions. Disputes are simulated (fixed seed): "
                   f"{fp:.0%} of wrong merges lead to a 'does not belong to me' dispute (counted in the dispute "
                   f"rate) and {fn:.0%} of missed duplicates lead to an 'I already have an account' complaint "
                   "(shown separately). Formulas follow the validation report; thresholds are in config.py.")

    st.markdown(f"""
**Monitoring set-up:** {len(C.MONITORING_PERIODS)} months ({C.MONITORING_PERIODS[0]} to {C.MONITORING_PERIODS[-1]}),
each with {C.MONITORING_MATCH_COUNT} FEBRL duplicates of existing people and {C.MONITORING_NO_MATCH_COUNT} new people,
none of them used in the test set. To simulate **data-quality drift**, duplicates are ordered by how many fields
FEBRL corrupted: the first month receives the cleanest records, the last month the most corrupted ones.
""")

    if st.button("Run Monthly Monitoring", type="primary"):
        with st.spinner("Matching 6 monthly batches..."):
            db.save_monitoring_results(P.run_monitoring(master))
        st.success("Monitoring run complete. Results saved to the database.")

    results = db.load_monitoring_results()
    if results.empty:
        st.info("Click 'Run Monthly Monitoring' to start.")
        return

    kpis = P.monthly_kpis(results)
    latest = kpis.iloc[-1]
    previous = kpis.iloc[-2] if len(kpis) > 1 else latest

    st.subheader(f"Latest month: {latest['period']}")
    keys = list(P.KPI_DICTIONARY)
    for row in (keys[:4], keys[4:]):
        cols = st.columns(4)
        for col, key in zip(cols, row):
            change = latest[key] - previous[key]
            col.metric(P.KPI_DICTIONARY[key]["label"], f"{latest[key]:.1%}",
                       f"{change:+.1%} vs {previous['period']} · {P.rag_status(key, latest[key])}",
                       delta_color="off")

    st.subheader("Monthly KPI table")
    table = pd.DataFrame({p: {P.KPI_DICTIONARY[k]["label"]: kpis.loc[i, k] for k in keys}
                          for i, p in enumerate(kpis["period"])})
    status = pd.DataFrame({p: {P.KPI_DICTIONARY[k]["label"]: P.rag_status(k, kpis.loc[i, k]) for k in keys}
                           for i, p in enumerate(kpis["period"])})
    styled = table.style.format("{:.1%}").apply(
        lambda col: [f"background-color: {RAG_COLOURS[s]}" for s in status[col.name]], axis=0)
    st.dataframe(styled, width="stretch")
    counts = kpis.set_index("period")[["records", "matches_generated", "confirmed", "actual_matches", "TP", "FP",
                                       "TN", "FN", "review", "disputed_matches", "duplicate_disputes"]]
    counts = counts.rename(columns={"records": "Records", "matches_generated": "Matches generated",
                                    "confirmed": "Confirmed matches", "actual_matches": "Actual matches",
                                    "review": "Manual review", "disputed_matches": "Disputed matches",
                                    "duplicate_disputes": "'I already have an account'"}).T
    st.dataframe(counts, width="stretch")

    st.subheader("Trend")
    label_to_key = {P.KPI_DICTIONARY[k]["label"]: k for k in keys}
    chosen = st.selectbox("KPI", list(label_to_key), index=list(label_to_key).index("Recall"))
    key = label_to_key[chosen]
    fig = px.line(kpis, x="period", y=key, markers=True, labels={"period": "Month", key: chosen})
    t = C.KPI_THRESHOLDS[key]
    fig.add_hline(y=t["green"], line_dash="dash", line_color="green", annotation_text="Green limit")
    fig.add_hline(y=t["amber"], line_dash="dash", line_color="orange", annotation_text="Amber limit")
    fig.update_layout(height=350, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Observations (rule-based)")
    notes = P.trend_observations(kpis)
    if not notes:
        st.success("All KPIs are Green and stable.")
    for note in notes:
        {"Red": st.error, "Amber": st.warning, "Green": st.info}[note["Status"]](
            f"**{note['Status']}** - {note['Observation']}")

    with st.expander("Drill-down: errors and disputes by month"):
        period = st.selectbox("Month", list(kpis["period"]), index=len(kpis) - 1)
        batch = results[(results["period"] == period) & (results["outcome"] != "Correct")]
        st.dataframe(batch[["test_record_id", "source_id", "scenario", "input_name", "best_candidate_id",
                            "true_entity_id", "match_score", "decision", "outcome", "disputed",
                            "decision_reasons"]], hide_index=True, width="stretch")


# ==================================================================
# PAGE 6 - INDEPENDENT VALIDATION
# ==================================================================

PERCENT_COLUMNS = V.PERCENT_COLUMNS   # shared with the PDF report
REPORT_LABELS = V.REPORT_LABELS


def agent_context(results, master, period, proposed_change=None):
    """Agent context with the coded validation results (challengers, calibration, data quality) when available."""
    import agent_tools as T
    benchmark, quality = db.load_benchmark_results(), None
    if benchmark.empty:
        benchmark = None
    else:
        quality = V.data_quality(results, get_master()[0])
    return T.live_context(results, master, period=period, proposed_change=proposed_change,
                          benchmark=benchmark, quality=quality)


def generate_full_report(master_df, master):
    """Write the full validation report with the AI agent (parts written in turn, each one verified)."""
    import validation_report as VR
    results, benchmark = db.load_monitoring_results(), db.load_benchmark_results()
    try:
        with st.status("AI agent writing the validation report (a few minutes on the Groq free tier)...",
                       expanded=True) as status:
            report = VR.build_report(results, benchmark, master_df, master, on_step=st.write)
            status.update(label="Validation report ready", state="complete", expanded=False)
    except Exception as e:  # missing key, every model rate-limited, network error
        st.error(f"The AI validation report could not be written: {e}")
        return None
    st.session_state["full_report"] = report
    return report


def full_report_files(report):
    """PDF and Word bytes of the full report, rebuilt only when the sign-off changes."""
    import report_docx as RD
    import report_pdf as RP
    approval = report.get("approval")
    key = (approval or {}).get("run_at")
    if "_files" not in report or report.get("_files_key") != key:
        report["_files"] = (RP.render_full_pdf(report, approval), RD.render_full_docx(report, approval))
        report["_files_key"] = key
    return report["_files"]


def report_pdf_bytes(run):
    """PDF of an AI run plus the coded validation results; cached on the run until it changes."""
    import report_pdf as R
    key = (run.get("approval") or {}).get("run_at")
    if "_pdf" not in run or run.get("_pdf_key") != key:
        master_df, master = get_master()
        results, benchmark = db.load_monitoring_results(), db.load_benchmark_results()
        sections = (None if results.empty or benchmark.empty
                    else V.report_sections(results, benchmark, master_df, master))
        run["_pdf"], run["_pdf_key"] = R.build_validation_pdf(run, sections), key
    return run["_pdf"]


def pdf_download(run, label="Download full report (PDF)", key="pdf"):
    st.download_button(label, report_pdf_bytes(run), file_name=f"validation_report_{run['period']}.pdf",
                       mime="application/pdf", type="primary", key=key)


def pct_table(df, columns=None):
    """Show a table with rate columns formatted as percentages."""
    df = df[columns] if columns else df
    fmt = {c: "{:.1%}" for c in df.columns if c in PERCENT_COLUMNS}
    renamed = df.rename(columns=REPORT_LABELS)
    fmt = {REPORT_LABELS.get(c, c): f for c, f in fmt.items()}
    st.dataframe(renamed.style.format(fmt), hide_index=True, width="stretch")


def page_validation(master_df, master):
    st.header("Independent Validation")
    st.write("A model validation of the Search & Match algorithm, structured like an independent validation "
             "report: data quality, hit and dispute rates, search path analysis, challenger benchmarking, "
             "Type I / Type II errors, dispute and incorrect merge analysis, root cause analysis and "
             "threshold calibration.")
    st.markdown(f"""
**Validation sample:** the {len(C.MONITORING_PERIODS)} monthly batches from the Monitoring page
({len(C.MONITORING_PERIODS) * (C.MONITORING_MATCH_COUNT + C.MONITORING_NO_MATCH_COUNT)} FEBRL records,
none used in the test set). Each month is treated as one reporting period. The ground truth comes from FEBRL,
disputes are simulated with a fixed seed, and everything is rule-based (no machine learning).
""")
    with st.expander("Metric definitions used on this page"):
        st.dataframe(V.METRIC_DEFINITIONS, hide_index=True, width="stretch")
        st.caption("These are the formulas of the validation report; the Performance and Monitoring pages use "
                   "the same definitions.")

    if st.button("Run Validation", type="primary"):
        with st.spinner("Running the validation sample and 5 challenger algorithms (about 20 seconds)..."):
            monitoring = P.run_monitoring(master)
            db.save_monitoring_results(monitoring)
            db.save_benchmark_results(V.run_benchmark(master, db.load_monitoring_results()))
        st.success("Validation complete. Results saved to the database.")
        if st.session_state.get("agent_run"):
            st.session_state["agent_run"].pop("_pdf", None)  # rebuild the PDF with the new results
        st.session_state.pop("full_report", None)            # the old report no longer matches the results

    results = db.load_monitoring_results()
    benchmark = db.load_benchmark_results()
    if results.empty or benchmark.empty:
        st.info("Click 'Run Validation' to start.")
        return

    periods = V.by_period(results)
    quality = V.data_quality(results, master_df)
    paths = V.search_path_analysis(results)
    contribution = V.attribute_contribution(results)
    exceptions = V.root_cause_analysis(results, master)
    rca = V.root_cause_summary(exceptions)

    st.subheader("1. Validation data summary")
    st.dataframe(V.data_summary(results, len(master_df)), hide_index=True, width="stretch")

    st.subheader("2. Data quality checks")
    st.write("Completeness = share of records where the attribute is filled. Validity = share of filled values "
             "that pass the rule.")
    pct_table(quality)

    st.subheader("3. Hit rate and dispute rate by period")
    pct_table(periods, ["period", "records", "matches_generated", "confirmed", "disputed_matches",
                        "hit_rate", "dispute_rate"])
    trend = periods.melt(id_vars="period", value_vars=["hit_rate", "dispute_rate", "type2_error"],
                         var_name="Metric", value_name="Rate")
    trend["Metric"] = trend["Metric"].map(REPORT_LABELS)
    fig = px.line(trend, x="period", y="Rate", color="Metric", markers=True, labels={"period": "Period"})
    fig.update_layout(height=320, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("4. Search path analysis")
    st.write("Which search rule found the correct person (correct automatic matches only). 'Found ONLY by "
             "this rule' shows how many matches would be lost if that rule were removed.")
    pct_table(paths)
    st.write("Share of the weighted match score contributed by each attribute:")
    col1, col2 = st.columns([1, 1])
    with col1:
        pct_table(contribution)
    with col2:
        fig = px.bar(contribution, x="Contribution", y="Matching attribute", orientation="h", text_auto=".1%")
        fig.update_layout(height=250, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("5. Challenger algorithm benchmarking")
    cutoffs = ", ".join(f"{k} ≥ {v}" for k, v in C.CHALLENGER_CUTOFFS.items())
    st.write("Each challenger replaces **one** production comparator; candidate search, weights and decision "
             "rules stay the same, so any difference comes from that algorithm alone. "
             f"Challenger cut-offs: {cutoffs}.")
    pct_table(benchmark, ["Algorithm", "matches_generated", "confirmed", "hit_rate", "precision", "recall",
                          "type1_error", "type2_error", "incorrect_merges", "dispute_rate", "review_rate"])

    st.subheader("6. Type I and Type II error assessment")
    st.write("Type I error (false positive): two different people matched. "
             "Type II error (false negative): the same person not matched.")
    pct_table(periods, ["period", "records", "TP", "FP", "TN", "FN", "type1_error", "type2_error"])

    st.subheader("7. Dispute and incorrect merge analysis")
    pct_table(periods, ["period", "matches_generated", "disputed_matches", "dispute_rate", "incorrect_merges",
                        "incorrect_merge_rate", "duplicate_disputes"])
    st.caption("Disputed matches = 'does not belong to me' complaints about automatic matches. Missed-duplicate "
               "disputes ('I already have an account') come from Type II errors and are shown separately.")

    st.subheader("8. Root cause analysis of exceptions")
    st.write(f"Exceptions = missed duplicates, wrong merges and manual reviews ({len(exceptions)} records). "
             "Each incoming record is compared with the true master record to find the primary cause.")
    pct_table(rca)
    st.write("Grouping by error pattern (records with exactly the same attributes differing):")
    st.dataframe(V.error_patterns(exceptions), hide_index=True, width="stretch")
    with st.expander("All exception records"):
        st.dataframe(exceptions[["period", "test_record_id", "source_id", "input_name", "true_entity_id",
                                 "best_candidate_id", "match_score", "decision", "outcome", "disputed",
                                 "root_cause", "attributes_affected"]], hide_index=True, width="stretch")

    st.subheader("9. Threshold calibration")
    st.write("Decisions are recomputed for other thresholds while everything else stays the same. Recall is "
             "measured on automatic decisions, so a lower REVIEW threshold turns missed matches into manual "
             "reviews - higher recall but more human workload.")
    calibration = V.threshold_calibration(results)
    for varied in ["MATCH threshold", "REVIEW threshold"]:
        st.markdown(f"**Varying the {varied}**")
        pct_table(calibration[calibration["Varied"] == varied],
                  ["MATCH threshold", "REVIEW threshold", "Current setting", "precision", "recall",
                   "review_rate", "type1_error", "type2_error", "incorrect_merges"])

    st.subheader("10. Validation summary")
    for note in V.observations(periods, contribution, benchmark, rca, quality):
        show = st.success if note["Status"] == "Satisfactory" else st.warning
        show(f"**{note['Area']} - {note['Status']}:** {note['Observation']}")

    st.info("The full validation report (PDF and Word), written by the AI agent from these results, is on the "
            "**AI Agent** page.")


def full_report_panel(master_df, master, has_key):
    """Generate, download and sign off the full validation report."""
    st.write("The AI agent writes a complete model validation report: background, executive summary, validation "
             "approach, qualitative validation, quantitative validation and validation summary. Tables, figures "
             "and ratings come from the coded validation; the agent writes the analysis, observations and "
             "recommendations in small parts, and every number it writes is checked against the evidence.")
    if st.button("Generate validation report", type="primary", disabled=not has_key):
        if db.load_benchmark_results().empty:
            with st.spinner("Running the coded validation first (about 20 seconds)..."):
                db.save_monitoring_results(P.run_monitoring(master))
                db.save_benchmark_results(V.run_benchmark(master, db.load_monitoring_results()))
        generate_full_report(master_df, master)
    report = st.session_state.get("full_report")
    if not report:
        st.info("Write the validation report to download it as PDF or Word.")
        return
    meta = report["meta"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall rating", meta["overall_rating"])
    c2.metric("AI text verified", f"{meta['ai_parts']} of {meta['parts']} parts")
    c3.metric("Model", meta["model"].split("/")[-1])
    failed = [r for r in report["log"] if r["result"] != "AI text verified"]
    if failed:
        st.warning(f"{len(failed)} part(s) failed the automatic checks and use rule-based text instead; they are "
                   "listed in Annexure B of the report.")
    pdf, docx_bytes = full_report_files(report)
    name = f"search_match_validation_report_{meta['period']}"
    d1, d2 = st.columns(2)
    d1.download_button("Download report (PDF)", pdf, file_name=f"{name}.pdf", mime="application/pdf",
                       type="primary", key="full_pdf")
    d2.download_button("Download report (Word)", docx_bytes, file_name=f"{name}.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       key="full_docx")
    with st.expander("AI generation and verification log"):
        st.dataframe(pd.DataFrame(report["log"]), hide_index=True, width="stretch")

    if st.session_state.get("signoff_message"):
        st.success(st.session_state.pop("signoff_message"))
    st.markdown("**Validator sign-off**")
    with st.form("full_report_signoff"):
        reviewer = st.text_input("Validator name")
        decision = st.radio("Decision", ["Approve", "Request changes", "Reject"], horizontal=True)
        comment = st.text_area("Comment")
        if st.form_submit_button("Record decision"):
            run_at = pd.Timestamp.now().isoformat(timespec="seconds")
            db.save_agent_run({
                "run_at": run_at, "period": meta["period"], "model": meta["model"],
                "overall_status": meta["overall_rating"], "breaches": "",
                "verification_passed": not failed, "decision": decision, "reviewer": reviewer, "comment": comment,
                "report": "Full validation report"})
            report["approval"] = {"run_at": run_at, "reviewer": reviewer, "decision": decision, "comment": comment}
            st.session_state["signoff_message"] = f"Decision '{decision}' recorded; the downloads now include it."
            st.rerun()  # rebuild the downloads with the signed sign-off page


# ==================================================================
# PAGE: AI AGENT (agentic monitoring)
# ==================================================================

SOLUTION_DEFINITION = pd.DataFrame([
    ("Model", "Search & Match algorithm (deterministic fuzzy matching, FEBRL-4 data)"),
    ("Agent", f"LLM agent on Groq ({C.GROQ_MODEL}, fallback {', '.join(C.GROQ_FALLBACK_MODELS)}), temperature 0"),
    ("Tools", "Deterministic Python tools: data integrity, logic review, KPIs, thresholds, trends, potential "
              "incorrect matches, sampling, challenger benchmarking and threshold calibration, change management, "
              "root causes, policy retrieval"),
    ("Guardrails", "Every number in the report must come from the evidence; status and breaches must match the "
                   "threshold check; all sections required; instructions inside data are ignored"),
    ("Self-correction", f"Failed checks are sent back to the agent (up to {C.AGENT_MAX_REVISIONS} revisions)"),
    ("Human in the loop", "The report is advisory: a validator approves, rejects or requests changes"),
], columns=["Component", "Implementation"])

def show_agent_run(run):
    ver = run["verification"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall status", run["overall_status"] or "-")
    c2.metric("Verification", "Passed" if ver["passed"] else "Failed")
    c3.metric("Revisions", run["revisions"])
    c4.metric("Run time", f"{run['seconds']} s")
    breaches = ", ".join(f"{b['kpi']} ({b['status']})" for b in run["breaches"]) or "none"
    st.caption(f"Model: {run['model']} · Numbers checked: {ver.get('numbers_checked', 0)} · Breaches: {breaches}")
    with st.expander("Agent trace (each workflow step)"):
        st.dataframe(pd.DataFrame([{"Step": s["step"], "Actor": s.get("actor", ""), "Stage": s.get("stage", s["type"]),
                                    "Tool": s.get("tool", ""), "Detail": s.get("detail", "")} for s in run["trace"]]),
                     hide_index=True, width="stretch")
    with st.expander("Guardrail checks"):
        st.dataframe(pd.DataFrame([{"Check": k.replace("_", " ").capitalize(), "Result": "Pass" if v else "Fail"}
                                   for k, v in ver.get("checks", {}).items()]), hide_index=True, width="stretch")
        for p in ver.get("problems", []):
            st.warning(p)
    st.markdown(run["report"] or "_No report was submitted._")
    col1, col2 = st.columns(2)
    with col1:
        pdf_download(run, key="pdf_agent")
    col2.download_button("Download report (Markdown)", run["report"], file_name=f"agent_report_{run['period']}.md")


def page_agent(master):
    import agent as A

    st.header("AI Agent - Agentic Monitoring")
    st.write("An LLM agent monitors and validates the Search & Match algorithm following the concept note "
             "process flow. The agent never calculates: deterministic tools produce every figure, the agent "
             "interprets them and writes the report, and guardrails check the report before a human approves it.")
    st.dataframe(SOLUTION_DEFINITION, hide_index=True, width="stretch")
    st.markdown("**Process flow:** Input data → Data integrity → Search and match logic review → Performance "
                "monitoring → Threshold and breach detection → Breaches, trends and anomalies → Sampling and "
                "validation → Change management → Findings → Report → Human approval")

    has_key = bool(A.get_api_key())
    if not has_key:
        st.warning("No Groq API key found. Add GROQ_API_KEY to the Streamlit secrets (or a local "
                   "groq_api_key.txt) to run the agent.")

    results = db.load_monitoring_results()
    tab_report, tab_run, tab_gov = st.tabs(["Validation report", "Monthly monitoring report", "Governance"])

    with tab_report:
        full_report_panel(get_master()[0], master, has_key)

    with tab_run:
        if results.empty:
            st.info("Run the monitoring first (Monitoring or Validation page).")
        else:
            periods = list(results["period"].unique())
            col1, col2 = st.columns(2)
            period = col1.selectbox("Assessed period", periods, index=len(periods) - 1)
            propose = col2.checkbox("Assess a proposed threshold change")
            change = None
            if propose:
                c1, c2 = st.columns(2)
                mt = c1.number_input("Proposed MATCH threshold", 50, 100, C.MATCH_THRESHOLD)
                rt = c2.number_input("Proposed REVIEW threshold", 40, 95, C.REVIEW_THRESHOLD)
                change = {k: int(v) for k, v, cur in [("match_threshold", mt, C.MATCH_THRESHOLD),
                                                      ("review_threshold", rt, C.REVIEW_THRESHOLD)] if v != cur}
            if st.button("Run monitoring agent", disabled=not has_key):
                try:
                    with st.status("Agent running...", expanded=True) as status:
                        ctx = agent_context(results, master, period, proposed_change=change or None)
                        run = A.run_agent(ctx, on_step=lambda s: st.write(
                            f"Step {s['step']}: {s.get('stage') or s['type']} "
                            f"{('- ' + s['tool']) if s.get('tool') else ''}"))
                        status.update(label="Agent finished", state="complete", expanded=False)
                    st.session_state["agent_run"] = run
                except Exception as e:
                    st.error(f"The agent could not finish: {e}")
            run = st.session_state.get("agent_run")
            if run:
                show_agent_run(run)
                st.subheader("Human approval")
                with st.form("approval"):
                    reviewer = st.text_input("Validator name")
                    decision = st.radio("Decision", ["Approve", "Request changes", "Reject"], horizontal=True)
                    comment = st.text_area("Comment")
                    recorded = st.form_submit_button("Record decision")
                    if recorded:
                        run_at = pd.Timestamp.now().isoformat(timespec="seconds")
                        db.save_agent_run({
                            "run_at": run_at, "period": run["period"],
                            "model": run["model"], "overall_status": run["overall_status"],
                            "breaches": ", ".join(f"{b['kpi']}:{b['status']}" for b in run["breaches"]),
                            "verification_passed": run["verification"]["passed"], "decision": decision,
                            "reviewer": reviewer, "comment": comment, "report": run["report"]})
                        run["approval"] = {"run_at": run_at, "reviewer": reviewer, "decision": decision,
                                           "comment": comment}
                        st.success(f"Decision '{decision}' recorded.")
                if run.get("approval"):
                    pdf_download(run, label="Download signed report (PDF)", key="pdf_signed")

    with tab_gov:
        st.markdown(f"""
- **Human in the loop:** the agent's report is advisory. Only a validator's recorded decision makes it final.
- **No autonomous changes:** the agent cannot change thresholds or data; it only reads tool outputs.
- **Change management:** threshold changes are classified by impact - up to 1.0 pp minor, up to 3.0 pp moderate,
  above that major (Model Risk / Governance Committee).
- **Data protection:** only aggregated KPIs and record IDs from the public FEBRL benchmark are sent to the LLM.
- **Audit trail:** every decision and expert review is stored in the database with time and reviewer.
- **Model:** `{C.GROQ_MODEL}` at temperature {C.AGENT_TEMPERATURE}; fallbacks when rate-limited:
  {', '.join(f'`{m}`' for m in C.GROQ_FALLBACK_MODELS)}.
""")
        decisions = db.load_agent_runs()
        st.subheader("Approval history")
        if decisions.empty:
            st.info("No decisions recorded yet.")
        else:
            st.dataframe(decisions.drop(columns=["report"]), hide_index=True, width="stretch")


# ==================================================================
# NAVIGATION
# ==================================================================

def main():
    master_df, master = get_master()
    st.sidebar.title("Search, Match & Merge")
    st.sidebar.caption("Phase 1 prototype - deterministic matching")
    page = st.sidebar.radio("Go to", ["Search & Match", "Test Data", "Performance", "Monitoring", "Validation",
                                      "AI Agent", "Methodology"])
    st.sidebar.divider()
    st.sidebar.caption(f"Dataset: {C.DATASET_NAME} benchmark")
    st.sidebar.caption(f"Master records: {len(master_df):,}")

    if page == "Search & Match":
        page_search(master_df, master)
    elif page == "Test Data":
        page_test_data(master_df, master)
    elif page == "Performance":
        page_performance()
    elif page == "Monitoring":
        page_monitoring(master)
    elif page == "Validation":
        page_validation(master_df, master)
    elif page == "AI Agent":
        page_agent(master)
    else:
        page_methodology()


main()
