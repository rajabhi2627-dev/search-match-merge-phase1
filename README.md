# Search, Match & Merge – Phase 1 Prototype

## 1. Project objective

Demonstrate a **deterministic, explainable** Search, Match & Merge (entity resolution) algorithm and
evaluate how well it performs on synthetic test data with automatically known ground truth.

## 2. Phase 1 scope

Included: a local SQLite master database (~1,000 synthetic entities), a Search & Match form, a
rule-based matching algorithm, a synthetic test-data generator with ground truth, a batch run on
100 test records, performance metrics, confusion matrices and a simple local dashboard.

Not included: AI agents, LLMs, RAG, machine learning, APIs, authentication, cloud deployment.
The same input always produces the same output.

## 3. Matching workflow

```
Incoming record
  -> Normalisation
  -> Candidate search (blocking)
  -> Exact matching
  -> Field-specific fuzzy matching
  -> Field-level scores
  -> Weighted overall score
  -> Conflict detection
  -> Candidate ranking
  -> Score gap
  -> Final decision (MATCH / MANUAL REVIEW / NO MATCH)
```

The Search & Match page and the 100-record test run call the **same** function: `matching.match_record()`.

**Normalisation** (identical for master and incoming records)

| Field | Rule |
|---|---|
| Text (name, address, company, city) | lowercase, trim, punctuation → space, collapse spaces |
| Address | expand abbreviations: Rd → road, St → street, MG → mahatma gandhi, Sec → sector, Ngr → nagar |
| Company | drop legal suffixes: Pvt, Private, Ltd, Limited, LLP, Inc |
| City | map old names: Bangalore → bengaluru, Bombay → mumbai, Gurgaon → gurugram, … |
| Phone | digits only; drop leading `00`, `91` country code (12 digits) or trunk `0` (11 digits) → 10 digits |
| Email | lowercase, remove all spaces |
| Date of birth | convert to `YYYY-MM-DD` (day-first formats such as `12/04/1998` accepted); unreadable → missing |

**Candidate search**: a master record becomes a candidate if at least one rule is true:
exact phone, exact email, similar name (WRatio ≥ 70), or same city + partly similar name (WRatio ≥ 50).
At most 25 candidates are fully scored.

## 4. Database

`data/matching.db` (SQLite, Python's built-in `sqlite3`). It is created automatically on first start.

| Table | Content |
|---|---|
| `master_records` | 1,000 entities: entity_id, name, email, phone, address, date_of_birth, company, city |
| `test_records` | the 100 generated incoming records + `true_entity_id`, `true_match_status`, `scenario` |
| `match_results` | the output of *Run Matching* (decision, scores, outcome) |

Entity `10001` is the documented example: Rahul Kumar, rahul.kumar@gmail.com, 9876543210,
12 MG Road Patna, 1998-04-12, ABC Technologies, Patna. To rebuild everything, delete `data/matching.db`.

## 5. Fuzzy matching methods

Every field is first compared exactly (identical normalised values → 100). Otherwise:

| Field | Method | Why |
|---|---|---|
| Name | RapidFuzz `WRatio` | tolerant of typos, word order, partial names |
| Email | Levenshtein similarity | strong identifier, character-level edits only |
| Phone | Digit comparison (custom) | numbers are not words; generic fuzzy matching is misleading |
| Address | RapidFuzz `token_set_ratio` | handles word order, extra/missing words |
| DOB | Exact | a different date means a different person |
| Company | RapidFuzz `WRatio` | handles spelling variants and abbreviations |
| City | Exact, then `ratio` | small typos only |

**Phone logic**: exact 10-digit match = 100; same length with 1 differing digit = 90; 2 differing digits = 75;
more = 0. If lengths differ, last 7 digits identical = 80, otherwise 0.

**DOB logic**: same = 100, different = 0, missing on either side = field excluded (see below).

**Missing values**: if a field is empty on either side, it is left out of the weighted score and the
remaining weights are rescaled to 100%. If the compared fields carry less than 60% of the total
weight, the result can be at most MANUAL REVIEW.

## 6. Distance parameters

A field **agrees** if `similarity ≥ threshold` **or** `edit distance ≤ max distance`; its score is then the
similarity. If it does not agree, the field score is **0**.

| Field | Method | Threshold | Max distance |
|---|---|---|---|
| Name | WRatio | 70 | 3 |
| Email | Levenshtein | 85 | 2 |
| Phone | Digit comparison | 75 (custom) | 2 digits |
| Address | Token Set Ratio | 60 | 5 |
| DOB | Exact | 100 | 0 |
| Company | WRatio | 70 | 3 |
| City | Ratio | 85 | 2 |

## 7. Field weights

Name 25%, Email 25%, Phone 25%, Address 10%, DOB 5%, Company 5%, City 5% (total 100%).

```
Overall = 0.25·Name + 0.25·Email + 0.25·Phone + 0.10·Address + 0.05·DOB + 0.05·Company + 0.05·City
```

## 8. Decision thresholds

**Conflict rules** (any of these blocks an automatic MATCH):

1. Strong identifiers disagree: one of name/email/phone ≥ 90 while another < 50.
2. Date of birth present on both sides and different.
3. Exact phone points to one entity, exact email to another.
4. Too little evidence (compared fields < 60% of the total weight).

**Score gap** = best score − second-best score. A gap < 5 means two candidates are too close →
MANUAL REVIEW, unless there is no meaningful second candidate (none, or its score < 65).

| Decision | Rule |
|---|---|
| MATCH | score ≥ 85, no conflict, score gap ≥ 5 |
| MANUAL REVIEW | score 65–85, or score gap < 5, or any conflict |
| NO MATCH | score < 65 (or no candidate) |

All values are in `config.py`.

## 9. Test data generation

*Generate 100 Test Records* creates exactly 100 incoming records (fixed seed 42 → reproducible).
Each record carries `test_record_id`, `true_entity_id` and `true_match_status`, so no manual labelling is needed.

| Scenario | Records | Ground truth |
|---|---|---|
| Clear match (formatting only) | 13 | MATCH |
| Name spelling error | 10 | MATCH |
| Email variation | 8 | MATCH |
| Phone formatting | 10 | MATCH |
| Address variation | 9 | MATCH |
| Company spelling variation | 8 | MATCH |
| Multiple noisy fields | 10 | MATCH |
| Changed contact details (new email *and* phone) | 2 | MATCH |
| Ambiguous – sparse record (no email/phone/DOB) | 5 | MATCH |
| Ambiguous – family member (same phone/address) | 5 | NO MATCH |
| New person | 15 | NO MATCH |
| Look-alike (same name and city) | 5 | NO MATCH |

## 10. Performance metrics

| Outcome | Meaning |
|---|---|
| TP | MATCH to the correct entity |
| FP | MATCH for a new person, or to the wrong entity (a false merge) |
| TN | NO MATCH for a new person |
| FN | NO MATCH although the person exists |
| REVIEW | MANUAL REVIEW – not an automatic decision |

MANUAL REVIEW is neither correct nor incorrect. It is excluded from the binary metrics and reported as
the **manual review rate**. The Performance page also shows a sensitivity view where review is treated as NO MATCH.

- Accuracy = (TP + TN) / (TP + FP + TN + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- False Positive Rate = FP / (FP + TN)
- False Negative Rate = FN / (FN + TP)

Division by zero returns 0. Two confusion matrices are shown: a binary MATCH vs NO MATCH view and the
full 3-decision distribution.

## 11. Local installation

```
cd search_match_project
pip install -r requirements.txt
```

## 12. Local execution

```
streamlit run app.py
```

Open http://localhost:8501. Streamlit usage statistics are switched off in `.streamlit/config.toml`.
No API keys are required. All data is synthetic.

## Online demo (Streamlit Community Cloud)

The app can be deployed from this GitHub repository at https://share.streamlit.io
(repository: this repo, branch: `main`, main file: `app.py`). The SQLite database is not stored in
the repository; it is generated automatically when the app starts.

Demo flow: **Search & Match** → **Test Data** (Generate 100 Test Records → Run Matching) → **Performance** → **Methodology**.

Run the tests:

```
python -m pytest -q
```

## Project structure

```
search_match_project/
├── app.py              Streamlit interface (4 pages)
├── database.py         SQLite creation and queries
├── matching.py         normalisation, candidate search, scoring, conflicts, decision
├── data_generator.py   master data and test data generation
├── performance.py      batch run, metrics, confusion matrices
├── config.py           all weights, thresholds and test-data settings
├── requirements.txt
├── data/matching.db    created automatically
└── tests/test_matching.py
```
