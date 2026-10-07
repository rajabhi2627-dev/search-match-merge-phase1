# Search, Match & Merge – Phase 1 Prototype + Week 4 Performance Monitoring

## 1. Project objective

Demonstrate a **deterministic, explainable** Search, Match & Merge (entity resolution) algorithm,
evaluate it on the public **FEBRL-4 record-linkage benchmark** with its official ground truth, and
monitor its performance over time with a defined set of KPIs.

## 2. Scope

Included: a SQLite master database, a Search & Match form, a rule-based matching algorithm,
100 benchmark test records with ground truth, performance metrics, confusion matrices,
Week 4 performance monitoring (KPI dictionary, 6 monthly batches, trends, Green/Amber/Red status)
and a simple Streamlit dashboard.

Not included: AI agents, LLMs, RAG, machine learning, APIs, authentication.
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

The Search & Match page, the 100-record test run and the monitoring run all call the **same**
function: `matching.match_record()`.

**Normalisation** (identical for master and incoming records)

| Field | Rule |
|---|---|
| Text (name, address, company, city) | lowercase, trim, punctuation → space, collapse spaces |
| Address | expand abbreviations: Rd → road, St → street, MG → mahatma gandhi, Sec → sector, Ngr → nagar |
| Company | drop legal suffixes: Pvt, Private, Ltd, Limited, LLP, Inc |
| City | map old names: Bangalore → bengaluru, Bombay → mumbai, … |
| Phone / ID | digits only; for phone numbers drop leading `00`, `91` country code or trunk `0` |
| Email | lowercase, remove all spaces |
| Date of birth | convert to `YYYY-MM-DD` (`YYYYMMDD` and day-first formats accepted); invalid → missing |

**Candidate search**: a master record becomes a candidate if at least one rule is true:
exact phone/ID, exact email, similar name (WRatio ≥ 70), or same city + partly similar name (WRatio ≥ 50).
At most 25 candidates are fully scored.

## 4. Dataset – FEBRL-4 benchmark

FEBRL (Freely Extensible Biomedical Record Linkage) is a public benchmark from the Australian National
University: P. Christen, *"Febrl – a freely available record linkage system with a graphical user interface"*,
Australasian Workshop on Health Data and Knowledge Management, 2008. FEBRL-4 contains 5,000 original person
records (`data/febrl/dataset4a.csv`) and one corrupted duplicate of each (`data/febrl/dataset4b.csv`) with
typos, phonetic changes, swapped or replaced names, missing values and changed digits. Record `rec-N-dup-0` is
the same person as `rec-N-org`, which gives the official ground truth. The files are the copies distributed with
the open-source `recordlinkage` Python package.

| Project field | FEBRL column(s) |
|---|---|
| Name | given_name + surname |
| Phone / ID | soc_sec_id (strong numeric identifier, compared digit by digit) |
| Address | street_number + address_1 + address_2 + postcode |
| Date of birth | date_of_birth |
| City | suburb |
| Email, Company | not in FEBRL → empty → excluded from the score |

**Database** `data/matching.db` (SQLite, built automatically on first start; rebuilt if the dataset changes):

| Table | Content |
|---|---|
| `master_records` | 1,000 FEBRL originals (fixed random sample); entity_id = FEBRL record number N |
| `test_records` | 100 incoming records + `true_entity_id`, `true_match_status`, FEBRL `source_id` |
| `match_results` | output of *Run Matching* |
| `monitoring_results` | output of the 6-month monitoring run |

## 5. Fuzzy matching methods

Every field is first compared exactly (identical normalised values → 100). Otherwise:

| Field | Method | Why |
|---|---|---|
| Name | RapidFuzz `WRatio` | tolerant of typos, word order, partial names |
| Email | Levenshtein similarity | strong identifier, character-level edits only |
| Phone / ID | Digit comparison (custom) | numbers are not words; generic fuzzy matching is misleading |
| Address | RapidFuzz `token_set_ratio` | handles word order, extra/missing words |
| DOB | Exact | a different date means a different person |
| Company | RapidFuzz `WRatio` | handles spelling variants and abbreviations |
| City | Exact, then `ratio` | small typos only |

**Phone / ID logic**: exact = 100; same length with 1 differing digit = 90; 2 differing digits = 75; more = 0.
If lengths differ, last 7 digits identical = 80, otherwise 0.

**Missing values**: a field that is empty on either side is left out and the remaining weights are rescaled.
With FEBRL (no email, no company) the effective weights are Name 35.7%, Phone/ID 35.7%, Address 14.3%,
DOB 7.1%, City 7.1%. If the compared fields carry less than 60% of the total weight, the result is at most
MANUAL REVIEW.

## 6. Distance parameters

A field **agrees** if `similarity ≥ threshold` **or** `edit distance ≤ max distance`; its score is then the
similarity. Otherwise the field score is **0**.

| Field | Method | Threshold | Max distance |
|---|---|---|---|
| Name | WRatio | 70 | 3 |
| Email | Levenshtein | 85 | 2 |
| Phone / ID | Digit comparison | 75 (custom) | 2 digits |
| Address | Token Set Ratio | 60 | 5 |
| DOB | Exact | 100 | 0 |
| Company | WRatio | 70 | 3 |
| City | Ratio | 85 | 2 |

## 7. Field weights

Name 25%, Email 25%, Phone/ID 25%, Address 10%, DOB 5%, Company 5%, City 5% (total 100%).

## 8. Decision thresholds

**Conflict rules** (any of these blocks an automatic MATCH):

1. Strong identifiers disagree: one of name/email/phone ≥ 90 while another < 50.
2. Date of birth present on both sides and different.
3. Exact phone points to one entity, exact email to another.
4. Too little evidence (compared fields < 60% of the total weight).

**Score gap** = best score − second-best score. A gap < 5 → MANUAL REVIEW, unless there is no meaningful
second candidate (none, or its score < 65).

| Decision | Rule |
|---|---|
| MATCH | score ≥ 85, no conflict, score gap ≥ 5 |
| MANUAL REVIEW | score 65–85, or score gap < 5, or any conflict |
| NO MATCH | score < 65 (or no candidate) |

All values are in `config.py`.

## 9. Test data

*Generate 100 Test Records* takes 75 FEBRL duplicates of people in the master database (true MATCH) and
25 duplicates of people not in the master database (true NO MATCH). The selection uses a fixed seed.
The "scenario" column shows how many fields FEBRL corrupted in each record.

## 10. Performance metrics

| Outcome | Meaning |
|---|---|
| TP | MATCH to the correct entity |
| FP | MATCH for a new person, or to the wrong entity (a false merge) |
| TN | NO MATCH for a new person |
| FN | NO MATCH although the person exists |
| REVIEW | MANUAL REVIEW – not an automatic decision |

MANUAL REVIEW is excluded from the binary metrics and reported as the manual review rate.

- Accuracy = (TP + TN) / (TP + FP + TN + FN)
- Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2PR / (P + R)
- False Positive Rate = FP / (FP + TN), False Negative Rate = FN / (FN + TP)

## 11. Week 4 – Performance monitoring

| KPI | Formula | Green | Amber | Red |
|---|---|---|---|---|
| HIT Rate | MATCH decisions / total records | ≥ 60% | ≥ 50% | < 50% |
| Precision | TP / (TP + FP) | ≥ 98% | ≥ 95% | < 95% |
| Recall | TP / (TP + FN) | ≥ 95% | ≥ 90% | < 90% |
| False Positive Rate | FP / (FP + TN) | ≤ 2% | ≤ 5% | > 5% |
| False Negative Rate | FN / (FN + TP) | ≤ 5% | ≤ 10% | > 10% |
| Manual Review Rate | MANUAL REVIEW / total records | ≤ 15% | ≤ 25% | > 25% |
| Merge Error Rate | FP / MATCH decisions | ≤ 1% | ≤ 3% | > 3% |
| Dispute Rate | disputes / total records | ≤ 1% | ≤ 3% | > 3% |

- **Monthly batches:** 6 months, each with 75 duplicates of existing people + 25 new people, none used in
  the test set.
- **Data-quality drift:** duplicates are ordered by the number of fields FEBRL corrupted; the first month
  receives the cleanest records, the last month the most corrupted.
- **Disputes (simulated, fixed seed):** 80% of wrong merges and 50% of missed duplicates lead to a customer
  complaint.
- **Observations:** rule-based – every KPI that is not Green in the latest month, or moved by ≥ 2 percentage
  points, is reported.

## 12. Installation and execution

```
cd search_match_project
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. No API keys are required.

Demo flow: **Search & Match** → **Test Data** (Generate 100 Test Records → Run Matching) → **Performance**
→ **Monitoring** (Run Monthly Monitoring) → **Methodology**.

Run the tests: `python -m pytest -q`

## Online demo (Streamlit Community Cloud)

The app can be deployed from this GitHub repository at https://share.streamlit.io
(branch `main`, main file `app.py`). The SQLite database is not stored in the repository; it is built
automatically from the bundled FEBRL files when the app starts.

## Project structure

```
search_match_project/
├── app.py              Streamlit interface (5 pages)
├── database.py         SQLite creation and queries
├── matching.py         normalisation, candidate search, scoring, conflicts, decision
├── data_generator.py   FEBRL loading: master, test and monitoring data with ground truth
├── performance.py      batch run, metrics, confusion matrices, monitoring KPIs
├── config.py           weights, thresholds, dataset and KPI settings
├── requirements.txt
├── data/febrl/         FEBRL-4 benchmark files
├── data/matching.db    created automatically
└── tests/test_matching.py
```
