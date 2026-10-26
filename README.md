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

The matching algorithm itself uses no machine learning: the same input always produces the same output.
An LLM is used only in the **AI Agent** page (section 13), which monitors and validates the algorithm
and is itself validated.

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
| Text (name, address, city) | lowercase, trim, punctuation → space, collapse spaces |
| Address | expand street abbreviations: Rd → road, St → street, Ave → avenue, Cres → crescent, Pl → place, … |
| Soc. Sec. ID | digits only (spaces and dashes removed) |
| Date of birth | convert to `YYYY-MM-DD` (`YYYYMMDD` and day-first formats accepted); invalid → missing |

**Candidate search**: a master record becomes a candidate if at least one rule is true:
exact Soc. Sec. ID, similar name (WRatio ≥ 70), or same city + partly similar name (WRatio ≥ 50).
At most 25 candidates are fully scored. Blocking only decides who is compared; the ID is then scored digit by
digit, so an ID with a typo still counts once the person is found by name.

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
| Soc. Sec. ID | soc_sec_id (strong numeric identifier, compared digit by digit) |
| Address | street_number + address_1 + address_2 + postcode |
| Date of birth | date_of_birth |
| City | suburb |

Only FEBRL's own fields are used - no extra fields are invented.

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
| Soc. Sec. ID | Digit comparison (custom) | numbers are not words; generic fuzzy matching is misleading |
| Address | RapidFuzz `token_set_ratio` | handles word order, extra/missing words |
| DOB | Exact | a different date means a different person |
| City | Exact, then `ratio` | small typos only |

**Soc. Sec. ID logic**: count the digit edits (change, add or remove one digit): 0 = 100, 1 = 90, 2 = 75,
more = 0 (a different ID).

**Missing values**: a field that is empty on either side is left out and the remaining weights are rescaled.
If the compared fields carry less than 70% of the total weight (for example the ID is missing), the result is
at most MANUAL REVIEW.

## 6. Distance parameters

A field **agrees** if `similarity ≥ threshold` **or** `edit distance ≤ max distance`; its score is then the
similarity. Otherwise the field score is **0**.

| Field | Method | Threshold | Max distance |
|---|---|---|---|
| Name | WRatio | 70 | 3 |
| Soc. Sec. ID | Digit comparison | 75 (custom) | 2 digits |
| Address | Token Set Ratio | 60 | 5 |
| DOB | Exact | 100 | 0 |
| City | Ratio | 85 | 2 |

## 7. Field weights

Name 35%, Soc. Sec. ID 35%, Address 15%, DOB 7.5%, City 7.5% (total 100%).

## 8. Decision thresholds

**Conflict rules** (any of these blocks an automatic MATCH):

1. Strong identifiers disagree: name or Soc. Sec. ID ≥ 90 while the other < 50.
2. Date of birth present on both sides and different.
3. The Soc. Sec. ID matches one entity exactly, but the best overall score belongs to another entity.
4. Too little evidence (compared fields < 70% of the total weight).
5. Look-alike guard: Soc. Sec. ID close but not identical and no date of birth to confirm it (possible relative).

**Score gap** = best score − second-best score. A gap < 5 → MANUAL REVIEW, unless there is no meaningful
second candidate (none, or its score < 65).

| Decision | Rule |
|---|---|
| MATCH | score ≥ 85, no conflict, score gap ≥ 5 |
| MANUAL REVIEW | score 65–85, or score gap < 5, or any conflict |
| NO MATCH | score < 65 (or no candidate), unless strongly corroborated |

**Strong corroboration:** a best candidate below 65 goes to MANUAL REVIEW (never to MATCH) when the name or the
Soc. Sec. ID scores ≥ 90, the date of birth is identical, and the address or the city agrees. The validation showed
that almost every missed match had one strong identifier completely different (a changed surname or a mistyped ID)
while everything else agreed. Effect on the validation sample (6 months, 600 records): Type II error 11.1% → 1.3%,
recall 87.9% → 98.4%, incorrect merges 5 → 0 (with the look-alike guard), manual review rate 6.0% → 14.2%.

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

**Validation-report definitions** (used on every page). *Matches generated* = MATCH + MANUAL REVIEW decisions;
*confirmed matches* = generated matches whose best candidate is the true person; *actual matches* = records
whose person exists in the master.

- Hit Rate = confirmed matches / total matches generated × 100
- Dispute Rate = disputed matches / total matches generated × 100
- Incorrect Merge Rate = incorrect merges / total matches generated × 100
- Type I Error Rate = false positives / total matches generated × 100
- Type II Error Rate = false negatives / total actual matches × 100

## 11. Week 4 – Performance monitoring

| KPI | Formula | Green | Amber | Red |
|---|---|---|---|---|
| Hit Rate | confirmed matches / total matches generated | ≥ 95% | ≥ 90% | < 90% |
| Precision | TP / (TP + FP) | ≥ 98% | ≥ 95% | < 95% |
| Recall | TP / (TP + FN) | ≥ 95% | ≥ 90% | < 90% |
| Type I Error Rate | false positives / total matches generated | ≤ 2% | ≤ 5% | > 5% |
| Type II Error Rate | false negatives / total actual matches | ≤ 5% | ≤ 10% | > 10% |
| Manual Review Rate | MANUAL REVIEW / total records | ≤ 15% | ≤ 25% | > 25% |
| Incorrect Merge Rate | incorrect merges / total matches generated | ≤ 0.5% | ≤ 1% | > 1% |
| Dispute Rate | disputed matches / total matches generated | ≤ 2% | ≤ 3% | > 3% |

- **Monthly batches:** 6 months, each with 75 duplicates of existing people + 25 new people, none used in
  the test set.
- **Data-quality drift:** duplicates are ordered by the number of fields FEBRL corrupted; the first month
  receives the cleanest records, the last month the most corrupted.
- **Hard cases (not in FEBRL):** from February a few new-person records are replaced by *look-alike relatives*
  (same name and address as an existing customer, DOB missing, Soc. Sec. ID 1 digit different) and *shared IDs*
  (a new person entered with an existing customer's ID). Since version 2.4 look-alikes are stopped by
  the look-alike guard (manual review) and shared IDs by the other conflict rules.
- **Disputes (simulated, fixed seed):** 80% of wrong merges lead to a "does not belong to me" dispute (counted
  in the dispute rate); 50% of missed duplicates lead to an "I already have an account" complaint (shown separately).
- **Observations:** rule-based – every KPI that is not Green in the latest month, or moved by ≥ 2 percentage
  points, is reported.

## 12. Independent validation

The **Validation** page follows the structure of a model validation report and uses the 600 monitoring records
(one reporting period per month):

| Analysis | What it shows |
|---|---|
| Validation data summary | records processed, matches generated, auto-merged, manual review, new subjects, disputes, incorrect merges |
| Data quality checks | completeness and validity of every attribute (incoming and master) |
| Hit rate and dispute rate by period | hit rate = confirmed / generated matches; dispute rate = disputed / generated matches |
| Search path analysis | which search rule found each correct match, and each attribute's share of the match score |
| Challenger benchmarking | Jaro-Winkler, Levenshtein, Damerau-Levenshtein, Soundex (name) and Jaccard (address), each replacing one production comparator |
| Type I / Type II errors | FP / (FP + TN) and FN / (FN + TP) by period |
| Dispute and incorrect merge analysis | disputed matches, incorrect merges and missed-duplicate disputes by period |
| Root cause analysis | primary cause of each exception (compared with the true master record) and grouping by error pattern |
| Threshold calibration | precision, recall and review rate when the MATCH or REVIEW threshold is changed |
| Validation summary | rule-based observations against the limits in `config.py` |

Root cause grouping is rule-based (records with the same differing attributes form a group); no clustering model is used.

## 13. Agentic monitoring and validation framework (AI Agent page)

Follows the concept note *AI Model Validation Framework* (Student 2: Search and Match algorithm with agentic
monitoring and validation).

| Step | Implementation |
|---|---|
| Evidence pipeline | deterministic tools (`agent_tools.py`): data integrity, logic review, KPIs, threshold and breach detection, trends and anomalies, potential incorrect matches, Cochran sampling, challenger benchmarking and threshold calibration (from the Validation page), change management, root causes |
| LLM agent | Groq `openai/gpt-oss-120b` (waits up to 45 s for it when rate-limited, then falls back to `gpt-oss-20b`; production models only), temperature 0; retrieves policy sections with a tool and writes the 10-section report (up to 1,000 words) |
| Guardrails | every number in the report must appear in the evidence; status and breaches must equal the threshold check; all sections present; data issues reported; instructions inside data are ignored |
| Self-correction | failed checks are returned to the agent (up to 2 revisions) |
| Human in the loop | the validator approves, rejects or requests changes; decisions are stored |
| PDF report | `report_pdf.py`: AI report + guardrail checks + all 10 coded validation sections + validator sign-off + agent trace |

**Validation report written by the AI agent.** On the **AI Agent** page, *Validation report* tab, click
*Generate validation report* (the coded validation runs first if needed). `validation_report.py` produces a
complete model validation report - Background, Executive Summary, Validation Approach, Qualitative Validation
(documentation, purpose, data quality, model design, algorithms, disputes, governance, monitoring), Quantitative
Validation (data overview, hit and dispute rates, search paths, challengers, Type I/II errors, disputes, root causes,
threshold calibration), Validation Summary, sign-off and annexures - downloadable as PDF (`report_pdf.py`) and Word
(`report_docx.py`). Tables, figures, the model description and every rating come from code; the agent writes the
analysis, observations and recommendations in eight small verified parts (numbers traced to the evidence, ratings equal
to the rule-based ratings, fixed review areas present). All text is written by the AI: a part that fails gets up to
3 correction rounds; if it still fails, the AI text is kept, marked in the report and listed in Annexure B. The validator records the decision on the same tab, and the downloads then include it.
Organisation and validator names are set in `config.py` (`REPORT_ORGANISATION`, `REPORT_VALIDATOR`).

**Validation of the agent** (`agent_validation.py`): 13 golden scenarios (baseline, all Green, dispute and hit
rate breaches, incorrect merges, rising Type II trend, missing period, out-of-range value, contradictory figures,
missing value, proposed threshold change, prompt injection). The expected answer comes from the deterministic
threshold check. Tests: hallucination, numeric integrity, stability (3 repeats), guardrails and refusal, retrieval
integrity, workflow consistency, adversarial / stress, and expert review (1-5 rubric). Each test is Green (100 %),
Amber (≥ 80 %) or Red; any Red = Fail, any Amber or pending review = Pass (Conditional), otherwise Pass.

Run the suite from the command line with `python run_agent_validation.py`; results are saved to
`data/agent_validation_results.csv` and shown on the dashboard.

**API key:** set `GROQ_API_KEY` as an environment variable, in Streamlit secrets, or in a `groq_api_key.txt`
next to the project (git-ignored). The key is never stored in the repository.
On a paid Groq plan, also set `GROQ_TOKENS_PER_MINUTE` (e.g. `250000`) in the secrets so the agent can use the
higher limit; the default `8000` matches the free tier. Only aggregated KPIs and public
FEBRL record IDs are sent to the LLM.

## 14. Installation and execution

```
cd search_match_project
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Only the AI Agent page needs a Groq API key.

Demo flow: **Search & Match** → **Test Data** (Generate 100 Test Records → Run Matching) → **Performance**
→ **Monitoring** (Run Monthly Monitoring) → **Validation** → **AI Agent** → **Methodology**.

Run the tests: `python -m pytest -q`

## Online demo (Streamlit Community Cloud)

The app can be deployed from this GitHub repository at https://share.streamlit.io
(branch `main`, main file `app.py`). The SQLite database is not stored in the repository; it is built
automatically from the bundled FEBRL files when the app starts. For the AI Agent page add
`GROQ_API_KEY = "..."` under **App settings → Secrets**.

## Project structure

```
search_match_project/
├── app.py              Streamlit interface (7 pages)
├── agent.py            AI agent: evidence pipeline, Groq LLM loop, guardrails, self-correction
├── agent_tools.py      deterministic tools and policy used by the agent
├── agent_validation.py golden scenarios and validation tests of the agent
├── run_agent_validation.py  command-line runner for the agent validation suite
├── database.py         SQLite creation and queries
├── matching.py         normalisation, candidate search, scoring, conflicts, decision
├── data_generator.py   FEBRL loading: master, test and monitoring data with ground truth
├── performance.py      batch run, metrics, confusion matrices, monitoring KPIs
├── validation.py       independent validation analyses and challenger algorithms
├── validation_report.py  full validation report: evidence, AI parts, guardrails, document structure
├── report_pdf.py       PDF rendering (monitoring report and full validation report)
├── report_docx.py      Word rendering of the full validation report
├── config.py           weights, thresholds, dataset and KPI settings
├── requirements.txt
├── data/febrl/         FEBRL-4 benchmark files
├── data/matching.db    created automatically
├── data/agent_validation_results.csv  latest agent validation run
└── tests/              test_matching.py, test_agent.py, test_report_pdf.py, test_validation_report.py (no LLM calls)
```

**Groq token limits.** Every request is sized so that prompt + reserved answer stay below the tokens-per-minute
limit (8,000 on the free tier) with headroom; a "request too large" reply shrinks the answer budget and retries.
The full report is written in eight small parts with word limits, a revision resends only the failed parts, and when
Groq is busy the call waits and retries. All report text is written by the AI.
