# Project state — Search, Match & Merge

Updated: 17 November 2026  
Authors: Abhinav Raj, Swagat Waghmare  
Model version: **2.5** (git `7e8a904` on `main`)  
Remote: https://github.com/rajabhi2627-dev/search-match-merge-phase1

Phase 1 prototype: a deterministic Search, Match & Merge model for person records, plus monthly monitoring, independent validation, and an AI-written quantitative validation report with automatic checks and human sign-off. Concept note: Project 3, AI Model Validation Framework. KPI structure follows the Grant Thornton Search & Match validation report.

## Current results (FEBRL-4, recomputed after v2.5)

Six months, January–June 2026, 100 records each (75 existing people, 25 new). Master: 1,000 people. Five fields: name, Soc. Sec. ID, address, date of birth, city.

| | Value |
|---|---|
| Decisions | 364 MATCH, 85 MANUAL REVIEW, 151 NO MATCH |
| Proposals (hit-rate denominator) | 449 = 364 + 85 |
| Confirmed proposals | 444 |
| Precision | 100% (0 wrong merges) |
| Hit rate | 98.9% (444 / 449) |
| Recall | 98.4% |
| Type II error | 1.3% (6 missed) |
| Manual review rate | 14.2% |
| Disputes | 0 |
| Test set (100) | 58 MATCH (all correct), 15 REVIEW, 27 NO MATCH, 2 missed; precision 100%, recall 96.7% |

June is the only Red month: manual review rate **32% (Red)**, recall **93.3% (Amber)**. Precision stays 100%. Overall validation rating is **Needs attention** because of that Red KPI, not because of wrong merges.

The five unconfirmed proposals are look-alike relatives (same name and address, ID one digit off, DOB missing). The look-alike guard sends them to review and suggests the existing person; the answer key says they are new people. Nothing is merged. Hit rate is divided by proposals, not by 75, so a month can show 74 or 76 proposals.

## Matching method (v2.4 rules, still current)

1. **Normalise.** Lower case, punctuation removed, address abbreviations expanded, ID digits only. Date of birth is **YYYY-MM-DD only**. Any other format is missing. FEBRL `YYYYMMDD` is converted once in `data_generator.to_record`.
2. **Search** (max 25 candidates). Exact ID; name WRatio ≥ 70; same city and name WRatio 50–69. If name and ID both fail and the city rule does not apply, the record is NO MATCH.
3. **Compare.** Name WRatio (agrees ≥ 70 or ≤ 3 edits, else score 0). ID digit edits: 0 → 100, 1 → 90, 2 → 75, 3+ → 0. Address Token Set Ratio (≥ 60 or ≤ 5 edits). DOB exact. City Ratio (≥ 85 or ≤ 2 edits).
4. **Score.** Weights 35 / 35 / 15 / 7.5 / 7.5. Missing fields are dropped and the rest re-scaled. At least 70% of the weight must be present.
5. **Conflicts.** Strong agree ≥ 90 vs strong disagree < 50 on name vs ID; different DOB; ID points to another person; look-alike guard; corroboration review (name or ID ≥ 90, same DOB, address or city agrees → review, never auto-match).
6. **Decide.** MATCH ≥ 85, no conflict, gap ≥ 5. REVIEW 65–85 or any conflict. NO MATCH < 65.

v2.4 (26 Oct 2026) added corroboration and the look-alike guard: Type II error 11.1% → 1.3%, wrong merges 5 → 0, review rate 6.0% → 14.2%.

## Comparison already done

Challengers replace one comparator only. Search, weights and decision rules stay the same. All keep 100% precision and 0 wrong merges.

| Algorithm | Recall | Type II | Missed | Review rate |
|---|---|---|---|---|
| Production (WRatio + Token Set Ratio) | 98.4% | 1.3% | 6 | 14.2% |
| Jaro-Winkler (name) | 94.2% | 4.4% | 20 | 18.8% |
| Levenshtein (name) | 92.4% | 5.6% | 25 | 20.7% |
| Damerau-Levenshtein (name) | 92.5% | 5.6% | 25 | 20.3% |
| Soundex (name) | 90.4% | 6.2% | 28 | 27.3% |
| Jaccard (address) | 97.5% | 1.8% | 8 | 23.2% |

Search paths on 364 correct automatic matches: exact ID 349 (95.9%), similar name 364, 15 found only by name (mistyped ID). City rule 0 on automatic matches. Attribute contribution is close to the weights (name 34.8%, ID 36.1%); nothing above 50%.

Calibration: MATCH 75 or 80 cuts review to 13.3% with the same recall and 0 wrong merges. MATCH 90 raises review to 18%. REVIEW 55 raises recall to 99.5% and cuts Type II to 0.4%. An alternative MATCH threshold is flagged only if recall rises by at least 1 percentage point (`CALIBRATION_MIN_RECALL_GAIN`).

Root causes of 91 exceptions: replaced name 40 (38 review, 2 missed), ID mismatch 23 (19 review, 4 missed), missing or different DOB 18 (all review), similar names 5 (the look-alikes).

## AI validation report

`validation_report.py` writes a **quantitative** report only (no qualitative section). Four Groq calls: `quant_a`, `quant_b`, `quant_c`, `summary`. Model `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b`. Up to 3 correction rounds. Ratings come from rules, not the model.

Guardrails in `check_answer`: numbers must be in the evidence; right KPI and unit; direction of change; no code field names; no table references; no “all limits met” while `latest_period_breaches` is non-empty; Red/Amber only on the breached month’s values; conclusion must name every Red KPI; challenger ranges must cover every challenger; “matches production” must be true; root-cause “missed / review” claims must match the counts; calibration “rises/falls to” must match the current setting.

## Where things live

| Item | Path |
|---|---|
| App | `app.py` (Streamlit) |
| Matching | `matching.py`, `config.py` |
| Data | `data_generator.py`, SQLite `data/matching.db` (not in git) |
| Monitoring KPIs | `performance.py` |
| Validation tables | `validation.py` |
| AI report | `validation_report.py`, `agent.py`, `report_pdf.py`, `report_docx.py` |
| Tests | `tests/` — 58 passing as of 16 Nov 2026 |
| Submission report and slides | `C:\Users\abhi9\Downloads\Search_Match_Merge_Project_Report.docx` and `.pdf`, `Search_Match_Merge_Presentation.pptx` |
| Presentation script | `C:\Users\abhi9\Downloads\Presentation_Script.pdf` |
| AI report preview | `C:\Users\abhi9\Downloads\my cursor\report_preview\` |

Do not commit `groq_api_key.txt`, `.env`, or the local database. Git author for this repo: `rajabhi2627-dev` / `338862296+rajabhi2627-dev@users.noreply.github.com`. Push to `main` only when asked.

## Known limits

FEBRL is synthetic. Disputes are simulated (80% of wrong merges, 50% of misses). The answer key stands in for a human reviewer. No ID or DOB challengers yet. Guardrails cover the error types already seen; a human still signs off.

## Not done

ID and DOB challengers, extra search rules (for example DOB plus postcode), a month selector on the AI report, and a real-data test. The phrase “36.1% of the match score” is corrected in the preview Word/PDF under `report_preview`; regenerate the in-app AI report if a new PDF is needed after the DOB change.
