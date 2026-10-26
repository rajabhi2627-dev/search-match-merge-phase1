"""
All tunable parameters for the Phase 1 Search, Match & Merge prototype.

Change values here - nothing else in the project hard-codes them.
"""
from pathlib import Path

# ------------------------------------------------------------------
# Storage
# ------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "matching.db"
AGENT_VALIDATION_CSV = DATA_DIR / "agent_validation_results.csv"

# ------------------------------------------------------------------
# Fields - exactly the person fields available in the FEBRL-4 benchmark
# ------------------------------------------------------------------
FIELDS = ["name", "soc_sec_id", "address", "date_of_birth", "city"]

FIELD_LABELS = {
    "name": "Name",
    "soc_sec_id": "Soc. Sec. ID",
    "address": "Address",
    "date_of_birth": "DOB",
    "city": "City",
}

# ------------------------------------------------------------------
# Field weights (must add up to 1.0)
# ------------------------------------------------------------------
NAME_WEIGHT = 0.35
SOC_SEC_ID_WEIGHT = 0.35
ADDRESS_WEIGHT = 0.15
DOB_WEIGHT = 0.075
CITY_WEIGHT = 0.075

FIELD_WEIGHTS = {
    "name": NAME_WEIGHT,
    "soc_sec_id": SOC_SEC_ID_WEIGHT,
    "address": ADDRESS_WEIGHT,
    "date_of_birth": DOB_WEIGHT,
    "city": CITY_WEIGHT,
}

# ------------------------------------------------------------------
# Field-specific matching methods
#   threshold    = minimum similarity (0-100) for the field to "agree"
#   max_distance = maximum edit distance that is still tolerated
# A field agrees if similarity >= threshold OR edit distance <= max_distance.
# If it does not agree, the field score is 0.
# ------------------------------------------------------------------
FIELD_CONFIG = {
    "name":          {"method": "WRatio",           "threshold": 70,  "max_distance": 3},
    "soc_sec_id":    {"method": "Digit comparison", "threshold": 75,  "max_distance": 2},
    "address":       {"method": "Token Set Ratio",  "threshold": 60,  "max_distance": 5},
    "date_of_birth": {"method": "Exact",            "threshold": 100, "max_distance": 0},
    "city":          {"method": "Ratio",            "threshold": 85,  "max_distance": 2},
}

# Soc. Sec. ID scoring: number of digit edits (change / insert / delete one digit) -> score.
# More edits than listed here -> 0.
ID_SCORE_BY_DIGIT_EDITS = {0: 100, 1: 90, 2: 75}

# ------------------------------------------------------------------
# Candidate search (blocking)
# ------------------------------------------------------------------
CANDIDATE_NAME_MIN = 70        # rule 2: name WRatio >= this
CANDIDATE_CITY_NAME_MIN = 50   # rule 3: same city AND name WRatio >= this
MAX_CANDIDATES = 25            # keep at most this many candidates for full scoring
TOP_N_DISPLAY = 3

# ------------------------------------------------------------------
# Conflict detection
# ------------------------------------------------------------------
STRONG_FIELDS = ["name", "soc_sec_id"]
STRONG_AGREE_SCORE = 90        # a strong field "clearly agrees" at or above this
STRONG_DISAGREE_SCORE = 50     # a strong field "clearly disagrees" below this
MIN_EVIDENCE_WEIGHT = 0.70     # fields present on both sides must carry >= 70% of the weight

# ------------------------------------------------------------------
# Decision thresholds
# ------------------------------------------------------------------
MATCH_THRESHOLD = 85
REVIEW_THRESHOLD = 65
SCORE_GAP_THRESHOLD = 5

MATCH = "MATCH"
MANUAL_REVIEW = "MANUAL REVIEW"
NO_MATCH = "NO MATCH"

# ------------------------------------------------------------------
# Dataset: FEBRL-4 public record-linkage benchmark
#   dataset4a.csv = 5,000 original person records
#   dataset4b.csv = 5,000 corrupted duplicates; rec-N-dup-0 is the same person as rec-N-org
# ------------------------------------------------------------------
DATASET_NAME = "FEBRL-4"
SCHEMA_VERSION = "4"           # change to force the database to be rebuilt
FEBRL_DIR = DATA_DIR / "febrl"
FEBRL_ORIGINALS = FEBRL_DIR / "dataset4a.csv"
FEBRL_DUPLICATES = FEBRL_DIR / "dataset4b.csv"
SAMPLE_SEED = 42

MASTER_RECORD_COUNT = 1000       # FEBRL originals loaded into the master database

# Test set: duplicates whose original IS in the master (true MATCH)
# plus duplicates whose original is NOT in the master (true NO MATCH = new person).
TEST_RECORD_COUNT = 100
TEST_MATCH_COUNT = 75
TEST_NO_MATCH_COUNT = TEST_RECORD_COUNT - TEST_MATCH_COUNT

# ------------------------------------------------------------------
# Week 4 - Performance monitoring
# ------------------------------------------------------------------
# One batch of 100 new FEBRL duplicates per month (75 existing people, 25 new people).
# Data-quality drift: the duplicates are ordered by how many fields FEBRL corrupted,
# and later months receive more heavily corrupted records.
MONITORING_PERIODS = ["Jan-26", "Feb-26", "Mar-26", "Apr-26", "May-26", "Jun-26"]
MONITORING_MATCH_COUNT = 75
MONITORING_NO_MATCH_COUNT = 25
# Hard cases that FEBRL lacks; they replace new-person records, so each month keeps 100 records.
#   look-alike relative: same name and address as a master customer, DOB missing, Soc. Sec. ID 1 digit different
#   shared ID: a new person entered with an existing customer's Soc. Sec. ID
MONITORING_HARD_CASES = {  # period: (look-alike relatives, shared IDs)
    "Jan-26": (0, 0), "Feb-26": (0, 1), "Mar-26": (1, 1), "Apr-26": (1, 1), "May-26": (1, 2), "Jun-26": (2, 2),
}

# Share of wrong decisions that customers actually complain about (simulated, fixed seed).
#   FP = wrong merge      -> "this is not my account"
#   FN = missed duplicate -> "I already have an account"
DISPUTE_RAISE_RATE = {"FP": 0.8, "FN": 0.5}

# ------------------------------------------------------------------
# Independent validation (Validation page)
# ------------------------------------------------------------------
# Challenger algorithms: each replaces ONE production comparator; search, weights and decision rules stay
# the same, so any difference in results comes from that algorithm alone.
# A challenger field agrees if its similarity (0-100) >= cut-off; otherwise the field score is 0.
CHALLENGER_CUTOFFS = {
    "Jaro-Winkler (name)": 85,
    "Levenshtein (name)": 70,
    "Damerau-Levenshtein (name)": 70,
    "Soundex (name)": 100,       # every word of the name must sound the same
    "Jaccard (address)": 50,
}

# Threshold calibration: decisions are recomputed for these values (all other rules unchanged).
CALIBRATION_MATCH_THRESHOLDS = [75, 80, 85, 90, 95]
CALIBRATION_REVIEW_THRESHOLDS = [55, 60, 65, 70, 75]

# Data quality checks on incoming records
ID_EXPECTED_LENGTH = 7           # FEBRL social security IDs have 7 digits

# Benchmark levels used in the validation observations
VALIDATION_LIMITS = {
    "type1_error": 0.05,         # Type I (false positive) error rate should stay below 5%
    "dispute_rate": 0.03,        # dispute rate on generated matches below 3%
    "incorrect_merge_rate": 0.01,  # incorrect merges below 1% of generated matches
    "max_attribute_share": 0.50,  # no single attribute should drive more than 50% of the match score
}

# ------------------------------------------------------------------
# AI validation agent (AI Agent page)
# ------------------------------------------------------------------
# The LLM interprets the evidence and writes the report. Every number comes from deterministic tools,
# and every number in the report is verified against the evidence before a human approves it.
GROQ_MODEL = "openai/gpt-oss-120b"
# Production models only (Groq Preview models such as Qwen are for evaluation and are retired at short notice).
# Each model has its own token limits; when the main model is rate-limited the agent first waits up to
# PRIMARY_MODEL_MAX_WAIT seconds for it, and only then moves to the fallback.
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-20b"]
PRIMARY_MODEL_MAX_WAIT = 45      # seconds
AGENT_TEMPERATURE = 0
AGENT_MAX_STEPS = 8              # maximum LLM calls per run
AGENT_MAX_OUTPUT_TOKENS = 6000
AGENT_REPORT_MAX_WORDS = 1000    # length limit for the AI report
AGENT_REPORT_MAX_WORDS_FREE_TIER = 500  # free tier: a revision (evidence + previous report) must still fit
FREE_TIER_TOKENS_PER_MINUTE = 8000
# Groq tokens per minute per model (prompt + output). 8,000 = free tier; the Developer plan allows 250,000.
# Override without changing code: set GROQ_TOKENS_PER_MINUTE in the Streamlit secrets or the environment.
GROQ_TOKENS_PER_MINUTE = 8000
AGENT_MAX_REVISIONS = 2          # self-correction rounds when verification fails
AGENT_STABILITY_RUNS = 3         # repeated runs per stability scenario

# Full validation report (Validation page, section 11): names on the cover and in the text.
REPORT_TITLE = "Search and Match Model Validation Report"
REPORT_ORGANISATION = "Credit Information Company"            # owner of the Search & Match model
REPORT_VALIDATOR = "Independent Model Validation Function"    # who performs the validation
REPORT_SECTION_REVISIONS = 1     # AI self-corrections per report part when a check fails

ANOMALY_CHANGE_PP = 5.0         # month-on-month KPI change (percentage points) flagged as an anomaly
TREND_PERIODS = 3                # consecutive deteriorations that count as an adverse trend
NEAR_THRESHOLD_MARGIN = 5        # MATCH decisions scoring < MATCH_THRESHOLD + margin are flagged
SMALL_GAP_MARGIN = 10            # MATCH decisions with score gap < this are flagged

# Sampling for manual validation (Cochran formula with finite population correction)
SAMPLE_Z = 1.96                  # 95% confidence
SAMPLE_MARGIN = 0.05             # +/- 5% margin of error
SAMPLE_EXPECTED_ERROR = 0.05     # expected error rate in matching decisions

# Change management: approval level by the largest impact (percentage points) on
# hit rate, Type I or Type II error rate.
CHANGE_APPROVAL_LEVELS = [
    (1.0, "Minor - approval by the Search & Match team lead"),
    (3.0, "Moderate - approval by the Head of Data Management"),
    (None, "Major - approval by the Model Risk / Governance Committee"),
]
CHANGE_LOG = [
    {"version": "1.0", "date": "2026-10-07", "change": "Initial prototype on synthetic data",
     "impact_assessed": True, "approved_by": None},
    {"version": "2.0", "date": "2026-10-07", "change": "Switched to the FEBRL-4 benchmark; Week 4 monitoring added",
     "impact_assessed": True, "approved_by": None},
    {"version": "2.1", "date": "2026-10-07",
     "change": "FEBRL fields only: email and company removed, Soc. Sec. ID digit comparison, weights 35/35/15/7.5/7.5",
     "impact_assessed": True, "approved_by": None},
    {"version": "2.2", "date": "2026-10-16", "change": "Independent validation page added",
     "impact_assessed": True, "approved_by": None},
    {"version": "2.3", "date": "2026-10-16",
     "change": "Hit rate, dispute rate, incorrect merge rate and Type I/II errors use the validation-report formulas",
     "impact_assessed": True, "approved_by": None},
]

# Acceptance criteria for the agent validation tests (share of scenario runs that must pass)
AGENT_TEST_GREEN = 1.00
AGENT_TEST_AMBER = 0.80
EXPERT_REVIEW_GREEN = 4.0        # average rubric score (1-5)
EXPERT_REVIEW_AMBER = 3.0

# KPI thresholds. direction "higher" = higher is better.
#   higher: Green >= green, Amber >= amber, otherwise Red
#   lower : Green <= green, Amber <= amber, otherwise Red
KPI_THRESHOLDS = {
    "hit_rate":             {"direction": "higher", "green": 0.95, "amber": 0.90},
    "precision":            {"direction": "higher", "green": 0.98, "amber": 0.95},
    "recall":               {"direction": "higher", "green": 0.95, "amber": 0.90},
    "type1_error":          {"direction": "lower",  "green": 0.02, "amber": 0.05},
    "type2_error":          {"direction": "lower",  "green": 0.05, "amber": 0.10},
    "review_rate":          {"direction": "lower",  "green": 0.15, "amber": 0.25},
    "incorrect_merge_rate": {"direction": "lower",  "green": 0.005, "amber": 0.01},
    "dispute_rate":         {"direction": "lower",  "green": 0.02, "amber": 0.03},
}
