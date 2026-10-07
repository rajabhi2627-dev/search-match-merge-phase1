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
SCHEMA_VERSION = "2"           # change to force the database to be rebuilt
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

# Share of wrong decisions that customers actually complain about (simulated, fixed seed).
#   FP = wrong merge      -> "this is not my account"
#   FN = missed duplicate -> "I already have an account"
DISPUTE_RAISE_RATE = {"FP": 0.8, "FN": 0.5}

# KPI thresholds. direction "higher" = higher is better.
#   higher: Green >= green, Amber >= amber, otherwise Red
#   lower : Green <= green, Amber <= amber, otherwise Red
KPI_THRESHOLDS = {
    "hit_rate":         {"direction": "higher", "green": 0.60, "amber": 0.50},
    "precision":        {"direction": "higher", "green": 0.98, "amber": 0.95},
    "recall":           {"direction": "higher", "green": 0.95, "amber": 0.90},
    "fpr":              {"direction": "lower",  "green": 0.02, "amber": 0.05},
    "fnr":              {"direction": "lower",  "green": 0.05, "amber": 0.10},
    "review_rate":      {"direction": "lower",  "green": 0.15, "amber": 0.25},
    "merge_error_rate": {"direction": "lower",  "green": 0.01, "amber": 0.03},
    "dispute_rate":     {"direction": "lower",  "green": 0.01, "amber": 0.03},
}
