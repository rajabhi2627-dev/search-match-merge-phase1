"""
All tunable parameters for the Phase 1 Search, Match & Merge prototype.

Change values here - nothing else in the project hard-codes them.
"""
from pathlib import Path

# ------------------------------------------------------------------
# Local storage (everything stays on this laptop)
# ------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DB_PATH = DATA_DIR / "matching.db"

# ------------------------------------------------------------------
# Field weights (must add up to 1.0)
# ------------------------------------------------------------------
NAME_WEIGHT = 0.25
EMAIL_WEIGHT = 0.25
PHONE_WEIGHT = 0.25
ADDRESS_WEIGHT = 0.10
DOB_WEIGHT = 0.05
COMPANY_WEIGHT = 0.05
CITY_WEIGHT = 0.05

FIELDS = ["name", "email", "phone", "address", "date_of_birth", "company", "city"]

FIELD_WEIGHTS = {
    "name": NAME_WEIGHT,
    "email": EMAIL_WEIGHT,
    "phone": PHONE_WEIGHT,
    "address": ADDRESS_WEIGHT,
    "date_of_birth": DOB_WEIGHT,
    "company": COMPANY_WEIGHT,
    "city": CITY_WEIGHT,
}

FIELD_LABELS = {
    "name": "Name",
    "email": "Email",
    "phone": "Phone",
    "address": "Address",
    "date_of_birth": "DOB",
    "company": "Company",
    "city": "City",
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
    "email":         {"method": "Levenshtein",      "threshold": 85,  "max_distance": 2},
    "phone":         {"method": "Digit comparison", "threshold": 75,  "max_distance": 2},
    "address":       {"method": "Token Set Ratio",  "threshold": 60,  "max_distance": 5},
    "date_of_birth": {"method": "Exact",            "threshold": 100, "max_distance": 0},
    "company":       {"method": "WRatio",           "threshold": 70,  "max_distance": 3},
    "city":          {"method": "Ratio",            "threshold": 85,  "max_distance": 2},
}

# Phone scoring (digit comparison on normalised 10-digit numbers)
PHONE_LENGTH = 10
PHONE_SCORE_BY_DIGIT_DIFF = {0: 100, 1: 90, 2: 75}   # differing digit positions -> score
PHONE_LAST_N_DIGITS = 7                              # last-N-digit fallback rule
PHONE_LAST_N_SCORE = 80

# ------------------------------------------------------------------
# Candidate search (blocking)
# ------------------------------------------------------------------
CANDIDATE_NAME_MIN = 70        # rule 3: name WRatio >= this
CANDIDATE_CITY_NAME_MIN = 50   # rule 4: same city AND name WRatio >= this
MAX_CANDIDATES = 25            # keep at most this many candidates for full scoring
TOP_N_DISPLAY = 3

# ------------------------------------------------------------------
# Conflict detection
# ------------------------------------------------------------------
STRONG_FIELDS = ["name", "email", "phone"]
STRONG_AGREE_SCORE = 90        # a strong field "clearly agrees" at or above this
STRONG_DISAGREE_SCORE = 50     # a strong field "clearly disagrees" below this
MIN_EVIDENCE_WEIGHT = 0.60     # fields present on both sides must carry >= 60% of weight

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
# Synthetic data generation
# ------------------------------------------------------------------
MASTER_RECORD_COUNT = 1000
MASTER_SEED = 2024
FIRST_ENTITY_ID = 10001

TEST_RECORD_COUNT = 100
TEST_SEED = 42

# Number of test records per scenario (must add up to TEST_RECORD_COUNT).
# true_status says what the ground truth is for that scenario.
TEST_SCENARIOS = {
    # ~60 records based on existing entities
    "Clear match":                {"count": 13, "true_status": MATCH},
    "Name spelling error":        {"count": 10, "true_status": MATCH},
    "Email variation":            {"count": 8,  "true_status": MATCH},
    "Phone formatting":           {"count": 10, "true_status": MATCH},
    "Address variation":          {"count": 9,  "true_status": MATCH},
    "Company spelling variation": {"count": 8,  "true_status": MATCH},
    # ~20 noisy / ambiguous records
    "Multiple noisy fields":      {"count": 10, "true_status": MATCH},
    "Changed contact details":    {"count": 2,  "true_status": MATCH},
    "Ambiguous - sparse record":  {"count": 5,  "true_status": MATCH},
    "Ambiguous - family member":  {"count": 5,  "true_status": NO_MATCH},
    # ~20 records that should be NO MATCH
    "New person":                 {"count": 15, "true_status": NO_MATCH},
    "Look-alike (same name)":     {"count": 5,  "true_status": NO_MATCH},
}
