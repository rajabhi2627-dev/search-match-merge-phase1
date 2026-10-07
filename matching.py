"""
Deterministic Search, Match & Merge logic.

Pipeline (used by BOTH the Search & Match page and the 100-record test run):

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
      -> Final decision
"""
import re
from datetime import datetime

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

import config as C

# ==================================================================
# 1. NORMALISATION
# ==================================================================

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "cres": "crescent",
    "cct": "circuit",
    "pl": "place",
    "ct": "court",
    "dr": "drive",
    "hwy": "highway",
    "apt": "apartment",
}

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%d %b %Y", "%d %B %Y", "%Y%m%d"]


def _is_blank(value):
    return value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""


def normalize_text(value):
    """Lowercase, strip, replace punctuation by spaces, collapse multiple spaces."""
    if _is_blank(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_id(value):
    """Keep digits only: "530 4218", "530-4218" -> "5304218"."""
    if _is_blank(value):
        return ""
    return re.sub(r"\D", "", str(value))


def normalize_dob(value):
    """Convert any supported date format to YYYY-MM-DD. Invalid or unparseable -> "" (missing).

    FEBRL uses YYYYMMDD. Day-first formats (DD/MM/YYYY) are assumed for slashes and dashes.
    """
    if _is_blank(value):
        return ""
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        text = text[:10]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def normalize_address(value):
    """Text normalisation + expansion of common abbreviations (Rd -> road, St -> street, Cres -> crescent)."""
    words = [ADDRESS_ABBREVIATIONS.get(w, w) for w in normalize_text(value).split()]
    return " ".join(words)


NORMALIZERS = {
    "name": normalize_text,
    "soc_sec_id": normalize_id,
    "address": normalize_address,
    "date_of_birth": normalize_dob,
    "city": normalize_text,
}


def normalize_record(record):
    """Apply the field-specific normaliser to every field of a record."""
    return {field: NORMALIZERS[field](record.get(field)) for field in C.FIELDS}


# ==================================================================
# 2. FIELD-SPECIFIC COMPARISON
# ==================================================================

def _result(field, score, similarity, distance, agrees, note):
    return {
        "field": field,
        "method": C.FIELD_CONFIG[field]["method"],
        "score": score,
        "similarity": similarity,
        "distance": distance,
        "agrees": agrees,
        "note": note,
    }


def _fuzzy_compare(field, a, b, similarity):
    """Shared threshold / max-distance rule for the fuzzy fields."""
    cfg = C.FIELD_CONFIG[field]
    similarity = round(float(similarity), 1)
    distance = Levenshtein.distance(a, b)
    agrees = similarity >= cfg["threshold"] or distance <= cfg["max_distance"]
    if agrees:
        note = f"similarity {similarity} (threshold {cfg['threshold']}), edit distance {distance} (max {cfg['max_distance']})"
        return _result(field, similarity, similarity, distance, True, note)
    note = (f"similarity {similarity} < threshold {cfg['threshold']} and edit distance "
            f"{distance} > max {cfg['max_distance']} -> score 0")
    return _result(field, 0.0, similarity, distance, False, note)


def compare_name(a, b):
    if a == b:
        return _result("name", 100.0, 100.0, 0, True, "exact match")
    return _fuzzy_compare("name", a, b, fuzz.WRatio(a, b))


def compare_address(a, b):
    if a == b:
        return _result("address", 100.0, 100.0, 0, True, "exact match")
    return _fuzzy_compare("address", a, b, fuzz.token_set_ratio(a, b))


def compare_city(a, b):
    if a == b:
        return _result("city", 100.0, 100.0, 0, True, "exact match")
    return _fuzzy_compare("city", a, b, fuzz.ratio(a, b))


def compare_id(a, b):
    """Digit-level comparison of the social security ID (no generic fuzzy matching).

    Count the digit edits (change, insert or delete one digit) needed to turn one ID into the other:
    0 edits -> 100, 1 edit -> 90, 2 edits -> 75, more -> 0.
    """
    if a == b:
        return _result("soc_sec_id", 100.0, 100.0, 0, True, "exact match")
    edits = Levenshtein.distance(a, b)
    score = float(C.ID_SCORE_BY_DIGIT_EDITS.get(edits, 0))
    agrees = score >= C.FIELD_CONFIG["soc_sec_id"]["threshold"]
    note = f"{edits} digit edit(s) -> {score:.0f}" if agrees else f"{edits} digit edits - different ID -> 0"
    return _result("soc_sec_id", score, score, edits, agrees, note)


def compare_dob(a, b):
    """Exact comparison only: same date -> 100, different date -> 0."""
    if a == b:
        return _result("date_of_birth", 100.0, 100.0, 0, True, "same date")
    return _result("date_of_birth", 0.0, 0.0, None, False, "different date")


COMPARATORS = {
    "name": compare_name,
    "soc_sec_id": compare_id,
    "address": compare_address,
    "date_of_birth": compare_dob,
    "city": compare_city,
}


def compare_field(field, a, b):
    """Compare one field. Missing on either side -> score None (field is excluded from the weighted score)."""
    if not a or not b:
        return _result(field, None, None, None, None, "missing - excluded from score")
    return COMPARATORS[field](a, b)


# ==================================================================
# 3. WEIGHTED SCORE
# ==================================================================

def weighted_score(field_scores):
    """Weighted average of the available field scores (0-100).

    Fields that are missing (None) are left out and the remaining weights are
    re-scaled so they add up to 100%. Returns (overall, evidence_weight, contributions).
    """
    present = {f: s for f, s in field_scores.items() if s is not None}
    evidence_weight = sum(C.FIELD_WEIGHTS[f] for f in present)
    if evidence_weight == 0:
        return 0.0, 0.0, {f: 0.0 for f in field_scores}
    contributions = {
        f: (present[f] * C.FIELD_WEIGHTS[f] / evidence_weight) if f in present else 0.0
        for f in field_scores
    }
    overall = round(sum(contributions.values()), 2)
    return overall, round(evidence_weight, 3), contributions


# ==================================================================
# 4. CANDIDATE SEARCH (BLOCKING)
# ==================================================================

def prepare_master(master_records):
    """Normalise every master record once. master_records: list of dicts."""
    return [{"entity_id": int(r["entity_id"]), "raw": r, "norm": normalize_record(r)} for r in master_records]


def find_candidates(norm_input, master):
    """Return [(master_item, [reasons])] for records that satisfy at least one rule:

    1. Exact Soc. Sec. ID match
    2. Similar name (WRatio >= CANDIDATE_NAME_MIN)
    3. Same city AND name WRatio >= CANDIDATE_CITY_NAME_MIN
    """
    found = []
    for item in master:
        m = item["norm"]
        reasons = []
        name_sim = fuzz.WRatio(norm_input["name"], m["name"]) if norm_input["name"] else 0
        if norm_input["soc_sec_id"] and norm_input["soc_sec_id"] == m["soc_sec_id"]:
            reasons.append("exact Soc. Sec. ID")
        if name_sim >= C.CANDIDATE_NAME_MIN:
            reasons.append(f"similar name ({name_sim:.0f})")
        elif norm_input["city"] and norm_input["city"] == m["city"] and name_sim >= C.CANDIDATE_CITY_NAME_MIN:
            reasons.append(f"same city + partly similar name ({name_sim:.0f})")
        if reasons:
            exact_hits = sum(r.startswith("exact") for r in reasons)
            found.append((exact_hits, name_sim, item, reasons))
    # exact identifier hits first, then by name similarity; keep the best MAX_CANDIDATES
    found.sort(key=lambda x: (-x[0], -x[1], x[2]["entity_id"]))
    return [(item, reasons) for _, _, item, reasons in found[: C.MAX_CANDIDATES]]


# ==================================================================
# 5. SCORING ONE CANDIDATE + CONFLICTS
# ==================================================================

def candidate_conflicts(field_scores):
    """Conflict rules for a single candidate.

    Rule 1: one strong identifier (name / Soc. Sec. ID) clearly agrees (>= 90)
            while the other clearly disagrees (< 50).
    Rule 2: date of birth present on both sides but different.
    """
    conflicts = []
    strong = {f: field_scores[f] for f in C.STRONG_FIELDS if field_scores[f] is not None}
    agree = [f for f, s in strong.items() if s >= C.STRONG_AGREE_SCORE]
    disagree = [f for f, s in strong.items() if s < C.STRONG_DISAGREE_SCORE]
    if agree and disagree:
        conflicts.append(
            "Strong identifiers disagree: "
            + ", ".join(f"{C.FIELD_LABELS[f]}={strong[f]:.0f}" for f in agree)
            + " vs "
            + ", ".join(f"{C.FIELD_LABELS[f]}={strong[f]:.0f}" for f in disagree)
        )
    if field_scores["date_of_birth"] == 0:
        conflicts.append("Date of birth is different")
    return conflicts


def score_candidate(norm_input, item, reasons):
    details = {f: compare_field(f, norm_input[f], item["norm"][f]) for f in C.FIELDS}
    field_scores = {f: details[f]["score"] for f in C.FIELDS}
    overall, evidence_weight, contributions = weighted_score(field_scores)
    return {
        "entity_id": item["entity_id"],
        "name": item["raw"].get("name"),
        "record": item["raw"],
        "norm": item["norm"],
        "search_reasons": reasons,
        "field_scores": field_scores,
        "details": details,
        "overall": overall,
        "evidence_weight": evidence_weight,
        "contributions": contributions,
        "conflicts": candidate_conflicts(field_scores),
    }


def cross_candidate_conflicts(scored):
    """Rule 3: the Soc. Sec. ID matches one entity exactly, but the best overall score belongs to another."""
    id_ids = [c["entity_id"] for c in scored if c["field_scores"]["soc_sec_id"] == 100]
    best_id = scored[0]["entity_id"]
    if id_ids and id_ids[0] != best_id:
        return [f"Soc. Sec. ID points to entity {id_ids[0]} but the best overall score is entity {best_id}"]
    return []


# ==================================================================
# 6. DECISION
# ==================================================================

def make_decision(best_score, score_gap, has_meaningful_second, conflicts, limited_evidence):
    """Return (decision, [reasons]).

    NO MATCH       : best score < REVIEW_THRESHOLD
    MANUAL REVIEW  : score in [REVIEW, MATCH) OR gap < SCORE_GAP_THRESHOLD (with a meaningful
                     second candidate) OR a conflict OR too little evidence
    MATCH          : score >= MATCH_THRESHOLD and none of the review reasons apply
    """
    if best_score < C.REVIEW_THRESHOLD:
        return C.NO_MATCH, [f"Best score {best_score:.1f} is below the review threshold ({C.REVIEW_THRESHOLD})"]
    reasons = []
    if best_score < C.MATCH_THRESHOLD:
        reasons.append(f"Score {best_score:.1f} is between {C.REVIEW_THRESHOLD} and {C.MATCH_THRESHOLD}")
    if has_meaningful_second and score_gap < C.SCORE_GAP_THRESHOLD:
        reasons.append(f"Score gap {score_gap:.1f} is below {C.SCORE_GAP_THRESHOLD} - two candidates are too close")
    if conflicts:
        reasons.append("Major conflict detected")
    if limited_evidence:
        reasons.append(f"Too few fields to compare (less than {C.MIN_EVIDENCE_WEIGHT:.0%} of total weight)")
    if reasons:
        return C.MANUAL_REVIEW, reasons
    return C.MATCH, [
        f"Score {best_score:.1f} >= {C.MATCH_THRESHOLD}, no conflicts and score gap {score_gap:.1f} >= {C.SCORE_GAP_THRESHOLD}"
    ]


# ==================================================================
# 7. FULL PIPELINE
# ==================================================================

def match_record(raw_input, master):
    """Run the complete Search & Match pipeline for one incoming record."""
    norm_input = normalize_record(raw_input)

    candidates = find_candidates(norm_input, master)
    scored = [score_candidate(norm_input, item, reasons) for item, reasons in candidates]
    scored.sort(key=lambda c: (-c["overall"], c["entity_id"]))

    result = {
        "input_raw": raw_input,
        "input_norm": norm_input,
        "candidates_found": len(scored),
        "candidates": scored,
        "top": scored[: C.TOP_N_DISPLAY],
        "best": scored[0] if scored else None,
        "second_score": None,
        "score_gap": None,
        "conflicts": [],
        "decision": C.NO_MATCH,
        "decision_reasons": ["No candidate satisfied any search rule"],
        "predicted_entity_id": None,
    }
    if not scored:
        return result

    best = scored[0]
    second_score = scored[1]["overall"] if len(scored) > 1 else 0.0
    score_gap = round(best["overall"] - second_score, 2)
    has_meaningful_second = len(scored) > 1 and second_score >= C.REVIEW_THRESHOLD
    conflicts = best["conflicts"] + cross_candidate_conflicts(scored)
    limited_evidence = best["evidence_weight"] < C.MIN_EVIDENCE_WEIGHT

    decision, reasons = make_decision(best["overall"], score_gap, has_meaningful_second, conflicts, limited_evidence)

    result.update(
        second_score=second_score,
        score_gap=score_gap,
        conflicts=conflicts,
        decision=decision,
        decision_reasons=reasons,
        predicted_entity_id=best["entity_id"] if decision == C.MATCH else None,
    )
    return result


# ==================================================================
# 8. MERGE PREVIEW (display only - nothing is written to the database)
# ==================================================================

def merge_preview(raw_input, master_record):
    """Survivorship rule: keep the master value; fill empty master fields from the incoming record."""
    rows = []
    for f in C.FIELDS:
        master_val = "" if _is_blank(master_record.get(f)) else str(master_record.get(f))
        incoming_val = "" if _is_blank(raw_input.get(f)) else str(raw_input.get(f))
        if not master_val and incoming_val:
            merged, action = incoming_val, "Filled from incoming record"
        elif incoming_val and NORMALIZERS[f](incoming_val) != NORMALIZERS[f](master_val):
            merged, action = master_val, "Kept master value (incoming differs)"
        else:
            merged, action = master_val, "Kept master value"
        rows.append({"Field": C.FIELD_LABELS[f], "Master": master_val, "Incoming": incoming_val,
                     "Merged": merged, "Action": action})
    return rows
