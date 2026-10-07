"""
Builds the master, test and monitoring data from the FEBRL-4 record-linkage benchmark.

FEBRL-4 (Christen, 2008) contains 5,000 original person records (dataset4a.csv) and one
corrupted duplicate of each (dataset4b.csv). rec-N-dup-0 is the same person as rec-N-org,
so the ground truth comes from the benchmark itself - nothing is labelled by hand.

* generate_master_records()      : 1,000 originals -> master database (entity_id = N)
* generate_test_records()        : 75 duplicates of master people (true MATCH)
                                   + 25 duplicates of people NOT in the master (true NO MATCH)
* generate_monitoring_batches()  : 6 monthly batches of new duplicates, with data-quality drift

All sampling uses fixed seeds, so the data is identical on every run.
"""
import random
import re
from functools import lru_cache

import pandas as pd

import config as C

FEBRL_FIELDS = ["given_name", "surname", "street_number", "address_1", "address_2",
                "suburb", "postcode", "state", "date_of_birth", "soc_sec_id"]


def _clean(value):
    return "" if pd.isna(value) else str(value).strip()


def febrl_number(rec_id):
    """rec-1070-org / rec-1070-dup-0 -> 1070"""
    return int(re.search(r"rec-(\d+)-", rec_id).group(1))


def _read(path):
    df = pd.read_csv(path, dtype=str, skipinitialspace=True, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["rec_id"] = df["rec_id"].str.strip()
    df["number"] = df["rec_id"].map(febrl_number)
    return df.set_index("number")


@lru_cache(maxsize=1)
def load_febrl():
    return _read(C.FEBRL_ORIGINALS), _read(C.FEBRL_DUPLICATES)


def to_record(row):
    """Map a FEBRL row onto the 5 matching fields."""
    name = " ".join(p for p in (_clean(row["given_name"]), _clean(row["surname"])) if p)
    address = " ".join(p for p in (_clean(row["street_number"]), _clean(row["address_1"]),
                                   _clean(row["address_2"]), _clean(row["postcode"])) if p)
    return {
        "name": name,
        "soc_sec_id": _clean(row["soc_sec_id"]),
        "address": address,
        "date_of_birth": _clean(row["date_of_birth"]),
        "city": _clean(row["suburb"]),
    }


def corrupted_fields(original, duplicate):
    """Number of FEBRL fields that differ between an original and its duplicate."""
    return sum(_clean(original[f]) != _clean(duplicate[f]) for f in FEBRL_FIELDS)


@lru_cache(maxsize=1)
def _split():
    """Fixed random split of the 5,000 FEBRL people into 'in master' and 'not in master'."""
    originals, _ = load_febrl()
    numbers = sorted(originals.index)
    random.Random(C.SAMPLE_SEED).shuffle(numbers)
    return tuple(numbers[: C.MASTER_RECORD_COUNT]), tuple(numbers[C.MASTER_RECORD_COUNT:])


def generate_master_records():
    originals, _ = load_febrl()
    in_master, _ = _split()
    return [{"entity_id": n, **to_record(originals.loc[n])} for n in sorted(in_master)]


def _incoming(number, is_match):
    originals, duplicates = load_febrl()
    dup = duplicates.loc[number]
    k = corrupted_fields(originals.loc[number], dup)
    return {
        "source_id": dup["rec_id"],
        **to_record(dup),
        "true_entity_id": number if is_match else None,
        "true_match_status": C.MATCH if is_match else C.NO_MATCH,
        "scenario": f"{k} field(s) corrupted" if is_match else "New person (not in master)",
    }


def _finalise(records, rng):
    rng.shuffle(records)
    for i, rec in enumerate(records, start=1):
        rec["test_record_id"] = f"T{i:03d}"
    columns = ["test_record_id", "source_id"] + C.FIELDS + ["true_entity_id", "true_match_status", "scenario"]
    return [{c: rec[c] for c in columns} for rec in records]


@lru_cache(maxsize=1)
def _test_numbers():
    in_master, outside = _split()
    rng = random.Random(C.SAMPLE_SEED + 1)
    return tuple(rng.sample(in_master, C.TEST_MATCH_COUNT)), tuple(rng.sample(outside, C.TEST_NO_MATCH_COUNT))


def generate_test_records(master_records=None):
    """100 incoming records with official FEBRL ground truth (master_records kept for API compatibility)."""
    match_nums, no_match_nums = _test_numbers()
    records = [_incoming(n, True) for n in match_nums] + [_incoming(n, False) for n in no_match_nums]
    return _finalise(records, random.Random(C.SAMPLE_SEED + 2))


def generate_monitoring_batches():
    """{period: records}. Existing-people duplicates are sorted by number of corrupted fields;
    month 1 gets the cleanest window, the last month the most corrupted one (data-quality drift)."""
    originals, duplicates = load_febrl()
    in_master, outside = _split()
    test_match, test_no_match = _test_numbers()

    pool = [n for n in in_master if n not in set(test_match)]
    pool.sort(key=lambda n: (corrupted_fields(originals.loc[n], duplicates.loc[n]), n))
    new_people = [n for n in outside if n not in set(test_no_match)]
    rng = random.Random(C.SAMPLE_SEED + 3)
    new_people = rng.sample(new_people, C.MONITORING_NO_MATCH_COUNT * len(C.MONITORING_PERIODS))

    months = len(C.MONITORING_PERIODS)
    size = C.MONITORING_MATCH_COUNT
    step = (len(pool) - size) / max(months - 1, 1)
    batches = {}
    for i, period in enumerate(C.MONITORING_PERIODS):
        start = round(i * step)
        matches = pool[start:start + size]
        news = new_people[i * C.MONITORING_NO_MATCH_COUNT:(i + 1) * C.MONITORING_NO_MATCH_COUNT]
        records = [_incoming(n, True) for n in matches] + [_incoming(n, False) for n in news]
        batches[period] = _finalise(records, random.Random(f"{C.SAMPLE_SEED}-{period}"))
    return batches
