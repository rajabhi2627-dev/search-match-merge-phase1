"""
Synthetic data generation (fixed random seeds -> fully reproducible).

* generate_master_records() : ~1,000 master / entity records
* generate_test_records()   : exactly 100 incoming records with automatic ground truth
"""
import random
from datetime import date, timedelta

import config as C

FIRST_NAMES = [
    "Rahul", "Amit", "Priya", "Neha", "Rohit", "Anjali", "Vikram", "Pooja", "Suresh", "Kavita",
    "Arjun", "Sneha", "Manish", "Divya", "Sanjay", "Ritu", "Deepak", "Swati", "Rajesh", "Meera",
    "Karan", "Nisha", "Ankit", "Shreya", "Vivek", "Komal", "Abhishek", "Payal", "Gaurav", "Simran",
    "Nikhil", "Aarti", "Harsh", "Tanvi", "Saurabh", "Isha", "Aditya", "Kriti", "Varun", "Megha",
    "Ravi", "Sunita", "Ajay", "Rekha", "Mohit", "Shalini", "Pankaj", "Jyoti", "Tarun", "Preeti",
    "Ashish", "Nandini", "Kunal", "Bhavna", "Yash", "Aishwarya", "Manoj", "Lakshmi", "Siddharth", "Farah",
]

LAST_NAMES = [
    "Kumar", "Sharma", "Singh", "Verma", "Gupta", "Patel", "Reddy", "Nair", "Iyer", "Mehta",
    "Joshi", "Mishra", "Yadav", "Chauhan", "Malhotra", "Agarwal", "Banerjee", "Chatterjee", "Das", "Pandey",
    "Srivastava", "Tiwari", "Saxena", "Kapoor", "Bhatia", "Menon", "Pillai", "Rao", "Desai", "Kulkarni",
    "Jain", "Sinha", "Thakur", "Chopra", "Khan", "Ansari", "Ghosh", "Bose", "Shetty", "Naidu",
]

CITY_STREETS = {
    "Patna": ["MG Road", "Boring Road", "Kankarbagh Main Road", "Ashok Rajpath", "Bailey Road"],
    "Delhi": ["Connaught Place", "Lajpat Nagar", "Rajouri Garden", "Karol Bagh", "Mayur Vihar"],
    "Mumbai": ["Linking Road", "Marine Drive", "Andheri East", "Hill Road", "Powai Lake Road"],
    "Bengaluru": ["MG Road", "Indiranagar 100 Feet Road", "Koramangala 5th Block", "Whitefield Main Road", "Jayanagar 4th Block"],
    "Chennai": ["Anna Salai", "T Nagar", "Adyar Main Road", "Velachery Road", "Nungambakkam High Road"],
    "Kolkata": ["Park Street", "Salt Lake Sector 5", "Gariahat Road", "Ballygunge Place", "Camac Street"],
    "Hyderabad": ["Banjara Hills Road", "Jubilee Hills", "Madhapur Main Road", "Ameerpet", "Kukatpally"],
    "Pune": ["FC Road", "Koregaon Park", "Baner Road", "Kothrud", "Viman Nagar"],
    "Ahmedabad": ["CG Road", "SG Highway", "Navrangpura", "Satellite Road", "Maninagar"],
    "Jaipur": ["MI Road", "Malviya Nagar", "Vaishali Nagar", "C Scheme", "Tonk Road"],
    "Lucknow": ["Hazratganj", "Gomti Nagar", "Aliganj", "Indira Nagar", "Aminabad"],
    "Bhopal": ["MP Nagar", "Arera Colony", "New Market", "Kolar Road", "Shahpura"],
    "Chandigarh": ["Sector 17", "Sector 22", "Sector 35", "Madhya Marg", "Sector 8"],
    "Kochi": ["MG Road", "Marine Drive", "Kakkanad", "Edappally", "Panampilly Nagar"],
    "Ranchi": ["Main Road", "Lalpur", "Harmu Road", "Kanke Road", "Doranda"],
}
CITIES = list(CITY_STREETS)

COMPANIES = [
    "ABC Technologies", "Infosys", "Tata Consultancy Services", "Wipro", "HCL Technologies",
    "Tech Mahindra", "Reliance Industries", "HDFC Bank", "ICICI Bank", "State Bank of India",
    "Axis Bank", "Larsen and Toubro", "Mahindra and Mahindra", "Bajaj Finance", "Asian Paints",
    "Hindustan Unilever", "ITC Limited", "Bharti Airtel", "Sun Pharma", "Dr Reddys Laboratories",
    "Maruti Suzuki", "Godrej Consumer Products", "Zomato", "Flipkart", "Paytm",
    "Byjus", "Ola Cabs", "Swiggy", "Deloitte India", "Grant Thornton Bharat",
    "Accenture India", "Capgemini India", "Mphasis", "Mindtree Solutions", "Persistent Systems",
    "Sunrise Traders", "Ganga Logistics", "Shree Ram Textiles", "Bharat Electronics", "Kotak Mahindra Bank",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "rediffmail.com", "hotmail.com"]

DOB_START = date(1960, 1, 1)
DOB_END = date(2004, 12, 31)


# ------------------------------------------------------------------
# Master data
# ------------------------------------------------------------------

def _random_dob(rng):
    return (DOB_START + timedelta(days=rng.randint(0, (DOB_END - DOB_START).days))).isoformat()


def _random_phone(rng, used_phones):
    while True:
        phone = str(rng.choice("6789")) + "".join(str(rng.randint(0, 9)) for _ in range(9))
        if phone not in used_phones:
            used_phones.add(phone)
            return phone


def _random_email(rng, first, last, used_emails, avoid_pattern=None):
    patterns = [
        lambda: f"{first}.{last}",
        lambda: f"{first}{last}{rng.randint(1, 99)}",
        lambda: f"{first[0]}{last}",
        lambda: f"{first}_{last}{rng.randint(70, 99)}",
        lambda: f"{last}.{first}",
    ]
    while True:
        idx = rng.randrange(len(patterns))
        if idx == avoid_pattern:
            continue
        email = f"{patterns[idx]()}@{rng.choice(EMAIL_DOMAINS)}".lower()
        if email not in used_emails:
            used_emails.add(email)
            return email


def _random_address(rng, city):
    return f"{rng.randint(1, 250)} {rng.choice(CITY_STREETS[city])} {city}"


def make_person(rng, used_emails, used_phones, first=None, last=None, city=None):
    first = first or rng.choice(FIRST_NAMES)
    last = last or rng.choice(LAST_NAMES)
    city = city or rng.choice(CITIES)
    return {
        "name": f"{first} {last}",
        "email": _random_email(rng, first.lower(), last.lower(), used_emails),
        "phone": _random_phone(rng, used_phones),
        "address": _random_address(rng, city),
        "date_of_birth": _random_dob(rng),
        "company": rng.choice(COMPANIES),
        "city": city,
    }


def generate_master_records(n=C.MASTER_RECORD_COUNT, seed=C.MASTER_SEED):
    """Generate n synthetic master records. Entity 10001 is the documented example (Rahul Kumar)."""
    rng = random.Random(seed)
    used_emails, used_phones = {"rahul.kumar@gmail.com"}, {"9876543210"}
    records = [{
        "entity_id": C.FIRST_ENTITY_ID, "name": "Rahul Kumar", "email": "rahul.kumar@gmail.com",
        "phone": "9876543210", "address": "12 MG Road Patna", "date_of_birth": "1998-04-12",
        "company": "ABC Technologies", "city": "Patna",
    }]
    for i in range(1, n):
        person = make_person(rng, used_emails, used_phones)
        records.append({"entity_id": C.FIRST_ENTITY_ID + i, **person})
    return records


# ------------------------------------------------------------------
# Variation helpers (each one imitates a realistic data-entry problem)
# ------------------------------------------------------------------

def typo(word, rng):
    """One small spelling error (delete / replace / swap / insert), never on the first letter."""
    if len(word) < 4:
        return word
    i = rng.randint(1, len(word) - 2)
    op = rng.choice(["delete", "replace", "swap", "insert"])
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "replace":
        return word[:i] + rng.choice("aeiourstn") + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rng.choice("aeiou") + word[i:]


def vary_name(name, rng):
    parts = name.split()
    idx = max(range(len(parts)), key=lambda k: len(parts[k])) if rng.random() < 0.5 else rng.randrange(len(parts))
    parts[idx] = typo(parts[idx], rng)
    return " ".join(parts)


def vary_email(email, rng):
    local, domain = email.split("@")
    option = rng.choice(["case_space", "domain_typo", "local_typo", "dot"])
    if option == "case_space":
        return f"  {email.upper()} "
    if option == "domain_typo":
        name, tld = domain.split(".", 1)
        return f"{local}@{typo(name, rng) if len(name) >= 4 else name + 'l'}.{tld}"
    if option == "dot" and "." in local:
        return f"{local.replace('.', '', 1)}@{domain}"
    return f"{typo(local, rng)}@{domain}"


def vary_phone_format(phone, rng):
    formats = [
        f"+91 {phone[:5]} {phone[5:]}",
        f"+91-{phone}",
        f"0{phone[:5]} {phone[5:]}",
        f"91 {phone}",
        f"({phone[:5]}) {phone[5:]}",
        f"{phone[:5]}-{phone[5:]}",
        f"0091 {phone}",
        f"{phone[:3]} {phone[3:6]} {phone[6:]}",
    ]
    return rng.choice(formats)


def phone_digit_typo(phone, rng):
    i = rng.randint(2, len(phone) - 1)
    new_digit = str((int(phone[i]) + rng.randint(1, 8)) % 10)
    return phone[:i] + new_digit + phone[i + 1:]


def vary_address(address, rng):
    replacements = [(" Road", " Rd"), (" Street", " St"), ("MG ", "Mahatma Gandhi "), ("Sector ", "Sec "),
                    (" Nagar", " Ngr")]
    option = rng.choice(["abbreviate", "punctuation", "landmark", "typo", "reorder"])
    if option == "abbreviate":
        for old, new in replacements:
            if old in address:
                return address.replace(old, new, 1)
        option = "punctuation"
    if option == "punctuation":
        parts = address.split(" ", 1)
        return f"{parts[0]}, {parts[1].rsplit(' ', 1)[0]}, {address.rsplit(' ', 1)[1]}"
    if option == "landmark":
        number, rest = address.split(" ", 1)
        return f"{number} {rest.rsplit(' ', 1)[0]} Near Bus Stand {rest.rsplit(' ', 1)[1]}"
    if option == "typo":
        words = address.split()
        idx = max(range(1, len(words)), key=lambda k: len(words[k]))
        words[idx] = typo(words[idx], rng)
        return " ".join(words)
    number, rest = address.split(" ", 1)
    return f"{rest} {number}"


def vary_company(company, rng):
    option = rng.choice(["typo", "suffix", "abbreviate"])
    if option == "abbreviate":
        for old, new in [("Technologies", "Tech"), ("Services", "Svcs"), ("Laboratories", "Labs"),
                         ("Industries", "Inds"), ("Solutions", "Soln")]:
            if old in company:
                return company.replace(old, new)
        option = "suffix"
    if option == "suffix":
        return company + rng.choice([" Pvt Ltd", " Ltd", " Private Limited"])
    words = company.split()
    idx = max(range(len(words)), key=lambda k: len(words[k]))
    words[idx] = typo(words[idx], rng)
    return " ".join(words)


def vary_dob_format(dob, rng):
    y, m, d = dob.split("-")
    month_abbr = date(int(y), int(m), int(d)).strftime("%b")
    return rng.choice([f"{d}/{m}/{y}", f"{d}-{m}-{y}", f"{d} {month_abbr} {y}", dob])


# ------------------------------------------------------------------
# Scenario builders: (master record or None) -> incoming record
# ------------------------------------------------------------------

def _copy(rec):
    return {f: rec[f] for f in C.FIELDS}


def scenario_clear(rec, rng, ctx):
    r = _copy(rec)
    r["name"] = rng.choice([r["name"], r["name"].upper(), f"  {r['name']} "])
    r["date_of_birth"] = vary_dob_format(r["date_of_birth"], rng)
    return r


def scenario_name(rec, rng, ctx):
    r = _copy(rec)
    r["name"] = vary_name(r["name"], rng)
    return r


def scenario_email(rec, rng, ctx):
    r = _copy(rec)
    r["email"] = vary_email(r["email"], rng)
    return r


def scenario_phone(rec, rng, ctx):
    r = _copy(rec)
    r["phone"] = vary_phone_format(r["phone"], rng)
    return r


def scenario_address(rec, rng, ctx):
    r = _copy(rec)
    r["address"] = vary_address(r["address"], rng)
    return r


def scenario_company(rec, rng, ctx):
    r = _copy(rec)
    r["company"] = vary_company(r["company"], rng)
    return r


def scenario_noisy(rec, rng, ctx):
    """Four or five problems at the same time, including real-life changes (new email, new phone, new job)."""
    r = _copy(rec)
    r["name"] = vary_name(r["name"], rng)
    if rng.random() < 0.3:
        r["name"] = vary_name(r["name"], rng)
    r["address"] = vary_address(r["address"], rng)
    first, last = rec["name"].lower().split(" ", 1)
    extra = rng.sample(["email_typo", "email_changed", "phone_format", "phone_typo", "phone_changed",
                        "company_changed", "missing_dob"], 2)
    for e in extra:
        if e == "email_typo":
            r["email"] = vary_email(r["email"], rng)
        elif e == "email_changed":
            r["email"] = _random_email(rng, first, last, ctx["used_emails"])
        elif e == "phone_format":
            r["phone"] = vary_phone_format(r["phone"], rng)
        elif e == "phone_typo":
            r["phone"] = phone_digit_typo(r["phone"], rng)
        elif e == "phone_changed":
            r["phone"] = _random_phone(rng, ctx["used_phones"])
        elif e == "company_changed":
            r["company"] = rng.choice([c for c in COMPANIES if c != rec["company"]])
        else:
            r["date_of_birth"] = ""
    if r["date_of_birth"]:
        r["date_of_birth"] = vary_dob_format(r["date_of_birth"], rng)
    return r


def scenario_changed_contact(rec, rng, ctx):
    """Same person, but both email and phone have changed since the master record was created."""
    first, last = rec["name"].lower().split(" ", 1)
    r = _copy(rec)
    r["email"] = _random_email(rng, first, last, ctx["used_emails"])
    r["phone"] = _random_phone(rng, ctx["used_phones"])
    return r


def scenario_sparse(rec, rng, ctx):
    """Only name, address, company and city are known - no email, phone or DOB."""
    r = _copy(rec)
    r["email"], r["phone"], r["date_of_birth"] = "", "", ""
    r["address"] = vary_address(r["address"], rng)
    return r


def scenario_family(rec, rng, ctx):
    """A different person in the same household: same surname, phone and address
    (about half of the households also share one email address)."""
    first, last = rec["name"].split(" ", 1)
    new_first = rng.choice([n for n in FIRST_NAMES if n != first])
    r = _copy(rec)
    r["name"] = f"{new_first} {last}"
    if rng.random() < 0.5:
        r["email"] = _random_email(rng, new_first.lower(), last.lower(), ctx["used_emails"])
    r["date_of_birth"] = _random_dob(rng)
    r["company"] = rng.choice(COMPANIES)
    return r


def scenario_new(rec, rng, ctx):
    return make_person(rng, ctx["used_emails"], ctx["used_phones"])


def scenario_lookalike(rec, rng, ctx):
    """Same name and city as an existing entity (sometimes even the same employer), but a different person."""
    first, last = rec["name"].split(" ", 1)
    person = make_person(rng, ctx["used_emails"], ctx["used_phones"], first=first, last=last, city=rec["city"])
    if rng.random() < 0.5:
        person["company"] = rec["company"]
    while person["address"] == rec["address"]:
        person["address"] = _random_address(rng, rec["city"])
    return person


SCENARIO_BUILDERS = {
    "Clear match": scenario_clear,
    "Name spelling error": scenario_name,
    "Email variation": scenario_email,
    "Phone formatting": scenario_phone,
    "Address variation": scenario_address,
    "Company spelling variation": scenario_company,
    "Multiple noisy fields": scenario_noisy,
    "Changed contact details": scenario_changed_contact,
    "Ambiguous - sparse record": scenario_sparse,
    "Ambiguous - family member": scenario_family,
    "New person": scenario_new,
    "Look-alike (same name)": scenario_lookalike,
}


def generate_test_records(master_records, seed=C.TEST_SEED, scenarios=C.TEST_SCENARIOS):
    """Generate the incoming test records with automatic ground truth.

    Every record carries: test_record_id, true_entity_id, true_match_status, scenario.
    """
    rng = random.Random(seed)
    ctx = {
        "used_emails": {r["email"] for r in master_records},
        "used_phones": {r["phone"] for r in master_records},
    }

    # Entities whose name appears more than once are used for the sparse (ambiguous) scenario.
    name_counts = {}
    for r in master_records:
        name_counts[r["name"]] = name_counts.get(r["name"], 0) + 1
    duplicate_named = [r for r in master_records if name_counts[r["name"]] > 1]
    pool = master_records[:]
    rng.shuffle(pool)
    rng.shuffle(duplicate_named)
    used_ids = set()

    def take(prefer=None):
        for candidate in (prefer or []) + pool:
            if candidate["entity_id"] not in used_ids:
                used_ids.add(candidate["entity_id"])
                return candidate
        raise ValueError("Not enough master records for the requested scenarios")

    records = []
    for scenario, spec in scenarios.items():
        builder = SCENARIO_BUILDERS[scenario]
        for _ in range(spec["count"]):
            source = None if scenario == "New person" else take(duplicate_named if "sparse" in scenario else None)
            incoming = builder(source, rng, ctx)
            is_match = spec["true_status"] == C.MATCH
            records.append({
                **incoming,
                "true_entity_id": source["entity_id"] if is_match else None,
                "true_match_status": spec["true_status"],
                "scenario": scenario,
            })

    rng.shuffle(records)
    for i, rec in enumerate(records, start=1):
        rec["test_record_id"] = f"T{i:03d}"
    columns = ["test_record_id"] + C.FIELDS + ["true_entity_id", "true_match_status", "scenario"]
    return [{c: rec[c] for c in columns} for rec in records]
