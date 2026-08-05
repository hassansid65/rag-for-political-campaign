"""
Metadata extraction: district, category, source, topic (+ candidate, party, schemes).

The assignment's driving example is an utterance like "I'm from Vijayawada", so
district resolution has to work on *aliases and constituency names*, not just
canonical district labels. That gazetteer lives here and is shared by both the
ingest path (tagging chunks) and the query path (inferring a filter from an
utterance) — one source of truth means a query can never infer a district label
the indexer never wrote.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --------------------------------------------------------------------- districts
# Canonical district -> aliases (cities, constituencies, spelling variants).
DISTRICT_GAZETTEER: dict[str, list[str]] = {
    # --- Andhra Pradesh (post-2022 26-district reorganisation) ---
    "Anakapalli": ["anakapalli", "anakapalle"],
    "Ananthapuramu": ["ananthapuramu", "anantapur", "anantapuram", "hindupur", "kadiri", "guntakal"],
    "Annamayya": ["annamayya", "annamaiah", "rayachoti", "madanapalle", "madanapalli"],
    "Bapatla": ["bapatla", "chirala", "repalle", "parchur"],
    "Chittoor": ["chittoor", "chittor", "palamaner", "punganur"],
    "Dr. B.R. Ambedkar Konaseema": ["konaseema", "amalapuram", "ambedkar konaseema", "razole"],
    "East Godavari": ["east godavari", "rajahmundry", "rajamahendravaram", "kovvur", "nidadavolu"],
    "Eluru": ["eluru", "ellore", "chintalapudi", "polavaram"],
    "Guntur": ["guntur", "guntoor", "tenali", "mangalagiri", "ponnur", "tadikonda"],
    "Kakinada": ["kakinada", "cocanada", "pithapuram", "peddapuram", "samalkot"],
    "Krishna": ["krishna", "machilipatnam", "bandar", "gudivada", "pedana", "avanigadda"],
    "Kurnool": ["kurnool", "kurnul", "nandyal", "adoni", "yemmiganur"],
    "Nandyal": ["nandyal", "nandikotkur", "allagadda", "dhone"],
    "NTR": ["ntr district", "vijayawada", "bezawada", "vijaywada", "vijaywada city", "nandigama", "jaggayyapeta", "mylavaram", "tiruvuru"],
    "Palnadu": ["palnadu", "narasaraopet", "sattenapalle", "gurazala", "macherla", "vinukonda"],
    "Parvathipuram Manyam": ["parvathipuram", "manyam", "salur", "kurupam", "palakonda"],
    "Prakasam": ["prakasam", "ongole", "markapur", "kanigiri", "darsi", "addanki"],
    "Sri Potti Sriramulu Nellore": ["nellore", "spsr nellore", "kavali", "gudur", "atmakur", "sarvepalli"],
    "Sri Sathya Sai": ["sri sathya sai", "puttaparthi", "dharmavaram", "penukonda", "kadiri"],
    "Srikakulam": ["srikakulam", "chicacole", "palasa", "ichchapuram", "tekkali"],
    "Tirupati": ["tirupati", "tirumala", "srikalahasti", "sullurpeta", "satyavedu", "renigunta"],
    "Visakhapatnam": ["visakhapatnam", "vizag", "vishakhapatnam", "waltair", "gajuwaka", "bheemili"],
    "Vizianagaram": ["vizianagaram", "vizianagram", "bobbili", "nellimarla", "gajapathinagaram"],
    "West Godavari": ["west godavari", "bhimavaram", "narsapur", "narasapuram", "tadepalligudem", "tanuku", "palakollu"],
    "YSR Kadapa": ["kadapa", "cuddapah", "ysr kadapa", "proddatur", "pulivendula", "jammalamadugu"],
    "Alluri Sitharama Raju": ["alluri sitharama raju", "asr district", "paderu", "rampachodavaram", "chintapalle"],
    "Bapatla Rural": ["karlapalem"],
    # --- Telangana (selected majors, for cross-state campaign docs) ---
    "Hyderabad": ["hyderabad", "secunderabad", "charminar", "hyd"],
    "Rangareddy": ["rangareddy", "ranga reddy", "shamshabad", "ibrahimpatnam"],
    "Medchal-Malkajgiri": ["medchal", "malkajgiri", "kukatpally", "uppal"],
    "Warangal": ["warangal", "hanamkonda", "kazipet"],
    "Karimnagar": ["karimnagar", "jagtial", "huzurabad"],
    "Nizamabad": ["nizamabad", "bodhan", "armoor"],
    "Khammam": ["khammam", "kothagudem", "palvancha"],
    "Nalgonda": ["nalgonda", "miryalaguda", "devarakonda"],
}

STATE_BY_DISTRICT: dict[str, str] = {}
for _d in DISTRICT_GAZETTEER:
    STATE_BY_DISTRICT[_d] = (
        "Telangana"
        if _d in {
            "Hyderabad", "Rangareddy", "Medchal-Malkajgiri", "Warangal",
            "Karimnagar", "Nizamabad", "Khammam", "Nalgonda",
        }
        else "Andhra Pradesh"
    )

# alias -> canonical district, longest-alias-first so "west godavari" beats "godavari"
_ALIAS_TO_DISTRICT: dict[str, str] = {}
for _district, _aliases in DISTRICT_GAZETTEER.items():
    _ALIAS_TO_DISTRICT[_district.lower()] = _district
    for _alias in _aliases:
        _ALIAS_TO_DISTRICT[_alias.lower()] = _district

_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_DISTRICT, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# -------------------------------------------------------------------- categories
CATEGORY_SIGNALS: dict[str, list[str]] = {
    "manifesto": [
        "manifesto", "our promises", "we pledge", "vision document", "poll promise",
        "election manifesto", "commitments to the people", "declaration",
    ],
    "district_info": [
        "district profile", "demographics", "constituency", "mandal", "population",
        "literacy rate", "assembly segment", "voter turnout", "electorate", "polling station",
    ],
    "candidate_profile": [
        "candidate profile", "biography", "born in", "educational qualification",
        "political career", "assets declaration", "affidavit", "mla", "mp",
        "contesting from", "profile of",
    ],
    "scheme": [
        "scheme", "yojana", "beneficiary", "eligibility criteria", "subsidy",
        "pension", "welfare programme", "welfare program", "financial assistance",
        "per annum", "disbursed", "rythu", "amma vodi", "direct benefit transfer",
    ],
    "faq": [
        "frequently asked", "faq", "q:", "q.", "question:", "answer:",
        "how do i", "what is the process", "who is eligible",
    ],
    "press_release": ["press release", "for immediate release", "media statement", "press note"],
    "speech": ["my dear brothers and sisters", "speech", "address to the", "namaskaram", "transcript of"],
}


# ------------------------------------------------------------------------ topics
TOPIC_SIGNALS: dict[str, list[str]] = {
    "agriculture": ["farmer", "rythu", "crop", "irrigation", "agriculture", "paddy", "msp", "minimum support price", "fertiliser", "fertilizer", "kisan"],
    "education": ["school", "college", "student", "scholarship", "education", "university", "amma vodi", "vidya", "teacher", "literacy"],
    "healthcare": ["hospital", "health", "aarogyasri", "medical", "doctor", "clinic", "insurance", "phc", "arogya"],
    "employment": ["job", "employment", "unemployment", "skill development", "youth", "recruitment", "vacancy", "livelihood"],
    "women_welfare": ["women", "mahila", "self help group", "shg", "girl child", "maternity", "widow", "dwcra"],
    "infrastructure": ["road", "bridge", "metro", "highway", "port", "airport", "flyover", "connectivity", "electrification"],
    "water": ["water", "drinking water", "canal", "reservoir", "polavaram", "godavari", "krishna river", "borewell", "tap"],
    "housing": ["house", "housing", "pucca", "shelter", "layout", "navaratnalu house", "ptm"],
    "pension_welfare": ["pension", "old age", "disability", "welfare", "ration", "bpl", "white card"],
    "industry": ["industry", "msme", "investment", "factory", "industrial corridor", "startup", "sez"],
    "energy": ["power", "electricity", "solar", "renewable", "free units", "discom"],
    "law_order": ["police", "crime", "law and order", "safety", "disha", "security"],
    "fisheries": ["fisher", "aqua", "fishing", "harbour", "harbor", "prawn"],
    "transport": ["bus", "rtc", "transport", "railway", "train", "auto"],
}


# ------------------------------------------------------------------- parties etc.
PARTY_SIGNALS: dict[str, list[str]] = {
    "YSRCP": ["ysrcp", "ysr congress", "yuvajana sramika rythu congress"],
    "TDP": ["tdp", "telugu desam", "telugu desam party"],
    "JSP": ["jana sena", "janasena", "jsp"],
    "BJP": ["bharatiya janata party", "bjp"],
    "INC": ["indian national congress", "congress party"],
    "BRS": ["brs", "bharat rashtra samithi", "trs", "telangana rashtra samithi"],
}

_SCHEME_PATTERN = re.compile(
    r"\b("
    r"(?:[A-Z][\w'’-]+\s+){0,3}"
    r"(?:Yojana|Yojna|Scheme|Bharosa|Vodi|Deevena|Sethu|Kanuka|Nestham|Aasara|Pension|Mission|Abhiyan|Card)"
    r")\b"
)

_PERSON_PATTERN = re.compile(
    r"\b(?:Sri|Shri|Smt|Dr|Mr|Ms|Mrs|Hon(?:'|’)?ble)\.?\s+"
    r"([A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3})"
)


# ============================================================================
@dataclass
class ExtractedMetadata:
    category: str = "other"
    district: Optional[str] = None
    districts: list[str] = field(default_factory=list)
    state: Optional[str] = None
    topic: Optional[str] = None
    topics: list[str] = field(default_factory=list)
    candidate: Optional[str] = None
    party: Optional[str] = None
    scheme_names: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


def find_districts(text: str) -> list[str]:
    """Canonical districts mentioned in `text`, ordered by mention frequency."""
    if not text:
        return []
    hits = Counter()
    for match in _ALIAS_PATTERN.finditer(text):
        canonical = _ALIAS_TO_DISTRICT.get(match.group(1).lower())
        if canonical:
            hits[canonical] += 1
    return [d for d, _ in hits.most_common()]


def resolve_district(name: str) -> Optional[str]:
    """Map a free-text place name to a canonical district (used by API filters)."""
    if not name:
        return None
    key = name.strip().lower()
    if key in _ALIAS_TO_DISTRICT:
        return _ALIAS_TO_DISTRICT[key]
    found = find_districts(name)
    return found[0] if found else None


def classify_category(text: str, filename: str = "") -> str:
    """Score category signals over text + filename; filename is weighted higher."""
    haystack = text[:8000].lower()
    fname = filename.lower()
    scores: Counter[str] = Counter()

    for category, signals in CATEGORY_SIGNALS.items():
        for signal in signals:
            occurrences = haystack.count(signal)
            if occurrences:
                scores[category] += min(occurrences, 5)
            if signal in fname:
                scores[category] += 8
        # Filenames are the strongest single signal: "krishna_district_info.md".
        if category.replace("_", "") in fname.replace("_", "").replace("-", ""):
            scores[category] += 12

    # FAQ documents are structurally obvious even without keyword hits.
    q_pairs = len(re.findall(r"^\s*(?:Q\d*[.:)]|Question\s*\d*[.:])", text[:8000], re.MULTILINE | re.IGNORECASE))
    if q_pairs >= 3:
        scores["faq"] += q_pairs * 3

    if not scores:
        return "other"
    return scores.most_common(1)[0][0]


def find_topics(text: str, limit: int = 4) -> list[str]:
    haystack = text.lower()
    scores: Counter[str] = Counter()
    for topic, signals in TOPIC_SIGNALS.items():
        for signal in signals:
            count = haystack.count(signal)
            if count:
                scores[topic] += count
    return [t for t, _ in scores.most_common(limit)]


def find_party(text: str) -> Optional[str]:
    haystack = text.lower()
    scores: Counter[str] = Counter()
    for party, signals in PARTY_SIGNALS.items():
        for signal in signals:
            scores[party] += haystack.count(signal)
    top = scores.most_common(1)
    return top[0][0] if top and top[0][1] else None


def find_schemes(text: str, limit: int = 8) -> list[str]:
    seen: dict[str, None] = {}
    for match in _SCHEME_PATTERN.finditer(text):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        # Drop bare category words matched with no qualifier ("Scheme", "Pension").
        if len(name.split()) < 2:
            continue
        seen.setdefault(name, None)
        if len(seen) >= limit:
            break
    return list(seen)


def find_people(text: str, limit: int = 6) -> list[str]:
    seen: dict[str, None] = {}
    for match in _PERSON_PATTERN.finditer(text):
        seen.setdefault(re.sub(r"\s+", " ", match.group(1)).strip(), None)
        if len(seen) >= limit:
            break
    return list(seen)


def extract_metadata(text: str, filename: str = "") -> ExtractedMetadata:
    """Document-level metadata. Chunk-level refinement happens in the chunker."""
    districts = find_districts(f"{filename} {text}")
    topics = find_topics(text)
    people = find_people(text)
    category = classify_category(text, filename)
    primary_district = districts[0] if districts else None

    candidate: Optional[str] = None
    if category == "candidate_profile" and people:
        candidate = people[0]

    return ExtractedMetadata(
        category=category,
        district=primary_district,
        districts=districts,
        state=STATE_BY_DISTRICT.get(primary_district or "", None),
        topic=topics[0] if topics else None,
        topics=topics,
        candidate=candidate,
        party=find_party(text),
        scheme_names=find_schemes(text),
        entities=people,
    )


def all_districts() -> list[str]:
    return sorted(DISTRICT_GAZETTEER)


def aliases_for(district: str) -> list[str]:
    return DISTRICT_GAZETTEER.get(district, [])


def district_alias_index() -> dict[str, str]:
    """Expose the alias map (read-only use) for query-side district inference."""
    return dict(_ALIAS_TO_DISTRICT)


def iter_alias_matches(text: str) -> Iterable[tuple[str, str]]:
    """Yield (matched_alias, canonical_district) pairs found in `text`."""
    for match in _ALIAS_PATTERN.finditer(text):
        canonical = _ALIAS_TO_DISTRICT.get(match.group(1).lower())
        if canonical:
            yield match.group(1), canonical
