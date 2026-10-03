"""Text normalisation for business names and addresses.

Everything here is pure-Python, rule-based and uses only the provided data
(no external lookups). Country is treated as an open set: country-specific
tables (US / India state names) are used only when the label matches, and any
other label (e.g. France) simply falls back to the generic rules.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Indic transliteration (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
# Telugu, Kannada, Malayalam). These Unicode blocks share the ISCII layout, so
# one offset table (relative to the block start) covers all of them.
# ---------------------------------------------------------------------------
_INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_IND_VOWELS = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
               0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
               0x60: "ri", 0x61: "li"}
_IND_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
             0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
             0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
             0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
             0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
             0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_IND_SIGNS = {0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri", 0x45: "e",
              0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li",
              0x63: "li", 0x57: "au"}
_IND_MODS = {0x01: "n", 0x02: "n", 0x03: "h"}
_VIRAMA = 0x4D
_MALAYALAM_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}


def _indic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp <= 0x0DFF:
        return cp - _INDIC_BLOCKS[(cp - 0x0900) // 0x80]
    return None


def has_indic(s):
    return any(0x0900 <= ord(c) <= 0x0DFF for c in s)


def transliterate_indic(s):
    """Rough phonetic romanisation. Inherent 'a' is added after a consonant
    unless a vowel sign / virama follows, and dropped at word end (Hindi-style
    schwa deletion) -- good enough for fuzzy comparison with the Latin name."""
    out = []
    pending_a = False           # a consonant was emitted and still carries its inherent 'a'
    for ch in s:
        cp = ord(ch)
        if cp in _MALAYALAM_CHILLU:
            if pending_a:
                out.append("a")
            out.append(_MALAYALAM_CHILLU[cp])
            pending_a = False
            continue
        off = _indic_offset(ch)
        if off is None:
            if pending_a and not ch.isspace():
                out.append("a")
            pending_a = False                 # word end: schwa deletion
            if 0x0964 <= cp <= 0x0965:
                out.append(" ")
            else:
                out.append(ch)
            continue
        if off in _IND_CONS:
            if pending_a:
                out.append("a")
            out.append(_IND_CONS[off])
            pending_a = True
        elif off in _IND_SIGNS:
            out.append(_IND_SIGNS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off in _IND_VOWELS:
            if pending_a:
                out.append("a")
            out.append(_IND_VOWELS[off])
            pending_a = False
        elif off in _IND_MODS:
            if pending_a:
                out.append("a")
            out.append(_IND_MODS[off])
            pending_a = False
        elif 0x66 <= off <= 0x6F:            # native digits
            if pending_a:
                out.append("a")
            out.append(str(off - 0x66))
            pending_a = False
        else:                                 # nukta, avagraha, om, ... -> drop
            continue
    return "".join(out)


# ---------------------------------------------------------------------------
# Generic Latin cleaning
# ---------------------------------------------------------------------------
_NON_ALNUM = re.compile(r"[^0-9a-z ]+")
_SPACES = re.compile(r"\s+")
_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")


def strip_accents(s):
    """Remove Latin diacritics (é->e) without touching Indic combining signs."""
    if s.isascii():
        return s
    d = unicodedata.normalize("NFKD", s)
    return "".join(c for c in d if not (0x0300 <= ord(c) <= 0x036F))


def base_clean(s):
    """lower, NFKC, transliterate Indic, strip accents, '&'->and, keep [0-9a-z ]."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    if has_indic(s):
        s = transliterate_indic(s)
    s = strip_accents(s)
    s = s.replace("&", " and ").replace("'", "").replace("’", "")
    s = _NON_ALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def join_single_letters(tokens):
    """['s','a','s'] -> ['sas'];  'e u r l' -> 'eurl' (dotted acronyms)."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
        else:
            if run:
                out.append("".join(run))
                run = []
            out.append(t)
    if run:
        out.append("".join(run))
    return out


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
LEGAL = {
    "inc", "incorporated", "incorporation", "corp", "corporation", "co", "company", "cie", "ltd", "limited",
    "llc", "llp", "lp", "plc", "pvt", "private", "pte", "gmbh", "sarl", "sas", "sasu", "sa", "sci", "snc",
    "eurl", "selarl", "scop", "scp", "sca", "ltda", "pllc", "pc", "pa", "lllp", "opc",
    # romanised Indic forms of the same words (after transliteration)
    "praivet", "praaivet", "praivaet", "limited", "limitaed", "limiteda", "limitad", "limitd", "elelpi",
    "eleelpi", "elaelapii", "elelpii", "elaelapi", "elaelasii", "kampanii", "kampani", "korporeshan",
    "praivettt", "privet", "praiveta", "praivetaa", "praiveet", "praivaeta", "praiveti", "limitedaa",
}
HONORIFIC = {"mr", "mrs", "ms", "miss", "dr", "shri", "sri", "shree", "smt", "ms", "messrs", "m s", "sh", "km",
             "prof", "the", "www", "com", "net", "org", "in", "co in", "http", "https"}
_DBA = re.compile(r"\b(?:dba|d b a|t a|ta|trading as|doing business as|aka|a k a)\b")
_ID_TAG = re.compile(r"\((?:id|ref|reg|no)[^)]*\)|\b(?:id|ref)\s*[:#]?\s*\d+\b", re.I)
_DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co\.in|in|co|biz|info|us|fr|io)\b", re.I)


def normalize_name(raw):
    """-> dict(full, core, is_domain, dba)

    full : cleaned name with legal words kept (tokens joined by space)
    core : legal suffixes / honorifics removed, duplicated tokens removed
    dba  : the part after 'dba' / 't/a' if present (else "")
    """
    if not raw:
        return "", "", 0, ""
    s = raw
    s = _ID_TAG.sub(" ", s)
    is_domain = 0
    m = _DOMAIN.search(s.lower())
    if m:
        is_domain = 1
        s = _DOMAIN.sub(lambda mm: " " + mm.group(1) + " ", s.lower())
    s = s.replace("/", " / ")
    c = base_clean(s)
    dba = ""
    parts = _DBA.split(c)
    if len(parts) > 1:
        dba = parts[-1].strip()
        c = " ".join(p.strip() for p in parts)
    toks = join_single_letters(c.split())
    full = " ".join(toks)
    core_toks, seen = [], set()
    for t in toks:
        if t in LEGAL or t in HONORIFIC:
            continue
        if t in seen:
            continue
        seen.add(t)
        core_toks.append(t)
    if not core_toks:                    # never let the core become empty
        core_toks = [t for t in toks if t not in HONORIFIC] or toks
    return full, " ".join(core_toks), is_domain, dba


_SKEL_H = re.compile(r"(?<=[bcdfgjkpqrstvxz])h")
_SKEL_MAP = str.maketrans({"c": "k", "q": "k", "g": "k", "x": "k", "z": "s", "w": "v", "b": "p", "d": "t",
                           "j": "k", "f": "p"})
_VOWELS = re.compile(r"[aeiouy]")
_REPEAT = re.compile(r"(.)\1+")


def skeleton_token(t):
    """Consonant skeleton, voicing merged: robust to transliteration & vowel typos.
    hospitality -> hsptlt ; 'hospitailiti' (romanised Devanagari) -> hsptlt."""
    if not t:
        return ""
    if t.isdigit():
        return t
    first = t[0]
    t = _SKEL_H.sub("", t).translate(_SKEL_MAP)
    t = _VOWELS.sub("", t[1:])
    t = (first.translate(_SKEL_MAP) + t)
    return _REPEAT.sub(r"\1", t)


def skeleton(s):
    return " ".join(skeleton_token(t) for t in s.split())


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
_ADDR_CANON = {}
for canon, variants in {
    "st": "street str st saint",
    "rd": "road rd",
    "av": "avenue ave av avn aven",
    "bd": "boulevard blvd bd boul bld",
    "dr": "drive dr drv",
    "ln": "lane ln",
    "ct": "court ct crt",
    "pl": "place pl plz plaza",
    "sq": "square sq",
    "hwy": "highway hwy",
    "pkwy": "parkway pkwy pky",
    "cir": "circle cir crcl",
    "ter": "terrace ter terr",
    "trl": "trail trl",
    "way": "way wy",
    "xing": "crossing xing",
    "expy": "expressway expy",
    "fwy": "freeway fwy",
    "ctr": "center centre ctr",
    "mt": "mount mt",
    "ft": "fort ft",
    "n": "north n",
    "s": "south s",
    "e": "east e",
    "w": "west w",
    "ne": "northeast ne",
    "nw": "northwest nw",
    "se": "southeast se",
    "sw": "southwest sw",
    "r": "rue r",
    "ch": "chemin ch",
    "imp": "impasse imp",
    "allee": "allee all al",
    "rte": "route rte",
    "fbg": "faubourg fbg",
    "qu": "quai qu",
    "crs": "cours crs",
    "ste": "sainte ste",
    "nagar": "nagar ngr",
    "sec": "sector sec sect",
    "ph": "phase ph",
    "mkt": "market mkt",
    "opp": "opposite opp",
    "nr": "near nr",
    "bldg": "building bldg",
    "fl": "floor flr fl",
    "apt": "apartment apt apts",
    "cplx": "complex cplx",
    "extn": "extension extn ext",
    "vill": "village vill vil",
    "dist": "district dist",
}.items():
    for v in variants.split():
        _ADDR_CANON[v] = canon

# unit / number designators: the same number is written "Shop No 9", "#9", "Door No 9", "Unit 9"
ADDR_DROP = {"no", "number", "num", "nos", "unit", "suite", "ste_", "door", "dno", "hno", "h", "d", "plot",
             "shop", "flat", "house", "po", "box", "pmb", "bldg_", "na", "n a", "and", "of", "the", "de", "du",
             "des", "la", "le", "les", "bis", "ter_", "near_"}

# --- state gazetteers (only used when the record's country label matches) ---
_US_STATES = """al alabama|ak alaska|az arizona|ar arkansas|ca california|co colorado|ct connecticut|
de delaware|dc district of columbia|fl florida|ga georgia|hi hawaii|id idaho|il illinois|
in indiana|ia iowa|ks kansas|ky kentucky|la louisiana|me maine|md maryland|ma massachusetts|
mi michigan|mn minnesota|ms mississippi|mo missouri|mt montana|ne nebraska|nv nevada|
nh new hampshire|nj new jersey|nm new mexico|ny new york|nc north carolina|nd north dakota|
oh ohio|ok oklahoma|or oregon|pa pennsylvania|ri rhode island|sc south carolina|
sd south dakota|tn tennessee|tx texas|ut utah|vt vermont|va virginia|wa washington|
wv west virginia|wi wisconsin|wy wyoming|pr puerto rico"""
_IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg ct",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od or orissa", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk ut uttaranchal", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py pondicherry",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
# native-script state names (romanised by transliterate_indic) -> canonical
_IN_NATIVE = {
    "आंध्र प्रदेश": "andhra pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ఆంధ్ర ప్రదేశ్": "andhra pradesh",
    "अरुणाचल प्रदेश": "arunachal pradesh", "असम": "assam", "অসম": "assam", "बिहार": "bihar",
    "छत्तीसगढ़": "chhattisgarh", "गोवा": "goa", "गुजरात": "gujarat", "ગુજરાત": "gujarat",
    "हरियाणा": "haryana", "हिमाचल प्रदेश": "himachal pradesh", "झारखंड": "jharkhand",
    "ಕರ್ನಾಟಕ": "karnataka", "कर्नाटक": "karnataka", "കേരളം": "kerala", "केरल": "kerala",
    "मध्य प्रदेश": "madhya pradesh", "महाराष्ट्र": "maharashtra", "मणिपुर": "manipur", "मेघालय": "meghalaya",
    "मिज़ोरम": "mizoram", "नागालैंड": "nagaland", "ଓଡ଼ିଶା": "odisha", "ओडिशा": "odisha", "ਪੰਜਾਬ": "punjab",
    "पंजाब": "punjab", "राजस्थान": "rajasthan", "सिक्किम": "sikkim", "தமிழ்நாடு": "tamil nadu",
    "तमिलनाडु": "tamil nadu", "తెలంగాణ": "telangana", "तेलंगाना": "telangana", "त्रिपुरा": "tripura",
    "उत्तर प्रदेश": "uttar pradesh", "उत्तराखंड": "uttarakhand", "পশ্চিমবঙ্গ": "west bengal",
    "पश्चिम बंगाल": "west bengal", "दिल्ली": "delhi", "जम्मू और कश्मीर": "jammu and kashmir",
    "चंडीगढ़": "chandigarh", "पुडुचेरी": "puducherry",
}


def _build_state_tables():
    tables = {"US": {}, "India": {}}
    for entry in _US_STATES.replace("\n", "").split("|"):
        abbr, name = entry.split(" ", 1)
        tables["US"][abbr] = abbr
        tables["US"][name] = abbr
    for name, abbrs in _IN_STATES.items():
        code = abbrs.split()[0]
        tables["India"][name] = code
        for a in abbrs.split():
            tables["India"][a] = code
    for native, name in _IN_NATIVE.items():
        tables["India"][base_clean(native)] = tables["India"][name]
    return tables


STATE_TABLES = _build_state_tables()
_STATE_RE = {c: re.compile(r"\b(" + "|".join(sorted((k for k in t if " " in k or len(k) > 2),
                                                        key=len, reverse=True)) + r")\b")
             for c, t in STATE_TABLES.items()}


def normalize_address(raw, country):
    """-> (addr_tokens_str, state, postal, house, city, nums_str)

    addr_tokens_str: canonical tokens (abbreviations unified, unit designators
    dropped, state names replaced by their code, digit/letter runs split).
    """
    if not raw:
        return "", "", "", "", "", ""
    table = STATE_TABLES.get(country)
    chunks = [base_clean(p) for p in raw.split(",")]
    chunks = [c for c in chunks if c]

    # state: rightmost chunk that is (entirely) a state name / code
    state, state_idx = "", -1
    if table:
        for i in range(len(chunks) - 1, -1, -1):
            letters = " ".join(w for w in chunks[i].split() if not w.isdigit())
            if letters in table:
                state, state_idx = table[letters], i
                break
    # city: the digit-free chunk adjacent to the state chunk
    city = ""
    if state_idx >= 0:
        for j in (state_idx - 1, state_idx + 1):
            if 0 <= j < len(chunks) and not any(ch.isdigit() for ch in chunks[j]) and chunks[j] not in table:
                city = chunks[j]
                break

    if state_idx >= 0:        # write the state chunk as its canonical code (tg / ts / telangana -> ts)
        chunks[state_idx] = " ".join([state] + [w for w in chunks[state_idx].split() if w.isdigit()])
    text = " ".join(chunks)
    if table:
        text = _STATE_RE[country].sub(lambda m: table[m.group(1)], text)
        if not state:
            m = _STATE_RE[country].findall(" ".join(chunks))
            if m:
                state = table[m[-1]]
    text = _DIGIT_ALPHA.sub(" ", text)
    toks = []
    for t in text.split():
        t = _ADDR_CANON.get(t, t)
        if t in ADDR_DROP:
            continue
        toks.append(t)
    nums = [t for t in toks if t.isdigit()]
    postal = next((n for n in nums if len(n) == 6), "") or next((n for n in nums[1:] if len(n) == 5), "")
    house = next((n for n in nums if n != postal), "")
    return " ".join(toks), state, postal, house, city, " ".join(nums)
