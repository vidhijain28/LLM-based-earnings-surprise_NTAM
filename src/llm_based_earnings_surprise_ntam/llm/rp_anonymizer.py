"""rp_anonymizer: mask what lets a language model recognize an earnings call.

Replaces, in a transcript whose speaker turns look like "Name (Role): text":
  - the company name (full legal name, and the short name without Inc., Corp., ...) -> [COMPANY]
  - the ticker                                                                      -> [TICKER]
  - speaker names (full names, and first or last names of Executives and Analysts)  -> [EXECUTIVE n] / [ANALYST n]
  - optionally four-digit years (MASK_YEARS = True)                                 -> [YEAR]

Usage:
    from rp_anonymizer import anonymize
    text_anon, stats = anonymize(text, company="Example Corp.", ticker="EXMP")
    # stats: counts of replacements, plus "leaks" (occurrences the rules meant to replace but did not)

Limits: this removes names, it does not guarantee the text is unidentifiable. Product names, brands and places stay.
Short one-word company names that are also ordinary words (for example "Target") are replaced only when capitalized
or upper case; the lower-case leftovers are counted in stats["ordinary_word_hits"].
The functions are copied unchanged from the scoring notebook rp_06.
"""
import re

ALIAS_REPLACEMENT = True     # also replace first and last names of speakers
MASK_YEARS = False           # also replace 4-digit years (2000-2030) with [YEAR]

ROLE_LABEL = {"Executive": "EXECUTIVE", "Analyst": "ANALYST"}
LINE_RE = re.compile(r"^(?P<name>[^\n():]{2,80}?) \((?P<role>Executive|Analyst|Operator|Shareholder|Attendee|Unknown)\): ", re.M)
CORP_SUFFIX = {"inc", "inc.", "corp", "corp.", "corporation", "co", "co.", "company", "ltd", "ltd.", "limited", "plc", "llc", "lp",
               "holdings", "holding", "group", "n.v.", "s.a.", "ag", "se", "the"}
COMMON_FIRST_NAMES = {"will", "mark", "grant", "bill", "rich", "frank", "dean", "hope", "faith", "joy", "rob", "pat", "ray", "gene"}

def company_variants(name, ticker):
    full = str(name).strip()
    tokens = [t for t in re.split(r"\s+", full) if t]
    if len(tokens) > 1 and tokens[0].lower() == "the":
        tokens = tokens[1:]
    core_tokens = tokens[:]
    while core_tokens and core_tokens[-1].lower().strip(",") in CORP_SUFFIX:
        core_tokens.pop()
    core = " ".join(core_tokens).strip(" ,")
    full_clean = " ".join(tokens)
    variants = sorted({v for v in [full, full_clean, full_clean.rstrip("."), core] if len(v) >= 3}, key=len, reverse=True)
    return variants, core, (str(ticker).strip() if isinstance(ticker, str) and len(str(ticker).strip()) >= 2 else None)

def is_loose(core):
    # Names that are unlikely to be everyday words are replaced in any letter case.
    return (" " in core) or any(ch.isupper() for ch in core[1:]) or len(core) >= 8

def company_regex(variant, core):
    if variant != core or is_loose(core):
        return re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.I)
    # Short one-word names (for example "Target" or "Block"): replace the capitalized and the upper-case forms only,
    # because the lower-case form is usually an ordinary word.
    return re.compile(rf"(?<!\w)(?:{re.escape(core)}|{re.escape(core.upper())})(?!\w)")

def anonymize(text, company, ticker):
    variants, core, tick = company_variants(company, ticker)
    stats = {"company": 0, "ticker": 0, "people": 0, "years": 0}
    people = {}
    for m in LINE_RE.finditer(text):
        name, role = m.group("name").strip(), m.group("role")
        if role in ROLE_LABEL and name not in people:
            people[name] = f"[{ROLE_LABEL[role]} {sum(1 for p in people.values() if p.startswith('[' + ROLE_LABEL[role]))+1}]"
    out = text
    if people:
        names = sorted(people, key=len, reverse=True)            # longest names first
        out, k = re.compile("|".join(re.escape(n) for n in names)).subn(lambda m: people[m.group(0)], out); stats["people"] += k
        if ALIAS_REPLACEMENT:
            part_map = {}
            for n in names:                                       # a part shared by two people goes to the longer name
                for part in n.split():
                    if len(part) >= 4 and part.lower() not in COMMON_FIRST_NAMES and part not in part_map:
                        part_map[part] = people[n]
            if part_map:
                part_re = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in sorted(part_map, key=len, reverse=True)) + r")\b")
                out, k = part_re.subn(lambda m: part_map[m.group(0)], out); stats["people"] += k
    for v in variants:
        out, k = company_regex(v, core).subn("[COMPANY]", out); stats["company"] += k
    if tick:
        out, k = re.subn(rf"\b{re.escape(tick)}\b", "[TICKER]", out); stats["ticker"] += k
    if MASK_YEARS:
        out, k = re.subn(r"\b(20[0-3]\d)\b", "[YEAR]", out); stats["years"] += k
    # Residual check. "leaks" counts occurrences the rules intend to replace but did not (should be near zero).
    leaks = len(company_regex(core, core).findall(out)) if len(core) >= 3 else 0
    long_names = [n for n in people if len(n) >= 4]
    if long_names:
        leaks += len(re.compile("|".join(re.escape(n) for n in long_names), re.I).findall(out))
    stats["leaks"] = leaks
    # Informational: lower-case occurrences of a short one-word name that were deliberately left alone (usually ordinary words)
    stats["ordinary_word_hits"] = 0 if (len(core) < 3 or is_loose(core)) else len(re.findall(rf"(?<!\w){re.escape(core.lower())}(?!\w)", out))
    return out, stats


if __name__ == "__main__":
    demo = ("Operator (Operator): Welcome to the Example Corp. call.\n\n"
            "Maria Lopez (Executive): Thanks. At Example, revenue grew 5% and EXMP raised guidance for 2026.\n\n"
            "John Smith (Analyst): Maria, how should we think about margins at Example Corp.?")
    out, st = anonymize(demo, company="Example Corp.", ticker="EXMP")
    print(out)
    print(st)
