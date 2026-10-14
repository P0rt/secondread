"""Rule-based meaning checks for MT segments: numbers/units and negation.

Both checks look for cheap, strong evidence (a number or a negation that is in
the source but not in the MT, or the other way round) and leave anything subtle
to the model. They are tuned for low false-positive rates: when a pattern is
ambiguous, the checks lean towards "matched" and stay silent.

Every finding is an ``Issue`` with:
  code        a stable machine code, safe for telemetry ("number_changed");
  confidence  a hand-set prior P(major error | issue). The app no longer combines it with the
              model: the aggregator uses weights measured on the train splits
              (secondread/rule_calibration.json, secondread/rulecal.py) and falls back to
              this prior only for a code that was never measured, with no model score;
  reason      human-readable text for the translator. It may quote numbers or
              words from the segment, so it is returned to the local client only
              and must never be logged or attached to telemetry;
  spans       (side, start, end) highlight ranges, side "src" or "mt", in Python
              str indices (code points), end exclusive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

__all__ = ["Issue", "CheckResult", "number_check", "negation_check", "split_lp"]


@dataclass(frozen=True)
class Issue:
    code: str
    confidence: float
    reason: str
    spans: tuple[tuple[str, int, int], ...] = ()


@dataclass
class CheckResult:
    name: str
    issues: list[Issue] = field(default_factory=list)
    skipped: str | None = None  # machine code explaining why the check did not run
    stats: dict = field(default_factory=dict)  # per-check details for the client (may quote text)

    @property
    def risk(self) -> float:
        """Strongest single finding; several weak hits do not add up to a strong one."""
        return max((i.confidence for i in self.issues), default=0.0)

    @property
    def codes(self) -> list[str]:
        return sorted({i.code for i in self.issues})

    def add(self, code: str, confidence: float, reason: str, spans: tuple = ()) -> None:
        if all(i.reason != reason for i in self.issues):  # the same "5" twice is one finding
            self.issues.append(Issue(code, confidence, reason, tuple(spans)))


def split_lp(lp: str) -> tuple[str, str]:
    src, _, tgt = lp.lower().partition("-")
    return src, tgt


# Hand-set prior confidences that a finding is a real major error ("worth a second look"),
# not measured precision. On the train splits a finding is Major in 10-50% of hits depending
# on code and pair (scripts/calibrate_rules.py, RESULTS.md "Rule calibration"); the measured
# values in secondread/rule_calibration.json replace these in the app's risk.
CONF_NUMBER_CHANGED = 0.75
CONF_NUMBER_MISSING = 0.55
CONF_NUMBER_EXTRA = 0.25
CONF_UNIT_CHANGED = 0.70
CONF_NEGATION_DROPPED = 0.75
CONF_NEGATION_FEWER = 0.40
CONF_NEGATION_ADDED = 0.35   # fires on lexical negatives ("offline" -> "sin conexión"): keep weak


def _norm(text: str) -> str:
    return text.lower().replace("’", "'").replace("ё", "е")


# ---------------------------------------------------------------- numbers ----

_SPACES = "    "
# A digit run with ".", "," or a space between digit groups. A space only joins
# groups when exactly three digits follow ("1 000"), so "pages 5 10" stays two numbers.
_NUM_RE = re.compile(rf"\d+(?:(?:[.,]|[{_SPACES}](?=\d{{3}}(?!\d)))\d+)*")
_SEP_RE = re.compile(rf"[.,{_SPACES}]")
# 12-hour clock right after the number or after its minutes: "3 p.m.", "10:26 PM"
_PM_RE = re.compile(r"(?::\d{2})?\s?p\.?\s?m\b", re.I)
_AM_RE = re.compile(r"(?::\d{2})?\s?a\.?\s?m\b", re.I)
_YEAR_RANGE_HEAD = re.compile(r"\d{4}\s?[-–/]\s?$")   # "1981-87"
_ROMAN_RE = re.compile(r"\b[IVXLC]{2,}\b")            # "Louis XIV" (2+ letters: "I" is a pronoun)
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _interpret(tok: str) -> frozenset[Decimal]:
    """All plausible numeric values of a token across en/ru/es conventions.

    "1,000" -> {1000, 1.0}, "1.000,5" -> {1000.5}, "1 000" -> {1000},
    "0,5" -> {0.5}. Tokens that fit no convention ("3.5.1", "192.168.0.1") yield
    their digit groups. Times like "10.30" also yield 10 and 30 to match "10:30".
    """
    parts = _SEP_RE.split(tok)
    seps = ["." if s == "." else "," if s == "," else " " for s in _SEP_RE.findall(tok)]
    if not seps:
        return frozenset({Decimal(tok)})
    vals: set[Decimal] = set()
    last = seps[-1]
    # (a) the last separator is the decimal mark, the earlier ones group thousands
    if last != " " and all(len(p) == 3 for p in parts[1:-1]) and all(s != last for s in seps[:-1]):
        vals.add(Decimal("".join(parts[:-1]) + "." + parts[-1]))
    # (b) every separator groups thousands
    if all(len(p) == 3 for p in parts[1:]) and len(set(seps)) == 1:
        vals.add(Decimal("".join(parts)))
    # (c) time-like "10.30"
    if len(parts) == 2 and len(parts[1]) == 2 and int(parts[0]) <= 24 and int(parts[1]) < 60:
        vals.update(Decimal(p) for p in parts)
    if not vals:
        vals.update(Decimal(p) for p in parts)
    return frozenset(vals)


# Cardinal number words, so "2" <-> "два"/"dos"/"two" counts as a match. Only
# used to *suppress* findings, so over-matching (Spanish "un" = 1) is harmless.
_WORDS: dict[str, dict[int, str]] = {
    "en": {0: "zero", 1: "one", 2: "two|twice", 3: "three", 4: "four", 5: "five", 6: "six",
           7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve|dozen",
           20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 100: "hundred", 1000: "thousand"},
    "ru": {0: r"ноль|нуля|нулю|нулем|нол[ьяюе]",
           1: r"один|одна|одно|одни|одного|одной|одному|одним|одном|одну|одних|одними",
           # "\w*" stems also catch compounds ("четырехминутный") and collectives ("семерка")
           2: r"два|две|двух\w*|двум|двумя|дважды|двое|двоих|двойк\w*",
           3: r"три|трех\w*|трем|тремя|трижды|трое|троих|тройк\w*",
           4: r"четыре|четырех\w*|четырем|четырьмя|четверо|четверк\w*", 5: r"пять|пяти\w*|пятью|пятерк\w*",
           6: r"шесть|шести\w*|шестью|шестерк\w*", 7: r"семь|семи\w*|семью|семерк\w*",
           8: r"восемь|восьми\w*|восемью|восьмью|восьмерк\w*", 9: r"девять|девяти\w*|девятью|девятк\w*",
           10: r"десять|десяти\w*|десятью|десятк\w*", 11: r"одиннадцат\w*",
           12: r"двенадцат\w*|дюжин\w*", 20: r"двадцат\w*", 30: r"тридцат\w*", 40: r"сорок\w*",
           50: r"пятьдесят|пятидесяти", 100: r"сто|ста|сотн\w*", 1000: r"тысяч\w*"},
    "es": {0: "cero", 1: "uno|una|un", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis",
           7: "siete", 8: "ocho", 9: "nueve", 10: "diez", 11: "once", 12: "doce|docena",
           20: "veinte", 30: "treinta", 40: "cuarenta", 50: "cincuenta", 100: "cien|ciento|cientos",
           1000: "mil"},
}
_WORD_RES = {lang: [(re.compile(rf"\b(?:{pat})\b"), Decimal(v)) for v, pat in table.items()]
             for lang, table in _WORDS.items()}

# Units that may follow a number, by canonical id and language. Bare one-letter
# English "m" (metres or millions?) and "s" are left out on purpose.
_UNITS: dict[str, dict[str, str]] = {
    "percent": {"en": r"%|per ?cent", "ru": r"%|процент\w*", "es": r"%|por ?ciento"},
    "km": {"en": r"km|kilomet(?:er|re)s?", "ru": r"км|километр\w*", "es": r"km|kil[oó]metros?"},
    "m": {"en": r"met(?:er|re)s?", "ru": r"м|метр\w*", "es": r"m|metros?"},
    "cm": {"en": r"cm|centimet(?:er|re)s?", "ru": r"см|сантиметр\w*", "es": r"cm|cent[ií]metros?"},
    "mm": {"en": r"mm|millimet(?:er|re)s?", "ru": r"мм|миллиметр\w*", "es": r"mm|mil[ií]metros?"},
    "mile": {"en": r"mi|miles?", "ru": r"мил[ьяие]\w*", "es": r"millas?"},
    "inch": {"en": r"inch(?:es)?", "ru": r"дюйм\w*", "es": r"pulgadas?"},
    "foot": {"en": r"ft|feet|foot", "ru": r"фут\w*", "es": r"pies?"},
    "kg": {"en": r"kg|kilo(?:gram(?:me)?)?s?", "ru": r"кг|килограмм\w*", "es": r"kg|kilo(?:gramo)?s?"},
    "gram": {"en": r"g|grams?|grammes?", "ru": r"г|грамм\w*", "es": r"g|gramos?"},
    "lb": {"en": r"lbs?", "ru": r"фунт\w*", "es": r"libras?"},
    "liter": {"en": r"lit(?:er|re)s?", "ru": r"л|литр\w*", "es": r"litros?"},
    "celsius": {"en": r"°\s?c|celsius", "ru": r"°\s?c|°\s?с|цельси\w*", "es": r"°\s?c|celsius"},
    "fahrenheit": {"en": r"°\s?f|fahrenheit", "ru": r"°\s?f|фаренгейт\w*", "es": r"°\s?f|fahrenheit"},
    "second": {"en": r"sec|seconds?", "ru": r"сек\w*", "es": r"seg|segundos?"},
    "minute": {"en": r"min|minutes?|mins", "ru": r"мин\w*", "es": r"min|minutos?"},
    "hour": {"en": r"h|hrs?|hours?", "ru": r"ч|час\w*", "es": r"h|horas?"},
    "day": {"en": r"days?", "ru": r"д(?:ень|ня|ней|ням|нями|нях)", "es": r"d[ií]as?"},
    "week": {"en": r"weeks?", "ru": r"недел\w*", "es": r"semanas?"},
    "month": {"en": r"months?", "ru": r"месяц\w*", "es": r"mes(?:es)?"},
    "year": {"en": r"years?|yrs?", "ru": r"год\w*|лет", "es": r"años?"},
    "kb": {"en": r"kb|kilobytes?", "ru": r"кб|kb|килобайт\w*", "es": r"kb|kilobytes?"},
    "mb": {"en": r"mb|megabytes?", "ru": r"мб|mb|мегабайт\w*", "es": r"mb|megabytes?"},
    "gb": {"en": r"gb|gigabytes?", "ru": r"гб|gb|гигабайт\w*", "es": r"gb|gigabytes?"},
    "tb": {"en": r"tb|terabytes?", "ru": r"тб|tb|терабайт\w*", "es": r"tb|terabytes?"},
    "usd": {"en": r"usd|dollars?", "ru": r"usd|долл\w*", "es": r"usd|d[oó]lar(?:es)?"},
    "eur": {"en": r"eur|euros?", "ru": r"eur|евро", "es": r"eur|euros?"},
    "gbp": {"en": r"gbp", "ru": r"gbp", "es": r"gbp"},
    "rub": {"en": r"rub|r[ou]+bles?", "ru": r"руб\w*", "es": r"rublos?"},
}
_UNIT_AFTER = {
    lang: re.compile(r"^[   ]?(?:" + "|".join(
        f"(?P<{uid}>{pats[lang]})" for uid, pats in _UNITS.items() if lang in pats) + r")(?!\w)", re.I)
    for lang in ("en", "ru", "es")
}
_CURRENCY_BEFORE = re.compile(r"(?:(?P<usd>\$)|(?P<eur>€)|(?P<gbp>£)|(?P<rub>₽))[  ]?$")
_CURRENCY_AFTER = re.compile(r"^[  ]?(?:(?P<usd>\$)|(?P<eur>€)|(?P<gbp>£)|(?P<rub>₽))")


@dataclass(frozen=True)
class _Num:
    text: str                    # surface form, e.g. "1,000.5"
    start: int
    end: int
    values: frozenset[Decimal]
    unit: str | None             # canonical unit id or None
    unit_text: str               # surface form of the unit, for reasons only
    unit_span: tuple[int, int]   # number plus unit, for highlighting
    ident: bool = False          # part of an identifier ("@user16", "G7", "mp3"): never reported


def _extract_numbers(text: str, lang: str) -> list[_Num]:
    out = []
    after_re = _UNIT_AFTER.get(lang)
    for m in _NUM_RE.finditer(text):
        tok, vals = m.group(), set(_interpret(m.group()))
        tail, head = text[m.end():m.end() + 24], text[max(0, m.start() - 6):m.start()]
        unit, unit_text, unit_span = None, "", (m.start(), m.end())
        for rx, s, before in ((_CURRENCY_BEFORE, head, True), (_CURRENCY_AFTER, tail, False),
                              (after_re, tail, False)):
            if rx is not None and (u := rx.search(s)):
                unit, unit_text = u.lastgroup, u.group().strip()
                unit_span = ((m.start() - len(u.group()), m.end()) if before
                             else (m.start(), m.end() + len(u.group())))
                break
        # Equivalent renderings that differ in digits:
        if _PM_RE.match(tail):                      # "3 p.m." == "15:00"
            vals |= {v + 12 for v in vals if v < 12}
        elif _AM_RE.match(tail):                    # "12 a.m." == "0:00"
            vals |= {Decimal(0) for v in vals if v == 12}
        if len(tok) == 2 and _YEAR_RANGE_HEAD.search(head):   # "1981-87" == "1981 a 1987"
            vals |= {Decimal(1900 + int(tok)), Decimal(2000 + int(tok))}
        if len(tok) == 4 and tok.isdigit():
            if tail[:1] == "s" or tail[:2] in ("'s", "’s"):  # "the 1970s" == "los años 70"
                vals.add(Decimal(int(tok) % 100))
            if tok[0] == "0":                                  # military time "0500" == "05:00"
                vals |= {Decimal(tok[:2]), Decimal(tok[2:])}
        prev = text[m.start() - 1] if m.start() else ""
        out.append(_Num(tok, m.start(), m.end(), frozenset(vals), unit, unit_text, unit_span,
                        ident=prev.isalpha() or prev in "@#_"))
    return out


def _roman_value(s: str) -> int:
    vals = [_ROMAN[c] for c in s]
    return sum(-v if i + 1 < len(vals) and v < vals[i + 1] else v for i, v in enumerate(vals))


def _word_values(text: str, lang: str) -> set[Decimal]:
    """Numbers written as words or Roman numerals ("two", "двух", "XIV")."""
    low = _norm(text)
    vals = {v for rx, v in _WORD_RES.get(lang, []) if rx.search(low)}
    vals |= {Decimal(_roman_value(m.group())) for m in _ROMAN_RE.finditer(text)}
    return vals


def _is_trivial(n: _Num) -> bool:
    # "1 hour" -> "час", "an hour" -> "1 час": a lone 1 moves in and out of
    # translations legitimately all the time, and so do "00" minutes ("3 p.m." -> "15:00").
    return n.values <= {Decimal(0), Decimal(1)}


def number_check(src: str, mt: str, lp: str) -> CheckResult:
    """Numbers (and their units) in the source that are missing or changed in the MT."""
    res = CheckResult("number_check", stats={"src_only": [], "mt_only": []})
    s_lang, t_lang = split_lp(lp)
    src_nums, mt_nums = _extract_numbers(src, s_lang), _extract_numbers(mt, t_lang)
    if not src_nums and not mt_nums:
        return res
    src_pool = set().union(*(n.values for n in src_nums)) | _word_values(src, s_lang)
    mt_pool = set().union(*(n.values for n in mt_nums)) | _word_values(mt, t_lang)

    lost = [n for n in src_nums if not n.values & mt_pool and not n.ident and not _is_trivial(n)]
    new = [n for n in mt_nums if not n.values & src_pool and not n.ident and not _is_trivial(n)]
    res.stats = {"src_only": [n.text for n in lost], "mt_only": [n.text for n in new]}
    for s, t in zip(lost, new, strict=False):  # one number went missing, another appeared: changed
        res.add("number_changed", CONF_NUMBER_CHANGED,
                f"Number {s.text} in the source does not match {t.text} in the MT",
                [("src", s.start, s.end), ("mt", t.start, t.end)])
    for s in lost[len(new):]:
        res.add("number_missing", CONF_NUMBER_MISSING, f"Number {s.text} from the source is missing in the MT",
                [("src", s.start, s.end)])
    for t in new[len(lost):]:
        res.add("number_extra", CONF_NUMBER_EXTRA, f"The MT has the number {t.text}, which is not in the source",
                [("mt", t.start, t.end)])

    # Units: the number survived, but its unit did not ("5 days" -> "5 недель").
    for s in src_nums:
        if s.unit is None or s in lost:
            continue
        twins = [t for t in mt_nums if t.values & s.values]
        if twins and all(t.unit is not None and t.unit != s.unit for t in twins):
            # Quote the MT twin whose unit no other source number explains ("$5 or 5 days").
            src_units = {x.unit for x in src_nums if x.values & s.values}
            t = next((t for t in twins if t.unit not in src_units), twins[0])
            res.add("unit_changed", CONF_UNIT_CHANGED,
                    f"Unit changed: the source has {s.text} {s.unit_text}, the MT has {t.text} {t.unit_text}",
                    [("src", *s.unit_span), ("mt", *t.unit_span)])
    return res


# --------------------------------------------------------------- negation ----

# "hard" markers are explicit negation words. "soft" markers carry negative
# meaning without one ("unable", "free of", "until" -> "пока не"); they only
# suppress findings: a soft source marker explains an MT negation, and a soft
# MT marker can express a source negation lexically ("not available" -> "недоступен").
_NEG: dict[str, dict[str, str]] = {
    "en": {
        "hard": r"\bnot\b|n't\b|\bno\b(?!\.?\s?\d)|\bnever\b|\bnone\b|\bnobody\b|\bno[- ]one\b|\bnothing\b"
                r"|\bnowhere\b|\bneither\b|\bnor\b|\bcannot\b|\bwithout\b",
        "soft": r"\bunable\b|\bimpossible\b|\bunless\b|\buntil\b|\btill\b|\bfail(?:s|ed|ing|ure)?\b|\black\w*"
                r"|\bprevent\w*|\bavoid\w*|\brefus\w*|\bden(?:y|ies|ied)\b|\bstop\w*|\bexcept\b|\binstead\b"
                r"|\brather than\b|\bfree\b|\w+less\b|\bhardly\b|\bbarely\b|\bscarcely\b|\bseldom\b|\brarely\b"
                r"|\bmissing\b|\babsen\w*|\bforbid\w*|\bprohibit\w*|\bban(?:s|ned)?\b|\bignor\w*"
                r"|\bwrong\b|\bissues?\b|\bproblems?\b|\btrouble\b|\berrors?\b|\bbroken\b|\boff(?:line)?\b"
                r"|\bun(?!der|til|it|ion|ique|iver|iform|ix)\w{3,}|\bnon\w{3,}"
                r"|\bdis(?:abl|allow|agre|lik|connect|continu|approv|appear|card|miss)\w*"
                r"|\bin(?:activ|correct|valid|complet|accessib|appropriat|capab|consisten|direct|effectiv"
                r"|efficien|sufficien|visib|abilit|accura|adequa|compatib|convenien|eligib|exact)\w*"
                r"|\bim(?:possib|proper|patien|mobil|matur|balanc|precis)\w*"
                r"|\bir(?:regular|relevant|revers|respons)\w*|\bil(?:legal|logic|legib)\w*",
    },
    "ru": {
        "hard": r"\bне\b|\bни\b|\bнет\b|\bнету\b|\bнельзя\b|\bбезо?\b|\bнекого\b|\bнечего\b|\bнегде\b|\bнекуда\b"
                r"|\bни(?:кто|кого|кому|кем|ком|что|чего|чему|чем|как|когда|где|куда|откуда|сколько"
                r"|какой|какая|какое|какие|какого|каких|каким|какую|какими|чей|чья|чье|чьи)\b",
        "soft": r"\bотсутств\w*|\bневозможн\w*|\bзапрещ\w*|\bпрекрат\w*|\bизбега\w*|\bкроме\b|\bвместо\b"
                r"|\bлишь\b|\bтолько\b|\bедва\b|\bредко\b|\bбе[зс]\w{3,}"
                r"|\bне(?:верн|доступ|правильн|корректн|известн|допустим|достаточн|удач|исправн|нужн"
                r"|активн|рабоч|полн|действительн|способн|ясн|возможн|обязательн|зависим|желательн)\w*",
    },
    "es": {
        "hard": r"\bno\b(?!\.?\s?\d)|\bnunca\b|\bjam[aá]s\b|\bnada\b|\bnadie\b|\bningun[oa]s?\b|\bning[uú]n\b"
                r"|\bni\b|\bsin\b|\btampoco\b",
        "soft": r"\bimposib\w*|\bincorrect\w*|\binv[aá]lid\w*|\bindisponib\w*|\bdesconoc\w*|\bincomplet\w*"
                r"|\binsuficient\w*|\binaccesib\w*|\binactiv\w*|\bdesactiv\w*|\bdeshabilit\w*|\bfalta\w*"
                r"|\bausen\w*|\bevit\w*|\bprohib\w*|\bimpid\w*|\bsalvo\b|\bexcepto\b|\bmenos\b|\bapenas\b"
                r"|\ben (?:lugar|vez) de\b|\bgratis\b|\bgratuit\w*|\bs[oó]lo\b|\bsolamente\b",
    },
}
_NEG_RES = {lang: {kind: re.compile(p) for kind, p in kinds.items()} for lang, kinds in _NEG.items()}


# Tag questions look negative but are not: "..., не так ли?" / "..., ¿no?" = "..., right?"
_TAG_QUESTIONS = re.compile(r"\bне так ли\b|¿\s*no\s*\?|,\s*no\s*\?")


def _markers(text: str, lang: str, kind: str) -> list[re.Match]:
    low = _TAG_QUESTIONS.sub(lambda m: " " * len(m.group()), _norm(text))  # keep offsets
    return list(_NEG_RES[lang][kind].finditer(low))


def negation_check(src: str, mt: str, lp: str) -> CheckResult:
    """Negation present in the source but absent in the MT (or added by the MT).

    Russian and Spanish use negative concord ("никто не", "no ... nadie"), so the
    MT usually has *more* markers than English; only a deficit or an MT
    negation with no source counterpart is suspicious.
    """
    res = CheckResult("negation_check")
    s_lang, t_lang = split_lp(lp)
    if s_lang not in _NEG or t_lang not in ("ru", "es"):
        res.skipped = "unsupported_lp"
        return res
    s_hard, s_soft = _markers(src, s_lang, "hard"), _markers(src, s_lang, "soft")
    t_hard, t_soft = _markers(mt, t_lang, "hard"), _markers(mt, t_lang, "soft")
    res.stats = {"src_count": len(s_hard), "mt_count": len(t_hard)}

    def quote(ms: list[re.Match]) -> str:
        return ", ".join(f"'{w}'" for w in dict.fromkeys(m.group() for m in ms))  # unique, in order

    def spans(side: str, text: str, ms: list[re.Match]) -> list[tuple[str, int, int]]:
        # Offsets come from the lower-cased text; skip them if lower-casing changed its length.
        return [(side, m.start(), m.end()) for m in ms] if len(_norm(text)) == len(text) else []

    if s_hard and not t_hard and not t_soft:
        res.add("negation_dropped", CONF_NEGATION_DROPPED,
                f"The source has a negation ({quote(s_hard)}) but the MT has none", spans("src", src, s_hard))
    elif len(s_hard) >= 2 and len(t_hard) + len(t_soft) < len(s_hard):
        res.add("negation_fewer", CONF_NEGATION_FEWER,
                f"The source has {len(s_hard)} negations ({quote(s_hard)}), the MT only "
                f"{len(t_hard) + len(t_soft)}", spans("src", src, s_hard) + spans("mt", mt, t_hard + t_soft))
    if t_hard and not s_hard and not s_soft:
        res.add("negation_added", CONF_NEGATION_ADDED,
                f"The MT adds a negation ({quote(t_hard)}) that is not in the source", spans("mt", mt, t_hard))
    return res
