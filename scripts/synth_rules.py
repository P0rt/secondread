"""Rule-based synthetic errors in numbers, units and dates, plus meaning-preserving
number variants. No LLM, seeded and deterministic.

usage: uv run python scripts/synth_rules.py [--lps en-ru en-es] [--splits train test]
           [--seed 13] [--max-major-per-seg 4] [--max-ok-per-seg 2] [--show 2]

Parents are clean units (label OK, no major/critical span) of the SAME split the output
is written for (see secondread/synth.py for the hygiene checks). Only numbers that are
anchored in the source are touched: a number in the MT is used only when exactly one
reading of it (thousands/decimal separators) equals a number in the English source, so
"6,9 млрд" for "$6.9 billion" is understood as 6.9 and "3 126" for "3,126" as 3126.

Major types (label Major, one synthetic span in `spans`):
  num_digit      one digit of a number changed           3 126 -> 3 526
  num_magnitude  x10 or /10                              300 -> 3000
  num_percent    a percentage changed                    15% -> 25%
  num_swap       two numbers exchange places             3 из 5 -> 5 из 3
  num_unit       unit or time period swapped             5 км -> 5 м, 2 года -> 2 месяца
  num_scale      scale word swapped                      6,9 млрд -> 6,9 млн
  num_currency   currency swapped                        $25 -> €25, 5 dólares -> 5 euros
  date_year      year shifted                            2021 -> 2023
  date_day       day of month changed                    13 января -> 17 января
  date_month     month changed                           13 de enero -> 13 de marzo
OK types (label OK, the edit keeps the meaning, so "any edit = error" is not learnable):
  ok_grouping    thousands separator per target locale   10,000 -> 10 000 (nbsp / narrow nbsp / space)
  ok_decimal     ru only: decimal point -> comma         2.5 -> 2,5
  ok_nbsp        space <-> no-break space after a number 5 км -> 5<NBSP>км (max 1 per segment)
  ok_percent     space before % added/removed            50% -> 50<NBSP>%
  ok_percent_word  % <-> word                            50% -> 50 процентов / 50 por ciento
  ok_currency    $ <-> dollar word                       $6,9 млрд -> 6,9 млрд долларов
  ok_abbrev      unit/scale word -> abbreviation         5 километров -> 5 км, 3,3 миллиарда -> 3,3 млрд

Output: data/synth/rules.<split>.jsonl (all requested language pairs in one file), same
schema as data/splits/*.jsonl plus synthetic=true, synth_type, synth_generator,
synth_parent and synth_edit.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from secondread.synth import (SYNTH_DIR, assert_hygiene, digit_string, error_span, make_record,  # noqa: E402
                              natural_mts, rng_for, segment_key, shift_spans, source_pool, target_lang,
                              write_jsonl)

GENERATOR = "rules-v1"
NBSP, NNBSP = "\u00a0", "\u202f"
WS = " " + NBSP + NNBSP

# --------------------------------------------------------------------------- numbers

# A run of digits with single separators between digit groups; never starts inside a word
# (so "B2B", "mp3", "site2" are ignored). Separators are resolved by `_readings`.
NUM_RE = re.compile(r"(?<![^\W_])\d+(?:[.,\u00a0\u202f ]\d+)*")
SEP_RE = re.compile(r"([.,\u00a0\u202f ])")

EN_GROUP, EN_DEC = {",", " ", NBSP, NNBSP}, {"."}
TGT_GROUP, TGT_DEC = {",", ".", " ", NBSP, NNBSP}, {",", "."}


@dataclass(frozen=True)
class Num:
    start: int
    end: int
    raw: str
    value: Decimal
    group: str | None  # thousands separator in use, if any
    dec: str | None    # decimal separator in use, if any
    frac: int          # digits after the decimal separator

    @property
    def int_digits(self) -> int:
        return len(re.sub(r"\D", "", self.raw)) - self.frac


def _valid_groups(groups: list[str]) -> bool:
    return 1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])


def _readings(raw: str, start: int, group_chars: set[str], dec_chars: set[str]) -> list[Num]:
    """Every consistent reading of a digit token: all separators grouping, or last one decimal."""
    parts = SEP_RE.split(raw)
    groups, seps = parts[0::2], parts[1::2]
    end = start + len(raw)
    if not seps:
        return [Num(start, end, raw, Decimal(raw), None, None, 0)]
    out = []
    if len(set(seps)) == 1 and seps[0] in group_chars and _valid_groups(groups):
        out.append(Num(start, end, raw, Decimal("".join(groups)), seps[0], None, 0))
    if seps[-1] in dec_chars:
        g, s = groups[:-1], seps[:-1]
        if not s or (len(set(s)) == 1 and s[0] in group_chars and s[0] != seps[-1] and _valid_groups(g)):
            value = Decimal("".join(g) + "." + groups[-1])
            out.append(Num(start, end, raw, value, s[0] if s else None, seps[-1], len(groups[-1])))
    return out


def _tokens(text: str):
    for m in NUM_RE.finditer(text):
        yield m.start(), m.group()


def source_numbers(src: str) -> dict[Decimal, list[Num]]:
    """English source numbers by value. Unparseable tokens contribute each digit run."""
    out: dict[Decimal, list[Num]] = collections.defaultdict(list)
    for start, raw in _tokens(src):
        readings = _readings(raw, start, EN_GROUP, EN_DEC)
        if not readings:
            pos = start
            for piece in SEP_RE.split(raw):
                if piece.isdigit():
                    readings.append(Num(pos, pos + len(piece), piece, Decimal(piece), None, None, 0))
                pos += len(piece)
        for n in readings:
            out[n.value].append(n)
    return out


def anchored_numbers(mt: str, src_vals) -> list[Num]:
    """MT numbers with exactly one reading equal to a source number (others are left alone)."""
    out = []

    def visit(start: int, raw: str, allow_split: bool) -> None:
        hits = [n for n in _readings(raw, start, TGT_GROUP, TGT_DEC) if n.value in src_vals]
        if len(hits) == 1:
            out.append(hits[0])
        elif not hits and allow_split and any(c in raw for c in WS):
            # "в 2022 100 человек": not one grouped number, try the space-separated pieces
            pos = start
            for piece in re.split(r"([ \u00a0\u202f])", raw):
                if piece and piece not in WS:
                    visit(pos, piece, False)
                pos += len(piece)

    for start, raw in _tokens(mt):
        visit(start, raw, True)
    return out


def render(value: Decimal, group: str | None, dec: str | None, frac: int) -> str:
    """Format a value with the given separators (grouping applied from 4 integer digits)."""
    q = value.quantize(Decimal(1).scaleb(-frac)) if frac else value.to_integral_value()
    sign, text = ("-", str(-q)) if q < 0 else ("", str(q))
    int_part, _, frac_part = text.partition(".")
    if group and len(int_part) >= 4:
        head = len(int_part) % 3 or 3
        int_part = group.join([int_part[:head]] + [int_part[i:i + 3] for i in range(head, len(int_part), 3)])
    return sign + int_part + ((dec or ".") + frac_part if frac else "")


def is_year(n: Num) -> bool:
    return n.frac == 0 and n.group is None and len(n.raw) == 4 and 1500 <= n.value <= 2100


# --------------------------------------------------------------------------- lexicon

MONTHS = {
    "en": ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"],
    "ru_gen": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
               "октября", "ноября", "декабря"],
    "ru_prep": ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре",
                "октябре", "ноябре", "декабре"],
    "ru_nom": ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
               "октябрь", "ноябрь", "декабрь"],
    "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
           "octubre", "noviembre", "diciembre"],
}
EN_MONTH_RE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|"
                         r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\b")
EN_MONTH_ABBR = {m[:3]: i for i, m in enumerate(MONTHS["en"])} | {"sept": 8}


def source_months(src: str) -> set[int]:
    out = set()
    for m in EN_MONTH_RE.finditer(src):
        w = m.group(1).lower()
        out.add(MONTHS["en"].index(w) if w in MONTHS["en"] else EN_MONTH_ABBR[w])
    return out


def month_lists(lang: str) -> list[list[str]]:
    return [MONTHS["ru_gen"], MONTHS["ru_prep"], MONTHS["ru_nom"]] if lang == "ru" else [MONTHS["es"]]


DAY_MONTH_RE = {
    "ru": re.compile(r"[ \u00a0]+(" + "|".join(MONTHS["ru_gen"]) + r")(?!\w)", re.IGNORECASE),
    "es": re.compile(r"[ \u00a0]+de[ \u00a0]+(" + "|".join(MONTHS["es"]) + r")(?!\w)", re.IGNORECASE),
}
PERCENT_RE = {
    "ru": re.compile(r"[ \u00a0\u202f]?%|[ \u00a0]+процент\w*", re.IGNORECASE),
    "es": re.compile(r"[ \u00a0\u202f]?%|[ \u00a0]+por[ \u00a0]?ciento", re.IGNORECASE),
}

RU_MASC = "|ам|ами|ах|а|у|ом|е|ы|ов"          # метр, метра, метров, ... (also миллион, доллар, фунт)
RU_FEM = "|ам|ами|ах|а|ы|е|у|ой"               # минута, минуты, минут, ...


def plural_ru(n: Decimal, one: str, few: str, many: str) -> str:
    if n != n.to_integral_value():
        return few  # 2,5 доллара
    k = int(n)
    if k % 10 == 1 and k % 100 != 11:
        return one
    return few if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14 else many


@dataclass(frozen=True)
class Swap:
    """Word after a number -> replacement. `stem` swaps keep the inflectional ending."""
    synth_type: str
    pattern: str
    repl: str | None = None       # fixed replacement (case of the first letter is kept)
    stem_to: str | None = None    # replace the matched stem, keep group "end"
    plural: tuple | None = None   # (one, few, many): Russian numeral agreement


def _stem(t, a, b, endings, **kw):
    return Swap(t, rf"(?P<stem>{a})(?P<end>{endings})", stem_to=b, **kw)


SWAPS = {
    "ru": [
        # scale words first so "6,9 млрд долларов" can also be read as scale + currency
        Swap("num_scale", r"млрд", "млн"), Swap("num_scale", r"млн", "млрд"), Swap("num_scale", r"тыс", "млн"),
        _stem("num_scale", "миллиард", "миллион", RU_MASC), _stem("num_scale", "миллион", "миллиард", RU_MASC),
        _stem("num_unit", "километр", "метр", RU_MASC), _stem("num_unit", "метр", "километр", RU_MASC),
        _stem("num_unit", "сантиметр", "миллиметр", RU_MASC), _stem("num_unit", "миллиметр", "сантиметр", RU_MASC),
        _stem("num_unit", "килограмм", "грамм", RU_MASC), _stem("num_unit", "грамм", "килограмм", RU_MASC),
        _stem("num_unit", "минут", "секунд", RU_FEM), _stem("num_unit", "секунд", "минут", RU_FEM),
        Swap("num_unit", r"км", "м"), Swap("num_unit", r"м", "км"), Swap("num_unit", r"см", "мм"),
        Swap("num_unit", r"мм", "см"), Swap("num_unit", r"кг", "т"), Swap("num_unit", r"мл", "л"),
        Swap("num_unit", r"л", "мл"), Swap("num_unit", r"ГБ", "МБ"), Swap("num_unit", r"МБ", "ГБ"),
        Swap("num_unit", r"ТБ", "ГБ"), Swap("num_unit", r"мин", "ч"), Swap("num_unit", r"ч", "мин"),
        Swap("num_unit", r"лет", "месяцев"),
        Swap("num_unit", r"года", "месяца"),
        Swap("num_unit", r"год", "месяц"), Swap("num_unit", r"месяцев", "лет"),
        Swap("num_unit", r"месяца", "года"), Swap("num_unit", r"месяц", "год"),
        Swap("num_unit", r"часов", "дней"), Swap("num_unit", r"часа", "дня"), Swap("num_unit", r"час", "день"),
        Swap("num_unit", r"дней", "часов"), Swap("num_unit", r"дня", "часа"), Swap("num_unit", r"день", "час"),
        Swap("num_unit", r"недель", "месяцев"), Swap("num_unit", r"недели", "месяца"),
        _stem("num_currency", "доллар", "фунт", RU_MASC), _stem("num_currency", "фунт", "доллар", RU_MASC),
        Swap("num_currency", r"евро", plural=("доллар", "доллара", "долларов")),
    ],
    "es": [
        Swap("num_scale", r"miles de millones", "millones"), Swap("num_scale", r"mil millones", "millones"),
        Swap("num_scale", r"millones", "mil millones"), Swap("num_scale", r"billones", "millones"),
        Swap("num_unit", r"kilómetros", "metros"), Swap("num_unit", r"kilómetro", "metro"),
        Swap("num_unit", r"metros", "kilómetros"), Swap("num_unit", r"metro", "kilómetro"),
        Swap("num_unit", r"centímetros", "milímetros"), Swap("num_unit", r"milímetros", "centímetros"),
        Swap("num_unit", r"kilogramos", "gramos"), Swap("num_unit", r"kilos", "gramos"),
        Swap("num_unit", r"gramos", "kilogramos"), Swap("num_unit", r"minutos", "segundos"),
        Swap("num_unit", r"minuto", "segundo"), Swap("num_unit", r"segundos", "minutos"),
        Swap("num_unit", r"horas", "semanas"), Swap("num_unit", r"semanas", "horas"),
        Swap("num_unit", r"años", "meses"), Swap("num_unit", r"año", "mes"), Swap("num_unit", r"meses", "años"),
        Swap("num_unit", r"mes", "año"), Swap("num_unit", r"días", "meses"), Swap("num_unit", r"día", "mes"),
        Swap("num_unit", r"km", "m"), Swap("num_unit", r"m", "km"), Swap("num_unit", r"cm", "mm"),
        Swap("num_unit", r"mm", "cm"), Swap("num_unit", r"kg", "g"), Swap("num_unit", r"g", "kg"),
        Swap("num_unit", r"ml", "l"), Swap("num_unit", r"GB", "MB"), Swap("num_unit", r"MB", "GB"),
        Swap("num_unit", r"TB", "GB"),
        Swap("num_currency", r"dólares", "euros"), Swap("num_currency", r"dólar", "euro"),
        Swap("num_currency", r"euros", "dólares"), Swap("num_currency", r"euro", "dólar"),
        Swap("num_currency", r"pesos", "dólares"), Swap("num_currency", r"USD", "EUR"),
        Swap("num_currency", r"EUR", "USD"),
    ],
}
SCALE_RE = {  # optional scale word between a number and its currency ("6,9 млрд долларов")
    "ru": re.compile(r"(?:млрд|млн|тыс\.?|(?:миллиард|миллион)\w*)(?!\w)", re.IGNORECASE),
    "es": re.compile(r"(?:miles de millones|mil millones|millones|millón|billones)(?!\w)", re.IGNORECASE),
}
CURRENCY_SYMBOLS = {"$": "€", "€": "$", "£": "$"}
# Words match case-insensitively; short abbreviations ("м", "ГБ", "mm") only exactly, so that a
# capital initial or another abbreviation is not mistaken for a unit.
_SWAP_RE = {lang: [(sw, re.compile(sw.pattern + r"(?!\w)", re.IGNORECASE if sw.stem_to or len(sw.pattern) > 3 else 0))
                   for sw in swaps] for lang, swaps in SWAPS.items()}


# Meaning-preserving abbreviations (word -> invariant abbreviation, so no case agreement needed).
ABBREV = {
    "ru": [(r"километр\w*", "км"), (r"сантиметр\w*", "см"), (r"миллиметр\w*", "мм"), (r"метр(?:а|ов|ах|ам)?", "м"),
           (r"килограмм\w*", "кг"), (r"минут\w*", "мин"), (r"час(?:а|ов)?", "ч"),
           (r"миллиард\w*", "млрд"), (r"миллион\w*", "млн"), (r"тысяч\w*", "тыс.")],
    "es": [(r"kilómetros?", "km"), (r"centímetros?", "cm"), (r"milímetros?", "mm"), (r"metros?", "m"),
           (r"kilogramos?", "kg"), (r"kilos?", "kg"), (r"gramos?", "g"), (r"minutos?", "min"), (r"horas?", "h")],
}
_ABBREV_RE = {lang: [(re.compile(p + r"(?!\w)", re.IGNORECASE), a) for p, a in pairs] for lang, pairs in ABBREV.items()}
DOLLAR_WORD_RE = {"ru": re.compile(r"доллар\w*(?!\w)", re.IGNORECASE),
                  "es": re.compile(r"(?:de[ \u00a0]+)?dólar(?:es)?(?!\w)", re.IGNORECASE)}
PERCENT_WORD_RE = {"ru": re.compile(r"[ \u00a0]+процент\w*", re.IGNORECASE),
                   "es": re.compile(r"[ \u00a0]+por[ \u00a0]?ciento", re.IGNORECASE)}


def dollar_word(n: Num, lang: str, after_scale: bool) -> str:
    if lang == "ru":
        return "долларов" if after_scale else plural_ru(n.value, "доллар", "доллара", "долларов")
    return ("de " if after_scale else "") + ("dólar" if n.value == 1 else "dólares")


def percent_word(n: Num, lang: str) -> str:
    return plural_ru(n.value, "процент", "процента", "процентов") if lang == "ru" else "por ciento"


@dataclass(frozen=True)
class Slots:
    word: int                # where a unit/scale word right after the number would start
    scale_end: int | None    # end of the scale word, if there is one ("6,9 млрд")
    currency: int            # where a currency word would start (after the scale, and "de" in es)


def number_slots(n: Num, mt: str, lang: str) -> Slots:
    skip_ws = lambda i: i + 1 if i < len(mt) and mt[i] in WS else i  # noqa: E731
    word = skip_ws(n.end)
    scale = SCALE_RE[lang].match(mt, word)
    if not scale:
        return Slots(word, None, word)
    cur = skip_ws(scale.end())
    if lang == "es" and (de := re.match(r"de[ \u00a0]", mt[cur:cur + 3])):
        cur += de.end()
    return Slots(word, scale.end(), cur)


def match_case(orig: str, repl: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return repl.upper()
    return repl[:1].upper() + repl[1:] if orig[:1].isupper() else repl


# --------------------------------------------------------------------------- edits

@dataclass
class Edit:
    synth_type: str
    start: int
    end: int
    repl: str
    label: str
    err: list[tuple[int, int]] = field(default_factory=list)  # error spans, offsets in the NEW mt

    def apply(self, mt: str) -> str:
        return mt[:self.start] + self.repl + mt[self.end:]


def _value_edit(n: Num, new: Decimal, synth_type: str, frac: int | None = None) -> Edit:
    """Replace a number by `new`, written in the same style (separators, decimals) as the original."""
    text = render(new, n.group, n.dec, n.frac if frac is None else frac)
    return Edit(synth_type, n.start, n.end, text, "Major", [(n.start, n.start + len(text))])


def in_compound(n: Num, mt: str) -> bool:
    """Part of a time, range, score, code or a name ("5:30", "1-2", "24/7", "COVID-19", "FTSE 250",
    "5<unk>7" from a broken MT dash): only a digit edit makes sense, no magnitude/unit/swap edits."""
    before, after = mt[max(0, n.start - 12):n.start], mt[n.end:n.end + 5]
    dash = r"[ \u00a0]?[\-–—−][ \u00a0]?"
    return bool(re.search(rf"\d(?:[:/]|{dash})$|[^\W\d_][\-−]$|\b[A-Z][A-Za-z]*[ \u00a0]$|<unk>$", before)
                or re.match(rf"(?:[:/]|{dash})\d|<unk>", after))


def _changed_digit(n: Num, rng) -> Decimal | None:
    digits = re.sub(r"\D", "", n.raw)
    lead = digits.lstrip("0")
    if not lead:
        return None
    offset = len(digits) - len(lead)
    # mostly the first two significant digits: "6 152 524 -> 6 152 525" is too subtle a lesson
    pos = offset + (rng.randrange(min(2, len(lead))) if rng.random() < 0.75 else rng.randrange(len(lead)))
    choices = [d for d in "0123456789" if d != digits[pos] and not (pos == offset and d == "0")]
    new = digits[:pos] + rng.choice(choices) + digits[pos + 1:]
    return Decimal(new[:len(new) - n.frac] + ("." + new[len(new) - n.frac:] if n.frac else ""))


def major_edits(parent: dict, rng) -> list[Edit]:
    src, mt, lang = parent["src"], parent["mt"], target_lang(parent["lp"])
    src_nums = source_numbers(src)
    nums = anchored_numbers(mt, src_nums)
    taken = set(src_nums)  # a corrupted value must not exist anywhere in the source
    edits: list[Edit] = []

    def fresh(v: Decimal | None) -> bool:
        return v is not None and v not in taken and v >= 0

    for n in nums:
        after = mt[n.end:]
        if PERCENT_RE[lang].match(after):
            new = n.value + rng.choice([-20, -10, -5, -2, 2, 5, 10, 20])
            if fresh(new) and new > 0 and (n.value > 100 or new <= 100):
                edits.append(_value_edit(n, new, "num_percent"))
            continue
        if DAY_MONTH_RE[lang].match(after) and n.frac == 0 and n.group is None and 1 <= n.value <= 31:
            days = [d for d in range(1, 29) if Decimal(d) not in taken]
            if days:
                edits.append(_value_edit(n, Decimal(rng.choice(days)), "date_day"))
            continue
        if is_year(n):
            new = n.value + rng.choice([-1, 1]) * rng.choice([1, 2, 3, 5, 10])
            if fresh(new) and new <= 2100:
                edits.append(_value_edit(n, new, "date_year"))
            continue  # a year gets no digit/magnitude/unit edits: "2022 года" is a date, not a duration
        new = _changed_digit(n, rng)
        if fresh(new):
            edits.append(_value_edit(n, new, "num_digit"))
        if in_compound(n, mt):
            continue
        # x10 or /10. Decimals move the point: 2,5 -> 25 and 1,35 -> 13,5
        if n.frac:
            new, frac = n.value * 10, n.frac - 1
        elif n.value % 10 == 0 and n.value >= 10 and rng.random() < 0.5:
            new, frac = n.value / 10, 0
        else:
            new, frac = n.value * 10, 0
        if fresh(new):
            edits.append(_value_edit(n, new, "num_magnitude", frac))
        edits += unit_edits(n, mt, lang)

    # two different numbers of the same kind exchange places, with real words in between (not a
    # range "1-2" or time "5:30"); ordinals like "1ra"/"5-й" and day-of-month numbers stay put
    def swappable(n: Num) -> bool:
        return (not re.match(r"[\-−]?[^\W\d_]", mt[n.end:n.end + 2]) and not DAY_MONTH_RE[lang].match(mt, n.end)
                and not in_compound(n, mt))

    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            between = mt[a.end:b.start]
            if (a.value != b.value and is_year(a) == is_year(b) and swappable(a) and swappable(b)
                    and 3 <= len(between) <= 60 and re.search(r"[^\W\d_]{2}", between)):
                repl = b.raw + between + a.raw
                second = a.start + len(b.raw) + len(between)
                edits.append(Edit("num_swap", a.start, b.end, repl, "Major",
                                  [(a.start, a.start + len(b.raw)), (second, second + len(a.raw))]))
    edits += month_edits(src, mt, lang, rng)
    return edits


def unit_edits(n: Num, mt: str, lang: str) -> list[Edit]:
    out = []
    slots = number_slots(n, mt, lang)
    pos = slots.word
    starts = [pos] + ([slots.currency] if slots.scale_end else [])
    for k, at in enumerate(starts):
        for sw, rx in _SWAP_RE[lang]:
            if k == 1 and sw.synth_type != "num_currency":
                continue  # after a scale word only currency swaps make sense
            m = rx.match(mt, at)
            # "фунтов стерлингов", "долларов США": the currency is two words, leave it alone
            if not m or re.match(r"[ \u00a0]+(?:стерлинг|США|US|americanos|estadounidenses)", mt[m.end():]):
                continue
            word = m.group()
            if sw.stem_to:
                repl = match_case(m.group("stem"), sw.stem_to) + m.group("end")
            elif sw.plural:
                repl = plural_ru(n.value, *sw.plural)
            else:
                repl = match_case(word, sw.repl)
            out.append(Edit(sw.synth_type, m.start(), m.end(), repl, "Major", [(m.start(), m.start() + len(repl))]))
            break  # first (longest/most specific) pattern wins at this position
    # currency symbol right before or after the number: "$25" / "25 $"
    before = mt[n.start - 1] if n.start else ""
    if before in CURRENCY_SYMBOLS:
        out.append(Edit("num_currency", n.start - 1, n.start, CURRENCY_SYMBOLS[before], "Major",
                        [(n.start - 1, n.start)]))
    elif pos < len(mt) and mt[pos] in CURRENCY_SYMBOLS:
        out.append(Edit("num_currency", pos, pos + 1, CURRENCY_SYMBOLS[mt[pos]], "Major", [(pos, pos + 1)]))
    return out


def month_edits(src: str, mt: str, lang: str, rng) -> list[Edit]:
    """Swap a month name that translates a month of the source to a month absent from it."""
    in_src = source_months(src)
    if not in_src:
        return []
    out = []
    for forms in month_lists(lang):
        for m in re.finditer(r"(?<!\w)(" + "|".join(forms) + r")(?!\w)", mt, re.IGNORECASE):
            i = forms.index(m.group(1).lower())
            if i not in in_src:
                continue
            j = rng.choice([k for k in range(12) if k not in in_src])
            repl = match_case(m.group(1), forms[j])
            out.append(Edit("date_month", m.start(), m.end(), repl, "Major", [(m.start(), m.start() + len(repl))]))
    return out


def ok_edits(parent: dict, rng) -> list[Edit]:
    src, mt, lang = parent["src"], parent["mt"], target_lang(parent["lp"])
    src_nums = source_numbers(src)
    edits = []
    for n in anchored_numbers(mt, src_nums):
        src_grouped = any(s.group == "," for s in src_nums[n.value])
        # thousands separator per target locale. Never "." grouping (reads as a decimal in es-MX)
        # and never "," in ru (decimal comma). 4-digit numbers may lose grouping, never gain it.
        if src_grouped and n.int_digits >= 4 and not is_year(n):
            styles = [" ", NBSP, NNBSP] if lang == "ru" else [",", " ", NBSP]
            if n.dec == ",":
                styles = [s for s in styles if s != ","]
            if n.int_digits == 4:
                styles = [None] if n.group else []
            styles = [s for s in styles if s != n.group]
            if styles:
                text = render(n.value, rng.choice(styles), n.dec, n.frac)
                edits.append(Edit("ok_grouping", n.start, n.end, text, "OK"))
        # ru: English decimal point -> Russian decimal comma (the reverse would be a format error)
        # "18.12" / "17.30" are a date or a time, not a decimal: leave two-digit fractions <= 31 alone
        looks_like_date = n.frac == 2 and n.value < 32
        if lang == "ru" and n.dec == "." and n.group in (None, " ", NBSP, NNBSP) and not looks_like_date:
            if any(s.dec == "." for s in src_nums[n.value]):
                edits.append(Edit("ok_decimal", n.start, n.end, render(n.value, n.group, ",", n.frac), "OK"))
        edits += word_variant_edits(n, src, mt, lang)
        after = mt[n.end:n.end + 2]
        if after[:1] == "%":
            edits.append(Edit("ok_percent", n.end, n.end, NBSP, "OK"))
        elif after[:1] in WS and after[1:2] == "%":
            edits.append(Edit("ok_percent", n.end, n.end + 1, "", "OK"))
        elif after[:1] == " " and re.match(r"[^\W\d_]|[$€£]", after[1:2]):
            edits.append(Edit("ok_nbsp", n.end, n.end + 1, NBSP, "OK"))
        elif after[:1] == NBSP and re.match(r"[^\W\d_]|[$€£]", after[1:2]):
            edits.append(Edit("ok_nbsp", n.end, n.end + 1, " ", "OK"))
    return edits


def word_variant_edits(n: Num, src: str, mt: str, lang: str) -> list[Edit]:
    """Visible but meaning-preserving rewrites next to a number, so OK examples are not all whitespace:
    "$50" <-> "50 долларов", "50%" <-> "50 процентов"/"50 por ciento", "5 километров" -> "5 км"."""
    out = []
    slots = number_slots(n, mt, lang)
    body_end = slots.scale_end or n.end            # number plus its scale word: "20 млрд"
    # dollars only when the source uses a plain "$" (not A$, C$, HK$)
    dollars = re.search(r"(?<![A-Za-z])\$\s?\d", src) and not re.search(r"[A-Z]\$", src)
    word_after = DOLLAR_WORD_RE[lang].match(mt, slots.currency)
    if dollars and mt[n.start - 1:n.start] == "$" and not mt[n.start - 2:n.start - 1].isalpha() and not word_after:
        # "$20 млрд" -> "20 млрд долларов", "$34 billones" -> "34 billones de dólares"
        text = mt[n.start:body_end] + " " + dollar_word(n, lang, bool(slots.scale_end))
        out.append(Edit("ok_currency", n.start - 1, body_end, text, "OK"))
    elif dollars and word_after and mt[n.start - 1:n.start] != "$":
        # "3,5 миллиарда долларов" -> "$3,5 миллиарда", "16 mil millones de dólares" -> "$16 mil millones"
        out.append(Edit("ok_currency", n.start, word_after.end(), "$" + mt[n.start:body_end], "OK"))
    # "50%" <-> "50 процентов" / "50 por ciento" (not "50%-ная")
    if pct := re.match(r"[ \u00a0\u202f]?%(?![-\w])", mt[n.end:]):
        out.append(Edit("ok_percent_word", n.end, n.end + pct.end(), " " + percent_word(n, lang), "OK"))
    elif m := PERCENT_WORD_RE[lang].match(mt, n.end):
        out.append(Edit("ok_percent_word", n.end, m.end(), "%", "OK"))
    # "5 километров" -> "5 км", "3,3 миллиарда" -> "3,3 млрд" (invariant abbreviations)
    for rx, abbr in _ABBREV_RE[lang]:
        if (m := rx.match(mt, slots.word)) and (slots.word > n.end or lang == "ru"):
            if m.group().lower() != abbr.rstrip("."):
                out.append(Edit("ok_abbrev", m.start(), m.end(), abbr, "OK"))
            break
    return out


# --------------------------------------------------------------------------- checks

def strip_format(text: str) -> str:
    return re.sub(r"[\s.,]", "", text)


def check(parent: dict, rec: dict) -> None:
    """Independent re-check of each output (cheap, catches regressions in the rules)."""
    old, new, t = parent["mt"], rec["mt"], rec["synth_type"]
    assert old != new, rec["id"]
    vals = lambda s: sorted(n.value for n in anchored_numbers(s, source_numbers(parent["src"])))  # noqa: E731
    if rec["label"] == "OK":
        if t in ("ok_grouping", "ok_decimal", "ok_nbsp", "ok_percent"):
            assert strip_format(old) == strip_format(new), (t, old, new)
        assert digit_string(old) == digit_string(new), (t, old, new)
        assert vals(old) == vals(new), (t, old, new)
        assert not any(sp["severity"] in ("major", "critical") for sp in rec["spans"]), rec["id"]
    elif t == "num_swap":
        assert sorted(digit_string(old)) == sorted(digit_string(new)) and digit_string(old) != digit_string(new)
    elif t in ("num_unit", "num_scale", "num_currency", "date_month"):
        assert digit_string(old) == digit_string(new), (t, old, new)
    else:  # a value changed: the multiset of source-anchored numbers must differ
        assert vals(old) != vals(new), (t, old, new)
    for sp in rec["spans"]:
        assert new[sp["start"]:sp["end"]] == sp["text"], (rec["id"], sp)


# --------------------------------------------------------------------------- selection

def build_record(parent: dict, e: Edit) -> dict:
    mt = e.apply(parent["mt"])
    spans = shift_spans(parent["spans"], e.start, e.end, len(e.repl))
    spans += [error_span(mt, s, t, e.synth_type) for s, t in e.err]
    spans.sort(key=lambda sp: (sp["start"], sp["end"]))
    edit = {"start": e.start, "end": e.start + len(e.repl), "text": e.repl, "orig": parent["mt"][e.start:e.end]}
    return make_record(parent, mt=mt, label=e.label, synth_type=e.synth_type, generator=GENERATOR,
                       edit=edit, spans=spans)


# At most this many per segment for types that are available almost everywhere.
TYPE_CAP_PER_SEG = {"ok_nbsp": 1}


def pick(parents: list[dict], cands: dict[str, list[Edit]], avail: collections.Counter, cap: int,
         seed: int, tag: str, banned: set, emitted: set) -> list[tuple[dict, Edit]]:
    """Up to `cap` edits for one segment: one per parent, preferring types not yet used in this
    segment and, among those, types that are rare overall (so num_digit does not swamp units/dates)."""
    used: collections.Counter = collections.Counter()
    out = []
    for p in parents:
        if len(out) >= cap:
            break
        rng = rng_for(seed, p["id"], tag, "pick")
        options = [e for e in cands[p["id"]] if (p["src"], e.apply(p["mt"])) not in emitted
                   and e.apply(p["mt"]) not in banned
                   and used[e.synth_type] < TYPE_CAP_PER_SEG.get(e.synth_type, cap)]
        if not options:
            continue
        t = min(sorted({e.synth_type for e in options}), key=lambda t: (used[t], avail[t] * rng.random()))
        e = rng.choice([e for e in options if e.synth_type == t])
        used[t] += 1
        emitted.add((p["src"], e.apply(p["mt"])))
        out.append((p, e))
    return out


def generate(lp: str, split: str, args) -> tuple[list[dict], dict]:
    pool, stats = source_pool(lp, split)
    natural = natural_mts(lp, split)
    majors = {p["id"]: major_edits(p, rng_for(args.seed, p["id"], "major")) for p in pool}
    oks = {p["id"]: ok_edits(p, rng_for(args.seed, p["id"], "ok")) for p in pool}
    avail_major = collections.Counter(t for es in majors.values() for t in {e.synth_type for e in es})
    avail_ok = collections.Counter(t for es in oks.values() for t in {e.synth_type for e in es})

    by_seg: dict[tuple, list[dict]] = collections.defaultdict(list)
    for p in pool:
        by_seg[segment_key(p)].append(p)
    records, emitted = [], set()
    for seg in sorted(by_seg):
        parents = sorted(by_seg[seg], key=lambda p: p["id"])
        rng_for(args.seed, lp, split, *seg).shuffle(parents)  # which systems get used is seeded
        banned = natural[seg[1]]  # never reproduce a real (human-rated) MT of this source
        chosen = pick(parents, majors, avail_major, args.max_major_per_seg, args.seed, "major", banned, emitted)
        chosen += pick(parents, oks, avail_ok, args.max_ok_per_seg, args.seed, "ok", banned, emitted)
        for p, e in chosen:
            rec = build_record(p, e)
            check(p, rec)
            records.append(rec)
    records.sort(key=lambda r: r["id"])
    assert_hygiene(records, lp, split)
    stats.update(parents_with_major_cand=sum(bool(v) for v in majors.values()),
                 parents_with_ok_cand=sum(bool(v) for v in oks.values()), segments=len(by_seg))
    return records, stats


def show(rec: dict) -> str:
    e = rec["synth_edit"]
    vis = lambda s: s.replace(NBSP, "⍽").replace(NNBSP, "␣")  # noqa: E731  make invisible spaces visible
    return (f"  [{rec['lp']} {rec['synth_type']} -> {rec['label']}] {vis(e['orig'])!r} -> {vis(e['text'])!r}\n"
            f"    SRC: {rec['src'][:160]}\n    MT : {vis(rec['mt'])[:160]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lps", nargs="+", default=["en-ru", "en-es"])
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-major-per-seg", type=int, default=4,
                    help="Major examples per source segment (each segment has up to 16 MT systems)")
    ap.add_argument("--max-ok-per-seg", type=int, default=2, help="OK variants per source segment")
    ap.add_argument("--out-dir", type=Path, default=SYNTH_DIR)
    ap.add_argument("--show", type=int, default=2, help="examples to print per (lp, type)")
    args = ap.parse_args()

    for split in args.splits:
        out = []
        for lp in args.lps:
            recs, stats = generate(lp, split, args)
            out += recs
            by_type = collections.Counter((r["label"], r["synth_type"]) for r in recs)
            n_major = sum(v for (lab, _), v in by_type.items() if lab == "Major")
            print(f"\n== {lp} {split}: {len(recs)} records (Major {n_major}, OK {len(recs) - n_major}) | " +
                  ", ".join(f"{k}={v}" for k, v in stats.items()))
            for (lab, t), v in sorted(by_type.items()):
                print(f"   {lab:5s} {t:14s} {v:5d}")
            if args.show:
                seen = collections.Counter()
                for r in recs:
                    if seen[r["synth_type"]] < args.show:
                        seen[r["synth_type"]] += 1
                        print(show(r))
        path = args.out_dir / f"rules.{split}.jsonl"
        write_jsonl(path, out)
        print(f"wrote {path} ({len(out)} records)")


if __name__ == "__main__":
    main()
