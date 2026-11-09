"""Постперевірка: кожне число у відповіді (з одиницею, якщо вона є) має бути в джерелах."""

import re
from dataclasses import dataclass, field

# число: 120 000 / 2,0 / 0.05 / 25
_NUM = r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?"
_UNITS = {
    "мм": "мм", "см": "см", "м": "м", "км/год": "км/год", "км": "км", "мкм": "мкм",
    "н·м": "нм", "нм": "нм", "н*м": "нм", "н.м": "нм", "n·m": "нм", "nm": "нм",
    "бар": "бар", "bar": "бар", "°c": "c", "° c": "c", "°": "°", "%": "%",
    "л": "л", "с": "с", "секунд": "с", "секунди": "с", "хв": "хв",
    "років": "рік", "роки": "рік", "рік": "рік", "місяців": "міс", "місяці": "міс", "міс": "міс",
    "зубів": "зуб", "шт": "шт", "км пробігу": "км",
}
_UNIT_RE = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
_NUM_UNIT = re.compile(rf"(?<![\w,.])({_NUM})(?:\s?({_UNIT_RE})(?![а-яіїєґa-z]))?", re.IGNORECASE)

# те, що не є фактичним числом відповіді
_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_PAGE_REF = re.compile(r"(?:с\.|стор\.|сторінк\w*|розділ\w*|п\.|пункт\w*|рис\.)\s*\d+", re.IGNORECASE)


def norm_number(raw: str) -> str:
    s = re.sub(r"[  ]", "", raw).replace(",", ".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def extract_numbers(text: str) -> list[tuple[str, str | None]]:
    text = _PAGE_REF.sub(" ", _CITATION.sub(" ", text))
    out = []
    for m in _NUM_UNIT.finditer(text):
        unit = m.group(2)
        out.append((norm_number(m.group(1)), _UNITS.get(unit.lower()) if unit else None))
    return out


@dataclass
class Verification:
    ok: bool
    unsupported: list[str] = field(default_factory=list)
    checked: int = 0


def verify_numbers(answer: str, sources: list[str]) -> Verification:
    src_pairs: set[tuple[str, str | None]] = set()
    for s in sources:
        src_pairs.update(extract_numbers(s))
    src_numbers = {n for n, _ in src_pairs}
    bare = {n for n, u in src_pairs if u is None}

    unsupported = []
    pairs = extract_numbers(answer)
    for num, unit in pairs:
        if num not in src_numbers:
            unsupported.append(num + (f" {unit}" if unit else ""))
            continue
        # число є, але з іншою одиницею (у джерелі «25 Н·м», у відповіді «25 мм») — помилка.
        # Якщо в джерелі число стоїть без одиниці (таблиця з одиницею в шапці), приймаємо.
        if unit and (num, unit) not in src_pairs and num not in bare:
            unsupported.append(f"{num} {unit}")
    return Verification(ok=not unsupported, unsupported=unsupported, checked=len(pairs))
