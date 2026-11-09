"""Нормалізація тексту, артикулів і лематизація української для повнотекстового пошуку."""

import re
from functools import lru_cache

# Кириличні літери, які на вигляд збігаються з латинськими. Користувач може набрати
# артикул «КВ-1415А» українською розкладкою, а в документі він латиницею.
_CYR_TO_LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "У": "Y",
})

_QUOTES = str.maketrans({
    "’": "'", "ʼ": "'", "‘": "'", "`": "'", "´": "'",
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",
    "−": "-",  # мінус
    "­": None,  # м'який перенос
})


def normalize_text(text: str) -> str:
    """Уніфікує лапки, апострофи, нерозривні пробіли і переноси рядків."""
    text = text.translate(_QUOTES)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_article(raw: str) -> str:
    """'4 510 220 017' -> '4510220017', 'fl 715/40' -> 'FL71540', 'КВ-1415А' -> 'KB1415A'."""
    s = raw.upper().translate(_CYR_TO_LAT)
    return re.sub(r"[\s\-./_]", "", s)


# 1) довгі цифрові артикули, можливо з пробілами/дефісами/крапками між групами: 4 510 220 017
_DIGITS = re.compile(r"(?<![\d,])\d(?:[ .\-]?\d){7,13}(?![\d,])")
# 2) літерний префікс + цифри (+ необов'язковий хвіст): FL 715/40, GD-3021, RT2041K1, KB-1415A.
#    Пробіл після префікса дозволяємо лише для латиниці, інакше «до 650» стане артикулом.
_ALNUM = re.compile(
    r"(?<![\w])(?:[A-Za-z]{1,4}[ \-]?|[A-Za-zА-Яа-яІіЇїЄє]{1,4}-?)\d{2,6}(?:[/\-]?[A-Za-zА-Яа-я0-9]{1,6})*(?![\w])"
)
# 3) цифри з літерним хвостом без префікса: 1028K2
# після дефіса чи слеша це хвіст довшого артикула (KB-1215A), а не окремий артикул
_NUM_SUFFIX = re.compile(r"(?<![\w\-/])\d{3,6}[A-Za-z]{1,2}\d{0,2}(?![\w])")

_NOT_ARTICLES = {"M10", "M12", "M14", "M20", "M22", "M24", "R12"}


def extract_articles(text: str) -> list[str]:
    """Знаходить кандидатів в артикули і повертає їх нормалізованими, без повторів."""
    found: list[str] = []
    for rx in (_DIGITS, _ALNUM, _NUM_SUFFIX):
        for m in rx.finditer(text):
            norm = normalize_article(m.group(0))
            if len(norm) < 4 or not any(ch.isdigit() for ch in norm):
                continue
            if not any(ch.isalpha() for ch in norm) and len(norm) < 8:
                continue
            if norm in _NOT_ARTICLES or norm in found:
                continue
            found.append(norm)
    return found


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer(lang="uk")


_WORD = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ']+|\d+(?:[.,]\d+)?")


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    if not re.search(r"[А-Яа-яІіЇїЄєҐґ]", word):
        return word.lower()
    return _morph().parse(word.lower())[0].normal_form


def lemmatize(text: str) -> str:
    """Повертає рядок лем, придатний для to_tsvector('simple', ...).

    До лем дописуються нормалізовані артикули, щоб «FL 715/40» і «fl71540»
    давали однаковий токен.
    """
    text = normalize_text(text)
    lemmas = [_lemma(w.strip("'")) for w in _WORD.findall(text) if w.strip("'")]
    lemmas += [a.lower() for a in extract_articles(text)]
    return " ".join(lemmas)
