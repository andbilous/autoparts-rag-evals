"""PDF -> список елементів (заголовок, абзац, таблиця, підпис до схеми) з номерами сторінок."""

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz

from ..config import get_settings
from ..normalize import normalize_text

CAPTION_RE = re.compile(r"^(Рис\.|Рисунок|Схема|Мал\.)\s*\d", re.IGNORECASE)
MARGIN_RATIO = 0.06  # верхні й нижні 6 % сторінки: тут короткий блок завжди колонтитул
REPEAT_BAND_RATIO = 0.12  # у цій смузі колонтитулом вважається блок, що повторюється на сторінках


@dataclass
class Element:
    kind: str  # heading | para | table | caption
    text: str
    page: int  # з 1
    header: list[str] = field(default_factory=list)  # для таблиць
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class OcrPage:
    page: int
    confidence: float
    accepted: bool


@dataclass
class ParsedDocument:
    page_count: int
    elements: list[Element]
    ocr_pages: list[OcrPage]


def _clean_cell(c: str | None) -> str:
    return normalize_text((c or "").replace("\n", " "))


def _overlaps(a: fitz.Rect, b: fitz.Rect) -> bool:
    return a.intersects(b) and (a & b).get_area() > 0.3 * min(a.get_area(), b.get_area())


def _text_page(page: fitz.Page, pno: int) -> list[tuple[float, Element, fitz.Rect]]:
    items: list[tuple[float, Element, fitz.Rect]] = []
    table_rects = []
    def cell_text(bbox) -> str:
        return _clean_cell(page.get_textbox(fitz.Rect(bbox))) if bbox else ""

    for tab in page.find_tables().tables:
        # tab.extract() у частини PDF губить пробіли й переставляє коми («131,0 мм» -> «1310мм ,»),
        # тобто псує числа. Тому текст кожної клітинки беремо за її координатами.
        rows = [[cell_text(c) for c in r.cells] for r in tab.rows]
        header = [cell_text(c) for c in tab.header.cells]
        if not tab.header.external and rows:
            rows = rows[1:]
        rect = fitz.Rect(tab.bbox)
        table_rects.append(rect)
        items.append((rect.y0, Element("table", "", pno, header=header, rows=rows), rect))

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        rect = fitz.Rect(block["bbox"])
        if any(_overlaps(rect, t) for t in table_rects):
            continue
        lines, max_size, bold = [], 0.0, True
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            lines.append("".join(s["text"] for s in line["spans"]))
            max_size = max(max_size, *(s["size"] for s in spans))
            bold &= all("Bold" in s["font"] or s["flags"] & 16 for s in spans)
        text = normalize_text(" ".join(lines))
        if not text:
            continue
        if max_size >= 13 or (bold and max_size >= 11.5 and len(text) < 150):
            kind = "heading"
        elif CAPTION_RE.match(text):
            kind = "caption"
        else:
            kind = "para"
        items.append((rect.y0, Element(kind, text, pno), rect))
    return items


def _ocr_page(page: fitz.Page, pno: int) -> tuple[list[tuple[float, Element, fitz.Rect]], float]:
    import pytesseract
    from PIL import Image

    zoom = 300 / 72
    pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(img, lang="ukr+eng", output_type=pytesseract.Output.DICT)

    paragraphs: dict[tuple[int, int], list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) >= 0:
            paragraphs.setdefault((data["block_num"][i], data["par_num"][i]), []).append(i)

    items, confs = [], []
    for idxs in paragraphs.values():
        words = [data["text"][i] for i in idxs]
        confs += [float(data["conf"][i]) for i in idxs]
        x0 = min(data["left"][i] for i in idxs) / zoom
        y0 = min(data["top"][i] for i in idxs) / zoom
        x1 = max(data["left"][i] + data["width"][i] for i in idxs) / zoom
        y1 = max(data["top"][i] + data["height"][i] for i in idxs) / zoom
        text = normalize_text(" ".join(words))
        kind = "caption" if CAPTION_RE.match(text) else "para"
        items.append((y0, Element(kind, text, pno), fitz.Rect(x0, y0, x1, y1)))
    conf = sum(confs) / len(confs) if confs else 0.0
    return items, conf


def _footer_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def parse_pdf(path: str) -> ParsedDocument:
    s = get_settings()
    doc = fitz.open(path)
    pages: list[list[tuple[float, Element, fitz.Rect]]] = []
    ocr_pages: list[OcrPage] = []
    heights = []

    for i, page in enumerate(doc):
        pno = i + 1
        heights.append(page.rect.height)
        if len(page.get_text("text").strip()) >= s.ocr_min_text_chars:
            pages.append(_text_page(page, pno))
            continue
        items, conf = _ocr_page(page, pno)
        accepted = conf >= s.ocr_min_confidence
        ocr_pages.append(OcrPage(pno, round(conf, 1), accepted))
        # сторінка з низькою впевненістю OCR чекає ручної перевірки і в індекс не йде
        pages.append(items if accepted else [])

    # колонтитули: короткі блоки, що (з точністю до цифр) повторюються на більшості сторінок
    counts = Counter(
        _footer_key(el.text) for items in pages for _, el, _ in items if el.kind != "table"
    )
    min_repeats = max(2, int(len(pages) * s.header_footer_ratio + 0.999))

    elements: list[Element] = []
    for items, height in zip(pages, heights):
        for _, el, rect in sorted(items, key=lambda t: t[0]):
            if el.kind != "table":
                def in_band(ratio: float) -> bool:
                    return rect.y1 < height * ratio or rect.y0 > height * (1 - ratio)

                # повтор шукаємо лише біля країв сторінки: абзаци «сторінка 1/2/3» з однаковим
                # текстом посередині сторінки після заміни цифр теж збіглися б
                repeated = (len(pages) >= 2 and in_band(REPEAT_BAND_RATIO)
                            and counts[_footer_key(el.text)] >= min_repeats)
                if len(el.text) < 100 and (repeated or in_band(MARGIN_RATIO)):
                    continue
            elements.append(el)
    return ParsedDocument(page_count=len(pages), elements=elements, ocr_pages=ocr_pages)
