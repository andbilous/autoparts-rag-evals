"""Розбиття елементів документа на chunk: за розділами, таблиці цілими або по рядках з шапкою."""

from dataclasses import dataclass, field

from ..config import get_settings
from .parse import Element


@dataclass
class Chunk:
    chunk_type: str  # text | table | figure_caption
    section_title: str | None
    page_from: int
    page_to: int
    body: str
    content: str = ""  # body з назвою документа і розділу; саме це йде в embedding і в LLM
    articles: list[str] = field(default_factory=list)


def table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"

    out = [line(header), "|" + "---|" * len(header)]
    out += [line(r) for r in rows]
    return "\n".join(out)


def merge_cross_page_tables(elements: list[Element]) -> list[tuple[Element, int]]:
    """Склеює частини однієї таблиці, розірвані розривом сторінки.

    Повертає пари (елемент, остання сторінка). Таблиці склеюються, якщо вони йдуть
    підряд, на сусідніх сторінках і мають однакову шапку.
    """
    merged: list[tuple[Element, int]] = []
    for el in elements:
        if merged and el.kind == "table":
            prev, last_page = merged[-1]
            if prev.kind == "table" and prev.header == el.header and el.page == last_page + 1:
                prev.rows = prev.rows + el.rows
                merged[-1] = (prev, el.page)
                continue
        merged.append((Element(el.kind, el.text, el.page, list(el.header), list(el.rows)), el.page))
    return merged


def split_table(header: list[str], rows: list[list[str]], max_rows: int) -> list[list[list[str]]]:
    if len(rows) <= max_rows:
        return [rows]
    return [rows[i : i + max_rows] for i in range(0, len(rows), max_rows)]


def chunk_elements(elements: list[Element], doc_title: str, max_chars: int | None = None,
                   overlap: int | None = None, table_max_rows: int | None = None) -> list[Chunk]:
    s = get_settings()
    max_chars = max_chars or s.chunk_max_chars
    overlap = s.chunk_overlap_paragraphs if overlap is None else overlap
    table_max_rows = table_max_rows or s.table_max_rows

    chunks: list[Chunk] = []
    section: str | None = None
    buf: list[Element] = []

    def flush(keep_overlap: bool = False) -> None:
        nonlocal buf
        if not buf:
            return
        chunks.append(Chunk("text", section, buf[0].page, buf[-1].page,
                            "\n\n".join(e.text for e in buf)))
        buf = buf[-overlap:] if keep_overlap and overlap else []

    for el, last_page in merge_cross_page_tables(elements):
        if el.kind == "heading":
            flush()
            section = el.text
        elif el.kind == "para":
            if buf and sum(len(e.text) for e in buf) + len(el.text) > max_chars:
                flush(keep_overlap=True)
            buf.append(el)
        elif el.kind == "caption":
            chunks.append(Chunk("figure_caption", section, el.page, el.page,
                                f"{el.text} (схема на с. {el.page} документа)"))
        elif el.kind == "table":
            flush()
            parts = split_table(el.header, el.rows, table_max_rows)
            for i, part in enumerate(parts):
                # для таблиці на кількох сторінках визначити точну сторінку частини не можна,
                # тому зберігаємо весь діапазон
                note = f"Таблиця, частина {i + 1} з {len(parts)}\n" if len(parts) > 1 else ""
                chunks.append(Chunk("table", section, el.page, last_page,
                                    note + table_to_markdown(el.header, part)))
    flush()

    for c in chunks:
        head = f"Документ: {doc_title}"
        if c.section_title and c.section_title != doc_title:
            head += f"\nРозділ: {c.section_title}"
        c.content = f"{head}\n\n{c.body}"
    return chunks
