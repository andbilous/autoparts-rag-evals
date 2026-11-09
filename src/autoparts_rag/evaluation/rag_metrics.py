"""Метрики RAG за id і сторінками (розділ 5): чи знайдено, чи дійшло до промпту, чи процитовано.

Усі детерміновані: порівнюємо документ і сторінку з expected.sources, без LLM."""

from ..db import connection

def source_hit(doc_key: str, page_from: int, page_to: int, sources: list[dict]) -> bool:
    return any(s["doc"] == doc_key and any(page_from <= p <= page_to for p in s["pages"])
               for s in sources)


def recall(cands, sources: list[dict]) -> float:
    """Частка очікуваних джерел (документ + сторінка), що потрапили в кандидатів."""
    if not sources:
        return 0.0
    found = sum(
        any(c.doc_key == s["doc"] and any(c.page_from <= p <= c.page_to for p in s["pages"]) for c in cands)
        for s in sources
    )
    return found / len(sources)


def chunk_texts(ids: list[int]) -> list[str]:
    if not ids:
        return []
    with connection() as conn:
        return [r["content"] for r in conn.execute(
            "SELECT content FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()]


def chunk_texts_ordered(ids: list[int]) -> list[str]:
    """Тексти фрагментів у тому самому порядку, що й ids (chunk_texts порядок не гарантує)."""
    if not ids:
        return []
    with connection() as conn:
        rows = {r["id"]: r["content"] for r in conn.execute(
            "SELECT id, content FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()}
    return [rows.get(i, "") for i in ids]
