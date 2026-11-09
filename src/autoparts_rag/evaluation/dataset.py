"""Тестсет: завантаження і lint (розділ 2.1 підготовки до заняття 9).

Тест з помилкою дає хибне число, тому набір перевіряємо до прогону, без виклику моделі."""

import json
from pathlib import Path

from ..db import connection

def load_items(path: Path, limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items[:limit] if limit else items


def lint(items: list[dict]) -> list[str]:
    """Перевірити тестсет до прогону (розділ 2.1 підготовки): тест з помилкою дає хибне число.
    Документ і сторінка існують, очікувані числа справді є на цих сторінках, у відповідних
    кейсів є еталон для судді, id не повторюються."""
    from ..verify import extract_numbers as nums
    issues, seen = [], set()
    with connection() as conn:
        docs = {r["doc_key"]: r["page_count"] for r in conn.execute(
            "SELECT doc_key, page_count FROM documents WHERE is_active").fetchall()}
        for it in items:
            exp, iid = it["expected"], it["id"]
            if iid in seen:
                issues.append(f"{iid}: id повторюється")
            seen.add(iid)
            if exp["behavior"] == "answer" and not exp.get("sources"):
                issues.append(f"{iid}: відповідь очікується, але джерела не вказано")
            if exp["behavior"] == "answer" and not it.get("reference"):
                issues.append(f"{iid}: немає reference, суддя правильності не зможе оцінити")
            texts = []
            for s in exp.get("sources", []):
                if s["doc"] not in docs:
                    issues.append(f"{iid}: документа {s['doc']} немає серед активних")
                    continue
                bad = [p for p in s["pages"] if not 1 <= p <= docs[s["doc"]]]
                if bad:
                    issues.append(f"{iid}: {s['doc']} має {docs[s['doc']]} с., а вказано {bad}")
                texts += [r["content"] for r in conn.execute(
                    """SELECT c.content FROM chunks c JOIN documents d ON d.id = c.document_id
                       WHERE d.is_active AND d.doc_key = %s AND c.page_from <= %s AND c.page_to >= %s""",
                    (s["doc"], max(s["pages"]), min(s["pages"]))).fetchall()]
            found = {n for t in texts for n, _ in nums(t)}
            for raw in exp.get("numbers", []):
                missing = [n for n, _ in nums(raw) if n not in found]
                if texts and missing:
                    issues.append(f"{iid}: число {raw} не знайдено на сторінках-джерелах")
    return issues
