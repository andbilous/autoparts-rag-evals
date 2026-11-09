"""Індексація: хеш файлу -> розбір -> chunk -> embedding -> запис однією транзакцією."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..db import connection
from ..models.embedder import get_embedder
from ..normalize import extract_articles, lemmatize, normalize_article
from .chunk import chunk_elements
from .parse import parse_pdf


@dataclass
class IngestStats:
    doc_key: str
    status: str  # indexed | skipped | reindexed
    version: int = 0
    pages: int = 0
    chunks: int = 0
    ocr_pages: int = 0
    ocr_rejected: int = 0


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest_file(path: Path, meta: dict, force: bool = False, chunk_kwargs: dict | None = None) -> IngestStats:
    digest = file_hash(path)
    with connection() as conn:
        current = conn.execute(
            "SELECT id, version, content_hash FROM documents WHERE doc_key = %s AND is_active",
            (meta["doc_key"],),
        ).fetchone()
    if current and current["content_hash"] == digest and not force:
        return IngestStats(meta["doc_key"], "skipped", current["version"])

    parsed = parse_pdf(str(path))
    chunks = chunk_elements(parsed.elements, meta["title"], **(chunk_kwargs or {}))
    vectors = get_embedder().embed([c.content for c in chunks]) if chunks else []
    chunk_articles = [extract_articles(c.content) for c in chunks]
    # артикули документа: з прив'язки (manifest), із заголовка і з усіх chunk. Тоді артикул,
    # згаданий лише в одному абзаці («серія GP-4510»), відкриває для пошуку весь документ.
    doc_articles = sorted(
        {normalize_article(a) for a in meta.get("articles", [])}
        | set(extract_articles(meta["title"]))
        | {a for arts in chunk_articles for a in arts}
    )

    with connection() as conn, conn.transaction():
        version = 1
        if current:
            # стара версія виходить з пошуку, але лишається в БД для аудиту
            conn.execute("UPDATE documents SET is_active = FALSE WHERE id = %s", (current["id"],))
            version = current["version"] + 1
        doc_id = conn.execute(
            """INSERT INTO documents (doc_key, file_name, title, manufacturer, doc_type, articles,
                                      content_hash, version, storage_path, page_count)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (meta["doc_key"], path.name, meta["title"], meta["manufacturer"], meta["doc_type"],
             doc_articles, digest, version, str(path.resolve()), parsed.page_count),
        ).fetchone()["id"]

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (document_id, chunk_index, page_from, page_to, section_title,
                                       chunk_type, content, content_lemmas, articles_norm, embedding,
                                       content_hash)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, i, c.page_from, c.page_to, c.section_title, c.chunk_type, c.content,
                     lemmatize(c.content), arts, vec,
                     hashlib.sha256(c.content.encode()).hexdigest())
                    for i, (c, arts, vec) in enumerate(zip(chunks, chunk_articles, vectors))
                ],
            )
        for op in parsed.ocr_pages:
            if not op.accepted:
                conn.execute(
                    "INSERT INTO ocr_review_queue (document_id, page, confidence) VALUES (%s, %s, %s)",
                    (doc_id, op.page, op.confidence),
                )

    return IngestStats(
        meta["doc_key"], "reindexed" if current else "indexed", version, parsed.page_count,
        len(chunks), len(parsed.ocr_pages), sum(not p.accepted for p in parsed.ocr_pages),
    )


def ingest_dir(directory: Path, force: bool = False, chunk_kwargs: dict | None = None) -> list[IngestStats]:
    manifest = json.loads((directory / "manifest.json").read_text())
    return [ingest_file(directory / m["file"], m, force=force, chunk_kwargs=chunk_kwargs) for m in manifest]
