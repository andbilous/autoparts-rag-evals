"""Гібридний пошук: FTS по лемах + точний збіг артикулів + вектори -> RRF -> rerank."""

import re
from dataclasses import asdict, dataclass, field

from .config import get_settings
from .db import connection
from .models.embedder import get_embedder
from .models.reranker import get_reranker
from .normalize import lemmatize, normalize_article

# Службові слова, які в OR-запиті лише шумлять. Список короткий навмисно:
# ts_rank_cd і так мало важить для частих слів, а зайве видалення шкодить.
STOPWORDS = {
    "і", "й", "та", "а", "але", "в", "у", "на", "з", "із", "зі", "до", "для", "по", "про", "при",
    "чи", "що", "як", "який", "яка", "яке", "які", "це", "цей", "ця", "ці", "той", "та", "ті",
    "є", "бути", "не", "ні", "так", "же", "ж", "би", "б", "мені", "я", "ви", "він", "вона", "воно",
    "вони", "мій", "ваш", "його", "її", "їх", "тут", "там", "де", "коли", "скільки", "чому",
    "можна", "треба", "потрібно", "якщо", "або", "від", "без", "під", "над", "за", "через",
}


@dataclass
class Candidate:
    chunk_id: int
    document_id: int
    doc_key: str
    doc_title: str
    page_from: int
    page_to: int
    section_title: str | None
    chunk_type: str
    content: str
    ranks: dict = field(default_factory=dict)
    rrf: float = 0.0
    rerank_score: float | None = None

    def brief(self) -> dict:
        d = asdict(self)
        d["content"] = self.content[:200]
        return d


@dataclass
class RetrievalResult:
    candidates: list[Candidate]  # після rerank, top context_top_k
    pre_rerank: list[Candidate]  # після RRF, top rerank_top_n
    scoped: bool  # пошук був обмежений документами товару/виробника
    widened: bool  # обмежений пошук нічого не дав і його розширили на всю базу
    max_score: float


def build_tsquery(query: str) -> str | None:
    tokens = []
    for lemma in lemmatize(query).split():
        if lemma in STOPWORDS or len(lemma) < 2:
            continue
        tok = re.sub(r"[^\w]", "", lemma)
        if tok and tok not in tokens:
            tokens.append(tok)
    return " | ".join(f"'{t}'" for t in tokens) or None


_SELECT = """
    SELECT c.id AS chunk_id, c.document_id, d.doc_key, d.title AS doc_title, c.page_from, c.page_to,
           c.section_title, c.chunk_type, c.content
    FROM chunks c JOIN documents d ON d.id = c.document_id
"""


def _scope_sql(articles: list[str], manufacturer: str | None) -> tuple[str, dict]:
    clauses, params = ["d.is_active"], {}
    scope = []
    if articles:
        scope.append("d.articles && %(scope_articles)s")
        params["scope_articles"] = articles
    if manufacturer:
        scope.append("d.manufacturer = %(scope_manufacturer)s")
        params["scope_manufacturer"] = manufacturer
    if scope:
        clauses.append("(" + " OR ".join(scope) + ")")
    return " AND ".join(clauses), params


def hybrid_search(query: str, query_articles: list[str], scope_articles: list[str] | None = None,
                  scope_manufacturer: str | None = None) -> list[Candidate]:
    s = get_settings()
    where, params = _scope_sql(scope_articles or [], scope_manufacturer)
    lists: dict[str, list[dict]] = {}
    qvec = get_embedder().embed([query])[0]

    with connection() as conn:
        conn.execute("SET hnsw.ef_search = 100")
        tsq = build_tsquery(query)
        if tsq:
            lists["fts"] = conn.execute(
                f"""{_SELECT} WHERE {where} AND c.tsv @@ to_tsquery('simple', %(tsq)s)
                    ORDER BY ts_rank_cd(c.tsv, to_tsquery('simple', %(tsq)s)) DESC LIMIT %(k)s""",
                {**params, "tsq": tsq, "k": s.fts_top_k},
            ).fetchall()
        if query_articles:
            lists["article"] = conn.execute(
                f"""{_SELECT} WHERE {where}
                      AND (c.articles_norm && %(qa)s OR d.articles && %(qa)s)
                    ORDER BY (c.articles_norm && %(qa)s) DESC, c.chunk_index LIMIT %(k)s""",
                {**params, "qa": query_articles, "k": s.fts_top_k},
            ).fetchall()
        lists["vector"] = conn.execute(
            f"""{_SELECT} WHERE {where} ORDER BY c.embedding <=> %(v)s LIMIT %(k)s""",
            {**params, "v": qvec, "k": s.vector_top_k},
        ).fetchall()

    return rrf_merge(lists, s.rrf_k)


def rrf_merge(lists: dict[str, list[dict]], k: int) -> list[Candidate]:
    by_id: dict[int, Candidate] = {}
    for name, rows in lists.items():
        for rank, row in enumerate(rows, start=1):
            cand = by_id.get(row["chunk_id"])
            if cand is None:
                cand = by_id[row["chunk_id"]] = Candidate(**row)
            cand.ranks[name] = rank
            cand.rrf += 1.0 / (k + rank)
    return sorted(by_id.values(), key=lambda c: c.rrf, reverse=True)


def rerank(query: str, candidates: list[Candidate], top_n: int, top_k: int) -> list[Candidate]:
    pool = candidates[:top_n]
    scores = get_reranker().score(query, [c.content for c in pool])
    for c, sc in zip(pool, scores):
        c.rerank_score = sc
    return sorted(pool, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_k]


def known_manufacturers() -> list[str]:
    with connection() as conn:
        rows = conn.execute("SELECT DISTINCT manufacturer FROM documents WHERE is_active").fetchall()
    return [r["manufacturer"] for r in rows]


def detect_manufacturer(text: str, manufacturers: list[str]) -> str | None:
    low = text.lower().replace("-", " ")
    for m in manufacturers:
        name = m.lower().replace("-", " ")
        # ціле слово з відмінковим закінченням: «Гальмар», «Гальмару», «Гальмара».
        # Основа без закінчення не годиться: «гальм» збігається з «гальмівні».
        if re.search(rf"(?<!\w){re.escape(name)}\w{{0,3}}(?!\w)", low):
            return m
    return None


def retrieve(query: str, query_articles: list[str], product_article: str | None = None,
             product_manufacturer: str | None = None, threshold: float | None = None) -> RetrievalResult:
    s = get_settings()
    threshold = s.rerank_threshold if threshold is None else threshold

    scope_articles = list(query_articles)
    if product_article and not scope_articles:
        scope_articles = [normalize_article(product_article)]
    manufacturer = product_manufacturer or detect_manufacturer(query, known_manufacturers())

    def run(scoped: bool) -> tuple[list[Candidate], list[Candidate]]:
        if scoped:
            merged = hybrid_search(query, query_articles, scope_articles, manufacturer)
        else:
            merged = hybrid_search(query, query_articles)
        return rerank(query, merged, s.rerank_top_n, s.context_top_k), merged[: s.rerank_top_n]

    if scope_articles or manufacturer:
        top, pre = run(scoped=True)
        best = max((c.rerank_score or 0.0 for c in top), default=0.0)
        if best >= threshold:
            return RetrievalResult(top, pre, scoped=True, widened=False, max_score=best)
        top, pre = run(scoped=False)
        best = max((c.rerank_score or 0.0 for c in top), default=0.0)
        return RetrievalResult(top, pre, scoped=True, widened=True, max_score=best)

    top, pre = run(scoped=False)
    best = max((c.rerank_score or 0.0 for c in top), default=0.0)
    return RetrievalResult(top, pre, scoped=False, widened=False, max_score=best)
