"""Конвеєр запиту (ТЗ 5.1). Видає події для SSE; остання подія done містить повний результат."""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from html import escape
from typing import AsyncIterator

import anthropic

from .config import get_settings
from .db import connection
from .models.llm import LLM, QueryAnalysis, RefusedByModel, Usage, get_llm
from .normalize import extract_articles
from .prompts import (
    DANGEROUS_DISCLAIMER,
    DANGEROUS_STEMS,
    NO_ANSWER_MARKER,
    REFUSAL_ABUSE,
    REFUSAL_NO_INFO,
    REFUSAL_OUT_OF_SCOPE,
    REFUSAL_UNAVAILABLE,
    STRICT_RETRY_NOTE,
)
from .retrieval import Candidate, RetrievalResult, retrieve
from .verify import verify_numbers

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class ChatRequest:
    session_id: str
    message: str
    product_article: str | None = None
    product_manufacturer: str | None = None
    history: list[dict] | None = None  # якщо None, береться з БД за session_id
    persist: bool = True  # eval не пише в messages, щоб не змішувати історії


@dataclass
class ChatResult:
    outcome: str = "error"
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    intent: dict | None = None
    rewritten_query: str | None = None
    query_articles: list[str] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    verification: dict | None = None
    dangerous: bool = False
    show_feedback_form: bool = False
    usage: Usage = field(default_factory=Usage)
    latency_ms: dict = field(default_factory=dict)
    trace_id: int | None = None
    context_ids: list[int] = field(default_factory=list)  # фрагменти, що реально пішли в промпт


def is_dangerous(analysis: QueryAnalysis, question: str) -> bool:
    text = f"{question} {analysis.rewritten_query}".lower()
    return analysis.dangerous or any(stem in text for stem in DANGEROUS_STEMS)


def build_user_content(question: str, rewritten: str, candidates: list[Candidate], widened: bool) -> str:
    docs = []
    for i, c in enumerate(candidates, start=1):
        docs.append(
            f'<document index="{i}" title="{escape(c.doc_title)}">\n{c.content}\n</document>'
        )
    note = ""
    if widened:
        note = ("\n<note>У документації товару, про який питають, релевантних фрагментів не "
                "знайдено. Наведені фрагменти з документації інших товарів.</note>\n")
    return (f"<documents>\n" + "\n".join(docs) + "\n</documents>\n" + note +
            f"\n<question>{question}</question>\n<standalone_question>{rewritten}</standalone_question>")


def extract_citations(answer: str, candidates: list[Candidate]) -> list[dict]:
    used: list[int] = []
    for m in CITE_RE.finditer(answer):
        for n in m.group(1).split(","):
            idx = int(n.strip())
            if 1 <= idx <= len(candidates) and idx not in used:
                used.append(idx)
    out = []
    for idx in used:
        c = candidates[idx - 1]
        out.append({
            "n": idx, "document_id": c.document_id, "doc_key": c.doc_key, "title": c.doc_title,
            "page_from": c.page_from, "page_to": c.page_to, "chunk_id": c.chunk_id,
            "url": f"/pdf/{c.document_id}#page={c.page_from}",
        })
    return out


def load_history(session_id: str, turns: int) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT %s",
            (session_id, turns),
        ).fetchall()
    return list(reversed(rows))


def save_trace(req: ChatRequest, res: ChatResult) -> int:
    r = res.retrieval
    with connection() as conn, conn.transaction():
        if req.persist:
            conn.execute("INSERT INTO messages (session_id, role, content) VALUES (%s, 'user', %s)",
                         (req.session_id, req.message))
            conn.execute("INSERT INTO messages (session_id, role, content) VALUES (%s, 'assistant', %s)",
                         (req.session_id, res.answer))
        return conn.execute(
            """INSERT INTO traces (session_id, question, intent, rewritten_query, query_articles,
                                   candidates, outcome, answer, citations, verification, usage,
                                   cost_usd, latency_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                req.session_id, req.message, json.dumps(res.intent, ensure_ascii=False),
                res.rewritten_query, res.query_articles,
                json.dumps({
                    "scoped": r.scoped, "widened": r.widened, "max_score": r.max_score,
                    "top": [c.brief() for c in r.candidates],
                    "pre_rerank": [c.brief() for c in r.pre_rerank],
                } if r else None, ensure_ascii=False),
                res.outcome, res.answer, json.dumps(res.citations, ensure_ascii=False),
                json.dumps(res.verification, ensure_ascii=False),
                json.dumps(res.usage.calls), res.usage.cost_usd, json.dumps(res.latency_ms),
            ),
        ).fetchone()["id"]


class Pipeline:
    def __init__(self, llm: LLM | None = None, threshold: float | None = None):
        self.llm = llm or get_llm()
        self.s = get_settings()
        self.threshold = self.s.rerank_threshold if threshold is None else threshold

    async def run(self, req: ChatRequest) -> AsyncIterator[dict]:
        res = ChatResult()
        t0 = time.perf_counter()

        def mark(name: str) -> None:
            res.latency_ms[name] = round((time.perf_counter() - t0) * 1000)

        def refuse(outcome: str, text: str, form: bool = True) -> dict:
            res.outcome, res.answer, res.show_feedback_form = outcome, text, form
            return {"event": "refusal", "data": {"outcome": outcome, "text": text,
                                                 "show_feedback_form": form}}

        try:
            async for ev in self._run(req, res, mark, refuse):
                yield ev
        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIStatusError) as e:
            # помилка могла статися посеред стріму: клієнт прибирає вже показаний текст
            yield {"event": "retract", "data": {}}
            yield refuse("error", REFUSAL_UNAVAILABLE)
            res.verification = {"error": f"{type(e).__name__}: {e}"}
        except RefusedByModel:
            yield {"event": "retract", "data": {}}
            yield refuse("refuse_no_info", REFUSAL_NO_INFO)

        mark("total")
        res.trace_id = await asyncio.to_thread(save_trace, req, res)
        yield {"event": "done", "data": {"trace_id": res.trace_id, "outcome": res.outcome},
               "result": res}

    async def _run(self, req: ChatRequest, res: ChatResult, mark, refuse) -> AsyncIterator[dict]:
        history = req.history
        if history is None:
            history = await asyncio.to_thread(load_history, req.session_id, self.s.history_turns)

        # 1-2. намір, небезпечна тема і переписування запиту одним викликом малої моделі
        product = " ".join(x for x in (req.product_manufacturer, req.product_article) if x) or None
        analysis = await self.llm.analyze(req.message, history[-self.s.history_turns:], res.usage,
                                          product=product)
        mark("analyze")
        res.intent = analysis.model_dump()
        res.rewritten_query = analysis.rewritten_query
        res.dangerous = analysis.intent == "technical" and is_dangerous(analysis, req.message)
        yield {"event": "meta", "data": {"intent": analysis.intent, "dangerous": res.dangerous}}

        if analysis.intent == "abuse":
            yield refuse("refuse_abuse", REFUSAL_ABUSE, form=False)
            return
        if analysis.intent == "out_of_scope":
            yield refuse("refuse_out_of_scope", REFUSAL_OUT_OF_SCOPE)
            return

        # 3-7. артикули, гібридний пошук з фільтром, rerank, поріг достатності
        query = analysis.rewritten_query
        res.query_articles = extract_articles(f"{req.message} {query}")
        r = await asyncio.to_thread(retrieve, query, res.query_articles, req.product_article,
                                    req.product_manufacturer, self.threshold)
        res.retrieval = r
        mark("retrieval")
        if r.max_score < self.threshold or not r.candidates:
            yield refuse("refuse_no_info", REFUSAL_NO_INFO)
            return
        context = [c for c in r.candidates if (c.rerank_score or 0) >= self.threshold * 0.5]
        res.context_ids = [c.chunk_id for c in context]

        # 8. генерація. Небезпечні теми буферизуються до кінця перевірки; інші стрімляться,
        # але перші символи притримуються, щоб не показати маркер «немає відповіді».
        user_content = build_user_content(req.message, query, context, r.widened)
        answer, streamed, hold = "", False, len(NO_ANSWER_MARKER) + 2
        async for text in self.llm.stream_answer(user_content, res.usage):
            answer += text
            if "first_token" not in res.latency_ms:
                mark("first_token")
            if res.dangerous:
                continue
            if not streamed and len(answer.lstrip()) >= hold:
                if answer.lstrip().startswith(NO_ANSWER_MARKER):
                    continue
                streamed = True
                yield {"event": "token", "data": {"text": answer}}
            elif streamed:
                yield {"event": "token", "data": {"text": text}}
        mark("generation")

        if NO_ANSWER_MARKER in answer:
            if streamed:
                yield {"event": "retract", "data": {}}
            yield refuse("refuse_no_info", REFUSAL_NO_INFO)
            return

        # 9. постперевірка чисел по процитованих фрагментах
        answer, ok = await self._verify_or_retry(answer, context, user_content, res)
        mark("verify")
        if not ok:
            if streamed:
                yield {"event": "retract", "data": {}}
            yield refuse("retracted", REFUSAL_NO_INFO)
            return

        final = answer.strip()
        if res.dangerous:  # небезпечні теми не стрімились, тож дисклеймер іде разом з текстом
            final = f"{final}\n\n{DANGEROUS_DISCLAIMER}"
        res.outcome, res.answer = "answer", final
        res.citations = extract_citations(final, context)

        if not streamed:
            yield {"event": "token", "data": {"text": final}}
        elif res.verification["retried"]:
            # стрімлена відповідь не пройшла перевірку, клієнт замінює її перегенерованою
            yield {"event": "retract", "data": {}}
            yield {"event": "token", "data": {"text": final}}
        yield {"event": "citations", "data": {"citations": res.citations,
                                              "from_other_product": r.widened}}

    async def _verify_or_retry(self, answer: str, context: list[Candidate], user_content: str,
                               res: ChatResult) -> tuple[str, bool]:
        def check(text: str):
            cited = extract_citations(text, context)
            ids = {c["chunk_id"] for c in cited}
            sources = [c.content for c in context if c.chunk_id in ids]
            v = verify_numbers(text, sources)
            return v, bool(cited)

        v, has_cites = check(answer)
        res.verification = {"ok": v.ok and has_cites, "unsupported": v.unsupported,
                            "has_citations": has_cites, "retried": False}
        if v.ok and has_cites:
            return answer, True

        numbers = ", ".join(v.unsupported) if v.unsupported else "відповідь без посилань [n]"
        retry_content = user_content + "\n\n" + STRICT_RETRY_NOTE.format(
            numbers=numbers, marker=NO_ANSWER_MARKER)
        retry = await self.llm.complete_answer(retry_content, res.usage)
        if NO_ANSWER_MARKER in retry:
            res.verification.update(retried=True, retry_ok=False)
            return retry, False
        v2, has_cites2 = check(retry)
        ok = v2.ok and has_cites2
        res.verification.update(retried=True, retry_ok=ok, retry_unsupported=v2.unsupported)
        return retry, ok
