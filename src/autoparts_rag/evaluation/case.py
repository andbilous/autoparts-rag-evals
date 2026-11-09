"""Оцінка одного кейсу: прогнати питання через пайплайн і порахувати метрики рядка.

Детерміновані метрики рахує код (числа, цитати, мітки, recall, in_context), недетерміновані
LLM-суддя (groundedness проти процитованих фрагментів, правильність проти еталона)."""

import time

from ..models.llm import Usage, get_llm
from ..pipeline import CITE_RE, ChatRequest, ChatResult, Pipeline
from ..verify import extract_numbers
from .rag_metrics import chunk_texts, chunk_texts_ordered, recall, source_hit

REFUSALS = {"refuse_no_info", "refuse_out_of_scope", "refuse_abuse", "retracted"}

async def run_item(item: dict, pipe: Pipeline, judge: bool) -> dict:
    req = ChatRequest(
        session_id=f"eval-{item['id']}", message=item["question"],
        product_article=item.get("product_article"), product_manufacturer=item.get("product_manufacturer"),
        history=item.get("history", []), persist=False,
    )
    res: ChatResult | None = None
    t0 = time.perf_counter()
    async for ev in pipe.run(req):
        if ev["event"] == "done":
            res = ev["result"]
    assert res is not None
    exp = item["expected"]
    sources = exp.get("sources", [])
    row = {
        "id": item["id"], "category": item["category"], "question": item["question"],
        "expected_behavior": exp["behavior"], "outcome": res.outcome, "answer": res.answer,
        "intent": res.intent, "rewritten_query": res.rewritten_query, "dangerous": res.dangerous,
        "expected_dangerous": exp.get("dangerous", False),
        "citations": res.citations, "verification": res.verification,
        "latency_ms": res.latency_ms, "cost_usd": res.usage.cost_usd,
        "wall_ms": round((time.perf_counter() - t0) * 1000),
        "max_score": res.retrieval.max_score if res.retrieval else None,
        "widened": res.retrieval.widened if res.retrieval else None,
    }
    if res.retrieval:
        # які фрагменти бачив суддя: потрібно для розмітки в rag calibrate --dump
        row["context_chunk_ids"] = [c.chunk_id for c in res.retrieval.candidates]
    row["context_ids"] = res.context_ids
    row["tokens_in"] = sum(c["input"] + c["cache_read"] + c["cache_write"] for c in res.usage.calls)
    row["tokens_out"] = sum(c["output"] for c in res.usage.calls)
    if res.retrieval and sources:
        row["recall@5"] = recall(res.retrieval.candidates, sources)
        row["recall@20_pre_rerank"] = recall(res.retrieval.pre_rerank, sources)
        if res.outcome == "answer":
            # in_context: потрібна сторінка серед фрагментів, які після порогу реально пішли в промпт
            ctx = [c for c in res.retrieval.candidates if c.chunk_id in set(res.context_ids)]
            row["in_context"] = recall(ctx, sources)

    if res.outcome == "answer":
        # формат: мітка [n], якої немає серед виданих фрагментів (extract_citations її мовчки відкидає)
        labels = [int(n) for m in CITE_RE.finditer(res.answer) for n in m.group(1).split(",")]
        row["invalid_citations"] = [n for n in labels if not 1 <= n <= len(res.context_ids)]
        if sources:
            hits = [source_hit(c["doc_key"], c["page_from"], c["page_to"], sources) for c in res.citations]
            row["citation_precision"] = sum(hits) / len(hits) if hits else 0.0
            row["citation_hit"] = any(hits)
        answer_nums = {n for n, _ in extract_numbers(res.answer)}
        expected_nums = [n for raw in exp.get("numbers", []) for n, _ in extract_numbers(raw)]
        if expected_nums:
            row["numbers_found"] = sum(n in answer_nums for n in expected_nums) / len(expected_nums)
        cited = chunk_texts([c["chunk_id"] for c in res.citations])
        src_nums = {n for t in cited for n, _ in extract_numbers(t)}
        row["numbers_unsupported"] = sorted(answer_nums - src_nums)
        if judge:
            usage = Usage()
            # groundedness (розділ 5.4 підготовки): твердження перевіряємо проти процитованих фрагментів,
            # а не всіх знайдених: користувач перевірятиме відповідь саме за посиланнями
            src = "\n\n".join(f"[{c['n']}] {t}" for c, t in
                               zip(res.citations, chunk_texts_ordered([c["chunk_id"] for c in res.citations])))
            j = await get_llm().judge(item["question"], res.answer, src, usage)
            if item.get("reference"):
                v = await get_llm().judge_correct(item["question"], item["reference"], res.answer, usage)
                row["correct_judge"] = v.correct
                row["correct_reason"] = v.reason
            row["judge_cost_usd"] = usage.cost_usd
            if j.claims:
                row["faithfulness"] = sum(c.supported for c in j.claims) / len(j.claims)
                row["unsupported_claims"] = [c.claim for c in j.claims if not c.supported]
    row["pass"] = case_pass(row)
    return row


def case_pass(row: dict) -> bool:
    """Кейс пройшов, якщо поведінка збіглася з очікуваною, а для відповіді ще й хоча б одна цитата
    веде на очікувану сторінку і всі очікувані числа є в тексті. Детерміновано, без судді:
    faithfulness рахується окремо, бо суддю спершу треба відкалібрувати (rag calibrate)."""
    if not behavior_ok(row):
        return False
    if row["expected_behavior"] != "answer":
        return True
    return row.get("citation_hit", True) and row.get("numbers_found", 1.0) == 1.0


def behavior_ok(r: dict) -> bool:
    exp, out = r["expected_behavior"], r["outcome"]
    if exp == "answer":
        return out == "answer"
    if exp == "refuse_out_of_scope":
        return out == "refuse_out_of_scope"
    return out in REFUSALS
