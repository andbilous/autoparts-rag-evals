"""Зведення рядків прогону в метрики набору і таблиця порогу «не знаю» (розділи 4–5)."""

from .case import REFUSALS
from .stats import mean, pct

def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["expected_behavior"] == "answer"]
    no_info = [r for r in rows if r["expected_behavior"] == "refuse_no_info"]
    oos = [r for r in rows if r["expected_behavior"] == "refuse_out_of_scope"]
    answered = [r for r in rows if r["outcome"] == "answer"]
    dangerous_answered = [r for r in answered if r["expected_dangerous"]]

    def collect(key, subset):
        return [r[key] for r in subset if r.get(key) is not None]

    nums_dangerous = collect("numbers_found", dangerous_answered)
    return {
        "n": len(rows),
        "pass": mean([float(r["pass"]) for r in rows]),
        "recall@5": mean(collect("recall@5", answerable)),
        "recall@20_pre_rerank": mean(collect("recall@20_pre_rerank", answerable)),
        "citation_precision": mean(collect("citation_precision", answerable)),
        "in_context": mean(collect("in_context", answered)),
        "correct_judge": mean([float(r["correct_judge"]) for r in rows if r.get("correct_judge") is not None]),
        "faithfulness": mean(collect("faithfulness", answered)),
        "answers_with_invalid_citations": sum(bool(r.get("invalid_citations")) for r in answered),
        "numbers_found": mean(collect("numbers_found", answered)),
        "numbers_found_dangerous": mean(nums_dangerous),
        "answers_with_unsupported_numbers": sum(bool(r.get("numbers_unsupported")) for r in answered),
        "dangerous_answers_with_unsupported_numbers": sum(bool(r.get("numbers_unsupported")) for r in dangerous_answered),
        "correct_refusal_no_info": mean([float(r["outcome"] in REFUSALS) for r in no_info]),
        "correct_refusal_out_of_scope": mean([float(r["outcome"] == "refuse_out_of_scope") for r in oos]),
        "false_refusal": mean([float(r["outcome"] != "answer") for r in answerable]),
        "dangerous_flag_recall": mean([float(r["dangerous"]) for r in rows
                                       if r["expected_dangerous"] and r["outcome"] == "answer"]),
        "latency_total_p50": pct(collect_nested(rows, "total"), 0.5),
        "latency_total_p95": pct(collect_nested(rows, "total"), 0.95),
        "latency_first_token_p50": pct(collect_nested(rows, "first_token"), 0.5),
        "latency_first_token_p95": pct(collect_nested(rows, "first_token"), 0.95),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 4),
        "cost_usd_per_question": round(sum(r["cost_usd"] for r in rows) / max(1, len(rows)), 5),
        "tokens_in_per_question": round(sum(r.get("tokens_in", 0) for r in rows) / max(1, len(rows))),
        "tokens_out_per_question": round(sum(r.get("tokens_out", 0) for r in rows) / max(1, len(rows))),
        "judge_cost_usd": round(sum(r.get("judge_cost_usd", 0) for r in rows), 4),
    }


def collect_nested(rows: list[dict], key: str) -> list[float]:
    return [r["latency_ms"][key] for r in rows if key in r.get("latency_ms", {})]


def threshold_sweep(rows: list[dict]) -> list[dict]:
    """Що було б з відмовами при іншому порозі (лише за max rerank score, без LLM)."""
    out = []
    scored = [r for r in rows if r["max_score"] is not None]
    for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6):
        ans = [r for r in scored if r["expected_behavior"] == "answer"]
        noinfo = [r for r in scored if r["expected_behavior"] == "refuse_no_info"]
        out.append({
            "threshold": t,
            "false_refusal": mean([float(r["max_score"] < t) for r in ans]),
            "no_info_blocked_by_threshold": mean([float(r["max_score"] < t) for r in noinfo]),
        })
    return out
