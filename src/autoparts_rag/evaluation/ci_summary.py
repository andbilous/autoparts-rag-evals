"""Markdown-звіт для Summary в GitHub Actions: те саме, що дашборд /evals, але текстом.

Gate, метрики з номерами розділів «Підготовки до заняття 9», хто рахує кожну, різниця з baseline
і кейси, що не пройшли. Нових викликів моделі немає: читаємо готовий JSON-звіт."""

import json
from pathlib import Path

from .regression import BASELINE, LOWER_IS_BETTER, gate

KIND = {"code": "код", "judge": "суддя", "measure": "вимір"}

# (ключ у summary, назва, розділ підготовки, хто рахує, формат)
AI = [
    ("pass", "Кейс пройшов", "4.2", "code", "frac"),
    ("numbers_found", "Правильність: числа з еталона", "4.2", "code", "frac"),
    ("correct_judge", "Правильність: суддя проти еталона", "4.2", "judge", "frac"),
    ("answers_with_invalid_citations", "Формат: невалідні мітки [n]", "4.3", "code", "int"),
    ("pass_at_1", "Стабільність: pass@1", "4.4", "code", "frac"),
    ("pass_all", "Стабільність: пройшли щоразу", "4.4", "code", "frac"),
    ("correct_refusal_no_info", "Abstain: немає в документах", "4.5", "code", "frac"),
    ("correct_refusal_out_of_scope", "Abstain: поза темою", "4.5", "code", "frac"),
    ("false_refusal", "False abstain", "4.5", "code", "frac"),
    ("cost_usd_per_question", "Ціна на питання", "4.6", "measure", "usd"),
    ("tokens_in_per_question", "Токени на питання (вхідні)", "4.6", "measure", "int"),
    ("latency_first_token_p50", "Перший токен, p50", "4.6", "measure", "ms"),
    ("latency_total_p95", "Повна відповідь, p95", "4.6", "measure", "ms"),
]
RAG = [
    ("recall@20_pre_rerank", "recall@20 до reranker", "5.1", "code", "frac"),
    ("recall@5", "recall@5", "5.1", "code", "frac"),
    ("in_context", "in_context", "5.2", "code", "frac"),
    ("citation_precision", "cited: точність цитат", "5.3", "code", "frac"),
    ("faithfulness", "Groundedness", "5.4", "judge", "frac"),
    ("answers_with_unsupported_numbers", "Числа без джерела", "5.4", "code", "int"),
    ("dangerous_answers_with_unsupported_numbers", "…у питаннях безпеки", "5.4", "code", "int"),
]


def fmt(v, f: str) -> str:
    if v is None:
        return "—"
    if f == "frac":
        return f"{v:.2f}"
    if f == "usd":
        return f"${v:.4f}"
    if f == "ms":
        return f"{v / 1000:.1f} с"
    return str(v)


def table(rows: list[tuple], summary: dict, base: dict | None, failed_keys: set[str],
          skip: tuple[str, ...]) -> list[str]:
    out = ["| Розділ | Метрика | Хто рахує | Зараз | Baseline | Gate |", "|---|---|---|---|---|---|"]
    for key, name, sec, kind, f in rows:
        if key not in summary and not (base and key in base):
            continue
        new, old = summary.get(key), (base or {}).get(key)
        arrow = " ↓" if key in LOWER_IS_BETTER else ""
        if skip and key.startswith(skip):
            status = "не порівнюємо"
        elif not base or old is None or new is None:
            status = "—"
        elif key in failed_keys:
            status = "❌ гірше за допуск"
        else:
            status = "✅"
        out.append(f"| {sec} | {name}{arrow} | {KIND[kind]} | {fmt(new, f)} | {fmt(old, f)} | {status} |")
    return out


def problem_cases(rows: list[dict]) -> list[str]:
    bad = [r for r in rows if not r["pass"] or r.get("numbers_unsupported") or r.get("correct_judge") is False
           or (r.get("faithfulness") is not None and r["faithfulness"] < 1)]
    out = [f"### Кейси з проблемами ({len(bad)})", ""]
    if not bad:
        return out + ["Немає."]
    out += ["| id | Питання | Очікували → вийшло | Що не так |", "|---|---|---|---|"]
    for r in bad:
        why = []
        if not r["pass"]:
            why.append("не пройшов")
        if r.get("correct_judge") is False:
            why.append("суддя: неправильно")
        if r.get("faithfulness") is not None and r["faithfulness"] < 1:
            why.append(f"groundedness {r['faithfulness']:.2f}")
        if r.get("numbers_unsupported"):
            why.append("числа без джерела: " + ", ".join(r["numbers_unsupported"]))
        q = r["question"].replace("|", "/")
        out.append(f"| {r['id']} | {q} | {r['expected_behavior']} → {r['outcome']} | {'; '.join(why)} |")
    return out


def render(report: dict, report_name: str, skip: tuple[str, ...] = ()) -> str:
    s, cfg = report["summary"], report["config"]
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    failures = gate(s, report["failed"], base, skip) if base else []
    failed_keys = {f.split(":")[0] for f in failures if not f.startswith("регресія")}
    regressions = [f.split(": ", 1)[1] for f in failures if f.startswith("регресія")]

    if base is None:
        lines = ["## ⚠️ Gate: немає eval/baseline.json", ""]
    elif failures:
        lines = [f"## ❌ Gate не пройдено проти baseline `{base['source']}`", ""]
        lines += [f"- `{f}`" for f in failures] + [""]
    else:
        lines = [f"## ✅ Gate пройдено проти baseline `{base['source']}`", ""]
    cost = s.get("cost_usd_all_runs", s.get("cost_usd_total"))
    lines += [f"{cfg['n']} питань × {cfg.get('repeat', 1)} · модель `{cfg['answer_model']}` · "
              f"поріг rerank {cfg['rerank_threshold']} · top-{cfg['context_top_k']} · "
              f"ціна прогону ${cost} · звіт `{report_name}`", ""]
    lint = report.get("lint")
    if lint is not None:
        lines += ["Lint тестсету (2.1): " + ("чисто" if not lint else f"{len(lint)} проблем: " + "; ".join(lint[:5])), ""]

    bs = base["summary"] if base else None
    lines += ["### Метрики AI-системи (розділ 4)", ""] + table(AI, s, bs, failed_keys, skip)
    lines += ["", "### Метрики RAG (розділ 5)", ""] + table(RAG, s, bs, failed_keys, skip)
    if cfg.get("repeat", 1) == 1:
        lines += ["", "Стабільність не рахувалась: один прогін. Повтори запускаються вручну, поле repeat у Run workflow."]
    elif report.get("stability"):
        lines += ["", f"Flaky (то проходять, то ні): {', '.join(report['stability']['flaky']) or 'немає'}"]
    if regressions:
        lines += ["", "Нові провали проти baseline: " + ", ".join(f"`{i}`" for i in regressions)]
    lines += [""] + problem_cases(report["rows"])
    lines += ["", "Повний звіт і JSON: артефакт `eval-report` унизу сторінки. Дашборд локально: "
              "`uv run uvicorn autoparts_rag.api:app`, сторінка `/evals`."]
    return "\n".join(lines)


def main(report_path: Path, skip: tuple[str, ...] = ()) -> str:
    return render(json.loads(report_path.read_text()), report_path.name, skip)
