"""Звіт прогону: markdown для людини і пошук останнього JSON-звіту."""

import json
from pathlib import Path

from ..config import ROOT
from .case import behavior_ok
from .stats import mean

def render(rows: list[dict], summary: dict, sweep: list[dict], config: dict, stab: dict | None = None) -> str:
    lines = [f"# Eval {config['started']}", "", "## Конфігурація", "", "```json",
             json.dumps(config, ensure_ascii=False, indent=2), "```", "", "## Метрики", "",
             "| Метрика | Значення |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    if stab:
        lines += ["", f"## Стабільність ({stab['runs']} прогони)", "",
                  f"pass@1 {stab['pass_at_1']}, пройшли щоразу {stab['pass_all']}. "
                  "Решта метрик вище з першого прогону.", ""]
        lines += [f"- {i}: пройшов у {round(stab['rates'][i] * stab['runs'])} з {stab['runs']}"
                  for i in stab["flaky"]] or ["Flaky кейсів немає."]
    lines += ["", "## За категоріями", "",
              "| Категорія | n | outcome = очікуваному | pass | recall@5 |", "|---|---|---|---|---|"]
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        ok = mean([float(behavior_ok(r)) for r in sub])
        lines.append(f"| {cat} | {len(sub)} | {ok} | {mean([float(r['pass']) for r in sub])} "
                     f"| {mean([r['recall@5'] for r in sub if 'recall@5' in r])} |")
    lines += ["", "## Поріг rerank (лише за max score)", "",
              "| Поріг | Хибні відмови | Питання без відповіді, відсічені порогом |", "|---|---|---|"]
    lines += [f"| {s['threshold']} | {s['false_refusal']} | {s['no_info_blocked_by_threshold']} |" for s in sweep]
    lines += ["", "## Проблемні випадки", ""]
    for r in rows:
        problems = []
        if not behavior_ok(r):
            problems.append(f"очікувано {r['expected_behavior']}, отримано {r['outcome']}")
        if stab and r["id"] in stab["flaky"]:
            problems.append(f"flaky: пройшов у {round(stab['rates'][r['id']] * stab['runs'])} з {stab['runs']}")
        if r.get("recall@5", 1) < 1:
            problems.append(f"recall@5={r['recall@5']}")
        if r.get("citation_precision", 1) < 1:
            problems.append(f"citation_precision={r['citation_precision']:.2f}")
        if r.get("numbers_found", 1) < 1:
            problems.append(f"numbers_found={r['numbers_found']:.2f}")
        if r.get("numbers_unsupported"):
            problems.append(f"числа без джерела: {r['numbers_unsupported']}")
        if r.get("faithfulness", 1) < 1:
            problems.append(f"faithfulness={r['faithfulness']:.2f}: {r.get('unsupported_claims')}")
        if problems:
            lines += [f"### {r['id']} ({r['category']})", "", f"Питання: {r['question']}", "",
                      f"Переписаний запит: {r['rewritten_query']}", "",
                      "- " + "\n- ".join(problems), "", "Відповідь:", "", "> " + r["answer"].replace("\n", "\n> "), ""]
    return "\n".join(lines)


def latest_report() -> Path:
    reports = sorted((ROOT / "eval" / "reports").glob("2*.json"))
    if not reports:
        raise SystemExit("в eval/reports немає звітів: спершу запустіть rag eval")
    return reports[-1]
