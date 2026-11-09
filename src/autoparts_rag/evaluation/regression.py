"""Gate проти baseline (розділ 9): зупинити гіршу версію до релізу.

Частки порівнюємо з абсолютним допуском, час, гроші й токени з відносним, лічильники без допуску.
Окремо кожен кейс, що проходив у baseline і тепер не проходить, рахується як регресія."""

import json
from pathlib import Path

from ..config import ROOT

BASELINE = ROOT / "eval" / "baseline.json"

# Gate. Частки 0..1 порівнюємо з абсолютним допуском: 0.05 це приблизно 2 з 39 питань з відповіддю,
# але 1 з 8 питань без відповіді вже 0.125, тож там допуск не рятує від жодного нового провалу.
TOLERANCE = 0.05
# Час і гроші шумлять сильніше, тому допуск відносний: +25 % від baseline.
RELATIVE_TOLERANCE = 0.25
# Лічильники відповідей з числами без джерела: допуск нуль, кожна нова така відповідь блокує.
COUNTS = {"answers_with_unsupported_numbers", "dangerous_answers_with_unsupported_numbers",
          "answers_with_invalid_citations"}
LOWER_IS_BETTER = {"false_refusal", *COUNTS,
                   "latency_total_p50", "latency_total_p95", "latency_first_token_p50",
                   "latency_first_token_p95", "cost_usd_total", "cost_usd_per_question",
                   "tokens_in_per_question", "tokens_out_per_question"}
# Не порівнюємо: розмір набору і службові суми.
NOT_GATED = {"n", "judge_cost_usd", "cost_usd_all_runs"}


def gate(summary: dict, failed: list[str], baseline: dict, skip: tuple[str, ...] = ()) -> list[str]:
    """Порівняти метрики з baseline. Повертає список причин провалу; порожній список = gate пройдено.
    skip: префікси метрик, які не порівнюємо. У CI це latency: раннер GitHub повільніший за машину,
    на якій знято baseline, і час там нічого не каже про якість версії."""
    failures = []
    old_summary = baseline.get("summary", {})
    for key, old in old_summary.items():
        new = summary.get(key)
        if key in NOT_GATED or (skip and key.startswith(tuple(skip))):
            continue
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        if key in COUNTS:
            allowed = 0
        elif key.startswith(("latency", "cost", "tokens")):
            allowed = abs(old) * RELATIVE_TOLERANCE
        else:
            allowed = TOLERANCE
        if key in LOWER_IS_BETTER and new > old + allowed:
            failures.append(f"{key}: {new} > baseline {old} (допуск {round(allowed, 4)})")
        elif key not in LOWER_IS_BETTER and new < old - allowed:
            failures.append(f"{key}: {new} < baseline {old} (допуск {round(allowed, 4)})")
    old_failed = set(baseline.get("failed", []))
    failures += [f"регресія: {i}" for i in failed if i not in old_failed]
    return failures


def baseline_from_report(report: dict, source: str) -> dict:
    return {"source": source, "config": report["config"], "summary": report["summary"],
            "failed": report["failed"]}


def save_baseline(report_path: Path) -> None:
    report = json.loads(report_path.read_text())
    BASELINE.write_text(json.dumps(baseline_from_report(report, report_path.name), ensure_ascii=False, indent=2))
    print(f"baseline збережено: eval/baseline.json (зі звіту {report_path.name})")


def run_gate(report_path: Path, skip: tuple[str, ...] = ()) -> bool:
    """Надрукувати результат gate. True, якщо пройдено."""
    if not BASELINE.exists():
        raise SystemExit("немає eval/baseline.json: спершу rag gate --save-baseline на версії, якій довіряєте")
    report, baseline = json.loads(report_path.read_text()), json.loads(BASELINE.read_text())
    old_cfg, new_cfg = baseline["config"], report["config"]
    if old_cfg.get("n") != new_cfg.get("n"):
        print(f"увага: різний розмір набору (baseline n={old_cfg.get('n')}, зараз n={new_cfg.get('n')}), "
              "порівняння частин неповне")
    if old_cfg.get("repeat", 1) != new_cfg.get("repeat", 1):
        print(f"увага: різна кількість повторів (baseline {old_cfg.get('repeat', 1)}, зараз "
              f"{new_cfg.get('repeat', 1)}): flaky-кейси без повторів не видно")
    changed = {k: (old_cfg.get(k), v) for k, v in new_cfg.items()
               if k not in ("started", "questions") and old_cfg.get(k) != v}
    for k, (old, new) in changed.items():
        print(f"змінено {k}: {old} -> {new}")
    if skip:
        print(f"не порівнюємо: {', '.join(k for k in baseline['summary'] if k.startswith(tuple(skip)))}")
    failures = gate(report["summary"], report["failed"], baseline, tuple(skip))
    print(f"\nGATE {'НЕ ПРОЙДЕНО' if failures else 'ПРОЙДЕНО'}: {report_path.name} проти baseline {baseline['source']}")
    for line in failures:
        print(f"  {line}")
    return not failures
