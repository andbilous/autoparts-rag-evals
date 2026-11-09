"""Eval автозапчастин (ТЗ 8.2, метрики заняття 9). Пакет розбито за кроками:

dataset      завантаження тестсету і lint
case         прогін одного кейсу і метрики рядка (код і LLM-суддя)
rag_metrics  recall, in_context, цитати за документом і сторінкою
aggregate    зведення в метрики набору, поріг «не знаю»
stability    pass@1, «пройшли щоразу», flaky
regression   gate проти eval/baseline.json
report       markdown-звіт
runner       rag eval: усе разом

Що означає кожна метрика, описано в eval/METRICS.md."""

from ..config import ROOT
from .aggregate import summarize, threshold_sweep
from .case import REFUSALS, case_pass, run_item
from .dataset import lint, load_items
from .rag_metrics import recall, source_hit
from .regression import (BASELINE, COUNTS, LOWER_IS_BETTER, NOT_GATED, RELATIVE_TOLERANCE, TOLERANCE,
                         baseline_from_report, gate, run_gate, save_baseline)
from .report import latest_report, render
from .runner import main
from .stability import failed_ids, stability

__all__ = [
    "ROOT", "summarize", "threshold_sweep", "REFUSALS", "case_pass", "run_item", "lint", "load_items",
    "recall", "source_hit", "BASELINE", "COUNTS", "LOWER_IS_BETTER", "NOT_GATED", "RELATIVE_TOLERANCE",
    "TOLERANCE", "baseline_from_report", "gate", "run_gate", "save_baseline", "latest_report", "render",
    "main", "failed_ids", "stability",
]
