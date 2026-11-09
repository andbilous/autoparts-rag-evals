"""rag eval: прогнати набір (з повторами), порахувати метрики і записати звіт у eval/reports."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ..config import ROOT, get_settings
from ..pipeline import Pipeline
from .aggregate import summarize, threshold_sweep
from .case import run_item
from .dataset import lint, load_items
from .report import render
from .stability import failed_ids, stability

async def main(questions: Path, threshold: float | None, limit: int | None, judge: bool,
               concurrency: int, tag: str, repeat: int = 1) -> Path:
    s = get_settings()
    items = load_items(questions, limit)
    lint_issues = lint(items)
    for msg in lint_issues:
        print(f"lint: {msg}")
    pipe = Pipeline(threshold=threshold)
    sem = asyncio.Semaphore(concurrency)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def guarded(item, run):
        async with sem:
            row = await run_item(item, pipe, judge)
            row["run"] = run
            print(f"{row['id']:6} #{run + 1} {row['expected_behavior']:20} -> {row['outcome']:20} "
                  f"r@5={row.get('recall@5', '-')} pass={row['pass']}", flush=True)
            return row

    # усі повтори йдуть через один семафор; порядок рядків у кожному прогоні = порядок питань
    flat = await asyncio.gather(*(guarded(i, run) for run in range(repeat) for i in items))
    runs = [flat[k * len(items):(k + 1) * len(items)] for k in range(repeat)]
    rows = runs[0]
    summary = summarize(rows)
    stab = stability(runs) if repeat > 1 else None
    if stab:
        summary["pass_at_1"], summary["pass_all"] = stab["pass_at_1"], stab["pass_all"]
        summary["cost_usd_all_runs"] = round(sum(r["cost_usd"] for r in flat), 4)
    sweep = threshold_sweep(rows)
    config = {
        "started": started, "questions": str(questions), "n": len(items), "repeat": repeat,
        "answer_model": s.llm_answer_model, "fast_model": s.llm_fast_model,
        "embedding_model": s.embedding_model, "reranker_model": s.reranker_model,
        "rerank_threshold": pipe.threshold, "chunk_max_chars": s.chunk_max_chars,
        "context_top_k": s.context_top_k, "rerank_top_n": s.rerank_top_n,
    }
    out_dir = ROOT / "eval" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    report = {"config": config, "summary": summary, "failed": failed_ids(rows, stab), "lint": lint_issues,
              "stability": stab, "sweep": sweep, "rows": rows, "extra_runs": runs[1:]}
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out_dir / f"{stem}.md").write_text(render(rows, summary, sweep, config, stab))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if stab:
        print(f"flaky: {' '.join(stab['flaky']) or 'немає'}")
    print(f"не пройшли: {' '.join(report['failed']) or 'немає'}")
    print(f"\nзвіт: eval/reports/{stem}.md")
    return path
