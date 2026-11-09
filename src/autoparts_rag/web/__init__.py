"""Веб-інтерфейс: чат для користувача (/) і дашборд метрик для команди (/evals).

Дашборд читає те, що вже пишуть rag eval, rag gate і rag calibrate (eval/reports/*.json,
eval/baseline.json), і живі дані з таблиць traces та feedback. Нічого не рахує заново через LLM.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..db import connection
from ..evaluation import BASELINE, LOWER_IS_BETTER, ROOT, gate

STATIC = Path(__file__).parent / "static"
REPORTS = ROOT / "eval" / "reports"

router = APIRouter()


@router.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC / "chat.html")


@router.get("/evals", include_in_schema=False)
def evals_page() -> FileResponse:
    return FileResponse(STATIC / "evals.html")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


@router.get("/api/evals/reports")
def list_reports() -> dict:
    """Звіти eval від найновішого. Для кожного короткий підсумок, щоб обрати, який відкрити."""
    out = []
    for p in sorted(REPORTS.glob("2*.json"), reverse=True):
        r = _read(p)
        out.append({"name": p.name, "started": r["config"]["started"], "n": r["config"]["n"],
                    "repeat": r["config"].get("repeat", 1), "pass": r["summary"].get("pass"),
                    "answer_model": r["config"].get("answer_model")})
    calib = sorted(REPORTS.glob("calibration-*.json"), reverse=True)
    return {"reports": out, "baseline": BASELINE.exists(),
            "calibration": calib[0].name if calib else None}


@router.get("/api/evals/report/{name}")
def report(name: str) -> dict:
    path = REPORTS / name
    if path.parent != REPORTS or not path.is_file():
        raise HTTPException(404, "report not found")
    r = _read(path)
    result = {"report": r, "gate": None, "baseline": None, "lower_is_better": sorted(LOWER_IS_BETTER)}
    if BASELINE.exists():
        base = _read(BASELINE)
        result["baseline"] = {"source": base["source"], "summary": base["summary"], "failed": base["failed"]}
        result["gate"] = gate(r["summary"], r["failed"], base)
    return result


@router.get("/api/evals/calibration/{name}")
def calibration(name: str) -> dict:
    path = REPORTS / name
    if path.parent != REPORTS or not path.is_file() or not name.startswith("calibration-"):
        raise HTTPException(404, "calibration not found")
    c = _read(path)
    return {k: c[k] for k in ("judge", "numbers_code", "judge_cost_usd") if k in c}


@router.get("/api/online")
def online(days: int = 7) -> dict:
    """Метрики з реального трафіку: те, чого офлайн-набір не бачить. Прогони eval не пишуться
    в traces (persist=False), тож тут лише запити з чату й CLI."""
    days = max(1, min(days, 90))
    with connection() as conn:
        total = conn.execute(
            """SELECT count(*) AS n,
                      percentile_cont(0.5) WITHIN GROUP (ORDER BY (latency_ms->>'total')::float) AS p50,
                      percentile_cont(0.95) WITHIN GROUP (ORDER BY (latency_ms->>'total')::float) AS p95,
                      percentile_cont(0.5) WITHIN GROUP (ORDER BY (latency_ms->>'first_token')::float) AS ft50,
                      coalesce(sum(cost_usd), 0) AS cost,
                      count(*) FILTER (WHERE outcome = 'answer'
                                       AND verification->>'ok' = 'false') AS unverified
               FROM traces WHERE created_at > now() - make_interval(days => %s)""", (days,)).fetchone()
        outcomes = conn.execute(
            """SELECT outcome, count(*) AS n FROM traces
               WHERE created_at > now() - make_interval(days => %s) GROUP BY outcome ORDER BY n DESC""",
            (days,)).fetchall()
        fb = conn.execute(
            """SELECT count(*) AS n, count(*) FILTER (WHERE f.useful) AS useful
               FROM feedback f JOIN traces t ON t.id = f.trace_id
               WHERE t.created_at > now() - make_interval(days => %s)""", (days,)).fetchone()
        bad = conn.execute(
            """SELECT t.id, t.question, t.outcome, f.comment, f.created_at
               FROM feedback f JOIN traces t ON t.id = f.trace_id
               WHERE NOT f.useful AND t.created_at > now() - make_interval(days => %s)
               ORDER BY f.created_at DESC LIMIT 20""", (days,)).fetchall()
    n = total["n"]
    return {
        "days": days, "requests": n,
        "latency_total_p50": total["p50"], "latency_total_p95": total["p95"],
        "latency_first_token_p50": total["ft50"],
        "cost_usd_total": float(total["cost"]),
        "cost_usd_per_request": float(total["cost"]) / n if n else None,
        "answers_unverified": total["unverified"],
        "outcomes": {r["outcome"]: r["n"] for r in outcomes},
        "feedback_n": fb["n"], "useful_rate": fb["useful"] / fb["n"] if fb["n"] else None,
        "not_useful": [{**r, "created_at": r["created_at"].isoformat()} for r in bad],
    }
