"""CLI: rag migrate | ingest | ask | eval | lint | gate | calibrate."""

import asyncio
import json
import uuid
from pathlib import Path

import typer

from .config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def migrate() -> None:
    """Створити схему БД."""
    from .db import migrate as run

    for name in run():
        typer.echo(f"applied {name}")


@app.command()
def ingest(directory: Path = typer.Argument(None), force: bool = False,
           chunk_chars: int = typer.Option(None, help="перевизначити chunk_max_chars")) -> None:
    """Проіндексувати PDF з директорії (за manifest.json). Незмінені файли пропускаються."""
    from .ingest.pipeline import ingest_dir

    kwargs = {"max_chars": chunk_chars} if chunk_chars else None
    for st in ingest_dir(directory or get_settings().data_dir, force=force, chunk_kwargs=kwargs):
        ocr = f", OCR стор.: {st.ocr_pages} (відхилено {st.ocr_rejected})" if st.ocr_pages else ""
        typer.echo(f"{st.status:9} {st.doc_key} v{st.version}: {st.pages} стор., {st.chunks} chunk{ocr}")


@app.command()
def ask(question: str, article: str = typer.Option(None), manufacturer: str = typer.Option(None),
        no_llm: bool = typer.Option(False, "--no-llm", help="лише пошук, без LLM"),
        session: str = typer.Option(None)) -> None:
    """Поставити питання з терміналу."""
    if no_llm:
        from .normalize import extract_articles
        from .retrieval import retrieve

        r = retrieve(question, extract_articles(question), article, manufacturer)
        typer.echo(f"scoped={r.scoped} widened={r.widened} max_score={r.max_score:.3f}")
        for i, c in enumerate(r.candidates, 1):
            typer.echo(f"\n[{i}] {c.rerank_score:.3f} {c.doc_key} с.{c.page_from}-{c.page_to} "
                       f"{c.chunk_type} ranks={c.ranks}\n{c.content[:400]}")
        return

    from .pipeline import ChatRequest, Pipeline

    async def run():
        req = ChatRequest(session or f"cli-{uuid.uuid4().hex[:8]}", question, article, manufacturer)
        async for ev in Pipeline().run(req):
            kind, data = ev["event"], ev["data"]
            if kind == "token":
                typer.echo(data["text"], nl=False)
            elif kind == "retract":
                typer.echo("\n[retract: попередній текст відкликано]")
            elif kind == "refusal":
                form = " [кнопка форми зворотного зв'язку]" if data["show_feedback_form"] else ""
                typer.echo(f"{data['text']}{form}")
            elif kind == "citations":
                typer.echo("\n\nДжерела:")
                for c in data["citations"]:
                    typer.echo(f"  [{c['n']}] {c['title']}, с. {c['page_from']}  {c['url']}")
            elif kind == "done":
                res = ev["result"]
                typer.echo(f"\n\n-- trace {data['trace_id']}, outcome={data['outcome']}, "
                           f"cost=${res.usage.cost_usd}, latency={json.dumps(res.latency_ms)}")

    asyncio.run(run())


@app.command("eval")
def run_eval(questions: Path = typer.Option(Path("eval/questions.jsonl")),
             threshold: float = typer.Option(None), limit: int = typer.Option(None),
             judge: bool = typer.Option(True, help="LLM-суддя для faithfulness"),
             concurrency: int = typer.Option(4), tag: str = typer.Option(""),
             repeat: int = typer.Option(1, min=1, help="прогнати кожне питання N разів: pass@1, flaky"),
             save_baseline: bool = typer.Option(False, "--save-baseline",
                                                help="записати цей звіт як eval/baseline.json"),
             gate: bool = typer.Option(False, "--gate", help="порівняти з baseline, exit 1 при регресії"),
             skip: list[str] = typer.Option([], "--skip", help="префікс метрик, які gate не порівнює (у CI: latency)")) -> None:
    """Прогнати eval-набір і записати звіт у eval/reports."""
    from .evaluation import main, run_gate
    from .evaluation import save_baseline as save

    path = asyncio.run(main(questions, threshold, limit, judge, concurrency, tag, repeat))
    if gate and not run_gate(path, tuple(skip)):
        raise typer.Exit(1)
    if save_baseline:
        save(path)


@app.command("lint")
def lint_cmd(questions: Path = typer.Option(Path("eval/questions.jsonl"))) -> None:
    """Перевірити тестсет без виклику моделі: джерела, сторінки, числа, еталони (exit 1, якщо є проблеми)."""
    from .evaluation import lint, load_items

    issues = lint(load_items(questions, None))
    for msg in issues:
        print(msg)
    print(f"{len(issues)} проблем" if issues else "тестсет чистий")
    if issues:
        raise typer.Exit(1)


@app.command("gate")
def run_gate_cmd(report: Path = typer.Argument(None, help="звіт eval/reports/*.json; за замовчуванням останній"),
                 save_baseline: bool = typer.Option(False, "--save-baseline",
                                                    help="не порівнювати, а записати звіт як baseline"),
                 skip: list[str] = typer.Option([], "--skip", help="префікс метрик, які не порівнюємо (у CI: latency)")) -> None:
    """Порівняти звіт eval з eval/baseline.json (exit 1 при регресії).

    Регресія: метрика погіршилась понад допуск або кейс, що проходив у baseline, тепер не проходить."""
    from .evaluation import latest_report, run_gate, save_baseline as save

    path = report or latest_report()
    if save_baseline:
        save(path)
        return
    if not run_gate(path, tuple(skip)):
        raise typer.Exit(1)


@app.command()
def calibrate(dump: bool = typer.Option(False, "--dump", help="дописати відповіді зі звіту в eval/labels.jsonl"),
              report: Path = typer.Option(None, help="звіт для --dump; за замовчуванням останній"),
              all_runs: bool = typer.Option(False, "--all-runs", help="для --dump: відповіді з усіх повторів")) -> None:
    """Порівняти LLM-суддю (faithfulness) з розміткою людини в eval/labels.jsonl."""
    from .calibration import dump as do_dump, run
    from .evaluation import latest_report

    if dump:
        do_dump(report or latest_report(), all_runs)
        return
    if not run():
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
