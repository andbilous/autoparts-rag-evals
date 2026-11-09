"""Калібрування LLM-судді (ТЗ 8.2): faithfulness судді проти розмітки людини в eval/labels.jsonl.

Порядок:
1. rag calibrate --dump: відповіді зі звіту eval потрапляють у eval/labels.jsonl з порожнім полем faithful.
2. Людина читає кожну відповідь і фрагменти, ставить faithful: true або false і коротко пише note.
3. rag calibrate: суддя заново оцінює ті самі відповіді, код рахує збіг з людиною.

Суддю викликаємо заново, а не беремо оцінку зі звіту: після зміни промпта судді калібрування
треба повторити на тій самій розмітці."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from .config import ROOT
from .db import connection
from .models.llm import Usage, get_llm
from .verify import verify_numbers

LABELS = ROOT / "eval" / "labels.jsonl"

HOW_TO_LABEL = """Що робити далі:
  1. Відкрийте eval/labels.jsonl. Один рядок = одна відповідь асистента і фрагменти, які бачив суддя.
  2. Для кожного рядка поставте "faithful": true, якщо КОЖНЕ твердження відповіді є в наведених
     фрагментах (sources), і false, якщо хоч одне не підтверджене. Правильність відповіді тут не
     оцінюємо: відповідь може бути неповною і все одно faithful.
  3. У "note" коротко напишіть, яке твердження не підтверджене. Це знадобиться при розборі розбіжностей.
  4. Розмічайте, не дивлячись на оцінку судді у звіті, інакше розмітка повторить суддю.
  5. Запустіть rag calibrate. Рядки з "faithful": null пропускаються."""


def calibrate(judged: list[bool], human: list[bool], ids: list[str]) -> dict:
    """judged[i] і human[i] = True, якщо відповідь i визнано faithful. false_pass: суддя пропустив
    відповідь, яку людина забракувала (найнебезпечніша помилка); false_fail: навпаки."""
    n = len(human)
    if not n:
        return {"n": 0, "agreement": None, "false_pass": None, "false_fail": None, "disagreements": []}
    return {
        "n": n,
        "agreement": round(sum(j == h for j, h in zip(judged, human)) / n, 3),
        "false_pass": round(sum(j and not h for j, h in zip(judged, human)) / n, 3),
        "false_fail": round(sum(h and not j for j, h in zip(judged, human)) / n, 3),
        "disagreements": [i for i, j, h in zip(ids, judged, human) if j != h],
    }


def label_items(rows: list[dict], chunks: dict[int, dict]) -> list[dict]:
    """Рядки звіту -> заготовки розмітки. Беремо лише відповіді (outcome=answer): у відмові немає
    тверджень, які можна перевірити. chunks: chunk_id -> {doc, pages, text}."""
    items = []
    for r in rows:
        if r["outcome"] != "answer":
            continue
        cited = {c["chunk_id"] for c in r.get("citations", [])}
        sources = [{"n": n, **chunks[cid], "cited": cid in cited}
                   for n, cid in enumerate(r.get("context_chunk_ids", []), 1) if cid in chunks]
        run = r.get("run", 0)
        items.append({
            "id": r["id"] if run == 0 else f"{r['id']}#{run + 1}",
            "case": r["id"], "question": r["question"], "answer": r["answer"], "sources": sources,
            "faithful": None, "note": "",
        })
    return items


def merge_labels(existing: list[dict], new: list[dict]) -> list[dict]:
    """Нові заготовки додаються, вже розмічені рядки лишаються як є."""
    have = {it["id"] for it in existing}
    return existing + [it for it in new if it["id"] not in have]


def read_labels(path: Path = LABELS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_labels(items: list[dict], path: Path = LABELS) -> None:
    path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))


def fetch_chunks(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    with connection() as conn:
        rows = conn.execute(
            """SELECT c.id, d.doc_key, c.page_from, c.page_to, c.content
               FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(%s)""",
            (ids,)).fetchall()
    return {r["id"]: {"doc": r["doc_key"], "pages": f"{r['page_from']}-{r['page_to']}", "text": r["content"]}
            for r in rows}


def dump(report_path: Path, all_runs: bool) -> None:
    report = json.loads(report_path.read_text())
    rows = report["rows"] + ([r for run in report.get("extra_runs", []) for r in run] if all_runs else [])
    if rows and "context_chunk_ids" not in rows[0]:
        raise SystemExit("у звіті немає context_chunk_ids: перезапустіть rag eval цією версією коду")
    chunks = fetch_chunks(sorted({cid for r in rows for cid in r.get("context_chunk_ids", [])}))
    existing = read_labels()
    items = merge_labels(existing, label_items(rows, chunks))
    write_labels(items)
    todo = sum(it["faithful"] is None for it in items)
    print(f"eval/labels.jsonl: {len(items)} відповідей ({len(items) - len(existing)} нових зі звіту "
          f"{report_path.name}), не розмічено {todo}.\n")
    print(HOW_TO_LABEL)


async def _judge_all(items: list[dict]) -> tuple[list[dict], float]:
    llm, usage = get_llm(), Usage()

    async def one(it):
        # як і в eval: суддя бачить лише процитовані фрагменти (розділ 5.4 підготовки)
        src = "\n\n".join(f"[{s['n']}] {s['text']}" for s in it["sources"] if s["cited"])
        return await llm.judge(it["question"], it["answer"], src, usage)

    return await asyncio.gather(*(one(it) for it in items)), usage.cost_usd


def run(path: Path = LABELS) -> bool:
    labelled = [it for it in read_labels(path) if it.get("faithful") is not None]
    if not labelled:
        print(f"у {path.relative_to(ROOT)} немає розмічених рядків.\n")
        print("Спершу: rag calibrate --dump (заготовки зі звіту eval), потім розмітка вручну.\n")
        print(HOW_TO_LABEL)
        return False

    verdicts, cost = asyncio.run(_judge_all(labelled))
    for it, v in zip(labelled, verdicts):
        it["judge_claims"] = len(v.claims)
        it["judge_unsupported"] = [c.claim for c in v.claims if not c.supported]
        it["judge_ok"] = not it["judge_unsupported"]
    # без тверджень суддя нічого не перевірив, тож у порівняння з людиною такий рядок не йде
    skipped = [it["id"] for it in labelled if not it["judge_claims"]]
    scored = [it for it in labelled if it["judge_claims"]]
    ids, human = [it["id"] for it in scored], [bool(it["faithful"]) for it in scored]

    # Детермінована перевірка для порівняння: усі числа відповіді є в процитованих фрагментах.
    # Вона бачить лише числа, тож пропускає вигадані твердження без чисел.
    code = [verify_numbers(it["answer"], [s["text"] for s in it["sources"] if s["cited"]]).ok for it in scored]
    res = {"judge": calibrate([it["judge_ok"] for it in scored], human, ids),
           "numbers_code": calibrate(code, human, ids)}

    print(f"{len(scored)} відповідей розмічено людиною, суддя ${cost:.4f}")
    if skipped:
        print(f"суддя не виділив тверджень, пропущено: {' '.join(skipped)}")
    print(f"людина: faithful {sum(human)}, не faithful {len(human) - sum(human)}\n")
    print(f"{'':<28}{'agreement':>10}{'false pass':>12}{'false fail':>12}")
    names = {"judge": "суддя (faithfulness)", "numbers_code": "код (лише числа)"}
    for key, c in res.items():
        print(f"{names[key]:<28}{c['agreement']:>10}{c['false_pass']:>12}{c['false_fail']:>12}")
    by_id = {it["id"]: it for it in scored}
    for i in res["judge"]["disagreements"]:
        it = by_id[i]
        print(f"\n{i}: людина faithful={it['faithful']}, суддя faithful={it['judge_ok']}")
        print(f"  відповідь: {it['answer'][:160]!r}")
        print(f"  людина: {it.get('note') or '-'}")
        print(f"  суддя, не підтверджено: {'; '.join(it['judge_unsupported']) or 'усе підтверджено'}")

    out = ROOT / "eval" / "reports" / f"calibration-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "judge_cost_usd": cost, "skipped": skipped, "items": labelled},
                              ensure_ascii=False, indent=2))
    print(f"\nдеталі: eval/reports/{out.name}")
    return True
