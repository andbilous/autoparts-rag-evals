"""Юніт-тести без мережі, БД і моделей."""

import pymupdf as fitz
import pytest

from autoparts_rag.calibration import calibrate, label_items, merge_labels
from autoparts_rag.evaluation import case_pass, failed_ids, gate, stability
from autoparts_rag.ingest.chunk import chunk_elements, merge_cross_page_tables, table_to_markdown
from autoparts_rag.ingest.parse import Element, parse_pdf
from autoparts_rag.normalize import extract_articles, lemmatize, normalize_article, normalize_text
from autoparts_rag.retrieval import build_tsquery, detect_manufacturer, rrf_merge
from autoparts_rag.verify import extract_numbers, norm_number, verify_numbers


# ---------------------------------------------------------------- артикули

@pytest.mark.parametrize("raw,expected", [
    ("4 510 220 017", "4510220017"),
    ("4510220017", "4510220017"),
    ("fl 715/40", "FL71540"),
    ("FL715-40", "FL71540"),
    ("КВ-1415А", "KB1415A"),  # набрано кирилицею
    ("rt2041k1", "RT2041K1"),
])
def test_normalize_article(raw, expected):
    assert normalize_article(raw) == expected


def test_extract_articles_variants():
    text = "що входить у комплект rt2041k1 і чи підійде фільтр FL 715/40 до колодок 4 510 220 017?"
    assert set(extract_articles(text)) == {"4510220017", "FL71540", "RT2041K1"}


def test_extract_articles_does_not_split_hyphenated_article():
    assert extract_articles("болт KB-1215A і KB-1415C") == ["KB1215A", "KB1415C"]


def test_extract_articles_ignores_plain_numbers_and_threads():
    text = "Затягніть моментом 120 Н·м, різьба M20×1,5, до 650 °C, пробіг 120 000 км"
    assert extract_articles(text) == []


def test_normalize_text_quotes_and_apostrophes():
    assert normalize_text("з’єднання «так» тут") == "з'єднання \"так\" тут"


# ---------------------------------------------------------------- лематизація

def test_lemmatize_inflected_forms_match():
    a = set(lemmatize("колодок").split())
    b = set(lemmatize("колодки").split())
    assert a & b, (a, b)


def test_lemmatize_adds_normalized_article_token():
    assert "fl71540" in lemmatize("фільтр FL 715/40").split()


def test_build_tsquery_drops_stopwords():
    q = build_tsquery("чи є датчик зносу в колодках?")
    assert "'чи'" not in q and "'датчик'" in q and " | " in q


# ---------------------------------------------------------------- chunker

def test_table_markdown_and_cross_page_merge():
    els = [
        Element("heading", "Розміри", 1),
        Element("table", "", 1, header=["A", "B"], rows=[["1", "2"]]),
        Element("table", "", 2, header=["A", "B"], rows=[["3", "4"]]),
    ]
    merged = merge_cross_page_tables(els)
    assert len(merged) == 2
    table, last_page = merged[1]
    assert table.rows == [["1", "2"], ["3", "4"]] and last_page == 2
    assert table_to_markdown(["A", "B"], [["1", "2"]]) == "| A | B |\n|---|---|\n| 1 | 2 |"


def test_big_table_split_repeats_header():
    rows = [[str(i), f"v{i}"] for i in range(10)]
    els = [Element("table", "", 3, header=["Артикул", "Значення"], rows=rows)]
    chunks = chunk_elements(els, "Док", table_max_rows=4)
    assert len(chunks) == 3
    assert all("| Артикул | Значення |" in c.content for c in chunks)
    assert all(c.chunk_type == "table" and c.page_from == 3 for c in chunks)


def test_text_chunks_split_by_section_and_carry_title():
    els = [
        Element("heading", "1. Комплектація", 1),
        Element("para", "Абзац один.", 1),
        Element("heading", "2. Встановлення", 2),
        Element("para", "Абзац два.", 2),
        Element("caption", "Рис. 1. Схема.", 2),
    ]
    chunks = chunk_elements(els, "Інструкція")
    texts = [c for c in chunks if c.chunk_type == "text"]
    assert [c.section_title for c in texts] == ["1. Комплектація", "2. Встановлення"]
    assert texts[0].content.startswith("Документ: Інструкція\nРозділ: 1. Комплектація")
    cap = [c for c in chunks if c.chunk_type == "figure_caption"][0]
    assert "с. 2" in cap.content


def test_long_section_split_with_overlap():
    paras = [Element("para", f"Абзац {i} " + "x" * 90, 1) for i in range(6)]
    chunks = chunk_elements([Element("heading", "Р", 1), *paras], "Д", max_chars=250, overlap=1)
    assert len(chunks) > 1
    # останній абзац попереднього chunk повторюється на початку наступного
    assert chunks[0].body.split("\n\n")[-1] == chunks[1].body.split("\n\n")[0]


# ---------------------------------------------------------------- парсер

def test_parse_removes_repeated_headers_and_footers(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 30), "Hal Tech docs", fontsize=8)
        page.insert_text((72, 150), f"Body text of page {i + 1} with enough characters to skip OCR.", fontsize=11)
        page.insert_text((450, 830), f"Page {i + 1}", fontsize=8)
    doc.save(path)
    parsed = parse_pdf(str(path))
    texts = [e.text for e in parsed.elements]
    assert len(texts) == 3 and all(t.startswith("Body text") for t in texts)
    assert [e.page for e in parsed.elements] == [1, 2, 3]
    assert parsed.ocr_pages == []


# ---------------------------------------------------------------- RRF і виробник

def _row(i):
    return dict(chunk_id=i, document_id=1, doc_key="d", doc_title="t", page_from=1, page_to=1,
                section_title=None, chunk_type="text", content=str(i))


def test_rrf_prefers_items_in_several_lists():
    merged = rrf_merge({"fts": [_row(1), _row(2)], "vector": [_row(2), _row(3)]}, k=60)
    assert [c.chunk_id for c in merged][0] == 2
    assert merged[0].ranks == {"fts": 2, "vector": 1}


def test_detect_manufacturer_whole_word_only():
    ms = ["Гальмар", "Колесо-Профі"]
    assert detect_manufacturer("гарантія Гальмару на диски", ms) == "Гальмар"
    assert detect_manufacturer("болти колесо-профі", ms) == "Колесо-Профі"
    assert detect_manufacturer("гальмівні колодки скриплять", ms) is None


# ---------------------------------------------------------------- постперевірка

def test_norm_number():
    assert norm_number("2,0") == "2"
    assert norm_number("120 000") == "120000"
    assert norm_number("0,05") == "0.05"


def test_extract_numbers_skips_citations_and_page_refs():
    nums = extract_numbers("Момент 28 Н·м [1], див. с. 3 і розділ 4.")
    assert nums == [("28", "нм")]


def test_verify_accepts_numbers_from_sources():
    src = ["| Болт | 25 Н·м + 45° |", "Гранична товщина накладки | 2,0 мм"]
    v = verify_numbers("Затягніть болт моментом 25 Н·м + 45° [1]. Межа 2 мм [2].", src)
    assert v.ok, v.unsupported


def test_verify_rejects_invented_number():
    v = verify_numbers("Момент затягування 30 Н·м [1].", ["Момент затягування 28 Н·м."])
    assert not v.ok and v.unsupported == ["30 нм"]


def test_verify_rejects_wrong_unit():
    v = verify_numbers("Товщина 25 мм [1].", ["Болт натяжного ролика 25 Н·м."])
    assert not v.ok


def test_verify_accepts_unit_from_table_header():
    # у таблиці одиниця в шапці («Мінімальна товщина, мм»), а в клітинці голе число
    v = verify_numbers("Мінімальна товщина 22,4 мм [1].", ["| GD-3021 | 288 | 25,0 | 22,4 | 5 |"])
    assert v.ok


# ---------------------------------------------------------------- eval: pass, стабільність, gate, калібрування


def _eval_row(i="q1", exp="answer", out="answer", passed=None, **kw):
    row = {"id": i, "expected_behavior": exp, "outcome": out, **kw}
    if passed is not None:
        row["pass"] = passed
    return row


def test_case_pass_answer_needs_citation_and_all_numbers():
    assert case_pass(_eval_row(citation_hit=True, numbers_found=1.0))
    assert not case_pass(_eval_row(citation_hit=False, numbers_found=1.0))
    assert not case_pass(_eval_row(citation_hit=True, numbers_found=0.5))
    assert not case_pass(_eval_row(out="refuse_no_info"))  # хибна відмова


def test_case_pass_refusals():
    assert case_pass(_eval_row(exp="refuse_no_info", out="retracted"))
    assert case_pass(_eval_row(exp="refuse_out_of_scope", out="refuse_out_of_scope"))
    assert not case_pass(_eval_row(exp="refuse_out_of_scope", out="refuse_no_info"))
    assert not case_pass(_eval_row(exp="refuse_no_info", out="answer"))


def test_stability_pass_at_1_pass_all_and_flaky():
    runs = [[_eval_row("a", passed=True), _eval_row("b", passed=True), _eval_row("c", passed=False)],
            [_eval_row("a", passed=True), _eval_row("b", passed=False), _eval_row("c", passed=False)]]
    st = stability(runs)
    assert st["rates"] == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert st["pass_at_1"] == 0.5 and st["pass_all"] == 0.333 and st["flaky"] == ["b"]
    # при повторах flaky теж вважається непройденим
    assert failed_ids(runs[0], st) == ["b", "c"]
    assert failed_ids(runs[0]) == ["c"]


def test_gate_directions_tolerances_and_regressions():
    baseline = {"summary": {"n": 53, "recall@5": 0.9, "false_refusal": 0.05, "answers_with_unsupported_numbers": 0,
                            "latency_total_p95": 4000, "cost_usd_total": 0.2, "judge_cost_usd": 0.01},
                "failed": ["q07"]}
    same = {**baseline["summary"], "n": 40, "judge_cost_usd": 1.0}
    assert gate(same, ["q07"], baseline) == []
    within = {**baseline["summary"], "recall@5": 0.86, "false_refusal": 0.09, "latency_total_p95": 4900}
    assert gate(within, [], baseline) == []
    worse = {**baseline["summary"], "recall@5": 0.8, "false_refusal": 0.11, "answers_with_unsupported_numbers": 1,
             "latency_total_p95": 5100, "cost_usd_total": 0.3}
    failures = gate(worse, ["q07", "q12"], baseline)
    assert [f.split(":")[0] for f in failures] == [
        "recall@5", "false_refusal", "answers_with_unsupported_numbers", "latency_total_p95", "cost_usd_total",
        "регресія"]
    assert failures[-1] == "регресія: q12"
    # покращення в будь-який бік не блокує
    better = {**baseline["summary"], "recall@5": 1.0, "false_refusal": 0.0, "latency_total_p95": 1000}
    assert gate(better, [], baseline) == []


def test_calibrate_agreement_false_pass_false_fail():
    c = calibrate([True, True, False, False], [True, False, True, False], ["a", "b", "c", "d"])
    assert c == {"n": 4, "agreement": 0.5, "false_pass": 0.25, "false_fail": 0.25, "disagreements": ["b", "c"]}
    assert calibrate([], [], [])["agreement"] is None


def test_label_items_only_answers_with_judge_context_and_no_labels():
    chunks = {10: {"doc": "d", "pages": "2-2", "text": "Момент 28 Н·м."}, 11: {"doc": "d", "pages": "3-3", "text": "x"}}
    rows = [
        {"id": "q1", "run": 0, "outcome": "answer", "question": "?", "answer": "28 Н·м [1]",
         "context_chunk_ids": [10, 11], "citations": [{"chunk_id": 10}]},
        {"id": "q1", "run": 1, "outcome": "answer", "question": "?", "answer": "28 Н·м [1]",
         "context_chunk_ids": [10], "citations": [{"chunk_id": 10}]},
        {"id": "q2", "run": 0, "outcome": "refuse_no_info", "question": "?", "answer": "Не знайшов."},
    ]
    items = label_items(rows, chunks)
    assert [it["id"] for it in items] == ["q1", "q1#2"]
    assert items[0]["sources"] == [{"n": 1, "doc": "d", "pages": "2-2", "text": "Момент 28 Н·м.", "cited": True},
                                   {"n": 2, "doc": "d", "pages": "3-3", "text": "x", "cited": False}]
    assert all(it["faithful"] is None for it in items)  # мітки ставить людина, не код
    done = [{**items[0], "faithful": False, "note": "n"}]
    merged = merge_labels(done, items)
    assert [it["id"] for it in merged] == ["q1", "q1#2"] and merged[0]["faithful"] is False


def test_gate_skip_prefix_ignores_latency():
    base = {"summary": {"pass": 0.9, "latency_total_p95": 1000}, "failed": []}
    now = {"pass": 0.9, "latency_total_p95": 5000}
    assert gate(now, [], base)  # без skip час блокує
    assert gate(now, [], base, ("latency",)) == []
    assert gate({"pass": 0.5, "latency_total_p95": 5000}, [], base, ("latency",))  # якість усе одно блокує
