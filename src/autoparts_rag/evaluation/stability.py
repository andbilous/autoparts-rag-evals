"""Стабільність між повторами (розділ 4.4): pass@1, «пройшли щоразу», flaky."""

import statistics

def stability(runs: list[list[dict]]) -> dict:
    """runs[i] = рядки i-го прогону. rates = частка прогонів, у яких кейс пройшов; pass_at_1 = середнє
    rates (шанс, що одна відповідь пройде); pass_all = частка кейсів, що пройшли в кожному прогоні;
    flaky = кейси, що то проходять, то ні."""
    ids = [r["id"] for r in runs[0]]
    passed = {i: [] for i in ids}
    for run in runs:
        for r in run:
            passed[r["id"]].append(bool(r["pass"]))
    rates = {i: sum(v) / len(v) for i, v in passed.items()}
    return {
        "runs": len(runs),
        "pass_at_1": round(statistics.mean(rates.values()), 3),
        "pass_all": round(sum(v == 1 for v in rates.values()) / len(rates), 3),
        "flaky": [i for i, v in rates.items() if 0 < v < 1],
        "rates": rates,
    }


def failed_ids(rows: list[dict], stab: dict | None = None) -> list[str]:
    """Кейси, що не пройшли. При повторах сюди потрапляє й flaky: користувач, якому випала погана
    відповідь, не знає, що двічі з трьох вона була б доброю."""
    if stab:
        return [i for i, v in stab["rates"].items() if v < 1]
    return [r["id"] for r in rows if not r["pass"]]
