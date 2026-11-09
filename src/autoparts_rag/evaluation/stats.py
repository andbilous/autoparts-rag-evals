"""Дрібні статистичні помічники: перцентиль і середнє з округленням."""

import statistics

def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[idx]


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None
