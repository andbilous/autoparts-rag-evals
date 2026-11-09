"""Адаптер reranker (cross-encoder). Повертає бали в [0, 1]."""

import threading
from functools import lru_cache
from typing import Protocol

from ..config import get_settings
from .embedder import pick_device


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import CrossEncoder

        # для моделі з одним виходом CrossEncoder за замовчуванням застосовує sigmoid
        self.model = CrossEncoder(model_name, device=device, max_length=1024)
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            scores = self.model.predict([(query, p) for p in passages], batch_size=8,
                                        show_progress_bar=False)
        return [float(x) for x in scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    s = get_settings()
    return CrossEncoderReranker(s.reranker_model, pick_device())
