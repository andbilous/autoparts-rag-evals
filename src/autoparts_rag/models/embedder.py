"""Тонкий адаптер embedding. Код пошуку залежить лише від протоколу Embedder,
тому заміна bge-m3 на хмарну модель зачепить тільки цей файл."""

import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from ..config import get_settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def pick_device() -> str:
    configured = get_settings().device
    if configured:
        return configured
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._lock = threading.Lock()  # модель на MPS не можна викликати з кількох потоків одночасно

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self.model.encode(
                texts, batch_size=8, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, pick_device())
