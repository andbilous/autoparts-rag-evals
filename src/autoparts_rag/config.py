from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5444/rag"
    anthropic_api_key: str | None = None

    # моделі
    llm_answer_model: str = "claude-sonnet-5-5"
    llm_fast_model: str = "claude-haiku-5-5"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str | None = None  # None = автовибір (mps/cuda/cpu)

    # ingestion
    chunk_max_chars: int = 2400  # орієнтовно 600-800 токенів українською
    chunk_overlap_paragraphs: int = 1
    table_max_rows: int = 25
    ocr_min_text_chars: int = 40
    ocr_min_confidence: float = 60.0
    header_footer_ratio: float = 0.6

    # пошук
    fts_top_k: int = 30
    vector_top_k: int = 30
    rrf_k: int = 60
    rerank_top_n: int = 20
    context_top_k: int = 5
    rerank_threshold: float = 0.2  # калібрується на eval
    history_turns: int = 6

    data_dir: Path = ROOT / "data" / "pdfs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
