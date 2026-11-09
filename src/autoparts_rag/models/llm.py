"""Адаптер Anthropic. Решта коду не знає про SDK: отримує Pydantic-об'єкти, текст і Usage."""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import AsyncIterator, Literal

import anthropic
from pydantic import BaseModel, Field

from ..config import get_settings
from ..prompts import ANALYZE_SYSTEM, ANSWER_SYSTEM, CORRECT_SYSTEM, JUDGE_SYSTEM

# $ за 1 млн токенів: input, output, cache read, cache write (5 хв).
# Джерело: таблиця моделей у skill claude-api (кеш від 2026-10-06), промпти до 100K токенів.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Usage:
    calls: list[dict] = field(default_factory=list)

    def add(self, step: str, model: str, u) -> None:
        self.calls.append({
            "step": step, "model": model,
            "input": u.input_tokens or 0, "output": u.output_tokens or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
        })

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for c in self.calls:
            p_in, p_out, p_cr, p_cw = PRICES.get(c["model"], (0, 0, 0, 0))
            total += (c["input"] * p_in + c["output"] * p_out
                      + c["cache_read"] * p_cr + c["cache_write"] * p_cw) / 1e6
        return round(total, 6)


class QueryAnalysis(BaseModel):
    intent: Literal["technical", "out_of_scope", "abuse"]
    dangerous: bool
    rewritten_query: str = Field(description="самодостатнє питання українською")


class Claim(BaseModel):
    claim: str
    supported: bool


class JudgeResult(BaseModel):
    claims: list[Claim]


class CorrectVerdict(BaseModel):
    reason: str  # спершу пояснення, потім вердикт: так суддя рідше вгадує
    correct: bool


class RefusedByModel(Exception):
    """Модель зупинилася зі stop_reason=refusal."""


def _history_text(history: list[dict]) -> str:
    lines = [f"{'Покупець' if m['role'] == 'user' else 'Асистент'}: {m['content']}" for m in history]
    return "\n".join(lines)


class LLM:
    def __init__(self) -> None:
        s = get_settings()
        self.s = s
        self.client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key or None, max_retries=2)

    async def analyze(self, question: str, history: list[dict], usage: Usage,
                      product: str | None = None) -> QueryAnalysis:
        parts = []
        if product:
            # без цього «а датчик зносу тут є?» зі сторінки товару не перепишеться в повне питання
            parts.append(f"<product_page>{product}</product_page>")
        if history:
            parts.append(f"<history>\n{_history_text(history)}\n</history>")
        parts.append(f"<message>\n{question}\n</message>")
        content = "\n\n".join(parts)
        resp = await self.client.messages.parse(
            model=self.s.llm_fast_model,
            max_tokens=1024,
            system=[{"type": "text", "text": ANALYZE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": content}],
            output_format=QueryAnalysis,
        )
        usage.add("analyze", self.s.llm_fast_model, resp.usage)
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RefusedByModel(str(resp.stop_details))
        return resp.parsed_output

    def _answer_params(self, user_content: str) -> dict:
        return dict(
            model=self.s.llm_answer_model,
            max_tokens=4096,
            system=[{"type": "text", "text": ANSWER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": user_content}],
            betas=[FALLBACK_BETA],
            # серверний fallback: якщо класифікатор безпеки моделі відмовить, API сам
            # перезапустить запит на іншій моделі
            fallbacks="default",
        )

    async def stream_answer(self, user_content: str, usage: Usage) -> AsyncIterator[str]:
        params = self._answer_params(user_content)
        async with self.client.beta.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        usage.add("answer", self.s.llm_answer_model, final.usage)
        if final.stop_reason == "refusal":
            raise RefusedByModel(str(final.stop_details))

    async def complete_answer(self, user_content: str, usage: Usage, step: str = "answer_retry") -> str:
        resp = await self.client.beta.messages.create(**self._answer_params(user_content))
        usage.add(step, self.s.llm_answer_model, resp.usage)
        if resp.stop_reason == "refusal":
            raise RefusedByModel(str(resp.stop_details))
        return "".join(b.text for b in resp.content if b.type == "text")

    async def judge(self, question: str, answer: str, sources: str, usage: Usage) -> JudgeResult:
        resp = await self.client.messages.parse(
            model=self.s.llm_fast_model,
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": (
                f"<sources>\n{sources}\n</sources>\n\n<question>{question}</question>\n\n"
                f"<answer>\n{answer}\n</answer>")}],
            output_format=JudgeResult,
        )
        usage.add("judge", self.s.llm_fast_model, resp.usage)
        return resp.parsed_output or JudgeResult(claims=[])


    async def judge_correct(self, question: str, reference: str, answer: str, usage: Usage) -> CorrectVerdict:
        resp = await self.client.messages.parse(
            model=self.s.llm_fast_model,
            max_tokens=1024,
            system=CORRECT_SYSTEM,
            messages=[{"role": "user", "content": (
                f"<question>{question}</question>\n\n<reference>\n{reference}\n</reference>\n\n"
                f"<answer>\n{answer}\n</answer>")}],
            output_format=CorrectVerdict,
        )
        usage.add("judge_correct", self.s.llm_fast_model, resp.usage)
        return resp.parsed_output or CorrectVerdict(reason="суддя не повернув вердикт", correct=False)


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM()
