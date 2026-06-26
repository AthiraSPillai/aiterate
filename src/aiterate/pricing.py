from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from aiterate.db import ModelPriceRecord, session_scope
from aiterate.domain import CostEstimate


PRICE_SOURCE_NOTE = "Approximate public list price. Verify provider billing before production use."

DEFAULT_MODEL_PRICES = [
    ("openai", "gpt-5.5", 5.00, 30.00, "https://developers.openai.com/api/docs/models"),
    ("openai", "gpt-5.4", 2.50, 15.00, "https://developers.openai.com/api/docs/models"),
    ("openai", "gpt-5.4-mini", 0.75, 4.50, "https://developers.openai.com/api/docs/models"),
    ("openai", "gpt-4.1", 2.00, 8.00, "https://openai.com/api/pricing/"),
    ("openai", "gpt-4.1-mini", 0.40, 1.60, "https://openai.com/api/pricing/"),
    ("openai", "gpt-4o", 2.50, 10.00, "https://openai.com/api/pricing/"),
    ("openai", "gpt-4o-mini", 0.15, 0.60, "https://openai.com/api/pricing/"),
    ("anthropic", "claude-3-5-sonnet-latest", 3.00, 15.00, "https://www.anthropic.com/pricing"),
    ("anthropic", "claude-3-5-haiku-latest", 0.80, 4.00, "https://www.anthropic.com/pricing"),
    ("anthropic", "claude-3-opus-latest", 15.00, 75.00, "https://www.anthropic.com/pricing"),
    ("aws_bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0", 3.00, 15.00, "https://aws.amazon.com/bedrock/pricing/"),
    ("aws_bedrock", "anthropic.claude-3-haiku-20240307-v1:0", 0.25, 1.25, "https://aws.amazon.com/bedrock/pricing/"),
    ("aws_bedrock", "amazon.nova-pro-v1:0", 0.80, 3.20, "https://aws.amazon.com/bedrock/pricing/"),
    ("aws_bedrock", "amazon.nova-lite-v1:0", 0.06, 0.24, "https://aws.amazon.com/bedrock/pricing/"),
    ("litellm", "anthropic/claude-3-5-sonnet-latest", 3.00, 15.00, "https://www.anthropic.com/pricing"),
    ("litellm", "gemini/gemini-1.5-pro", 1.25, 5.00, "https://ai.google.dev/gemini-api/docs/pricing"),
    ("litellm", "openai/gpt-4.1", 2.00, 8.00, "https://openai.com/api/pricing/"),
    ("litellm", "openai/gpt-5.5", 5.00, 30.00, "https://developers.openai.com/api/docs/models"),
    ("litellm", "openai/gpt-5.4", 2.50, 15.00, "https://developers.openai.com/api/docs/models"),
    ("litellm", "openai/gpt-5.4-mini", 0.75, 4.50, "https://developers.openai.com/api/docs/models"),
]


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model_id: str
    currency: str
    input_per_1m_tokens: float
    output_per_1m_tokens: float
    source: str
    notes: str


class RunCostMeter:
    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, system: str, user: str, output: str) -> None:
        self.input_tokens += estimate_tokens(system) + estimate_tokens(user)
        self.output_tokens += estimate_tokens(output)

    def estimate(self) -> CostEstimate:
        price = get_model_price(self.provider, self.model_id)
        input_cost = (self.input_tokens / 1_000_000) * price.input_per_1m_tokens
        output_cost = (self.output_tokens / 1_000_000) * price.output_per_1m_tokens
        return CostEstimate(
            provider=self.provider,
            model=self.model_id,
            currency=price.currency,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(input_cost + output_cost, 6),
            input_per_1m_tokens=price.input_per_1m_tokens,
            output_per_1m_tokens=price.output_per_1m_tokens,
            pricing_source=price.source,
            approximate=True,
            notes=price.notes,
        )


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text or "") / 4))


def list_model_prices() -> dict[str, list[dict]]:
    with session_scope() as session:
        _seed_default_prices(session)
        records = session.scalars(
            select(ModelPriceRecord)
            .where(ModelPriceRecord.enabled == 1)
            .order_by(ModelPriceRecord.provider, ModelPriceRecord.model_id)
        ).all()
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record.provider, []).append(_record_to_dict(record))
    return grouped


def get_model_price(provider: str, model_id: str) -> ModelPrice:
    with session_scope() as session:
        _seed_default_prices(session)
        record = _find_price_record(session, provider, model_id)
    if record:
        return ModelPrice(
            provider=record.provider,
            model_id=record.model_id,
            currency=record.currency,
            input_per_1m_tokens=record.input_per_1m_tokens,
            output_per_1m_tokens=record.output_per_1m_tokens,
            source=record.source,
            notes=record.notes,
        )
    return ModelPrice(
        provider=provider,
        model_id=model_id,
        currency="USD",
        input_per_1m_tokens=1.00,
        output_per_1m_tokens=4.00,
        source="",
        notes="No configured price found. Used conservative fallback estimate.",
    )


def _find_price_record(session: Session, provider: str, model_id: str) -> ModelPriceRecord | None:
    exact = session.get(ModelPriceRecord, _price_id(provider, model_id))
    if exact:
        return exact
    if provider == "litellm" and "/" in model_id:
        routed_provider, routed_model = model_id.split("/", 1)
        exact = session.get(ModelPriceRecord, _price_id(routed_provider, routed_model))
        if exact:
            return exact
    return session.scalars(
        select(ModelPriceRecord)
        .where(ModelPriceRecord.provider == provider, ModelPriceRecord.enabled == 1)
        .order_by(ModelPriceRecord.model_id)
        .limit(1)
    ).first()


def _seed_default_prices(session: Session) -> None:
    now = datetime.now(UTC)
    for provider, model_id, input_price, output_price, source in DEFAULT_MODEL_PRICES:
        session.merge(
            ModelPriceRecord(
                id=_price_id(provider, model_id),
                provider=provider,
                model_id=model_id,
                currency="USD",
                input_per_1m_tokens=input_price,
                output_per_1m_tokens=output_price,
                source=source,
                notes=PRICE_SOURCE_NOTE,
                updated_at=now,
            )
        )
    session.flush()


def _record_to_dict(record: ModelPriceRecord) -> dict:
    return {
        "provider": record.provider,
        "model": record.model_id,
        "currency": record.currency,
        "input_per_1m_tokens": record.input_per_1m_tokens,
        "output_per_1m_tokens": record.output_per_1m_tokens,
        "source": record.source,
        "notes": record.notes,
    }


def _price_id(provider: str, model_id: str) -> str:
    return f"{provider}:{model_id}"
