from __future__ import annotations

import httpx

from aiterate.config import settings
from aiterate.domain import ProviderKind
from aiterate.providers.base import ModelProvider, ProviderError


class OpenAIProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install aiterate[providers] to use OpenAI.") from exc

        client = OpenAI(
            api_key=_api_key_value(self.config.api_key.get_secret_value() if self.config.api_key else None, "OpenAI"),
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            http_client=_http_client(),
        )
        try:
            response = client.responses.create(
                model=self.config.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenAI request failed: {_provider_error_message(exc)}") from exc
        return response.output_text


class AzureOpenAIProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ProviderError("Install aiterate[providers] to use Azure OpenAI.") from exc

        if not self.config.base_url:
            raise ProviderError("Azure OpenAI requires base_url/endpoint.")
        client = AzureOpenAI(
            api_key=_api_key_value(
                self.config.api_key.get_secret_value() if self.config.api_key else None,
                "Azure OpenAI",
            ),
            azure_endpoint=self.config.base_url,
            api_version=self.config.api_version,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            http_client=_http_client(),
        )
        try:
            response = client.responses.create(
                model=self.config.deployment or self.config.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Azure OpenAI request failed: {_provider_error_message(exc)}") from exc
        return response.output_text


SUPPORTED_NATIVE = {ProviderKind.OPENAI, ProviderKind.AZURE_OPENAI}


def _http_client() -> httpx.Client:
    return httpx.Client(trust_env=settings.trust_env_proxy)


def _api_key_value(value: str | None, provider_name: str) -> str:
    if not value or not value.strip():
        raise ProviderError(f"{provider_name} API key is missing. Add a saved credential or paste a key for this run.")
    cleaned = value.strip()
    placeholders = {
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "YOUR_API_KEY",
        "YOUR_OPENAI_API_KEY",
        "YOUR_AZURE_OPENAI_API_KEY",
    }
    if cleaned.upper() in placeholders or cleaned.upper().endswith("_API_KEY"):
        raise ProviderError(
            f"{provider_name} needs the actual API key value. Open credentials and paste the key value for this provider."
        )
    if any(character.isspace() for character in cleaned):
        raise ProviderError(f"{provider_name} API key contains whitespace. Remove spaces or line breaks and try again.")
    return cleaned


def _provider_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__
