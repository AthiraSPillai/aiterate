from __future__ import annotations

from aiterate.domain import ProviderKind
from aiterate.providers.base import ModelProvider, ProviderError


class OpenAIProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError("Install aiterate[providers] to use OpenAI.") from exc

        client = OpenAI(
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else None,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        response = client.responses.create(
            model=self.config.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
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
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else None,
            azure_endpoint=self.config.base_url,
            api_version=self.config.api_version,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        response = client.responses.create(
            model=self.config.deployment or self.config.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.output_text


SUPPORTED_NATIVE = {ProviderKind.OPENAI, ProviderKind.AZURE_OPENAI}

