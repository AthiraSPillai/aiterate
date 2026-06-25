from __future__ import annotations

from aiterate.providers.base import ModelProvider, ProviderError


class AnthropicProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ProviderError("Install aiterate[providers] to use Anthropic.") from exc

        client = Anthropic(
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else None,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )
        response = client.messages.create(
            model=self.config.model,
            system=system,
            max_tokens=1200,
            temperature=0.2,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
