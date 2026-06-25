from __future__ import annotations

from aiterate.providers.base import ModelProvider, ProviderError


class LiteLLMProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise ProviderError("Install aiterate[providers] to use LiteLLM.") from exc

        response = completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            api_base=self.config.base_url,
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else None,
            timeout=self.config.timeout_seconds,
            num_retries=self.config.max_retries,
        )
        return response["choices"][0]["message"]["content"]

