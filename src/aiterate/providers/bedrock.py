from __future__ import annotations

import json

from aiterate.providers.base import ModelProvider, ProviderError


class BedrockProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        try:
            import boto3
        except ImportError as exc:
            raise ProviderError("Install aiterate[providers] to use AWS Bedrock.") from exc

        session_kwargs = {}
        if self.config.profile:
            session_kwargs["profile_name"] = self.config.profile
        if self.config.region:
            session_kwargs["region_name"] = self.config.region
        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock-runtime")
        payload = {
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "system": [{"text": system}],
            "inferenceConfig": {"maxTokens": 1200, "temperature": 0.2},
        }
        response = client.converse(modelId=self.config.model, **payload)
        chunks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(chunk.get("text", "") for chunk in chunks)
        if text:
            return text
        return json.dumps(response, default=str)

