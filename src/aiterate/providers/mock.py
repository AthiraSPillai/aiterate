from __future__ import annotations

from aiterate.providers.base import ModelProvider


class MockProvider(ModelProvider):
    def generate(self, system: str, user: str) -> str:
        return (
            "You are an AI assistant optimized by AIterate.\n\n"
            "Follow the weighted policies exactly, cite available sources, and escalate when data is "
            "incomplete.\n\n"
            f"Optimization brief:\n{user[:1200]}"
        )

