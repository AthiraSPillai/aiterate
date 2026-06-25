from __future__ import annotations

from abc import ABC, abstractmethod

from aiterate.domain import ProviderConfig


class ModelProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Generate text from a model provider."""


class ProviderError(RuntimeError):
    pass

