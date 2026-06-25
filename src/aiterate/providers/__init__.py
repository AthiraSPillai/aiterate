from aiterate.domain import ProviderConfig, ProviderKind
from aiterate.providers.anthropic_native import AnthropicProvider
from aiterate.providers.base import ModelProvider
from aiterate.providers.bedrock import BedrockProvider
from aiterate.providers.litellm_adapter import LiteLLMProvider
from aiterate.providers.mock import MockProvider
from aiterate.providers.openai_native import AzureOpenAIProvider, OpenAIProvider


def build_provider(config: ProviderConfig) -> ModelProvider:
    if config.kind == ProviderKind.OPENAI:
        return OpenAIProvider(config)
    if config.kind == ProviderKind.ANTHROPIC:
        return AnthropicProvider(config)
    if config.kind == ProviderKind.AZURE_OPENAI:
        return AzureOpenAIProvider(config)
    if config.kind == ProviderKind.AWS_BEDROCK:
        return BedrockProvider(config)
    if config.kind == ProviderKind.LITELLM:
        return LiteLLMProvider(config)
    return MockProvider(config)
