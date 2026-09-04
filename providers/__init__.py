from enum import Enum
from typing import Dict

__all__ = [
    "Provider",
    "ModelNames",
]

class Provider(Enum):
    OpenAI = "OpenAI"
    Anthropic = "Anthropic"
    Gemini = "Gemini"
    xAI = "xAI"

# Framework assumes only using single best LLM from each provider for now.
ModelNames: Dict[Provider, str] = {
    Provider.OpenAI: "gpt-6-astra",
    Provider.Anthropic: "claude-fable-5.1",
    Provider.Gemini: "gemini-3.8-flash",
    Provider.xAI: "grok-4.6",
}

def get_AgentNode_impl(provider: Provider) -> type:
    # Only import the specific AgentNode subtype when needed
    # to avoid unnecessary dependencies by apps that don't use all providers.
    if provider == Provider.Anthropic:
        from .anthropic import AnthropicAgentNode
        return AnthropicAgentNode
    elif provider == Provider.Gemini:
        from .gemini import GeminiAgentNode
        return GeminiAgentNode
    elif provider == Provider.OpenAI:
        raise NotImplementedError("todo: develop the OpenAIAgentNode subtype.")
    elif provider == Provider.xAI:
        raise NotImplementedError("todo: develop the XAIAgentNode subtype.")
    else:
        raise ValueError(f"Unknown provider: {provider}")
