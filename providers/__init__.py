from enum import Enum
from typing import Dict

class Provider(Enum):
    OpenAI = "OpenAI"
    Anthropic = "Anthropic"
    Gemini = "Gemini"
    xAI = "xAI"

# Framework assumes only using single best LLM from each provider for now.
ModelNames: Dict[Provider, str] = {
    Provider.OpenAI: "gpt-5-2025-08-07",
    Provider.Anthropic: "claude-opus-4-1-20250805",
    Provider.Gemini: "gemini-2.5-pro",
    Provider.xAI: "grok-4",
}

def get_AgentNode_impl(provider: Provider) -> type:
    from .anthropic import AnthropicAgentNode
    from .gemini import GeminiAgentNode

    if provider == Provider.Anthropic:
        return AnthropicAgentNode
    elif provider == Provider.Gemini:
        return GeminiAgentNode
    elif provider == Provider.OpenAI:
        raise NotImplementedError("todo: develop the OpenAIAgentNode subtype.")
    elif provider == Provider.xAI:
        raise NotImplementedError("todo: develop the xAIAgentNode subtype.")
    else:
        raise ValueError(f"Unknown provider: {provider}")
