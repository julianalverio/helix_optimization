"""Thin Anthropic Messages-API wrapper for the lead-optimization agent.

The agent driver maintains the conversation history and tool-result
plumbing; this module only handles authentication and the model-call
boundary. We use the streaming-free Messages API because per-turn
latency is dominated by the GPU forward pass on each tool execution,
not by token generation."""
from __future__ import annotations

import anthropic


class ClaudeClient:
    """Wraps `anthropic.Anthropic().messages.create` with the model and
    max-tokens-per-turn baked in. The Anthropic SDK reads `ANTHROPIC_API_KEY`
    from the environment by default."""

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 4096):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def send(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
