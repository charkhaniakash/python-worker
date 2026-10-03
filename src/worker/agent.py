
"""LiveKit `Agent` — a minimal wrapper that seeds the initial system prompt."""

from __future__ import annotations

from livekit.agents import Agent

from .config_builder import AgentConfig


class TelephonyAssistant(Agent):
    """
    Voice agent used by AgentSession. Instruction handling:
      • BharatGPT chat SSE owns the assistant persona server-side, so the agent
        instruction here is only a minimal voice-formatting rule.
      • For direct LLM (Gemini) we push the persona's full system prompt.
    """

    def __init__(self, config: AgentConfig, *, use_chat_backend: bool) -> None:
        if use_chat_backend:
            voice_formatting_rule = "You are a voice interface. Keep replies short and conversational."
            instructions = f"{voice_formatting_rule} {config.llm.system_prompt}"
        else:
            instructions = config.llm.system_prompt

        super().__init__(instructions=instructions)
        print(f"Agent initialized: {config.agent_name}", flush=True)
