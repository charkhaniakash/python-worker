from __future__ import annotations

import os

from dotenv import load_dotenv
from livekit.agents import WorkerOptions, cli

from .config_builder import apply_outbound_call_opening
from .entrypoint import entrypoint, prewarm
from .settings import get_settings


def cli_main() -> None:
    load_dotenv()  # populate os.environ from .env
    settings = get_settings()
    agent_name = settings.agent_name
    print(
        f"[BOOT] Registering worker as agent_name=\"{agent_name}\" "
        f"on {os.environ.get('LIVEKIT_URL', '(no LIVEKIT_URL)')} — "
        f"your backend dispatch MUST target this exact name, or no jobs will arrive.",
        flush=True,
    )
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=agent_name,
        )
    )


if __name__ == "__main__":
    cli_main()
