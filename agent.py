"""
LiveKit voice agent with swappable STT providers.

Joins a LiveKit room and transcribes speech via Speechmatics, Deepgram, or
Soniox. With AGENT_MODE on it also runs an LLM and TTS so it can hold an
actual conversation rather than just printing transcripts.

Environment variables:
    AGENT_MODE  set to "1" to enable LLM + TTS (default: transcribe-only)
    STT         which provider to use: "smx" | "dg" | "soniox" (default: "smx")
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else touches os.environ
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
os.environ.setdefault("NUM_CPUS", "2")

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import (
    deepgram,
    elevenlabs,
    openai,
    silero,
    soniox,
    speechmatics,
)

# ── Configuration ──────────────────────────────────────────────────────────────

AGENT_MODE = os.getenv("AGENT_MODE", "0") == "1"
STT = os.getenv("STT", "smx")

# ── Logging ────────────────────────────────────────────────────────────────────
#
# LiveKit and its plugins are incredibly noisy at INFO level. This squashes
# everything we don't own down to ERROR, then sets up a separate logger for
# our own output so it doesn't get buried.

_NOISY_LOGGERS = (
    "livekit",
    "livekit.agents",
    "livekit.plugins.speechmatics",
    "speechmatics",
    "asyncio",
)
for name in _NOISY_LOGGERS:
    logging.getLogger(name).setLevel(logging.ERROR)

# Separate logger for our stuff, unaffected by the blanket silencing above
log = logging.getLogger("boost-agent")
log.setLevel(logging.DEBUG)


def _log(tag: str, message: str):
    """Fixed-width tag + message so the console output lines up nicely."""
    log.debug(f"{tag:<13}  {message}")


# ── Agent ──────────────────────────────────────────────────────────────────────


class SpeechmaticsAgent(Agent):
    """Thin wrapper around Agent that adds a greeting on room entry."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly bot. Keep responses short, helpful, and polite. "
                "Answer user questions succinctly and ask a follow-up question when appropriate."
            ),
        )

    async def on_enter(self):
        """Say hello when we join the room. Only fires in agent mode."""
        if AGENT_MODE:
            await self.session.generate_reply(
                instructions="Greet the user warmly and ask how you can help."
            )


# ── Entrypoint ─────────────────────────────────────────────────────────────────


async def entrypoint(ctx: JobContext):
    """Set up the STT/LLM/TTS pipeline, hook up event logging, and go."""

    await ctx.connect()

    # Low min_speech_duration so it picks up short utterances (the default is too high)
    vad = silero.VAD.load()  # min_silence_duration=0.15, min_speech_duration=0.15)

    # Lazy-init so we only connect to the provider we're actually using
    stt_providers = {
        "smx": lambda: speechmatics.STT(
            language="en",
            turn_detection_mode=speechmatics.TurnDetectionMode.EXTERNAL,
        ),
        "dg": lambda: deepgram.STT(),
        "soniox": lambda: soniox.STT(),
    }
    stt = stt_providers[STT]()
    _log("STT", f"Using {STT} provider")

    llm = openai.LLM(model="gpt-4.1-nano")
    tts = elevenlabs.TTS(voice_id="9BWtsMINqrJLrRacOk9x")

    # Without AGENT_MODE the LLM and TTS are left out (transcribe-only)
    session = AgentSession(
        stt=stt,
        llm=llm if AGENT_MODE else NOT_GIVEN,
        tts=tts if AGENT_MODE else NOT_GIVEN,
        turn_detection="vad",
        vad=vad,
    )

    # ── Session event handlers ─────────────────────────────────────────────

    @session.on("user_input_transcribed")
    def on_transcription(ev):
        # Interims are too noisy, only log finals
        if ev.is_final:
            speaker = ev.speaker_id or "UU"
            _log("STT FINAL", f'{speaker} "{ev.transcript}"')

    @session.on("user_state_changed")
    def on_user_state(ev):
        _log("USER STATE", f"{ev.old_state} -> {ev.new_state}")
        # Flush the STT buffer as soon as the user stops talking
        if ev.old_state == "speaking" and ev.new_state == "listening":
            stt.finalize()

    @session.on("agent_state_changed")
    def on_agent_state(ev):
        _log("AGENT STATE", f"{ev.old_state} -> {ev.new_state}")

    @session.on("error")
    def on_error(ev):
        _log("ERROR", str(ev.error))

    @session.on("close")
    def on_close(ev):
        _log("CLOSE", f"reason={ev.reason}")

    # ── Go ─────────────────────────────────────────────────────────────────

    await ctx.wait_for_participant()

    await session.start(
        agent=SpeechmaticsAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, agent_name="speechmatics-tester-sam")
    )
