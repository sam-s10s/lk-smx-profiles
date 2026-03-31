import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
os.environ.setdefault("NUM_CPUS", "2")

from livekit.agents import (  # noqa: E402
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import (
    deepgram,
    soniox,
    speechmatics,
)  # noqa: E402

logger = logging.getLogger("speechmatics-tester")
logger.setLevel(logging.WARNING)

# Dedicated logger for our own output (replaces print statements)
out = logging.getLogger("lk-smx-profiles")
out.setLevel(logging.DEBUG)

STT = os.getenv("STT", "smx")


def log(tag: str, message: str):
    out.debug(f"{tag:<13}  {message}")


for x in [
    "livekit",
    "livekit.agents",
    "livekit.plugins.speechmatics",
    "speechmatics",
    "asyncio",
]:
    logging.getLogger(x).setLevel(logging.ERROR)


class TestAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=("Nothing!"),
        )


async def entrypoint(ctx: JobContext):

    await ctx.connect()

    stt_providers = {
        "smx": speechmatics.STT(
            language="en",
            turn_detection_mode=speechmatics.TurnDetectionMode.FIXED,
            end_of_utterance_silence_trigger=0.9,
        ),
        "dg": deepgram.STT(),
        "soniox": soniox.STT(),
    }

    stt = stt_providers[STT]
    log("STT", f"Using {STT} provider")

    session = AgentSession(
        stt=stt,
        turn_detection="stt",
    )

    @session.on("user_input_transcribed")
    def on_transcription(ev):
        tag = "STT FINAL" if ev.is_final else "STT INTERIM"
        if tag == "STT FINAL":
            speaker = ev.speaker_id if ev.speaker_id else "UU"
            log(tag, f'{speaker} "{ev.transcript}"')

    @session.on("user_state_changed")
    def on_user_state(ev):
        log("USER STATE", f"{ev.old_state} -> {ev.new_state}")

    @session.on("agent_state_changed")
    def on_agent_state(ev):
        log("AGENT STATE", f"{ev.old_state} -> {ev.new_state}")

    @session.on("error")
    def on_error(ev):
        log("ERROR", str(ev.error))

    @session.on("close")
    def on_close(ev):
        log("SESSION CLOSE", f"reason={ev.reason}")

    await ctx.wait_for_participant()

    await session.start(
        agent=TestAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, agent_name="speechmatics-tester-sam")
    )
