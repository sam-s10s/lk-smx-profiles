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
    TurnHandlingOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import elevenlabs, openai, speechmatics  # noqa: E402

logger = logging.getLogger("speechmatics-tester")
logger.setLevel(logging.WARNING)

# Dedicated logger for our own output (replaces print statements)
out = logging.getLogger("out")
out.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
)
out.addHandler(_handler)
out.propagate = False


def log(tag: str, message: str):
    out.debug(f"{tag:<13} - {message}")


# Silence logging noise
for x in [
    "livekit",
    "livekit.agents",
    "livekit.plugins.speechmatics",
    "speechmatics",
    "asyncio",
]:
    logging.getLogger(x).setLevel(logging.ERROR)


class SpeechmaticsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly bot. Keep responses short, helpful, and polite. "
                "Answer user questions succinctly and ask a follow-up question when appropriate."
            ),
        )

    # async def on_enter(self):
    #     # greet when entering the room
    #     await self.session.generate_reply(
    #         instructions="Greet the user warmly and ask how you can help."
    #     )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # vad = silero.VAD.load( min_speech_duration=0.03)

    stt = speechmatics.STT(
        language="en",
        # External endpointing: Silero controls end-of-speech, then we call `stt.finalize()`.
        # turn_detection_mode=speechmatics.TurnDetectionMode.EXTERNAL,
        turn_detection_mode=speechmatics.TurnDetectionMode.ADAPTIVE,
        # end_of_utterance_silence_trigger=0.3,  # default is 0.5s, reduce to 0.3s for more aggressive endpointing
    )

    # Log STT events: metrics, errors, and all SpeechEventType values
    from livekit.agents.stt.stt import SpeechEventType

    for ev_name in ("metrics_collected", "error") + tuple(
        e.value for e in SpeechEventType
    ):

        def _log(ev, _name=ev_name):
            logger.info(f"STT event -> {_name}: {ev}")

        stt.on(ev_name, _log)

    # Create session (with new turn_handling)
    session = AgentSession(
        stt=stt,
        # llm=openai.LLM(model="gpt-4.1-nano"),
        # tts=elevenlabs.TTS(voice_id="9BWtsMINqrJLrRacOk9x"),
        turn_handling=TurnHandlingOptions(interruptions=True, turn_detection="stt"),
    )

    @session.on("user_input_transcribed")
    def on_transcription(ev):
        tag = "STT FINAL" if ev.is_final else "STT INTERIM"
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

    # -- start ----------------------------------------------------------------

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
