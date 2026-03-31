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
from livekit.plugins import elevenlabs, openai, speechmatics  # noqa: E402

logger = logging.getLogger("speechmatics-tester")
logger.setLevel(logging.INFO)


class SpeechmaticsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly bot. Keep responses short, helpful, and polite. "
                "Answer user questions succinctly and ask a follow-up question when appropriate."
            ),
        )

    async def on_enter(self):
        # greet when entering the room
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help."
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    stt = speechmatics.STT(
        language="en",
        turn_detection_mode=speechmatics.TurnDetectionMode.FIXED,
        end_of_utterance_silence_trigger=0.6,
    )

    # Log STT events: metrics, errors, and all SpeechEventType values
    from livekit.agents.stt.stt import SpeechEventType

    for ev_name in ("metrics_collected", "error") + tuple(
        e.value for e in SpeechEventType
    ):

        def _log(ev, _name=ev_name):
            logger.info(f"STT event {_name}: {ev}")

        stt.on(ev_name, _log)
    session = AgentSession(
        stt=stt,
        llm=openai.LLM(model="gpt-4.1-nano"),
        tts=elevenlabs.TTS(voice_id="9BWtsMINqrJLrRacOk9x"),
        turn_detection="stt",
        allow_interruptions=False,
    )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        print(f"User state changed: {ev.old_state} -> {ev.new_state}")

    await ctx.wait_for_participant()

    await session.start(
        agent=SpeechmaticsAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, agent_name="speechmatics-tester-ali")
    )
