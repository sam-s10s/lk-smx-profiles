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
    RunContext,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import function_tool  # noqa: E402
from livekit.plugins import elevenlabs, openai, speechmatics, silero  # noqa: E402

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
        await self.session.generate_reply(instructions="Greet the user warmly and ask how you can help.")

    # @function_tool
    # async def echo(self, context: RunContext, text: str):
    #     """Simple function exposed to the LLM to repeat text back."""
    #     await self.session.say(f"You said: {text}")
    #     return {"ok": True}


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    vad = silero.VAD.load( 
        #min_speech_duration=0.03
        )

    stt = speechmatics.STT(
        language="en",
        # External endpointing: Silero controls end-of-speech, then we call `stt.finalize()`.
        turn_detection_mode=speechmatics.TurnDetectionMode.EXTERNAL,
        # turn_detection_mode=speechmatics.TurnDetectionMode.ADAPTIVE,
        

    )
    
    # Log STT events: metrics, errors, and all SpeechEventType values
    from livekit.agents.stt.stt import SpeechEventType

    for ev_name in ("metrics_collected", "error") + tuple(e.value for e in SpeechEventType):
        def _log(ev, _name=ev_name):
            logger.info(f"STT event {_name}: {ev}")

        stt.on(ev_name, _log)
    session = AgentSession(
        turn_detection="vad",
        vad=vad,
        stt=stt,
        llm=openai.LLM(model="gpt-4.1-nano"),
        tts=elevenlabs.TTS(voice_id="9BWtsMINqrJLrRacOk9x"),
        allow_interruptions=False,
        
    )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev):
        print(f"User state changed: {ev.old_state} -> {ev.new_state}")
        if ev.old_state == "speaking":# and ev.new_state == "listening":
            try:
                stt.finalize()
                
            except Exception:
                logger.exception("Failed to finalize Speechmatics STT turn")

    await ctx.wait_for_participant()

    await session.start(
        agent=SpeechmaticsAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="speechmatics-tester-ali"))

