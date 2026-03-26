#!/usr/bin/env python3
"""
Publish an audio file into a LiveKit room at realtime speed.

Joins the room as a fake "microphone" participant, streams the PCM audio
frame-by-frame, then disconnects once playout finishes. The agent running
in the same room picks up the audio via its STT pipeline.

Accepts any audio format supported by ffmpeg (wav, mp3, ogg, flac, m4a, etc.).
WAV and raw ulaw files are streamed at their native sample rate and channel
count. Everything else is converted to 16 kHz, 16-bit, mono PCM.

Usage:
    uv run publish_wav.py [OPTIONS] [AUDIO_FILE]

    AUDIO_FILE defaults to samples/audio_01_16kHz.wav when omitted.

Options:
    --room NAME        LiveKit room to join (default: test-room)
    --seconds N        Seconds to wait before streaming (default: 5)

Examples:
    uv run publish_wav.py
    uv run publish_wav.py --room my-room
    uv run publish_wav.py --room my-room samples/audio_01_16kHz.wav
    uv run publish_wav.py /path/to/recording.mp3

Requires LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env or
the environment. Requires ffmpeg on PATH for non-WAV formats.
"""

import argparse
import asyncio
import logging
import os
import warnings
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

# Suppress pydub regex SyntaxWarnings and livekit shutdown errors
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")
logging.getLogger("livekit").setLevel(logging.CRITICAL)

DEFAULT_WAV = Path(__file__).parent / "samples" / "audio_01_16kHz.wav"
DEFAULT_ROOM = "test-room"
IDENTITY = "wav-publisher"

# WebRTC sends audio in 20 ms frames
FRAME_DURATION_MS = 20
FRAMES_PER_SECOND = 1000 // FRAME_DURATION_MS

# Fallback format for non-WAV/ulaw files
FALLBACK_RATE = 16000
FALLBACK_CHANNELS = 1


def load_audio(file_path: str) -> tuple[bytes, int, int]:
    """Load an audio file and return (pcm_data, sample_rate, num_channels).

    WAV files are read natively. Everything else is converted via pydub/ffmpeg
    to 16 kHz, 16-bit, mono PCM.
    """
    ext = Path(file_path).suffix.lower()

    if ext in (".wav", ".wave"):
        with wave.open(file_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            pcm_data = wf.readframes(wf.getnframes())
        if sample_width != 2:
            raise ValueError(f"WAV must be 16-bit PCM, got {sample_width * 8}-bit")
        return pcm_data, sample_rate, num_channels

    # Everything else: convert via pydub (requires ffmpeg)
    from pydub import AudioSegment

    if ext == ".ulaw":
        audio = AudioSegment.from_file(file_path, codec="pcm_mulaw", sample_width=1, channels=1, frame_rate=8000)
    else:
        audio = AudioSegment.from_file(file_path)

    audio = audio.set_frame_rate(FALLBACK_RATE).set_channels(FALLBACK_CHANNELS).set_sample_width(2)
    return audio.raw_data, FALLBACK_RATE, FALLBACK_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an audio file into a LiveKit room at realtime speed."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=str(DEFAULT_WAV),
        help=f"Path to an audio file (default: {DEFAULT_WAV.name})",
    )
    parser.add_argument(
        "--room",
        default=DEFAULT_ROOM,
        help=f"LiveKit room name to join (default: {DEFAULT_ROOM})",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="Seconds to wait after connecting before streaming audio (default: 5)",
    )
    return parser.parse_args()


async def publish_audio(file_path: str, room_name: str, delay_seconds: int = 5):
    # -- load the audio file ---------------------------------------------------

    pcm_data, sample_rate, num_channels = load_audio(file_path)
    total_samples = len(pcm_data) // (num_channels * 2)
    duration = total_samples / sample_rate

    print(f"Audio: {file_path}")
    print(f"       {sample_rate} Hz, {num_channels}ch, {duration:.1f}s")

    # -- connect to the room --------------------------------------------------

    url = os.environ["LIVEKIT_URL"]
    token = (
        api.AccessToken()
        .with_identity(IDENTITY)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()
    await room.connect(url, token)
    print(f"Connected to room '{room_name}' as '{IDENTITY}'")

    # -- create an audio track that looks like a microphone --------------------
    #
    # The agent's RoomInputOptions only accepts SOURCE_MICROPHONE by default,
    # so we must tag the track accordingly.

    source = rtc.AudioSource(sample_rate, num_channels, queue_size_ms=1000)
    track = rtc.LocalAudioTrack.create_audio_track("audio-publish", source)

    publish_opts = rtc.TrackPublishOptions()
    publish_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, publish_opts)
    print("Publishing audio track...")

    # -- wait before streaming -------------------------------------------------
    if delay_seconds > 0:
        print(f"Waiting {delay_seconds}s before streaming...")
        await asyncio.sleep(delay_seconds)

    # -- stream pcm frames at realtime speed ----------------------------------
    #
    # capture_frame() feeds into an internal queue. When the queue is full the
    # await blocks, which naturally paces us to realtime without manual sleeps.
    # Each frame is also written to the local speakers via sounddevice.

    samples_per_frame = sample_rate // FRAMES_PER_SECOND
    bytes_per_frame = samples_per_frame * num_channels * 2  # 2 bytes per int16
    offset = 0
    frames_sent = 0

    # open a local playback stream on the default speakers
    playback_stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=num_channels,
        dtype="int16",
        blocksize=samples_per_frame,
    )
    playback_stream.start()

    try:
        while offset < len(pcm_data):
            chunk = pcm_data[offset : offset + bytes_per_frame]

            # pad the last chunk with silence if it's shorter than a full frame
            if len(chunk) < bytes_per_frame:
                chunk += b"\x00" * (bytes_per_frame - len(chunk))

            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=samples_per_frame,
            )
            await source.capture_frame(frame)

            # play the same chunk through the local speakers
            samples = np.frombuffer(chunk, dtype=np.int16).reshape(-1, num_channels)
            playback_stream.write(samples)

            offset += bytes_per_frame
            frames_sent += 1

            # progress update every 5 seconds
            if frames_sent % (FRAMES_PER_SECOND * 5) == 0:
                elapsed = frames_sent * FRAME_DURATION_MS / 1000
                print(f"  {elapsed:.0f}s / {duration:.0f}s sent")

        # -- finish up --------------------------------------------------------

        await source.wait_for_playout()
        print(f"Done -- published {duration:.1f}s of audio ({frames_sent} frames)")
        await asyncio.sleep(2)  # let the agent finish processing the tail end

    except (asyncio.CancelledError, KeyboardInterrupt):
        elapsed = frames_sent * FRAME_DURATION_MS / 1000
        print(f"\nInterrupted after {elapsed:.0f}s / {duration:.0f}s")

    finally:
        playback_stream.stop()
        playback_stream.close()
        await room.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(publish_audio(args.audio_file, args.room, args.seconds))
