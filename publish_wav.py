#!/usr/bin/env python3
"""
Publish a WAV file into a LiveKit room at realtime speed.

Joins the room as a fake "microphone" participant, streams the PCM audio
frame-by-frame, then disconnects once playout finishes. The agent running
in the same room picks up the audio via its STT pipeline.

Usage:
    uv run publish_wav.py [OPTIONS] [AUDIO_FILE]

    AUDIO_FILE defaults to samples/audio_01_16kHz.wav when omitted.

Options:
    --room NAME    LiveKit room to join (default: test-room)

Examples:
    uv run publish_wav.py
    uv run publish_wav.py --room my-room
    uv run publish_wav.py --room my-room samples/audio_01_16kHz.wav
    uv run publish_wav.py /path/to/other.wav

Requires LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env or
the environment.
"""

import argparse
import asyncio
import os
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv()

DEFAULT_WAV = Path(__file__).parent / "samples" / "audio_01_16kHz.wav"
DEFAULT_ROOM = "test-room"
IDENTITY = "wav-publisher"

# WebRTC sends audio in 20 ms frames
FRAME_DURATION_MS = 20
FRAMES_PER_SECOND = 1000 // FRAME_DURATION_MS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a WAV file into a LiveKit room at realtime speed."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=str(DEFAULT_WAV),
        help=f"Path to a 16-bit PCM WAV file (default: {DEFAULT_WAV.name})",
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


async def publish_wav(wav_path: str, room_name: str, delay_seconds: int = 5):
    # -- read the wav file ----------------------------------------------------

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        total_frames = wf.getnframes()
        pcm_data = wf.readframes(total_frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM, got {sample_width * 8}-bit")

    duration = total_frames / sample_rate
    print(f"WAV: {wav_path}")
    print(f"     {sample_rate} Hz, {num_channels}ch, {duration:.1f}s")

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
    track = rtc.LocalAudioTrack.create_audio_track("wav-audio", source)

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

    # -- finish up ------------------------------------------------------------

    await source.wait_for_playout()
    playback_stream.stop()
    playback_stream.close()
    print(f"Done -- published {duration:.1f}s of audio ({frames_sent} frames)")

    await asyncio.sleep(2)  # let the agent finish processing the tail end
    await room.disconnect()
    print("Disconnected")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(publish_wav(args.audio_file, args.room, args.seconds))
