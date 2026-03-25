# LiveKit Agent Profile Testing

A test harness for LiveKit voice agent profiles using Speechmatics for speech-to-text, OpenAI for the LLM, and ElevenLabs for text-to-speech. Includes a script for feeding WAV files into a room so you can test without a microphone.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A `.env` file in this directory with the following keys:

```
LIVEKIT_URL=wss://...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

OPENAI_API_KEY=...
ELEVEN_API_KEY=...

SPEECHMATICS_API_KEY=...
SPEECHMATICS_RT_URL=wss://...
```

## Setup

Install dependencies:

```sh
uv sync
```

That's it. Everything else is handled by `uv run`.

## Running the agent

Start the voice agent and connect it to a room:

```sh
uv run speechmatics_1.4.5_no_ext_vad.py connect --room sam-test-3
```

The agent will join the room, greet any participant that arrives, and respond to spoken questions using the Speechmatics STT pipeline.

## Sending audio from a WAV file

In a separate terminal, publish a WAV file into the same room:

```sh
uv run publish_wav.py samples/audio_01_16kHz.wav --room sam-test-3
```

This joins the room as a fake microphone participant, streams the audio at realtime speed, and plays it through your local speakers so you can hear what's being sent.

By default it waits 5 seconds after connecting before it starts streaming, giving the agent time to settle in. You can change that:

```sh
uv run publish_wav.py samples/audio_01_16kHz.wav --room sam-test-3 --seconds 10
```

Or skip the wait entirely with `--seconds 0`.

## Tips

- The WAV file must be 16-bit PCM. Sample rate and channel count are read from the file header automatically.
- To suppress library log noise, the agent script silences most loggers by default. Only `print()` output shows up in the terminal.
- You can also set `LOG_LEVEL=WARNING` in `.env` for a quieter experience across the board.
