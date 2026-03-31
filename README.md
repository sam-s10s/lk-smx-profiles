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

## Running the agent (console)

Start the voice agent with LLM and TTS (`AGENT_MODE-1` will enable the LLM and TTS):

```sh
AGENT_MODE=1 uv run agent.py console
```

## Running the agent

Start the voice agent and connect it to a room (using `smx` for Speechmatics, `dg` for Deepgram, or `soniox` for Soniox):

```sh
STT=smx uv run agent.py connect --room smx-test-001
```

The agent will join the room, greet any participant that arrives, and respond to spoken questions using the Speechmatics STT pipeline.

## Sending audio from a WAV file

In a separate terminal, publish a WAV file into the same room:

```sh
uv run publish.py samples/bst_sample_1.wav --room smx-test-001
```

This joins the room as a fake microphone participant, streams the audio at realtime speed, and plays it through your local speakers so you can hear what's being sent.

By default it waits 5 seconds after connecting before it starts streaming, giving the agent time to settle in. You can change that:

```sh
uv run publish.py samples/bst_sample_1.wav --room smx-test-001 --seconds 1
```

Or skip the wait entirely with `--seconds 0`.

## Tips

- The WAV file must be 16-bit PCM. Sample rate and channel count are read from the file header automatically.
- To suppress library log noise, the agent script silences most loggers by default. Only `print()` output shows up in the terminal.
- You can also set `LOG_LEVEL=WARNING` in `.env` for a quieter experience across the board.
