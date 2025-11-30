## LiveKit Voice Intelligence Agent 
# Note: This repository is forked from https://github.com/Dark-Sys-Jenkins/agents-assignment and all the base codes are completely owned by the actual owner. I have just made the required changes in examples/voice_agents/basic_agent.py

<!--BEGIN_DESCRIPTION-->
A modular, real-time conversational AI agent built using LiveKit, OpenAI, and Python.

This repository contains a complete implementation of a voice-enabled conversational agent capable of real-time speech interaction, tool calling, structured reasoning, and multi-modal extension.
<!--END_DESCRIPTION-->

## Features

- **Real-time Voice Interaction**: Low-latency microphone streaming, Live transcription + Dynamic TTS, Natural turn-taking with interruption handling
- **Tool Calling Capabilities**: Define tools using Python functions, Structured function outputs, Run background tasks, Interrupt long-running actions
- **Extensible Architecture**: Add external APIs (weather, search, email), Plug-and-play nodes (TTS, ASR, LLM, Turn Detection), Build custom pipelines

## 📁 Project Structure
.
├── basic_agent.py
├── tools/
│   ├── weather_tool.py
│   ├── email_tool.py
│   └── ...
├── pipeline/
│   ├── llm_node.py
│   ├── tts_control.py
│   └── turn_detector.py
├── utils/
│   ├── audio.py
│   └── helpers.py
└── README.md

## Installation

# 1) Installing Dependencies

```bash
pip install "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.0"
```
```bash
pip install -r requirements.txt
```

# 2) Create a .env File

```bash
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
OPENAI_API_KEY=your_key
```

# 3) Run the Agent

```bash
python basic_agent.py dev
```

## How it Works

#🔹 1. Audio Input

LiveKit captures your voice → streams to ASR (OpenAI/Whisper).

#🔹 2. Language Model

User intent is processed by a realtime LLM with:

function calling

structured outputs

interruption support

#🔹 3. Tool Execution

Depending on user queries, the agent may:

fetch weather

send an email

retrieve information

generate content

#🔹 4. Voice Response

TTS converts LLM output → real-time audio output.

## Add your own Tools

Create a new file inside the tools/ directory:
```bash
from livekit.agents import llm

@llm.tool()
def get_time(city: str):
    """Returns local time in a given city."""
    ...
```
Then register it in basic_agent.py.

## Go to the livekit.io website, make a project and set the api keys and then run the agent in Livekit Agents Playgroud

## References used for this task

- LiveKit Agents Docs
- OpenAI Realtime API
- https://medium.com/
- Stack Overflow
- OpenAI, ``ChatGPT-5,'' Large Language Model, 2025. [Online]. Available: https://chatgpt.com

## License
This project is for educational and assignment use only.

