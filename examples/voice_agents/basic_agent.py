import logging
import time
from typing import Set

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

def _parse_word_list(env_value: str | None, default: Set[str]) -> Set[str]:
    if not env_value:
        return default
    return {w.strip().lower() for w in env_value.split(",") if w.strip()}

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=("Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user"),
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    @staticmethod
    def soft_words() -> Set[str]:
        from os import getenv

        default = {"yeah", "ya", "ok", "okay", "hmm", "uh-huh", "uh", "mm", "right", "sure"}
        return _parse_word_list(getenv("IGNORE_WORDS"), default)
    
    @staticmethod
    def hard_words() -> Set[str]:
        from os import getenv

        default = {"stop", "wait", "hold", "holdon", "hold-on", "pause", "cancel", "no"}
        return _parse_word_list(getenv("INTERRUPT_WORDS"), default)

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # For Tracking speaking state
    speaking_state = {
        "agent_is_speaking": False,
        "last_speaking_start": 0.0,
        "last_speaking_end": 0.0,
    }

    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="deepgram/nova-3",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        # Treating very short speech with few words as likely false
        min_interruption_words=2,
    )

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        old = ev.old_state
        new= ev.new_state
        now=time.time()

        logger.info(
            f"Agent state changed: old_state='{old}' new_state='{new}' "
            f"created at={ev.created_at}"
        )

        if new == "speaking":
            speaking_state["agent_is_speaking"] = True
            speaking_state["last_speaking_start"] = now
        elif old == "speaking" and new != "speaking":
            speaking_state["agent_is_speaking"] = False
            speaking_state["last_speaking_end"] = now

    # IntelliGent Interruption Handling
    soft_words = MyAgent.soft_words()
    hard_words = MyAgent.hard_words()

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        # We only need to consider final transcripts from STT
        if not ev.is_final:
            return

        text_raw = ev.transcript or ""
        text = text_raw.strip().lower()
        now=time.time()

        if not text:
            return

        # Rough tokenization
        tokens = [t.strip(".,!?;:") for t in text.split() if t.strip()]
        token_set = set(tokens)

        # Soft / hard detection (single words | multi-word phrases)
        has_soft = any(t in soft_words for t in token_set)
        has_hard = any(t in hard_words for t in token_set)

        # "overlap" is when agent is speaking OR just finished speaking 
        recently_spoke = now - speaking_state["last_speaking_end"] < 2.0
        overlapping = speaking_state["agent_is_speaking"] or recently_spoke

        logger.info(
            "user_input_transcribed: %r (final=%s, overlapping=%s, soft=%s, hard=%s)",
            text_raw,
            ev.is_final,
            overlapping,
            has_soft,
            has_hard,
        )

        # Hard interruption when the agent is speaking 

        if overlapping and has_hard:
            logger.info("Hard interrupt detected during/after agent speech -> interrupting")
            # explicitly cut off any current speech; this also prevents auto-resume
            session.interrupt()
            return

        # Backchannel while the agent is talking 
        if overlapping and has_soft and not has_hard:
            logger.debug(
                "Soft backchannel during/after agent speech -> "
                "clearing user turn and letting TTS continue"
            )
            # discard this overlapping mini-utterance so it doesn't
            # start a new user turn or change the LLM context
            session.clear_user_turn()
            return

        # When the agent is silent, it will be treated normally 
        logger.debug("Agent silent or far from last speech -> normal handling")

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
