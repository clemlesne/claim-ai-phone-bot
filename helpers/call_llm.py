from typing import Awaitable, Callable, Optional, Tuple, Type

from pydantic import ValidationError
from helpers.config import CONFIG
from helpers.logging import build_logger
from models.call import CallStateModel
from models.message import (
    extract_message_style,
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
    remove_message_action,
    StyleEnum as MessageStyleEnum,
    ToolModel as MessageToolModel,
)
from fastapi import BackgroundTasks
from helpers.llm_tools import LlmPlugins
import asyncio
from helpers.llm_worker import (
    completion_model_sync,
    completion_stream,
    completion_sync,
    ModelType,
    safety_check,
    SafetyCheckError,
)
from openai import APIError
from openai.types.chat import ChatCompletionSystemMessageParam


_logger = build_logger(__name__)
_db = CONFIG.database.instance()


async def llm_completion(text: Optional[str], call: CallStateModel) -> Optional[str]:
    """
    Run LLM completion from a system prompt and a Call model.

    If the system prompt is None, no completion will be run and None will be returned. Otherwise, the response of the LLM will be returned.
    """
    _logger.info("Running LLM completion")

    if not text:
        return None

    system = _llm_completion_system(text, call)
    content = None

    try:
        content = await completion_sync(
            max_tokens=1000,
            messages=call.messages,
            system=system,
        )
    except APIError as e:
        _logger.warning(f"OpenAI API call error: {e}")
    except SafetyCheckError as e:
        _logger.warning(f"OpenAI safety check error: {e}")

    return content


async def llm_model(
    text: Optional[str], call: CallStateModel, model: Type[ModelType]
) -> Optional[ModelType]:
    """
    Run LLM completion from a system prompt, a Call model, and an expected model type as a return.

    The logic will try its best to return a model of the expected type, but it is not guaranteed. It it fails, `None` will be returned.
    """
    _logger.debug("Running LLM model")

    if not text:
        return None

    system = _llm_completion_system(text, call)
    res = None

    try:
        res = await completion_model_sync(
            max_tokens=1000,
            messages=call.messages,
            model=model,
            system=system,
        )
    except APIError as e:
        _logger.warning(f"OpenAI API call error: {e}")
    except ValidationError as e:
        _logger.debug(f"Parsing error: {e.errors()}")

    return res


def _llm_completion_system(
    system: str, call: CallStateModel
) -> list[ChatCompletionSystemMessageParam]:
    messages = [
        ChatCompletionSystemMessageParam(
            content=CONFIG.prompts.llm.default_system(call=call),
            role="system",
        ),
        ChatCompletionSystemMessageParam(
            content=system,
            role="system",
        ),
    ]
    _logger.debug(f"Messages: {messages}")
    return messages


async def load_llm_chat(
    background_tasks: BackgroundTasks,
    call: CallStateModel,
    post_call_intelligence: Callable[[CallStateModel, BackgroundTasks], None],
    _iterations_remaining: int = 3,
) -> CallStateModel:
    """
    Handle the intelligence of the call, including: LLM chat, TTS, and media play.

    Play the loading sound while waiting for the intelligence to be processed. If the intelligence is not processed after few seconds, play the timeout sound. If the intelligence is not processed after more seconds, stop the intelligence processing and play the error sound.

    Returns the updated call model.
    """
    _logger.info("Loading LLM chat")
    should_play_sound = True
    voice = CONFIG.voice.instance()

    async def _user_callback(text: str, style: MessageStyleEnum) -> None:
        """
        Send back the TTS to the user.
        """
        nonlocal should_play_sound

        try:
            await safety_check(text)
        except SafetyCheckError as e:
            _logger.warning(f"Unsafe text detected, not playing: {e}")
            return

        should_play_sound = False
        await voice.aplay_text(
            background_tasks=background_tasks,
            call=call,
            store=False,
            style=style,
            text=text,
        )

    chat_task = asyncio.create_task(
        _execute_llm_chat(
            background_tasks=background_tasks,
            call=call,
            post_call_intelligence=post_call_intelligence,
            use_tools=_iterations_remaining > 0,
            user_callback=_user_callback,
        )
    )

    soft_timeout_triggered = False
    soft_timeout_task = asyncio.create_task(
        asyncio.sleep(CONFIG.workflow.intelligence_soft_timeout_sec)
    )
    hard_timeout_task = asyncio.create_task(
        asyncio.sleep(CONFIG.workflow.intelligence_hard_timeout_sec)
    )

    is_error = True
    continue_chat = True
    should_user_answer = True
    try:
        while True:
            _logger.debug(f"Chat task status: {chat_task.done()}")
            if chat_task.done():  # Break when chat coroutine is done
                # Clean up
                soft_timeout_task.cancel()
                hard_timeout_task.cancel()
                # Store updated chat model
                is_error, continue_chat, should_user_answer, call = chat_task.result()
                break

            if hard_timeout_task.done():  # Break when hard timeout is reached
                _logger.warning(
                    f"Hard timeout of {CONFIG.workflow.intelligence_hard_timeout_sec}s reached"
                )
                # Clean up
                chat_task.cancel()
                soft_timeout_task.cancel()
                break

            if should_play_sound:  # Catch timeout if async loading is not started
                if (
                    soft_timeout_task.done() and not soft_timeout_triggered
                ):  # Speak when soft timeout is reached
                    _logger.warning(
                        f"Soft timeout of {CONFIG.workflow.intelligence_soft_timeout_sec}s reached"
                    )
                    soft_timeout_triggered = True
                    await voice.aplay_text(
                        background_tasks=background_tasks,
                        call=call,
                        store=False,  # Do not store timeout prompt as it perturbs the LLM and makes it hallucinate
                        text=await CONFIG.prompts.tts.timeout_loading(call),
                    )

                else:  # Do not play timeout prompt plus loading, it can be frustrating for the user
                    await voice.aplay_audio(
                        background_tasks=background_tasks,
                        call=call,
                        url=CONFIG.prompts.sounds.loading(),
                    )  # Play loading sound

            # Wait to not block the event loop and play too many sounds
            await asyncio.sleep(5)

    except Exception:
        _logger.warning("Error loading intelligence", exc_info=True)

    if is_error:  # Error during chat
        if not continue_chat or _iterations_remaining < 1:  # Maximum retries reached
            _logger.warning("Maximum retries reached, stopping chat")
            should_user_answer = True
            content = await CONFIG.prompts.tts.error(call)
            style = MessageStyleEnum.NONE
            await _user_callback(content, style)
            call.messages.append(
                MessageModel(
                    content=content,
                    persona=MessagePersonaEnum.ASSISTANT,
                    style=style,
                )
            )

        else:  # Retry chat after an error
            _logger.info(f"Retrying chat, {_iterations_remaining - 1} remaining")
            return await load_llm_chat(
                background_tasks=background_tasks,
                call=call,
                post_call_intelligence=post_call_intelligence,
                _iterations_remaining=_iterations_remaining - 1,
            )

    elif continue_chat:  # Contiue chat
        _logger.info(f"Continuing chat, {_iterations_remaining - 1} remaining")
        return await load_llm_chat(
            background_tasks=background_tasks,
            call=call,
            post_call_intelligence=post_call_intelligence,
            _iterations_remaining=_iterations_remaining - 1,
        )

    if should_user_answer:
        await voice.arecognize_speech(
            background_tasks=background_tasks,
            call=call,
        )

    return call


async def _execute_llm_chat(
    background_tasks: BackgroundTasks,
    call: CallStateModel,
    post_call_intelligence: Callable[[CallStateModel, BackgroundTasks], None],
    use_tools: bool,
    user_callback: Callable[[str, MessageStyleEnum], Awaitable],
) -> Tuple[bool, bool, bool, CallStateModel]:
    """
    Perform the chat with the LLM model.

    This function will handle:

    - The chat with the LLM model (incl system prompts, tools, and user callback)
    - Retry as possible if the LLM model fails to return a response

    Returns a tuple with:

    1. `bool`, notify error
    2. `bool`, should retry chat
    3. `bool`, if the chat should continue
    4. `CallStateModel`, the updated model
    """
    _logger.debug("Running LLM chat")
    content_full = ""
    should_user_answer = True
    voice = CONFIG.voice.instance()

    async def _tools_callback(text: str, style: MessageStyleEnum) -> None:
        nonlocal content_full
        content_full += f" {text}"
        await user_callback(text, style)

    async def _content_callback(
        buffer: str, style: MessageStyleEnum
    ) -> MessageStyleEnum:
        # Remove tool calls from buffer content and detect style
        local_style, local_content = extract_message_style(
            remove_message_action(buffer)
        )
        new_style = local_style or style
        if local_content:
            await user_callback(local_content, new_style)
        return new_style

    async def _tools_cancellation_callback() -> None:
        nonlocal should_user_answer
        _logger.info("Chat stopped by tool")
        should_user_answer = False

    # Build RAG
    trainings = await call.trainings()
    _logger.info(f"Enhancing LLM chat with {len(trainings)} trainings")
    _logger.debug(f"Trainings: {trainings}")

    # Build system prompts
    system = [
        ChatCompletionSystemMessageParam(
            content=CONFIG.prompts.llm.default_system(call=call),
            role="system",
        ),
        ChatCompletionSystemMessageParam(
            content=CONFIG.prompts.llm.chat_system(
                call=call,
                trainings=trainings,
            ),
            role="system",
        ),
    ]

    # Build plugins
    plugins = LlmPlugins(
        background_tasks=background_tasks,
        call=call,
        cancellation_callback=_tools_cancellation_callback,
        post_call_intelligence=lambda call: post_call_intelligence(
            call, background_tasks
        ),
        user_callback=_tools_callback,
    )

    tools = []
    if not use_tools:
        _logger.warning("Tools disabled for this chat")
    else:
        tools = await plugins.to_openai(call)
        _logger.debug(f"Tools: {tools}")

    # Execute LLM inference
    content_buffer_pointer = 0
    tool_calls_buffer: dict[int, MessageToolModel] = {}
    try:
        async for delta in completion_stream(
            max_tokens=350,
            messages=call.messages,
            system=system,
            tools=tools,
        ):
            if not delta.content:
                for piece in delta.tool_calls or []:
                    tool_calls_buffer[piece.index] = tool_calls_buffer.get(
                        piece.index, MessageToolModel()
                    )
                    tool_calls_buffer[piece.index] += piece
            else:
                # Store whole content
                content_full += delta.content
                for sentence in voice.tts_sentence_split(
                    content_full[content_buffer_pointer:], False
                ):
                    content_buffer_pointer += len(sentence)
                    plugins.style = await _content_callback(sentence, plugins.style)
    except APIError as e:
        _logger.warning(f"OpenAI API call error: {e}")
        return True, True, should_user_answer, call

    # Flush the remaining buffer
    if content_buffer_pointer < len(content_full):
        plugins.style = await _content_callback(
            content_full[content_buffer_pointer:], plugins.style
        )

    # Convert tool calls buffer
    tool_calls = [tool_call for _, tool_call in tool_calls_buffer.items()]

    # Delete action and style from the message as they are in the history and LLM hallucinates them
    _, content_full = extract_message_style(remove_message_action(content_full))

    _logger.debug(f"Chat response: {content_full}")
    _logger.debug(f"Tool calls: {tool_calls}")

    # OpenAI GPT-4 Turbo sometimes return wrong tools schema, in that case, retry within limits
    # TODO: Tries to detect this error earlier
    # See: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653
    if any(
        tool_call.function_name == "multi_tool_use.parallel" for tool_call in tool_calls
    ):
        _logger.warning(f'LLM send back invalid tool schema "multi_tool_use.parallel"')
        return True, True, should_user_answer, call

    # OpenAI GPT-4 Turbo tends to return empty content, in that case, retry within limits
    if not content_full and not tool_calls:
        _logger.warning("Empty content, retrying")
        return True, True, should_user_answer, call

    # Execute tools
    tool_tasks = [tool_call.execute_function(plugins) for tool_call in tool_calls]
    await asyncio.gather(*tool_tasks)

    # Store message
    call.messages.append(
        MessageModel(
            content=content_full.strip(),
            persona=MessagePersonaEnum.ASSISTANT,
            style=plugins.style,
            tool_calls=tool_calls,
        )
    )

    # Recusive call if needed
    if tool_calls and should_user_answer:
        return False, True, should_user_answer, call

    return False, False, should_user_answer, call
