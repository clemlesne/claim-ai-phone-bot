from typing import Any, Optional
from azure.communication.callautomation import (
    DtmfTone,
    RecognitionChoice,
)  # TODO: Should be abstracted in the persistence layer
from helpers.config import CONFIG
from helpers.logging import build_logger, TRACER
from models.synthesis import SynthesisModel
from models.call import CallStateModel
from models.message import (
    ActionEnum as MessageActionEnum,
    extract_message_style,
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
    remove_message_action,
)
from persistence.ivoice import ContextEnum as VoiceContextEnum
from fastapi import BackgroundTasks
import asyncio
from models.next import NextModel
from helpers.call_llm import llm_completion, llm_model, load_llm_chat


_logger = build_logger(__name__)
_db = CONFIG.database.instance()
_sms = CONFIG.sms.instance()


@TRACER.start_as_current_span("on_incoming_call")
async def on_incoming_call(
    call: CallStateModel,
    incoming_context: str,
    phone_number: str,
    callback_url: str,
    background_tasks: BackgroundTasks,
) -> bool:
    _logger.debug(f"Incoming call handler caller ID: {phone_number}")
    voice = CONFIG.voice.instance()
    await voice.aanswer(
        background_tasks=background_tasks,
        call=call,
        callback_url=callback_url,
        incoming_context=incoming_context,
    )
    _logger.info(f"Answered call with {phone_number} (id {call.voice_id})")
    return True


@TRACER.start_as_current_span("on_call_connected")
async def on_call_connected(
    call: CallStateModel, background_tasks: BackgroundTasks
) -> None:
    _logger.info("Call connected")
    call.voice_recognition_retry = 0  # Reset recognition retry counter
    call.messages.append(
        MessageModel(
            action=MessageActionEnum.CALL,
            content="",
            persona=MessagePersonaEnum.HUMAN,
        )
    )  # Add call action to the messages
    await _handle_ivr_language(
        background_tasks=background_tasks,
        call=call,
    )  # Every time a call is answered, confirm the language
    await _db.call_aset(
        call
    )  # Persist to save the recognition retry count and the new message


@TRACER.start_as_current_span("on_call_disconnected")
async def on_call_disconnected(
    background_tasks: BackgroundTasks, call: CallStateModel
) -> None:
    _logger.info("Call disconnected")
    await _handle_hangup(background_tasks=background_tasks, call=call)


@TRACER.start_as_current_span("on_speech_recognized")
async def on_speech_recognized(
    background_tasks: BackgroundTasks,
    call: CallStateModel,
    text: str,
) -> None:
    _logger.info(f"On speech: {text}")
    call.messages.append(MessageModel(content=text, persona=MessagePersonaEnum.HUMAN))
    call = await load_llm_chat(
        background_tasks=background_tasks,
        call=call,
        post_call_intelligence=_post_call_intelligence,
    )
    await _db.call_aset(call)  # Persist to save the new messages


@TRACER.start_as_current_span("on_speech_timeout_error")
async def on_speech_timeout_error(
    call: CallStateModel,
    background_tasks: BackgroundTasks,
) -> None:
    if call.voice_recognition_retry < 10:
        voice = CONFIG.voice.instance()
        await voice.arecognize_speech(
            background_tasks=background_tasks,
            call=call,
            store=False,  # Do not store timeout prompt as it perturbs the LLM and makes it hallucinate
            text=await CONFIG.prompts.tts.timeout_silence(call),
        )
        call.voice_recognition_retry += 1
    else:
        await _handle_goodbye(call=call, background_tasks=background_tasks)
    await _db.call_aset(call)  # Persist to save the retry count


@TRACER.start_as_current_span("on_speech_unknown_error")
async def on_speech_unknown_error(
    call: CallStateModel,
    background_tasks: BackgroundTasks,
) -> None:
    await _handle_generic_error(call=call, background_tasks=background_tasks)


@TRACER.start_as_current_span("on_play_completed")
async def on_play_completed(
    background_tasks: BackgroundTasks,
    call: CallStateModel,
    context: Optional[str],
) -> None:
    _logger.debug("Play completed")
    voice = CONFIG.voice.instance()
    if (
        context == VoiceContextEnum.TRANSFER_FAILED
        or context == VoiceContextEnum.GOODBYE
    ):  # Call ended
        _logger.info("Ending call")
        await _handle_hangup(background_tasks=background_tasks, call=call)
    elif context == VoiceContextEnum.CONNECT_AGENT:  # Call transfer
        _logger.info("Initiating transfer call initiated")
        agent_phone_number = call.initiate.transfer_phone_number
        await voice.atransfer(
            background_tasks=background_tasks,
            call=call,
            phone_number=agent_phone_number,
        )


@TRACER.start_as_current_span("on_play_error")
async def on_play_error(
    call: CallStateModel,
    background_tasks: BackgroundTasks,
) -> None:
    await _handle_generic_error(call=call, background_tasks=background_tasks)


@TRACER.start_as_current_span("on_ivr_recognized")
async def on_ivr_recognized(
    background_tasks: BackgroundTasks,
    call: CallStateModel,
    label: str,
) -> None:
    voice = CONFIG.voice.instance()

    try:
        lang = next(
            (x for x in call.initiate.lang.availables if x.short_code == label),
            call.lang,
        )
    except ValueError:
        _logger.warning(f"Unknown IVR {label}, code not implemented")
        return

    _logger.info(f"Setting call language to {lang}")
    call.lang = lang.short_code
    await _db.call_aset(
        call
    )  # Persist language change, if the user calls back before the first message, the language will be set

    if len(call.messages) <= 1:  # First call, or only the call action
        await voice.arecognize_speech(
            background_tasks=background_tasks,
            call=call,
            text=await CONFIG.prompts.tts.hello(call),
        )

    else:  # Returning call
        await voice.aplay_text(
            background_tasks=background_tasks,
            call=call,
            text=await CONFIG.prompts.tts.welcome_back(call),
        )
        call = await load_llm_chat(
            background_tasks=background_tasks,
            call=call,
            post_call_intelligence=_post_call_intelligence,
        )

    await _db.call_aset(call)  # Persist to save the new message


@TRACER.start_as_current_span("on_transfer_completed")
async def on_transfer_completed(
    call: CallStateModel, background_tasks: BackgroundTasks
) -> None:
    _logger.info("Call transfer accepted event")
    _post_call_intelligence(call, background_tasks)


@TRACER.start_as_current_span("on_transfer_error")
async def on_transfer_error(
    call: CallStateModel, background_tasks: BackgroundTasks
) -> None:
    voice = CONFIG.voice.instance()
    await voice.aplay_text(
        background_tasks=background_tasks,
        call=call,
        context=VoiceContextEnum.TRANSFER_FAILED,
        text=await CONFIG.prompts.tts.calltransfer_failure(call),
    )
    await _db.call_aset(call)  # Persist to save the new message


async def _handle_generic_error(
    call: CallStateModel,
    background_tasks: BackgroundTasks,
):
    if call.voice_recognition_retry < 10:
        voice = CONFIG.voice.instance()
        await voice.arecognize_speech(
            background_tasks=background_tasks,
            call=call,
            store=False,  # Do not store error prompt as it perturbs the LLM and makes it hallucinate
            text=await CONFIG.prompts.tts.error(call),
        )
        call.voice_recognition_retry += 1
    else:
        await _handle_goodbye(call=call, background_tasks=background_tasks)
    await _db.call_aset(call)  # Persist to save the retry count


async def _handle_goodbye(
    call: CallStateModel,
    background_tasks: BackgroundTasks,
) -> None:
    voice = CONFIG.voice.instance()
    await voice.aplay_text(
        background_tasks=background_tasks,
        call=call,
        context=VoiceContextEnum.GOODBYE,
        store=False,  # Do not store error prompt as it perturbs the LLM and makes it hallucinate
        text=await CONFIG.prompts.tts.goodbye(call),
    )


async def _handle_hangup(
    background_tasks: BackgroundTasks, call: CallStateModel
) -> None:
    _logger.debug("Hanging up call")
    voice = CONFIG.voice.instance()
    await voice.ahangup(
        call=call,
        everyone=True,
    )  # Hang up the call
    call.messages.append(
        MessageModel(
            content="",
            persona=MessagePersonaEnum.HUMAN,
            action=MessageActionEnum.HANGUP,
        )
    )  # Add hangup action to the messages
    _post_call_intelligence(call, background_tasks)


def _post_call_intelligence(
    call: CallStateModel, background_tasks: BackgroundTasks
) -> None:
    """
    Shortcut to run all post-call intelligence tasks in background.
    """
    if (
        len(call.messages) >= 3
        and call.messages[-3].action == MessageActionEnum.CALL
        and call.messages[-2].persona == MessagePersonaEnum.ASSISTANT
        and call.messages[-1].action == MessageActionEnum.HANGUP
    ):
        _logger.info(
            "Call ended before any interaction, skipping post-call intelligence"
        )
        return
    background_tasks.add_task(
        _post_call_next,
        call=call,
    )
    background_tasks.add_task(
        _post_call_sms,
        call=call,
    )
    background_tasks.add_task(
        _post_call_synthesis,
        call=call,
    )


async def _post_call_sms(call: CallStateModel) -> None:
    """
    Send an SMS report to the customer.
    """
    _logger.info("Post-call: SMS report")
    content = await llm_completion(
        text=CONFIG.prompts.llm.sms_summary_system(call),
        call=call,
    )

    # Delete action and style from the message as they are in the history and LLM hallucinates them
    _, content = extract_message_style(remove_message_action(content or ""))

    if not content:
        _logger.warning("Error generating SMS report")
        return
    _logger.debug(f"SMS report: {content}")

    # Send the SMS to both the current caller and the policyholder
    success = False
    for number in set(
        [
            call.initiate.phone_number,
            call.customer_file.get("caller_phone", None),
        ]
    ):
        if not number:
            continue
        res = await _sms.asend(content, number)
        if not res:
            _logger.warning(f"Failed sending SMS report to {number}")
            continue
        success = True

    if success:
        # Store the SMS in the call messages
        call.messages.append(
            MessageModel(
                action=MessageActionEnum.SMS,
                content=content,
                persona=MessagePersonaEnum.ASSISTANT,
            )
        )
        await _db.call_aset(call)


async def _post_call_synthesis(call: CallStateModel) -> None:
    """
    Synthesize the call and store it to the model.
    """
    _logger.info("Post-call: Synthesis")
    short, long = await asyncio.gather(
        llm_completion(
            call=call,
            text=CONFIG.prompts.llm.synthesis_short_system(call),
        ),
        llm_completion(
            call=call,
            text=CONFIG.prompts.llm.citations_system(
                call=call,
                text=await llm_completion(
                    call=call,
                    text=CONFIG.prompts.llm.synthesis_long_system(call),
                ),
            ),
        ),
    )
    if not short or not long:
        _logger.warning("Error generating synthesis")
        return
    _logger.debug(f"Short synthesis: {short}")
    _logger.debug(f"Long synthesis: {long}")
    call.synthesis = SynthesisModel(
        long=long,
        short=short,
    )
    await _db.call_aset(call)


async def _post_call_next(call: CallStateModel) -> None:
    """
    Generate next action for the call.
    """
    _logger.info("Post-call: Next action")
    next = await llm_model(
        call=call,
        model=NextModel,
        text=CONFIG.prompts.llm.next_system(call),
    )
    if not next:
        _logger.warning("Error generating next action")
        return
    _logger.debug(f"Next action: {next}")
    call.next = next
    await _db.call_aset(call)


async def _handle_ivr_language(
    call: CallStateModel, background_tasks: BackgroundTasks
) -> None:
    voice = CONFIG.voice.instance()
    tones = [
        DtmfTone.ONE,
        DtmfTone.TWO,
        DtmfTone.THREE,
        DtmfTone.FOUR,
        DtmfTone.FIVE,
        DtmfTone.SIX,
        DtmfTone.SEVEN,
        DtmfTone.EIGHT,
        DtmfTone.NINE,
    ]
    choices = []
    for i, lang in enumerate(call.initiate.lang.availables):
        choices.append(
            RecognitionChoice(
                label=lang.short_code,
                phrases=lang.pronunciations_en,
                tone=tones[i],
            )
        )
    await voice.arecognize_ivr(
        background_tasks=background_tasks,
        call=call,
        choices=choices,
        text=await CONFIG.prompts.tts.ivr_language(call),
    )
