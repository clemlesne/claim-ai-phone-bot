from azure.communication.callautomation import (
    CallAutomationClient,
    CallConnectionClient,
    DtmfTone,
    RecognitionChoice,
    PhoneNumberIdentifier,
)
from helpers.config import CONFIG
from helpers.logging import build_logger
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)
from models.synthesis import SynthesisModel
from models.call import CallModel
from models.message import (
    ActionEnum as MessageActionEnum,
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
)
from helpers.call_utils import (
    ContextEnum as CallContextEnum,
    handle_play,
    handle_recognize_ivr,
    handle_recognize_text,
)
from fastapi import BackgroundTasks
import asyncio
from azure.communication.sms import SmsClient
from models.next import NextModel
from helpers.call_llm import llm_completion, llm_model, load_llm_chat


_logger = build_logger(__name__)
_sms_client = SmsClient(
    credential=CONFIG.communication_service.access_key.get_secret_value(),
    endpoint=CONFIG.communication_service.endpoint,
)
_db = CONFIG.database.instance()


async def on_new_call(
    client: CallAutomationClient, context: str, phone_number: str, callback_url: str
) -> bool:
    _logger.debug(f"Incoming call handler caller ID: {phone_number}")

    try:
        answer_call_result = client.answer_call(
            callback_url=callback_url,
            cognitive_services_endpoint=CONFIG.cognitive_service.endpoint,
            incoming_call_context=context,
        )
        _logger.info(
            f"Answered call with {phone_number} ({answer_call_result.call_connection_id})"
        )
        return True

    except ClientAuthenticationError as e:
        _logger.error(
            "Authentication error with Communication Services, check the credentials",
            exc_info=True,
        )

    except HttpResponseError as e:
        if "lifetime validation of the signed http request failed" in e.message.lower():
            _logger.debug("Old call event received, ignoring")
        else:
            _logger.error(
                f"Unknown error answering call with {phone_number}", exc_info=True
            )

    return False


async def on_call_connected(
    call: CallModel,
    client: CallConnectionClient,
) -> None:
    _logger.info("Call connected")
    call.recognition_retry = 0  # Reset recognition retry counter

    call.messages.append(
        MessageModel(
            action=MessageActionEnum.CALL,
            content="",
            persona=MessagePersonaEnum.HUMAN,
        )
    )

    await _handle_ivr_language(
        call=call, client=client
    )  # Every time a call is answered, confirm the language


async def on_call_disconnected(
    background_tasks: BackgroundTasks,
    call: CallModel,
    client: CallConnectionClient,
) -> None:
    _logger.info("Call disconnected")
    await _handle_hangup(background_tasks, client, call)


async def on_speech_recognized(
    background_tasks: BackgroundTasks,
    call: CallModel,
    client: CallConnectionClient,
    text: str,
) -> None:
    _logger.info(f"Voice recognition: {text}")
    call.messages.append(MessageModel(content=text, persona=MessagePersonaEnum.HUMAN))
    call = await load_llm_chat(
        background_tasks=background_tasks,
        call=call,
        client=client,
        post_call_intelligence=_post_call_intelligence,
    )


async def on_speech_timeout_error(
    call: CallModel,
    client: CallConnectionClient,
) -> None:
    if call.recognition_retry < 10:
        await handle_recognize_text(
            call=call,
            client=client,
            store=False,  # Do not store timeout prompt as it perturbs the LLM and makes it hallucinate
            text=await CONFIG.prompts.tts.timeout_silence(call),
        )
        call.recognition_retry += 1
    else:
        await handle_play(
            call=call,
            client=client,
            context=CallContextEnum.GOODBYE,
            text=await CONFIG.prompts.tts.goodbye(call),
            store=False,  # Do not store goodbye prompt as it perturbs the LLM and makes it hallucinate
        )


async def on_speech_unknown_error(
    call: CallModel,
    client: CallConnectionClient,
    error_code: int,
) -> None:
    if error_code == 8511:  # Failure while trying to play the prompt
        _logger.warn("Failed to play prompt")
    else:
        _logger.warn(
            f"Recognition failed with unknown error code {error_code}, answering with default error"
        )
    await handle_recognize_text(
        call=call,
        client=client,
        store=False,  # Do not store error prompt as it perturbs the LLM and makes it hallucinate
        text=await CONFIG.prompts.tts.error(call),
    )


async def on_play_completed(
    background_tasks: BackgroundTasks,
    call: CallModel,
    client: CallConnectionClient,
    context: str,
) -> None:
    _logger.debug("Play completed")

    if (
        context == CallContextEnum.TRANSFER_FAILED or context == CallContextEnum.GOODBYE
    ):  # Call ended
        _logger.info("Ending call")
        await _handle_hangup(background_tasks, client, call)

    elif context == CallContextEnum.CONNECT_AGENT:  # Call transfer
        _logger.info("Initiating transfer call initiated")
        agent_caller = PhoneNumberIdentifier(str(CONFIG.workflow.agent_phone_number))
        client.transfer_call_to_participant(
            target_participant=agent_caller,  # type: ignore
        )


async def on_play_error(
    error_code: int,
) -> None:
    _logger.debug("Play failed")
    # See: https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/communication-services/how-tos/call-automation/play-action.md
    if error_code == 8535:  # Action failed, file format
        _logger.warn("Error during media play, file format is invalid")
    elif error_code == 8536:  # Action failed, file downloaded
        _logger.warn("Error during media play, file could not be downloaded")
    elif error_code == 8565:  # Action failed, AI services config
        _logger.error(
            "Error during media play, impossible to connect with Azure AI services"
        )
    elif error_code == 9999:  # Unknown
        _logger.warn("Error during media play, unknown internal server error")
    else:
        _logger.warn(f"Error during media play, unknown error code {error_code}")


async def on_ivr_recognized(
    client: CallConnectionClient,
    call: CallModel,
    label: str,
    background_tasks: BackgroundTasks,
) -> None:
    try:
        lang = next(
            (x for x in CONFIG.workflow.lang.availables if x.short_code == label),
            CONFIG.workflow.lang.default_lang,
        )
    except ValueError:
        _logger.warn(f"Unknown IVR {label}, code not implemented")
        return

    _logger.info(f"Setting call language to {lang}")
    call.lang = lang
    await _db.call_aset(
        call
    )  # Persist language change, if the user calls back before the first message, the language will be set

    if not call.messages:  # First call
        await handle_recognize_text(
            call=call,
            client=client,
            text=await CONFIG.prompts.tts.hello(call),
        )

    else:  # Returning call
        await handle_play(
            call=call,
            client=client,
            text=await CONFIG.prompts.tts.welcome_back(call),
        )
        call = await load_llm_chat(
            background_tasks=background_tasks,
            call=call,
            client=client,
            post_call_intelligence=_post_call_intelligence,
        )


async def on_transfer_completed() -> None:
    _logger.info("Call transfer accepted event")
    # TODO: Is there anything to do here?


async def on_transfer_error(
    call: CallModel,
    client: CallConnectionClient,
    error_code: int,
) -> None:
    _logger.info(f"Error during call transfer, subCode {error_code}")
    await handle_play(
        call=call,
        client=client,
        context=CallContextEnum.TRANSFER_FAILED,
        text=await CONFIG.prompts.tts.calltransfer_failure(call),
    )


async def _handle_hangup(
    background_tasks: BackgroundTasks, client: CallConnectionClient, call: CallModel
) -> None:
    _logger.debug("Hanging up call")
    try:
        client.hang_up(is_for_everyone=True)
    except ResourceNotFoundError:
        _logger.debug("Call already hung up")
    except HttpResponseError as e:
        if "call already terminated" in e.message.lower():
            _logger.debug("Call hung up before playing")
        else:
            raise e

    call.messages.append(
        MessageModel(
            content="",
            persona=MessagePersonaEnum.HUMAN,
            action=MessageActionEnum.HANGUP,
        )
    )

    _post_call_intelligence(call, background_tasks)


def _post_call_intelligence(call: CallModel, background_tasks: BackgroundTasks) -> None:
    """
    Shortcut to run all post-call intelligence tasks in background.
    """
    background_tasks.add_task(_post_call_next, call)
    background_tasks.add_task(_post_call_sms, call)
    background_tasks.add_task(_post_call_synthesis, call)


async def _post_call_sms(call: CallModel) -> None:
    """
    Send an SMS report to the customer.
    """
    content = await llm_completion(
        text=CONFIG.prompts.llm.sms_summary_system(call),
        call=call,
    )

    if not content:
        _logger.warn("Error generating SMS report")
        return

    _logger.info(f"SMS report: {content}")
    try:
        responses = _sms_client.send(
            from_=str(CONFIG.communication_service.phone_number),
            message=content,
            to=call.phone_number,
        )
        response = responses[0]

        if response.successful:
            _logger.debug(f"SMS report sent {response.message_id} to {response.to}")
            call.messages.append(
                MessageModel(
                    action=MessageActionEnum.SMS,
                    content=content,
                    persona=MessagePersonaEnum.ASSISTANT,
                )
            )
            await _db.call_aset(call)
        else:
            _logger.warn(
                f"Failed SMS to {response.to}, status {response.http_status_code}, error {response.error_message}"
            )

    except ClientAuthenticationError:
        _logger.error(
            "Authentication error for SMS, check the credentials", exc_info=True
        )
    except HttpResponseError as e:
        _logger.error(f"Error sending SMS: {e}")
    except Exception:
        _logger.warn(f"Failed SMS to {call.phone_number}", exc_info=True)


async def _post_call_synthesis(call: CallModel) -> None:
    """
    Synthesize the call and store it to the model.
    """
    _logger.debug("Synthesizing call")

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
        _logger.warn("Error generating synthesis")
        return

    _logger.info(f"Short synthesis: {short}")
    _logger.info(f"Long synthesis: {long}")

    call.synthesis = SynthesisModel(
        long=long,
        short=short,
    )
    await _db.call_aset(call)


async def _post_call_next(call: CallModel) -> None:
    """
    Generate next action for the call.
    """
    next = await llm_model(
        call=call,
        model=NextModel,
        text=CONFIG.prompts.llm.next_system(call),
    )

    if not next:
        _logger.warn("Error generating next action")
        return

    _logger.info(f"Next action: {next}")
    call.next = next
    await _db.call_aset(call)


async def _handle_ivr_language(
    client: CallConnectionClient,
    call: CallModel,
) -> None:
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
    for i, lang in enumerate(CONFIG.workflow.lang.availables):
        choices.append(
            RecognitionChoice(
                label=lang.short_code,
                phrases=lang.pronunciations_en,
                tone=tones[i],
            )
        )
    await handle_recognize_ivr(
        call=call,
        client=client,
        choices=choices,
        text=await CONFIG.prompts.tts.ivr_language(call),
    )