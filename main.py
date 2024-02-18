# First imports, to make sure the following logs are first
from helpers.logging import build_logger
from helpers.config import CONFIG


_logger = build_logger(__name__)
_logger.info(f"claim-ai v{CONFIG.version}")


# General imports
from typing import (
    Any,
    Callable,
    Coroutine,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
)
from azure.communication.callautomation import (
    CallAutomationClient,
    CallConnectionClient,
    DtmfTone,
    PhoneNumberIdentifier,
    RecognitionChoice,
    RecognizeInputType,
)
from azure.communication.sms import SmsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.core.messaging import CloudEvent
from azure.eventgrid import EventGridEvent, SystemEventNames
from azure.identity import DefaultAzureCredential
from enum import Enum
from fastapi import FastAPI, status, Request, HTTPException, BackgroundTasks, Response
from fastapi.responses import JSONResponse, HTMLResponse
from helpers.config_models.database import ModeEnum as DatabaseMode
from helpers.config_models.cache import ModeEnum as CacheMode
from helpers.logging import build_logger
from jinja2 import Environment, FileSystemLoader, select_autoescape
from models.action import ActionModel, IndentEnum as IndentAction
from models.call import CallModel
from models.next import ActionEnum as NextAction
from models.next import NextModel
from models.reminder import ReminderModel
from models.synthesis import SynthesisModel
from openai import APIError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from persistence.ai_search import AiSearchSearch, TrainingModel as AiSearchTrainingModel
from persistence.cosmos import CosmosStore
from persistence.memory import MemoryCache
from persistence.redis import RedisCache
from persistence.sqlite import SqliteStore
from urllib.parse import quote_plus
import asyncio
import html
import re
from models.message import (
    ActionEnum as MessageAction,
    MessageModel,
    PersonaEnum as MessagePersona,
    StyleEnum as MessageStyle,
    ToolModel as MessageToolModel,
)
from helpers.llm import (
    completion_model_sync,
    completion_stream,
    completion_sync,
    build_kernel,
    ModelType,
    safety_check,
    SafetyCheckError,
)
from models.claim import ClaimModel
from pydantic import ValidationError
from uuid import UUID
import json
import mistune
from helpers.call import (
    handle_media,
    handle_play,
    handle_recognize_ivr,
    handle_recognize_text,
)
from sk_plugins.call import CallPlugin
from sk_plugins.claim import ClaimPlugin
from sk_plugins.reminder import ReminderPlugin


# Jinja configuration
jinja = Environment(
    autoescape=select_autoescape(),
    enable_async=True,
    loader=FileSystemLoader("public_website"),
)
# Jinja custom functions
jinja.filters["quote_plus"] = lambda x: quote_plus(str(x)) if x else ""
jinja.filters["markdown"] = lambda x: mistune.create_markdown(escape=False, plugins=["abbr", "speedup", "url"])(x) if x else ""  # type: ignore

# Azure Communication Services
source_caller = PhoneNumberIdentifier(CONFIG.communication_service.phone_number)
_logger.info(f"Using phone number {str(CONFIG.communication_service.phone_number)}")
# Cannot place calls with RBAC, need to use access key (see: https://learn.microsoft.com/en-us/azure/communication-services/concepts/authentication#authentication-options)
call_automation_client = CallAutomationClient(
    endpoint=CONFIG.communication_service.endpoint,
    credential=AzureKeyCredential(
        CONFIG.communication_service.access_key.get_secret_value()
    ),
)
sms_client = SmsClient(
    credential=DefaultAzureCredential(), endpoint=CONFIG.communication_service.endpoint
)

# Persistence
cache = (
    MemoryCache(CONFIG.cache.memory)
    if CONFIG.cache.mode == CacheMode.MEMORY
    else RedisCache(CONFIG.cache.redis)
)
db = (
    SqliteStore(CONFIG.database.sqlite)
    if CONFIG.database.mode == DatabaseMode.SQLITE
    else CosmosStore(CONFIG.database.cosmos_db)
)
search = AiSearchSearch(cache, CONFIG.ai_search)

# FastAPI
_logger.info(f'Using root path "{CONFIG.api.root_path}"')
api = FastAPI(
    contact={
        "url": "https://github.com/clemlesne/claim-ai-phone-bot",
    },
    description="AI-powered call center solution with Azure and OpenAI GPT.",
    license_info={
        "name": "Apache-2.0",
        "url": "https://github.com/clemlesne/claim-ai-phone-bot/blob/master/LICENCE",
    },
    root_path=CONFIG.api.root_path,
    title="claim-ai-phone-bot",
    version=CONFIG.version,
)


CALL_EVENT_URL = f'{CONFIG.api.events_domain.strip("/")}/call/event/{{phone_number}}/{{callback_secret}}'
MESSAGE_ACTION_R = rf"action=([a-z_]*)( .*)?"
MESSAGE_STYLE_R = rf"style=([a-z_]*)( .*)?"
FUNC_NAME_SANITIZER_R = r"[^a-zA-Z0-9_-]"


class ContextEnum(str, Enum):
    TRANSFER_FAILED = "transfer_failed"
    CONNECT_AGENT = "connect_agent"
    GOODBYE = "goodbye"


@api.get(
    "/health/liveness",
    status_code=status.HTTP_204_NO_CONTENT,
    description="Liveness healthckeck, always returns 204, used to check if the API is up.",
)
async def health_liveness_get() -> None:
    pass


@api.get(
    "/report/{phone_number}",
    description="Display the history of calls in a web page.",
)
async def report_history_get(phone_number: str) -> HTMLResponse:
    calls = await db.call_asearch_all(phone_number) or []

    template = jinja.get_template("history.html.jinja")
    render = await template.render_async(
        bot_company=CONFIG.workflow.bot_company,
        bot_name=CONFIG.workflow.bot_name,
        calls=calls,
        phone_number=phone_number,
        version=CONFIG.version,
    )
    return HTMLResponse(content=render)


@api.get(
    "/report/{phone_number}/{call_id}",
    description="Display the call report in a web page.",
)
async def report_call_get(phone_number: str, call_id: UUID) -> HTMLResponse:
    call = await db.call_aget(call_id)
    if not call or call.phone_number != phone_number:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call {call_id} for phone number {phone_number} not found",
        )

    template = jinja.get_template("report.html.jinja")
    render = await template.render_async(
        bot_company=CONFIG.workflow.bot_company,
        bot_name=CONFIG.workflow.bot_name,
        call=call,
        next_actions=[action for action in NextAction],
        version=CONFIG.version,
    )
    return HTMLResponse(content=render)


@api.get(
    "/call",
    description="Get all calls by phone number.",
)
async def call_get(phone_number: str) -> List[CallModel]:
    return await db.call_asearch_all(phone_number) or []


@api.get(
    "/call/initiate",
    status_code=status.HTTP_204_NO_CONTENT,
    description="Initiate an outbound call to a phone number.",
)
async def call_initiate_get(phone_number: str) -> None:
    _logger.info(f"Initiating outbound call to {phone_number}")
    call_connection_properties = call_automation_client.create_call(
        callback_url=await callback_url(phone_number),
        cognitive_services_endpoint=CONFIG.cognitive_service.endpoint,
        source_caller_id_number=source_caller,
        target_participant=PhoneNumberIdentifier(phone_number),
    )
    _logger.info(
        f"Created call with connection id: {call_connection_properties.call_connection_id}"
    )


@api.post(
    "/call/inbound",
    description="Handle incoming call from a Azure Event Grid event originating from Azure Communication Services.",
)
async def call_inbound_post(request: Request):
    responses = await asyncio.gather(
        *[call_inbound_worker(event_dict) for event_dict in await request.json()]
    )
    for response in responses:
        if response:
            return response
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def call_inbound_worker(event_dict: dict[str, Any]) -> Optional[JSONResponse]:
    event = EventGridEvent.from_dict(event_dict)
    event_type = event.event_type

    _logger.debug(f"Call inbound event {event_type} with data {event.data}")

    if event_type == SystemEventNames.EventGridSubscriptionValidationEventName:
        validation_code = event.data["validationCode"]
        _logger.info(f"Validating Event Grid subscription ({validation_code})")
        return JSONResponse(
            content={"validationResponse": event.data["validationCode"]},
            status_code=status.HTTP_200_OK,
        )

    elif event_type == SystemEventNames.AcsIncomingCallEventName:
        if event.data["from"]["kind"] == "phoneNumber":
            phone_number = event.data["from"]["phoneNumber"]["value"]
        else:
            phone_number = event.data["from"]["rawId"]

        _logger.debug(f"Incoming call handler caller ID: {phone_number}")
        call_context = event.data["incomingCallContext"]

        try:
            answer_call_result = call_automation_client.answer_call(
                callback_url=await callback_url(phone_number),
                cognitive_services_endpoint=CONFIG.cognitive_service.endpoint,
                incoming_call_context=call_context,
            )
            _logger.info(
                f"Answered call with {phone_number} ({answer_call_result.call_connection_id})"
            )
        except HttpResponseError as e:
            if (
                "lifetime validation of the signed http request failed"
                in e.message.lower()
            ):
                _logger.debug("Old call event received, ignoring")
            else:
                raise e


@api.post(
    "/call/event/{phone_number}/{secret}",
    description="Handle callbacks from Azure Communication Services.",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def call_event_post(
    request: Request,
    background_tasks: BackgroundTasks,
    phone_number: str,
    secret: str,
) -> None:
    await asyncio.gather(
        *[
            communication_event_worker(
                background_tasks, event_dict, phone_number, secret
            )
            for event_dict in await request.json()
        ]
    )


async def communication_event_worker(
    background_tasks: BackgroundTasks,
    event_dict: dict,
    phone_number: str,
    secret: str,
) -> None:
    call = await db.call_asearch_one(phone_number)
    if not call or call.callback_secret.get_secret_value() != secret:
        _logger.warn(f"Call with {phone_number} not found")
        return

    event = CloudEvent.from_dict(event_dict)
    connection_id = event.data["callConnectionId"]
    operation_context = event.data.get("operationContext", None)
    client = call_automation_client.get_call_connection(
        call_connection_id=connection_id
    )
    event_type = event.type

    _logger.debug(f"Call event received {event_type} for call {call}")
    _logger.debug(event.data)

    if event_type == "Microsoft.Communication.CallConnected":  # Call answered
        _logger.info(f"Call connected ({call.call_id})")
        call.recognition_retry = 0  # Reset recognition retry counter

        call.messages.append(
            MessageModel(
                action=MessageAction.CALL,
                content="",
                persona=MessagePersona.HUMAN,
            )
        )

        await handle_ivr_language(
            call=call, client=client
        )  # Every time a call is answered, confirm the language

    elif event_type == "Microsoft.Communication.CallDisconnected":  # Call hung up
        _logger.info(f"Call disconnected ({call.call_id})")
        await handle_hangup(background_tasks, client, call)

    elif (
        event_type == "Microsoft.Communication.RecognizeCompleted"
    ):  # Speech recognized
        recognition_result = event.data["recognitionType"]

        if recognition_result == "speech":  # Handle voice
            speech_text = event.data["speechResult"]["speech"]
            _logger.info(f"Voice recognition ({call.call_id}): {speech_text}")

            if speech_text is not None and len(speech_text) > 0:
                call.messages.append(
                    MessageModel(content=speech_text, persona=MessagePersona.HUMAN)
                )
                call = await intelligence(background_tasks, call, client)

        elif recognition_result == "choices":  # Handle IVR
            label_detected = event.data["choiceResult"]["label"]

            try:
                lang = next(
                    (
                        x
                        for x in CONFIG.workflow.lang.availables
                        if x.short_code == label_detected
                    ),
                    CONFIG.workflow.lang.default_lang,
                )
                _logger.info(f"IVR recognition ({call.call_id}): {lang}")
            except ValueError:
                _logger.warn(f"Unknown IVR {label_detected}, code not implemented")
                return

            _logger.info(f"Setting call language to {lang} ({call.call_id})")
            call.lang = lang
            await db.call_aset(
                call
            )  # Persist language change, if the user calls back before the first message, the language will be set

            if len(call.messages) == 1:  # First call
                await handle_recognize_text(
                    call=call,
                    client=client,
                    text=await CONFIG.prompts.tts.hello(call),
                )

            if len(call.messages) > 1:  # Returning call
                await handle_play(
                    call=call,
                    client=client,
                    text=await CONFIG.prompts.tts.welcome_back(call),
                )
                call = await intelligence(background_tasks, call, client)

    elif (
        event_type == "Microsoft.Communication.RecognizeFailed"
    ):  # Speech recognition failed
        result_information = event.data["resultInformation"]
        error_code = result_information["subCode"]

        # Error codes:
        # 8510 = Action failed, initial silence timeout reached
        # 8532 = Action failed, inter-digit silence timeout reached
        # 8512 = Unknown internal server error
        # See: https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/communication-services/how-tos/call-automation/recognize-action.md#event-codes
        if error_code in (8510, 8532, 8512):  # Timeout retry
            if call.recognition_retry < 10:
                await handle_recognize_text(
                    call=call,
                    client=client,
                    text=await CONFIG.prompts.tts.timeout_silence(call),
                )
                call.recognition_retry += 1
            else:
                await handle_play(
                    call=call,
                    client=client,
                    context=ContextEnum.GOODBYE,
                    text=await CONFIG.prompts.tts.goodbye(call),
                )

        else:  # Other recognition error
            if error_code == 8511:  # Failure while trying to play the prompt
                _logger.warn(f"Failed to play prompt ({call.call_id})")
            else:
                _logger.warn(
                    f"Recognition failed with unknown error code {error_code}, answering with default error ({call.call_id})"
                )
            await handle_recognize_text(
                call=call,
                client=client,
                text=await CONFIG.prompts.tts.error(call),
            )

    elif event_type == "Microsoft.Communication.PlayCompleted":  # Media played
        _logger.debug(f"Play completed ({call.call_id})")

        if (
            operation_context == ContextEnum.TRANSFER_FAILED
            or operation_context == ContextEnum.GOODBYE
        ):  # Call ended
            _logger.info(f"Ending call ({call.call_id})")
            await handle_hangup(background_tasks, client, call)

        elif operation_context == ContextEnum.CONNECT_AGENT:  # Call transfer
            _logger.info(f"Initiating transfer call initiated ({call.call_id})")
            agent_caller = PhoneNumberIdentifier(
                str(CONFIG.workflow.agent_phone_number)
            )
            client.transfer_call_to_participant(target_participant=agent_caller)

    elif event_type == "Microsoft.Communication.PlayFailed":  # Media play failed
        _logger.debug(f"Play failed ({call.call_id})")

        result_information = event.data["resultInformation"]
        error_code = result_information["subCode"]

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

    elif (
        event_type == "Microsoft.Communication.CallTransferAccepted"
    ):  # Call transfer accepted
        _logger.info(f"Call transfer accepted event ({call.call_id})")
        # TODO: Is there anything to do here?

    elif (
        event_type == "Microsoft.Communication.CallTransferFailed"
    ):  # Call transfer failed
        _logger.debug(f"Call transfer failed event ({call.call_id})")
        result_information = event.data["resultInformation"]
        sub_code = result_information["subCode"]
        _logger.info(f"Error during call transfer, subCode {sub_code} ({call.call_id})")
        await handle_play(
            call=call,
            client=client,
            context=ContextEnum.TRANSFER_FAILED,
            text=await CONFIG.prompts.tts.calltransfer_failure(call),
        )

    await db.call_aset(call)


async def intelligence(
    background_tasks: BackgroundTasks, call: CallModel, client: CallConnectionClient
) -> CallModel:
    """
    Handle the intelligence of the call, including: LLM chat, TTS, and media play.

    Play the loading sound while waiting for the intelligence to be processed. If the intelligence is not processed after 15 seconds, play the timeout sound. If the intelligence is not processed after 30 seconds, stop the intelligence processing and play the error sound.
    """
    has_started = False

    async def tts_callback(text: str, style: MessageStyle) -> None:
        nonlocal has_started

        try:
            await safety_check(text)
        except SafetyCheckError as e:
            _logger.warn(f"Unsafe text detected, not playing ({call.call_id}): {e}")
            return

        has_started = True
        await handle_play(
            call=call,
            client=client,
            store=False,
            style=style,
            text=text,
        )

    chat_task = asyncio.create_task(
        llm_chat(background_tasks, call, tts_callback, client)
    )
    soft_timeout_task = asyncio.create_task(
        asyncio.sleep(CONFIG.workflow.intelligence_soft_timeout_sec)
    )
    soft_timeout_triggered = False
    hard_timeout_task = asyncio.create_task(
        asyncio.sleep(CONFIG.workflow.intelligence_hard_timeout_sec)
    )
    chat_action = None

    try:
        while True:
            _logger.debug(f"Chat task status ({call.call_id}): {chat_task.done()}")
            if chat_task.done():  # Break when chat coroutine is done
                # Clean up
                soft_timeout_task.cancel()
                hard_timeout_task.cancel()
                # Answer with chat result
                call, chat_action = chat_task.result()
                break
            if hard_timeout_task.done():  # Break when hard timeout is reached
                _logger.warn(
                    f"Hard timeout of {CONFIG.workflow.intelligence_hard_timeout_sec}s reached ({call.call_id})"
                )
                # Clean up
                chat_task.cancel()
                soft_timeout_task.cancel()
                break
            if not has_started:  # Catch timeout if async loading is not started
                if (
                    soft_timeout_task.done() and not soft_timeout_triggered
                ):  # Speak when soft timeout is reached
                    _logger.warn(
                        f"Soft timeout of {CONFIG.workflow.intelligence_soft_timeout_sec}s reached ({call.call_id})"
                    )
                    soft_timeout_triggered = True
                    await handle_play(
                        call=call,
                        client=client,
                        text=await CONFIG.prompts.tts.timeout_loading(call),
                    )
                else:  # Do not play timeout prompt plus loading, it can be frustrating for the user
                    await handle_media(
                        call=call,
                        client=client,
                        sound_url=CONFIG.prompts.sounds.loading(),
                    )  # Play loading sound
            # Wait to not block the event loop and play too many sounds
            await asyncio.sleep(5)
    except Exception:
        _logger.warn(f"Error loading intelligence ({call.call_id})", exc_info=True)

    # For any error reason, answer with error
    if not chat_action:
        _logger.debug(
            f"Error loading intelligence ({call.call_id}), answering with default error"
        )
        chat_action = ActionModel(
            content=await CONFIG.prompts.tts.error(call),
            intent=IndentAction.CONTINUE,
        )

    _logger.debug(f"Chat ({call.call_id}): {chat_action}")

    if chat_action.intent == IndentAction.TALK_TO_HUMAN:
        await handle_play(
            call=call,
            client=client,
            context=ContextEnum.CONNECT_AGENT,
            text=await CONFIG.prompts.tts.end_call_to_connect_agent(call),
        )

    elif chat_action.intent == IndentAction.END_CALL:
        await handle_play(
            call=call,
            client=client,
            context=ContextEnum.GOODBYE,
            text=await CONFIG.prompts.tts.goodbye(call),
        )

    elif chat_action.intent in (
        IndentAction.NEW_CLAIM,
        IndentAction.UPDATED_CLAIM,
        IndentAction.NEW_OR_UPDATED_REMINDER,
    ):
        # Save in DB for new claims and allowing demos to be more "real-time"
        await db.call_aset(call)
        # Recursively call intelligence to continue the conversation
        call = await intelligence(background_tasks, call, client)

    else:
        await handle_recognize_text(
            call=call,
            client=client,
        )

    return call


async def llm_completion(system: Optional[str], call: CallModel) -> Optional[str]:
    """
    Run LLM completion from a system prompt and a Call model.

    If the system prompt is None, no completion will be run and None will be returned. Otherwise, the response of the LLM will be returned.
    """
    _logger.debug(f"Running LLM completion ({call.call_id})")

    if not system:
        return None

    messages = _oai_completion_messages(system, call)
    content = None

    try:
        content = await completion_sync(
            max_tokens=1000,
            messages=messages,
        )
    except APIError:
        _logger.warn(f"OpenAI API call error", exc_info=True)
    except SafetyCheckError as e:
        _logger.warn(f"OpenAI safety check error: {e}")

    return content


async def llm_model(
    system: Optional[str], call: CallModel, model: Type[ModelType]
) -> Optional[ModelType]:
    """
    Run LLM completion from a system prompt, a Call model, and an expected model type as a return.

    The logic will try its best to return a model of the expected type, but it is not guaranteed. It it fails, `None` will be returned.
    """
    _logger.debug(f"Running LLM model ({call.call_id})")

    if not system:
        return None

    messages = _oai_completion_messages(system, call)
    res = None

    try:
        res = await completion_model_sync(
            max_tokens=1000,
            messages=messages,
            model=model,
        )
    except APIError:
        _logger.warn(f"OpenAI API call error", exc_info=True)

    return res


def _oai_completion_messages(
    system: str, call: CallModel
) -> List[ChatCompletionMessageParam]:
    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            content=CONFIG.prompts.llm.default_system(
                phone_number=call.phone_number,
            ),
            role="system",
        ),
        ChatCompletionSystemMessageParam(
            content=system,
            role="system",
        ),
    ]
    _logger.debug(f"Messages: {messages}")
    return messages


async def llm_chat(
    background_tasks: BackgroundTasks,
    call: CallModel,
    user_callback: Callable[[str, MessageStyle], Coroutine[Any, Any, None]],
    client: CallConnectionClient,
    _retry_attempt: int = 0,
) -> Tuple[CallModel, ActionModel]:
    _logger.debug(f"Running LLM chat ({call.call_id})")

    def _remove_message_actions(text: str) -> str:
        """
        Remove action from content. AI often adds it by mistake event if explicitly asked not to.
        """
        res = re.match(MESSAGE_ACTION_R, text)
        if not res:
            return text.strip()
        content = res.group(2)
        return content.strip() if content else ""

    def _extract_message_style(text: str) -> Tuple[Optional[MessageStyle], str]:
        """
        Detect the style of a message.
        """
        res = re.match(MESSAGE_STYLE_R, text)
        if not res:
            return None, text
        try:
            content = res.group(2)
            return MessageStyle(res.group(1)), content.strip() if content else ""
        except ValueError:
            return None, text

    async def _buffer_user_callback(buffer: str, style: MessageStyle) -> MessageStyle:
        # Remove tool calls from buffer content and detect style
        local_style, local_content = _extract_message_style(
            _remove_message_actions(buffer)
        )
        new_style = local_style or style
        # Batch current user return
        if local_content:
            await user_callback(local_content, new_style)
        return new_style

    async def _error_response() -> Tuple[CallModel, ActionModel]:
        content = await CONFIG.prompts.tts.error(call)
        style = MessageStyle.NONE
        await user_callback(content, style)
        call.messages.append(
            MessageModel(
                content=content,
                persona=MessagePersona.ASSISTANT,
                style=style,
            )
        )
        return (
            call,
            ActionModel(
                content=content,
                intent=IndentAction.CONTINUE,
            ),
        )

    kernel = build_kernel()

    kernel.import_plugin(
        CallPlugin(
            call=call,
            client=client,
        ),
        "Call",
    )
    kernel.import_plugin(
        ClaimPlugin(
            background_tasks=background_tasks,
            call=call,
            post_call_next=post_call_next,
            post_call_synthesis=post_call_synthesis,
        ),
        "Claim",
    )
    kernel.import_plugin(
        ReminderPlugin(
            call=call,
            client=client,
        ),
        "Reminder",
    )

    function = CONFIG.prompts.llm.chat_system(kernel=kernel, call=call)

    full_content = ""
    buffer_content = ""
    default_style = MessageStyle.NONE
    tool_calls = {}
    try:
        async for delta in completion_stream(
            max_tokens=350,
            messages=messages,
            tools=tools,
        ):
            if delta.content is None:
                for piece in delta.tool_calls or []:
                    tool_calls[piece.index] = tool_calls.get(
                        piece.index,
                        {
                            "function": {"arguments": "", "name": ""},
                            "id": None,
                            "type": "function",
                        },
                    )
                    if piece.id:
                        tool_calls[piece.index]["id"] = piece.id
                    if piece.function:
                        if piece.function.name:
                            tool_calls[piece.index]["function"][
                                "name"
                            ] = piece.function.name
                        tool_calls[piece.index]["function"][
                            "arguments"
                        ] += piece.function.arguments
            else:
                # Store whole content
                full_content += delta.content
                buffer_content += delta.content
                for local_content in _sentence_split(buffer_content):
                    buffer_content = buffer_content[
                        len(local_content) :
                    ]  # Remove consumed content from buffer
                    default_style = await _buffer_user_callback(
                        local_content, default_style
                    )

        if buffer_content:
            default_style = await _buffer_user_callback(buffer_content, default_style)

        # Get data from full content to be able to store it in the DB
        _, full_content = _extract_message_style(_remove_message_actions(full_content))

        _logger.debug(f"Chat response: {full_content}")
        _logger.debug(f"Tool calls: {tool_calls}")

        # OpenAI GPT-4 Turbo sometimes return wrong tools schema, in that case, retry within limits
        # TODO: Tries to detect this error earlier
        # See: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653
        if any(
            x["function"]["name"] == "multi_tool_use.parallel"
            for _, x in tool_calls.items()
        ):
            _logger.debug(f"Invalid tool schema: {tool_calls}")
            if _retry_attempt > 3:
                _logger.warn(
                    f'LLM send back invalid tool schema "multi_tool_use.parallel", retry limit reached'
                )
                return await _error_response()
            _logger.warn(
                f'LLM send back invalid tool schema "multi_tool_use.parallel", retrying'
            )
            return await llm_chat(
                background_tasks, call, user_callback, client, _retry_attempt + 1
            )

        # OpenAI GPT-4 Turbo tends to return empty content, in that case, retry within limits
        if not full_content and not tool_calls:
            _logger.debug(f"Empty content, retrying")
            if _retry_attempt > 3:
                _logger.warn(f"LLM send back empty content, retry limit reached")
                return await _error_response()
            _logger.warn(f"LLM send back empty content, retrying")
            return await llm_chat(
                background_tasks,
                call,
                user_callback,
                client,
                _retry_attempt + 1,
            )

        intent = IndentAction.CONTINUE
        models = []
        if tool_calls:
            # TODO: Catch tool error individually
            for _, tool_call in tool_calls.items():
                name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                _logger.info(f"Tool call {name} with parameters {arguments}")

                model = MessageToolModel(
                    content="",
                    function_arguments=arguments,
                    function_name=name,
                    tool_id=tool_call["id"],
                )

                if name == IndentAction.TALK_TO_HUMAN.value:
                    intent = IndentAction.TALK_TO_HUMAN

                elif name == IndentAction.END_CALL.value:
                    intent = IndentAction.END_CALL

                elif name == IndentAction.UPDATED_CLAIM.value:
                    intent = IndentAction.UPDATED_CLAIM
                    try:
                        parameters = json.loads(arguments)
                    except Exception:
                        _logger.warn(
                            f'LLM send back invalid JSON for "{arguments}", ignoring this tool call.'
                        )
                        continue

                    if not customer_response_prop in parameters:
                        _logger.warn(
                            f"Missing {customer_response_prop} prop in {arguments}, please fix this!"
                        )
                    else:
                        local_content = parameters[customer_response_prop]
                        full_content += local_content + " "
                        await user_callback(local_content, default_style)

                    field = parameters["field"]
                    value = parameters["value"]
                    if not field in ClaimModel.editable_fields():
                        content = f'Failed to update a non-editable field "{field}".'
                    else:
                        try:
                            # Define the field and force to trigger validation
                            copy = call.claim.model_dump()
                            copy[field] = value
                            call.claim = ClaimModel.model_validate(copy)
                            content = (
                                f'Updated claim field "{field}" with value "{value}".'
                            )
                        except ValidationError as e:  # Catch error to inform LLM
                            content = f'Failed to edit field "{field}": {e.json()}'
                    model.content = content

                elif name == IndentAction.NEW_CLAIM.value:
                    intent = IndentAction.NEW_CLAIM
                    try:
                        parameters = json.loads(arguments)
                    except Exception:
                        _logger.warn(
                            f'LLM send back invalid JSON for "{arguments}", ignoring this tool call.'
                        )
                        continue

                    if not customer_response_prop in parameters:
                        _logger.warn(
                            f"Missing {customer_response_prop} prop in {arguments}, please fix this!"
                        )
                    else:
                        local_content = parameters[customer_response_prop]
                        full_content += local_content + " "
                        await user_callback(local_content, default_style)

                    # Generate next action
                    background_tasks.add_task(post_call_next, call)
                    # Generate synthesis
                    background_tasks.add_task(post_call_synthesis, call)

                    # Add context of the last message, if not, LLM messed up and loop on this action
                    last_message = call.messages[-1]
                    call = CallModel(phone_number=call.phone_number)
                    call.messages.append(last_message)
                    model.content = "Claim, reminders and messages reset."

                elif name == IndentAction.NEW_OR_UPDATED_REMINDER.value:
                    intent = IndentAction.NEW_OR_UPDATED_REMINDER
                    try:
                        parameters = json.loads(arguments)
                    except Exception:
                        _logger.warn(
                            f'LLM send back invalid JSON for "{arguments}", ignoring this tool call.'
                        )
                        continue

                    if not customer_response_prop in parameters:
                        _logger.warn(
                            f'Missing "{customer_response_prop}" prop in "{arguments}", please fix this!'
                        )
                    else:
                        local_content = parameters[customer_response_prop]
                        full_content += local_content + " "
                        await user_callback(local_content, default_style)

                    updated = False
                    title = parameters["title"]
                    for reminder in call.reminders:
                        if reminder.title == title:
                            try:
                                reminder.description = parameters["description"]
                                reminder.due_date_time = parameters["due_date_time"]
                                reminder.owner = parameters["owner"]
                                content = f'Reminder "{title}" updated.'
                            except ValidationError as e:  # Catch error to inform LLM
                                content = (
                                    f'Failed to edit reminder "{title}": {e.json()}'
                                )
                            model.content = content
                            updated = True
                            break

                    if not updated:
                        try:
                            reminder = ReminderModel(
                                description=parameters["description"],
                                due_date_time=parameters["due_date_time"],
                                title=title,
                                owner=parameters["owner"],
                            )
                            call.reminders.append(reminder)
                            content = f'Reminder "{title}" created.'
                        except ValidationError as e:  # Catch error to inform LLM
                            content = f'Failed to create reminder "{title}": {e.json()}'
                        model.content = content

                models.append(model)

        call.messages.append(
            MessageModel(
                content=full_content,
                persona=MessagePersona.ASSISTANT,
                style=default_style,
                tool_calls=models,
            )
        )

        return (
            call,
            ActionModel(
                content=full_content,
                intent=intent,
            ),
        )

    except APIError:
        _logger.warn(f"OpenAI API call error", exc_info=True)

    return await _error_response()


async def handle_hangup(
    background_tasks: BackgroundTasks, client: CallConnectionClient, call: CallModel
) -> None:
    _logger.debug(f"Hanging up call ({call.call_id})")
    try:
        client.hang_up(is_for_everyone=True)
    except ResourceNotFoundError:
        _logger.debug(f"Call already hung up ({call.call_id})")
    except HttpResponseError as e:
        if "call already terminated" in e.message.lower():
            _logger.debug(f"Call hung up before playing ({call.call_id})")
        else:
            raise e

    call.messages.append(
        MessageModel(
            content="",
            persona=MessagePersona.HUMAN,
            action=MessageAction.HANGUP,
        )
    )

    # Start post-call intelligence
    background_tasks.add_task(post_call_next, call)
    background_tasks.add_task(post_call_sms, call)
    background_tasks.add_task(post_call_synthesis, call)


async def post_call_sms(call: CallModel) -> None:
    """
    Send an SMS report to the customer.
    """
    content = await llm_completion(
        system=CONFIG.prompts.llm.sms_summary_system(call),
        call=call,
    )

    if not content:
        _logger.warn(f"Error generating SMS report ({call.call_id})")
        return

    _logger.info(f"SMS report ({call.call_id}): {content}")
    try:
        responses = sms_client.send(
            from_=str(CONFIG.communication_service.phone_number),
            message=content,
            to=call.phone_number,
        )
        response = responses[0]

        if response.successful:
            _logger.debug(
                f"SMS report sent {response.message_id} to {response.to} ({call.call_id})"
            )
            call.messages.append(
                MessageModel(
                    action=MessageAction.SMS,
                    content=content,
                    persona=MessagePersona.ASSISTANT,
                )
            )
            await db.call_aset(call)
        else:
            _logger.warn(
                f"Failed SMS to {response.to}, status {response.http_status_code}, error {response.error_message} ({call.call_id})"
            )

    except ClientAuthenticationError:
        _logger.error("Authentication error for SMS, check the credentials")
    except Exception:
        _logger.warn(
            f"Failed SMS to {call.phone_number} ({call.call_id})", exc_info=True
        )


async def callback_url(caller_id: str) -> str:
    """
    Generate the callback URL for a call.

    If the caller has already called, use the same call ID, to keep the conversation history. Otherwise, create a new call ID.
    """
    call = await db.call_asearch_one(caller_id)
    if not call:
        call = CallModel(phone_number=caller_id)
        await db.call_aset(call)
    return CALL_EVENT_URL.format(
        callback_secret=html.escape(call.callback_secret.get_secret_value()),
        phone_number=html.escape(call.phone_number),
    )


async def post_call_synthesis(call: CallModel) -> None:
    """
    Synthesize the call and store it to the model.
    """
    _logger.debug(f"Synthesizing call ({call.call_id})")

    short, long = await asyncio.gather(
        llm_completion(
            call=call,
            system=CONFIG.prompts.llm.synthesis_short_system(call),
        ),
        llm_completion(
            call=call,
            system=CONFIG.prompts.llm.citations_system(
                call=call,
                text=await llm_completion(
                    call=call,
                    system=CONFIG.prompts.llm.synthesis_long_system(call),
                ),
            ),
        ),
    )

    if not short or not long:
        _logger.warn(f"Error generating synthesis ({call.call_id})")
        return

    _logger.info(f"Short synthesis ({call.call_id}): {short}")
    _logger.info(f"Long synthesis ({call.call_id}): {long}")

    call.synthesis = SynthesisModel(
        long=long,
        short=short,
    )
    await db.call_aset(call)


async def post_call_next(call: CallModel) -> None:
    """
    Generate next action for the call.
    """
    next = await llm_model(
        call=call,
        model=NextModel,
        system=CONFIG.prompts.llm.next_system(call),
    )

    if not next:
        _logger.warn(f"Error generating next action ({call.call_id})")
        return

    _logger.info(f"Next action ({call.call_id}): {next}")
    call.next = next
    await db.call_aset(call)


async def handle_ivr_language(
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
