from azure.communication.sms import SmsSendResult
from azure.communication.sms.aio import SmsClient
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from contextlib import asynccontextmanager
from helpers.config_models.communication_services import CommunicationServicesModel
from helpers.logging import build_logger
from helpers.pydantic_types.phone_numbers import PhoneNumber
from models.readiness import ReadinessStatus
from persistence.isms import ISms
from models.call import CallStateModel
from persistence.ivoice import IVoice
from fastapi import BackgroundTasks
from typing import AsyncGenerator, Generator, Optional
from azure.communication.callautomation.aio import (
    CallAutomationClient,
    CallConnectionClient,
)
from azure.communication.callautomation import (
    CommunicationIdentifier,
    FileSource,
    PhoneNumberIdentifier,
    RecognitionChoice,
    RecognizeInputType,
    SsmlSource,
)
from models.message import StyleEnum as MessageStyleEnum
from helpers.config import CONFIG
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from helpers.config_models.prompts import SoundModel
import re
from textwrap import dedent


_logger = build_logger(__name__)
_db = CONFIG.database.instance()
_TTS_SANITIZER_R = re.compile(
    r"[^\w\s'«»“”\"\"‘’''(),.!?;\-\+_@/]"
)  # Sanitize text for TTS


class CommunicationServicesVoice(IVoice):
    _config: CommunicationServicesModel
    _sound: SoundModel
    _source_caller: PhoneNumberIdentifier

    def __init__(self, config: CommunicationServicesModel, sound: SoundModel):
        _logger.info(f"Using Communication Services from number {config.phone_number}")
        self._config = config
        self._sound = sound
        self._source_caller = PhoneNumberIdentifier(config.phone_number)

    async def areadiness(self) -> ReadinessStatus:
        """
        Check the readiness of the Communication Services call service.
        """
        # TODO: SDK does not provide a way to check the readiness of the call service.
        return ReadinessStatus.OK

    async def acreate(
        self,
        call: CallStateModel,
        callback_url: str,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
    ) -> None:
        async with self._use_automation_client() as client:
            res = await client.create_call(
                callback_url=callback_url,
                cognitive_services_endpoint=self._config.endpoint,
                source_caller_id_number=self._source_caller,
                target_participant=PhoneNumberIdentifier(phone_number),  # type: ignore
            )
        assert res.call_connection_id, "Call ID not returned"
        call.voice_id = res.call_connection_id  # Store call ID
        await _db.call_aset(call)

    async def aanswer(
        self,
        call: CallStateModel,
        callback_url: str,
        incoming_context: str,
        background_tasks: BackgroundTasks,
    ) -> None:
        try:
            async with self._use_automation_client() as client:
                res = await client.answer_call(
                    callback_url=callback_url,
                    cognitive_services_endpoint=self._config.endpoint,
                    incoming_call_context=incoming_context,
                )
        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before answering")
            return
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before answering")
                return
            else:
                raise e

        assert res.call_connection_id, "Call ID not returned"
        call.voice_id = res.call_connection_id  # Store call ID
        await _db.call_aset(call)

    async def atransfer(
        self,
        call: CallStateModel,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        try:
            async with self._use_connection_client(call) as client:
                await client.transfer_call_to_participant(
                    operation_context=context,
                    target_participant=PhoneNumberIdentifier(phone_number),  # type: ignore
                )
        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before transferring")
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before transferring")
            else:
                raise e

    async def ahangup(
        self,
        call: CallStateModel,
        everyone: bool,
    ) -> None:
        try:
            async with self._use_connection_client(call) as client:
                await client.hang_up(is_for_everyone=everyone)
        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before hanging up")
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before hanging up")
            else:
                raise e

    async def aplay_audio(
        self,
        call: CallStateModel,
        url: str,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        try:
            async with self._use_connection_client(call) as client:
                await client.play_media(
                    operation_context=context,
                    play_source=FileSource(url=url),
                )
        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before playing")
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before playing")
            else:
                raise e

    async def aplay_text(
        self,
        call: CallStateModel,
        text: str,
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        lang = call.lang.short_code
        voice = call.lang.voice

        # Store text if requested
        if store:
            self._store_message_in_call(
                call=call,
                style=style,
                text=text,
            )

        # Play each chunk
        try:
            async with self._use_connection_client(call) as client:
                for chunk in self._chuncks_from_text(text):
                    _logger.info(f"Playing text: {text} ({lang}, {style})")
                    play_prompt = self._audio_from_text(
                        lang=lang,
                        style=style,
                        text=chunk,
                        voice=voice,
                    )
                    await client.play_media(
                        operation_context=context,
                        play_source=play_prompt,
                    )
        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before playing")
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before playing")
            else:
                raise e

    async def arecognize_ivr(
        self,
        call: CallStateModel,
        text: str,
        choices: list[RecognitionChoice],
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
    ) -> None:
        lang = call.lang.short_code
        voice = call.lang.voice
        target_participant: CommunicationIdentifier = PhoneNumberIdentifier(call.initiate.phone_number)  # type: ignore
        play_prompt = self._audio_from_text(
            lang=lang,
            style=style,
            text=text,
            voice=voice,
        )

        try:
            async with self._use_connection_client(call) as client:
                _logger.info(f"Playing text: {text} ({lang}, {style})")
                await client.start_recognizing_media(
                    choices=choices,
                    end_silence_timeout=20,
                    input_type=RecognizeInputType.CHOICES,
                    interrupt_prompt=True,
                    operation_context=context,
                    play_prompt=play_prompt,
                    speech_language=lang,
                    target_participant=target_participant,
                )
        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before recognizing")
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before recognizing")
            else:
                raise e

    async def arecognize_speech(
        self,
        call: CallStateModel,
        background_tasks: BackgroundTasks,
        text: Optional[str] = None,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        target_participant: CommunicationIdentifier = PhoneNumberIdentifier(call.initiate.phone_number)  # type: ignore
        play_prompt = FileSource(url=self._sound.ready())
        lang = call.lang.short_code

        try:
            if text:
                await self.aplay_text(
                    background_tasks=background_tasks,
                    call=call,
                    context=context,
                    store=store,
                    style=style,
                    text=text,
                )

            async with self._use_connection_client(call) as client:
                await client.start_recognizing_media(
                    end_silence_timeout=3,  # Sometimes user includes breaks in their speech
                    input_type=RecognizeInputType.SPEECH,
                    interrupt_prompt=True,
                    operation_context=context,
                    play_prompt=play_prompt,
                    speech_language=lang,
                    target_participant=target_participant,
                )

        except ResourceNotFoundError:
            _logger.debug(f"Call hung up before recognizing")
        except HttpResponseError as e:
            if "call already terminated" in e.message.lower():
                _logger.debug(f"Call hung up before recognizing")
            else:
                raise e

    def _audio_from_text(
        self, text: str, style: MessageStyleEnum, lang: str, voice: str
    ) -> SsmlSource:
        """
        Generate an audio source that can be read by Azure Communication Services SDK.

        Text requires to be SVG escaped, and SSML tags are used to control the voice. Plus, text is slowed down by 5% to make it more understandable for elderly people. Text is also truncated to 400 characters, as this is the limit of Azure Communication Services TTS, but a warning is logged.
        """
        # Azure Speech Service TTS limit is 400 characters
        if len(text) > 400:
            _logger.warning(
                f"Text is too long to be processed by TTS, truncating to 400 characters, fix this!"
            )
            text = text[:400]
        ssml = dedent(
            f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{lang}">
                <voice name="{voice}" effect="eq_telecomhp8k">
                    <lexicon uri="{CONFIG.resources.public_url}/lexicon.xml" />
                    <mstts:express-as style="{style.value}" styledegree="0.5">
                        <prosody rate="0.95">{text}</prosody>
                    </mstts:express-as>
                </voice>
            </speak>
        """
        )
        _logger.debug(f"SSML: {ssml}")
        return SsmlSource(ssml_text=ssml)

    def _chuncks_from_text(self, text: str) -> Generator[str, None, None]:
        # Sanitize text for TTS
        text = re.sub(_TTS_SANITIZER_R, "", text)

        # Split text in chunks of max 400 characters, separated by sentence
        buffer = ""
        for sentence in self.tts_sentence_split(text, True):
            if len(buffer) + len(sentence) >= 400:
                yield buffer.strip()  # Remove trailing space
                buffer = ""
            buffer += sentence

    @asynccontextmanager
    async def _use_connection_client(
        self, call: CallStateModel
    ) -> AsyncGenerator[CallConnectionClient, None]:
        assert call.voice_id, "Call has no voice ID"
        client = CallConnectionClient(
            api_version="2023-10-15",
            endpoint=CONFIG.communication_services.endpoint,
            call_connection_id=call.voice_id,
            credential=AzureKeyCredential(
                CONFIG.communication_services.access_key.get_secret_value()
            ),
        )
        try:
            yield client
        finally:
            await client.close()

    @asynccontextmanager
    async def _use_automation_client(
        self,
    ) -> AsyncGenerator[CallAutomationClient, None]:
        client = CallAutomationClient(
            # api_version="2023-10-15",
            endpoint=CONFIG.communication_services.endpoint,
            credential=AzureKeyCredential(
                CONFIG.communication_services.access_key.get_secret_value()
            ),
        )
        try:
            yield client
        finally:
            await client.close()


class CommunicationServicesSms(ISms):
    _config: CommunicationServicesModel

    def __init__(self, config: CommunicationServicesModel):
        _logger.info(f"Using Communication Services from number {config.phone_number}")
        self._config = config

    async def areadiness(self) -> ReadinessStatus:
        """
        Check the readiness of the Communication Services SMS service.
        """
        # TODO: How to check the readiness of the SMS service? We could send a SMS for each test, but that would be damm expensive.
        return ReadinessStatus.OK

    async def asend(self, content: str, phone_number: PhoneNumber) -> bool:
        _logger.info(f"Sending SMS to {phone_number}")
        success = False
        _logger.info(f"SMS content: {content}")
        try:
            async with self._use_client() as client:
                responses: list[SmsSendResult] = await client.send(
                    from_=str(self._config.phone_number),
                    message=content,
                    to=str(phone_number),
                )
                response = responses[0]
                if response.successful:
                    _logger.debug(f"SMS sent {response.message_id} to {response.to}")
                    success = True
                else:
                    _logger.warning(
                        f"Failed SMS to {response.to}, status {response.http_status_code}, error {response.error_message}"
                    )
        except ClientAuthenticationError:
            _logger.error(
                "Authentication error for SMS, check the credentials", exc_info=True
            )
        except HttpResponseError as e:
            _logger.error(f"Error sending SMS: {e}")
        except Exception:
            _logger.warning(f"Failed SMS to {phone_number}", exc_info=True)
        return success

    @asynccontextmanager
    async def _use_client(self) -> AsyncGenerator[SmsClient, None]:
        client = SmsClient(
            credential=self._config.access_key.get_secret_value(),
            endpoint=self._config.endpoint,
        )
        try:
            yield client
        finally:
            await client.close()
