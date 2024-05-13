from azure.communication.callautomation import RecognitionChoice
from fastapi import BackgroundTasks
from helpers.logging import build_logger
from helpers.pydantic_types.phone_numbers import PhoneNumber
from models.call import CallStateModel
from models.message import StyleEnum as MessageStyleEnum
from models.readiness import ReadinessStatus
from persistence.ivoice import IVoice
from typing import Callable, Optional


_logger = build_logger(__name__)


class VoiceMock(IVoice):
    _text_callback: Callable[[str], None]

    def __init__(self, text_callback: Callable[[str], None]) -> None:
        _logger.warning(
            "Mock voice is used, no real calls will be made, this is for testing only"
        )
        self._text_callback = text_callback

    async def areadiness(self) -> ReadinessStatus:
        return ReadinessStatus.OK

    async def acreate(
        self,
        call: CallStateModel,
        callback_url: str,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
    ) -> None:
        _logger.info("acreate, ignoring")

    async def aanswer(
        self,
        call: CallStateModel,
        callback_url: str,
        incoming_context: str,
        background_tasks: BackgroundTasks,
    ) -> None:
        _logger.info("aanswer, ignoring")

    async def atransfer(
        self,
        call: CallStateModel,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        _logger.info("atransfer, ignoring")

    async def ahangup(
        self,
        call: CallStateModel,
        everyone: bool,
    ) -> None:
        _logger.info("ahangup, ignoring")

    async def aplay_audio(
        self,
        call: CallStateModel,
        url: str,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        _logger.info("aplay_audio, ignoring")

    async def aplay_text(
        self,
        call: CallStateModel,
        text: str,
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        self._text_callback(text.strip())

    async def arecognize_ivr(
        self,
        call: CallStateModel,
        text: str,
        choices: list[RecognitionChoice],
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
    ) -> None:
        _logger.info("arecognize_ivr, ignoring")

    async def arecognize_speech(
        self,
        call: CallStateModel,
        background_tasks: BackgroundTasks,
        text: Optional[str] = None,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        if text:
            self._text_callback(text.strip())
