from azure.communication.callautomation import RecognitionChoice, DtmfTone
from fastapi import BackgroundTasks
from helpers.logging import build_logger
from helpers.pydantic_types.phone_numbers import PhoneNumber
from models.call import CallStateModel
from models.message import StyleEnum as MessageStyleEnum
from models.readiness import ReadinessStatus
from persistence.ivoice import IVoice
from typing import Optional


_logger = build_logger(__name__)


class ConsoleVoice(IVoice):
    def __init__(self):
        _logger.warning("Using console as voice, no real calls will be made")

    async def areadiness(self) -> ReadinessStatus:
        """
        Check the readiness of the console voice.
        """
        return ReadinessStatus.OK  # Always ready, it's memory :)

    async def acreate(
        self,
        call: CallStateModel,
        callback_url: str,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
    ) -> None:
        from helpers.call_events import on_call_connected

        _logger.info(f"📞 New call")
        background_tasks.add_task(
            on_call_connected,
            background_tasks=background_tasks,
            call=call,
        )

    async def aanswer(
        self,
        call: CallStateModel,
        callback_url: str,
        incoming_context: str,
        background_tasks: BackgroundTasks,
    ) -> None:
        from helpers.call_events import on_call_connected

        _logger.info(f"🤖 Answering call (context {incoming_context})")
        background_tasks.add_task(
            on_call_connected,
            background_tasks=background_tasks,
            call=call,
        )

    async def atransfer(
        self,
        call: CallStateModel,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        from helpers.call_events import on_transfer_completed

        _logger.info(f"👩‍💼 Transferring call to {phone_number}")
        background_tasks.add_task(
            on_transfer_completed,
            background_tasks=background_tasks,
            call=call,
        )

    async def ahangup(
        self,
        call: CallStateModel,
        everyone: bool,
    ) -> None:
        _logger.info(f"📞 Hangup")

    async def aplay_audio(
        self,
        call: CallStateModel,
        url: str,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        from helpers.call_events import on_play_completed

        _logger.info(f"🔈 {url}")
        background_tasks.add_task(
            on_play_completed,
            background_tasks=background_tasks,
            call=call,
            context=context,
        )

    async def aplay_text(
        self,
        call: CallStateModel,
        text: str,
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        from helpers.call_events import on_play_completed

        # Store text if requested
        if store:
            self._store_message_in_call(
                call=call,
                style=style,
                text=text,
            )

        _logger.info(f"🤖 {text}")
        background_tasks.add_task(
            on_play_completed,
            background_tasks=background_tasks,
            call=call,
            context=context,
        )

    async def arecognize_ivr(
        self,
        call: CallStateModel,
        text: str,
        choices: list[RecognitionChoice],
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
    ) -> None:
        from helpers.call_events import on_ivr_recognized

        _logger.info(f"🤖 {text}")
        for i, choice in enumerate(choices):
            assert isinstance(choice.tone, DtmfTone)
            _logger.info(f"{i}: {choice.phrases[0]} ({choice.label})")
        res = input("🤖 Select a choice: ")
        selected = (
            choices[int(res)]
            if res.isdigit() and int(res) < len(choices)
            else choices[0]
        ).label
        _logger.info(f"Selected choice: {selected}")
        background_tasks.add_task(
            on_ivr_recognized,
            call=call,
            label=selected,
            background_tasks=background_tasks,
        )

    async def arecognize_speech(
        self,
        call: CallStateModel,
        background_tasks: BackgroundTasks,
        text: Optional[str] = None,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        from helpers.call_events import on_speech_recognized

        if text:
            await self.aplay_text(
                call=call,
                text=text,
                background_tasks=background_tasks,
                style=style,
                context=context,
                store=store,
            )

        res = input("Enter response: ")
        background_tasks.add_task(
            on_speech_recognized,
            call=call,
            text=res,
            background_tasks=background_tasks,
        )
