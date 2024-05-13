from abc import ABC, abstractmethod
from azure.communication.callautomation import RecognitionChoice
from enum import Enum
from fastapi import BackgroundTasks
from helpers.pydantic_types.phone_numbers import PhoneNumber
from models.call import CallStateModel
from models.message import (
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
    StyleEnum as MessageStyleEnum,
)
from models.readiness import ReadinessStatus
from typing import Generator, Optional
import re


_SENTENCE_PUNCTUATION_R = r"(\. |\.$|[!?;])"  # Split by sentence by punctuation


class ContextEnum(str, Enum):
    """
    Enum for call context.

    Used to track the operation context of a call in Azure Communication Services.
    """

    CONNECT_AGENT = "connect_agent"  # Transfer to agent
    GOODBYE = "goodbye"  # Hang up
    TRANSFER_FAILED = "transfer_failed"  # Transfer failed


class IVoice(ABC):

    @abstractmethod
    async def areadiness(self) -> ReadinessStatus:
        pass

    @abstractmethod
    async def acreate(
        self,
        call: CallStateModel,
        callback_url: str,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
    ) -> None:
        pass

    @abstractmethod
    async def aanswer(
        self,
        call: CallStateModel,
        callback_url: str,
        incoming_context: str,
        background_tasks: BackgroundTasks,
    ) -> None:
        pass

    @abstractmethod
    async def atransfer(
        self,
        call: CallStateModel,
        phone_number: PhoneNumber,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    async def ahangup(
        self,
        call: CallStateModel,
        everyone: bool,
    ) -> None:
        pass

    @abstractmethod
    async def aplay_audio(
        self,
        call: CallStateModel,
        url: str,
        background_tasks: BackgroundTasks,
        context: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    async def aplay_text(
        self,
        call: CallStateModel,
        text: str,
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        pass

    @abstractmethod
    async def arecognize_ivr(
        self,
        call: CallStateModel,
        text: str,
        choices: list[RecognitionChoice],
        background_tasks: BackgroundTasks,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    async def arecognize_speech(
        self,
        call: CallStateModel,
        background_tasks: BackgroundTasks,
        text: Optional[str] = None,
        style: MessageStyleEnum = MessageStyleEnum.NONE,
        context: Optional[str] = None,
        store: bool = True,
    ) -> None:
        pass

    @staticmethod
    def tts_sentence_split(text: str, include_last: bool) -> Generator[str, None, None]:
        """
        Split a text into sentences.
        """
        # Split by sentence by punctuation
        splits = re.split(_SENTENCE_PUNCTUATION_R, text)
        for i, split in enumerate(splits):
            if i % 2 == 1:  # Skip punctuation
                continue
            if not split:  # Skip empty lines
                continue
            if i == len(splits) - 1:  # Skip last line in case of missing punctuation
                if include_last:
                    yield split
            else:  # Add punctuation back
                yield split + splits[i + 1]

    @staticmethod
    def _store_message_in_call(
        call: CallStateModel, text: str, style: MessageStyleEnum
    ) -> None:
        """
        Store text in call messages.
        """
        if (
            call.messages and call.messages[-1].persona == MessagePersonaEnum.ASSISTANT
        ):  # If the last message was from the assistant, append to it
            call.messages[-1].content += f" {text}"
        else:  # Otherwise, create a new message
            call.messages.append(
                MessageModel(
                    content=text,
                    persona=MessagePersonaEnum.ASSISTANT,
                    style=style,
                )
            )
