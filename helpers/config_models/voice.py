from enum import Enum
from functools import cache
from persistence.ivoice import IVoice
from pydantic import field_validator, BaseModel, ValidationInfo
from typing import Callable, Optional


class ModeEnum(str, Enum):
    COMMUNICATION_SERVICES = "communication_services"
    CONSOLE = "console"
    MOCK = "mock"


class CommunicationServicesModel(BaseModel, frozen=True):
    """
    Represents the configuration for the Communication Services API.

    Model is purely empty to fit to the `IVoice` interface and the "mode" enum code organization. As the Communication Services is also used as the only call interface, it is not necessary to duplicate the models.
    """

    @cache
    def instance(self) -> IVoice:
        from persistence.communication_services import CommunicationServicesVoice
        from helpers.config import CONFIG

        return CommunicationServicesVoice(
            CONFIG.communication_services, CONFIG.prompts.sounds
        )


class ConsoleModel(BaseModel, frozen=True):
    """
    Represents the configuration for the Console voice.
    """

    @cache
    def instance(self) -> IVoice:
        from persistence.console import ConsoleVoice

        return ConsoleVoice()


class MockModel(BaseModel, frozen=True):
    """
    Represents the configuration for the Mock voice.
    """

    text_callback: Callable[[str], None]

    @cache
    def instance(self) -> IVoice:
        from persistence.mock import VoiceMock

        return VoiceMock(self.text_callback)


class VoiceModel(BaseModel):
    communication_services: Optional[CommunicationServicesModel] = (
        CommunicationServicesModel()
    )  # Object is fully defined by default
    console: Optional[ConsoleModel] = (
        ConsoleModel()
    )  # Object is fully defined by default
    mock: Optional[MockModel] = None
    mode: ModeEnum = ModeEnum.COMMUNICATION_SERVICES

    @field_validator("communication_services")
    def _validate_communication_services(
        cls,
        communication_services: Optional[CommunicationServicesModel],
        info: ValidationInfo,
    ) -> Optional[CommunicationServicesModel]:
        if (
            not communication_services
            and info.data.get("mode", None) == ModeEnum.COMMUNICATION_SERVICES
        ):
            raise ValueError("Communication Services config required")
        return communication_services

    @field_validator("mock")
    def _validate_mock(
        cls, mock: Optional[MockModel], info: ValidationInfo
    ) -> Optional[MockModel]:
        if not mock and info.data.get("mode", None) == ModeEnum.MOCK:
            raise ValueError("Mock config required")
        return mock

    @field_validator("console")
    def _validate_console(
        cls, console: Optional[ConsoleModel], info: ValidationInfo
    ) -> Optional[ConsoleModel]:
        if not console and info.data.get("mode", None) == ModeEnum.CONSOLE:
            raise ValueError("Console config required")
        return console

    def instance(self) -> IVoice:
        if self.mode == ModeEnum.COMMUNICATION_SERVICES:
            assert self.communication_services
            return self.communication_services.instance()

        if self.mode == ModeEnum.CONSOLE:
            assert self.console
            return self.console.instance()

        assert self.mock
        return self.mock.instance()
