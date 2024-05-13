from enum import Enum
from functools import cache
from helpers.pydantic_types.phone_numbers import PhoneNumber
from persistence.isms import ISms
from pydantic import field_validator, SecretStr, BaseModel, ValidationInfo
from typing import Optional


class ModeEnum(str, Enum):
    COMMUNICATION_SERVICES = "communication_services"
    TWILIO = "twilio"


class CommunicationServicesModel(BaseModel, frozen=True):
    """
    Represents the configuration for the Communication Services API.

    Model is purely empty to fit to the `ISms` interface and the "mode" enum code organization. As the Communication Services is also used as the only call interface, it is not necessary to duplicate the models.
    """

    @cache
    def instance(self) -> ISms:
        from persistence.communication_services import CommunicationServicesSms
        from helpers.config import CONFIG

        return CommunicationServicesSms(CONFIG.communication_services)


class TwilioModel(BaseModel, frozen=True):
    account_sid: str
    auth_token: SecretStr
    phone_number: PhoneNumber

    @cache
    def instance(self) -> ISms:
        from persistence.twilio import TwilioSms

        return TwilioSms(self)


class SmsModel(BaseModel):
    communication_services: Optional[CommunicationServicesModel] = (
        CommunicationServicesModel()
    )  # Object is fully defined by default
    mode: ModeEnum = ModeEnum.COMMUNICATION_SERVICES
    twilio: Optional[TwilioModel] = None

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

    @field_validator("twilio")
    def _validate_twilio(
        cls,
        twilio: Optional[TwilioModel],
        info: ValidationInfo,
    ) -> Optional[TwilioModel]:
        if not twilio and info.data.get("mode", None) == ModeEnum.TWILIO:
            raise ValueError("Twilio config required")
        return twilio

    def instance(self) -> ISms:
        if self.mode == ModeEnum.COMMUNICATION_SERVICES:
            assert self.communication_services
            return self.communication_services.instance()

        assert self.twilio
        return self.twilio.instance()
