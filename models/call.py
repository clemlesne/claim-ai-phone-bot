from datetime import datetime, UTC, tzinfo
from helpers.config_models.workflow import DefaultInitiateModel, LanguageEntryModel
from helpers.pydantic_types.phone_numbers import PhoneNumber
from models.message import MessageModel
from models.next import NextModel
from models.reminder import ReminderModel
from models.synthesis import SynthesisModel
from models.training import TrainingModel
from pydantic import BaseModel, Field, computed_field, field_validator, ValidationInfo
from pydantic_extra_types.phone_numbers import PhoneNumber as PydanticPhoneNumber
from pytz import country_timezones, timezone, utc
from typing import Any, Optional
from uuid import UUID, uuid4
import asyncio
import phonenumbers
import random
import string


class CallInitiateModel(DefaultInitiateModel):
    phone_number: PhoneNumber


class CallGetModel(BaseModel):
    # Immutable fields
    call_id: UUID = Field(default_factory=uuid4, frozen=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    # Editable fields
    initiate: CallInitiateModel
    customer_file: dict[str, Any] = Field(
        default={},
        validation_alias="claim",  # Compatibility with v1
    )  # Place after "initiate" as it depends on it for validation
    messages: list[MessageModel] = []
    next: Optional[NextModel] = None
    reminders: list[ReminderModel] = []
    synthesis: Optional[SynthesisModel] = None

    @computed_field
    @property
    def lang(self) -> LanguageEntryModel:  # type: ignore
        if self.lang_short_code:
            return next(
                (
                    lang
                    for lang in self.initiate.lang.availables
                    if lang.short_code == self.lang_short_code
                ),
                self.initiate.lang.selected,
            )
        return self.initiate.lang.selected

    @lang.setter
    def lang(self, short_code: str) -> None:
        self.lang_short_code = short_code

    @field_validator("customer_file")
    def _validate_customer_file(
        cls, customer_file: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        initiate: CallInitiateModel = info.data["initiate"]
        model = initiate.customer_file_model()
        return model.model_validate(customer_file, strict=True).model_dump(
            exclude_none=True
        )

    def tz(self) -> tzinfo:
        parsed = phonenumbers.parse(self.initiate.phone_number)
        if not parsed.country_code:
            return utc
        region_code = phonenumbers.region_code_for_country_code(parsed.country_code)
        tz_name = country_timezones[region_code][0]
        return timezone(tz_name)


class CallStateModel(CallGetModel, validate_assignment=True):
    # Immutable fields
    callback_secret: str = Field(
        default="".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(16)
        ),
        frozen=True,
    )
    # Editable fields
    lang_short_code: Optional[str] = Field(default=None)
    voice_id: Optional[str] = None
    voice_recognition_retry: int = Field(
        default=0,
        validation_alias="recognition_retry",  # Compatibility with v1
    )

    async def trainings(self) -> list[TrainingModel]:
        """
        Get the trainings from the last messages.

        Is using query expansion from last messages. Then, data is sorted by score.
        """
        from helpers.config import CONFIG
        from helpers.logging import TRACER

        with TRACER.start_as_current_span("trainings"):
            search = CONFIG.ai_search.instance()
            trainings_tasks = await asyncio.gather(
                *[
                    search.training_asearch_all(message.content, self)
                    for message in self.messages[-CONFIG.ai_search.expansion_k :]
                ],
            )  # Get trainings from last messages
            trainings = sorted(
                set(
                    training
                    for trainings in trainings_tasks
                    for training in trainings or []
                )
            )  # Flatten, remove duplicates, and sort by score
            return trainings
