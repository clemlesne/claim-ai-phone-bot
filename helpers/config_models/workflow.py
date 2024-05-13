from datetime import datetime
from enum import Enum
from helpers.pydantic_types.phone_numbers import PhoneNumber
from pydantic import BaseModel, EmailStr, Field, create_model
from pydantic.fields import FieldInfo
from typing import Annotated, Any, Optional, Tuple, Union


class CrmTypeEnum(str, Enum):
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    TEXT = "text"


class CrmFieldModel(BaseModel):
    description: Optional[str] = None
    name: str
    type: CrmTypeEnum


class LanguageEntryModel(BaseModel):
    """
    Language entry, containing the standard short code, an human name and the Azure Text-to-Speech voice name.

    See: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts#supported-languages
    """

    pronunciations_en: list[str]
    short_code: str
    voice: str

    @property
    def human_name(self) -> str:
        return self.pronunciations_en[0]

    def __str__(self):  # Pretty print for logs
        return self.short_code


class LanguageModel(BaseModel):
    """
    Manage language for the workflow.
    """

    selected_short_code: str = "fr-FR"
    # Voice list from Azure TTS
    # See: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    availables: list[LanguageEntryModel] = [
        LanguageEntryModel(
            pronunciations_en=["French", "FR", "France"],
            short_code="fr-FR",
            # Use voice optimized for conversational use
            # See: https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/introducing-more-multilingual-ai-voices-optimized-for/ba-p/4012832
            voice="fr-FR-VivienneMultilingualNeural",
        ),
        LanguageEntryModel(
            short_code="en-US",
            pronunciations_en=["English", "EN", "United States"],
            # Use voice optimized for conversational use
            # See: https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/introducing-more-multilingual-ai-voices-optimized-for/ba-p/4012832
            voice="en-US-AvaMultilingualNeural",
        ),
        LanguageEntryModel(
            short_code="es-ES",
            pronunciations_en=["Spanish", "ES", "Spain"],
            # Use voice optimized for conversational use
            # See: https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/introducing-7-new-realistic-ai-voices-optimized-for/ba-p/3971966
            voice="es-ES-XimenaNeural",
        ),
        LanguageEntryModel(
            short_code="zh-CN",
            pronunciations_en=["Chinese", "ZH", "China"],
            # Use voice optimized for conversational use
            # See: https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/introducing-more-multilingual-ai-voices-optimized-for/ba-p/4012832
            voice="zh-CN-XiaoxiaoMultilingualNeural",
        ),
    ]

    @property
    def selected(self) -> LanguageEntryModel:
        return next(
            (
                lang
                for lang in self.availables
                if self.selected_short_code == lang.short_code
            ),
            self.availables[0],
        )


class DefaultInitiateModel(BaseModel):
    bot_company: str
    bot_name: str
    customer_file: list[CrmFieldModel] = [
        CrmFieldModel(name="extra_details", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="incident_datetime", type=CrmTypeEnum.DATETIME),
        CrmFieldModel(name="incident_description", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="incident_location", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="injuries", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="medical_records", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="parties", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="policy_number", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="pre_existing_damages", type=CrmTypeEnum.TEXT),
        CrmFieldModel(name="witnesses", type=CrmTypeEnum.TEXT),
    ]
    lang: LanguageModel = LanguageModel()  # Object is fully defined by default
    task: str = """
        Assistant will help the customer with their insurance claim. Assistant requires data from the customer to fill the claim. Claim data is located in the customer file. Assistant role is not over until all the relevant data is gathered.
    """
    transfer_phone_number: PhoneNumber

    def customer_file_model(self) -> type[BaseModel]:
        return fields_to_pydantic(
            name="CrmEntryModel",
            fields=[
                *self.customer_file,
                CrmFieldModel(
                    description="Email of the customer",
                    name="caller_email",
                    type=CrmTypeEnum.EMAIL,
                ),
                CrmFieldModel(
                    description="First and last name of the customer",
                    name="caller_name",
                    type=CrmTypeEnum.TEXT,
                ),
                CrmFieldModel(
                    description="Phone number of the customer",
                    name="caller_phone",
                    type=CrmTypeEnum.PHONE_NUMBER,
                ),
                CrmFieldModel(
                    description="Relevant details gathered in the conversation than can be useful to solve the task",
                    name="extra_details",
                    type=CrmTypeEnum.TEXT,
                ),
            ],
        )


class WorkflowModel(BaseModel):
    conversation_timeout_hour: int = 72  # 3 days
    default_initiate: DefaultInitiateModel
    intelligence_hard_timeout_sec: int = 180  # 3 minutes
    intelligence_soft_timeout_sec: int = 30  # 30 seconds


def fields_to_pydantic(name: str, fields: list[CrmFieldModel]) -> type[BaseModel]:
    field_definitions = {field.name: field_to_pydantic(field) for field in fields}
    return create_model(
        name,
        **field_definitions,  # type: ignore
    )


def field_to_pydantic(
    field: CrmFieldModel,
) -> Union[Annotated[Any, ...], Tuple[type, FieldInfo]]:
    type = type_to_pydantic(field.type)
    return (
        Optional[type],
        Field(
            default=None,
            description=field.description,
        ),
    )


def type_to_pydantic(
    data: CrmTypeEnum,
) -> Union[type, Annotated[Any, ...]]:
    if data == CrmTypeEnum.DATETIME:
        return datetime
    elif data == CrmTypeEnum.EMAIL:
        return EmailStr
    elif data == CrmTypeEnum.PHONE_NUMBER:
        return PhoneNumber
    elif data == CrmTypeEnum.TEXT:
        return str
    else:
        raise ValueError(f"Unsupported data: {data}")
