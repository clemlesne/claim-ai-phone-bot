from enum import Enum
from pydantic import BaseModel


class IndentEnum(str, Enum):
    CONTINUE = "continue"
    NEW_OR_UPDATED_REMINDER = "new_or_updated_reminder"
    TALK_TO_HUMAN = "talk_to_human"


class ActionModel(BaseModel):
    content: str
    intent: IndentEnum
