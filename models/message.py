from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import List
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
import re


FUNC_NAME_SANITIZER_R = r"[^a-zA-Z0-9_-]"


class StyleEnum(str, Enum):
    """
    Voice styles the Azure AI Speech Service supports.

    Doc:
    - Speaking styles: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#use-speaking-styles-and-roles
    - Support by language: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts#voice-styles-and-roles
    """

    CHEERFUL = "cheerful"
    NONE = "none"  # This is not a valid style, but we use it in the code to indicate no style
    SAD = "sad"


class ActionEnum(str, Enum):
    CALL = "call"
    HANGUP = "hangup"
    SMS = "sms"
    TALK = "talk"


class PersonaEnum(str, Enum):
    ASSISTANT = "assistant"
    HUMAN = "human"
    TOOL = "tool"


class ToolModel(BaseModel):
    content: str
    function_arguments: str
    function_name: str
    tool_id: str


class MessageModel(BaseModel):
    # Immutable fields
    created_at: datetime = Field(default_factory=datetime.utcnow, frozen=True)
    # Editable fields
    action: ActionEnum = ActionEnum.TALK
    content: str
    persona: PersonaEnum
    style: StyleEnum = StyleEnum.NONE
    tool_calls: List[ToolModel] = []

    def to_openai(self) -> List[ChatCompletionMessageParam]:
        if self.persona == PersonaEnum.HUMAN:
            return [
                ChatCompletionUserMessageParam(
                    content=f"action={self.action.value} style={self.style.value} {self.content}",
                    role="user",
                )
            ]

        elif self.persona == PersonaEnum.ASSISTANT:
            if not self.tool_calls:
                return [
                    ChatCompletionAssistantMessageParam(
                        content=f"action={self.action.value} style={self.style.value} {self.content}",
                        role="assistant",
                    )
                ]

        res = []
        res += ChatCompletionAssistantMessageParam(
            content=f"action={self.action.value} style={self.style.value} {self.content}",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=tool_call.tool_id,
                    type="function",
                    function={
                        "arguments": tool_call.function_arguments,
                        "name": "-".join(
                            re.sub(
                                FUNC_NAME_SANITIZER_R,
                                "-",
                                tool_call.function_name,
                            ).split("-")
                        ),  # Sanitize with dashes then deduplicate dashes, backward compatibility with old models
                    },
                )
                for tool_call in self.tool_calls
            ],
        )
        for tool_call in self.tool_calls:
            res += ChatCompletionToolMessageParam(
                content=tool_call.content,
                role="tool",
                tool_call_id=tool_call.tool_id,
            )
        return res
