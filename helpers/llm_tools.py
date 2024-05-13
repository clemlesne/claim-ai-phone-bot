from fastapi import BackgroundTasks
from helpers.config import CONFIG
from helpers.llm_utils import function_schema
from helpers.logging import build_logger
from inspect import getmembers, isfunction
from models.call import CallStateModel
from models.message import StyleEnum as MessageStyleEnum
from models.reminder import ReminderModel
from openai.types.chat import ChatCompletionToolParam
from persistence.ivoice import ContextEnum as VoiceContextEnum
from pydantic import ValidationError
from typing import Awaitable, Callable, Annotated, Literal, TypedDict
import asyncio


_logger = build_logger(__name__)
_search = CONFIG.ai_search.instance()


class UpdateCustomerFileDict(TypedDict):
    field: str
    value: str


class LlmPlugins:
    background_tasks: BackgroundTasks
    call: CallStateModel
    cancellation_callback: Callable[[], Awaitable]
    post_call_intelligence: Callable[[CallStateModel], None]
    style: MessageStyleEnum = MessageStyleEnum.NONE
    user_callback: Callable[[str, MessageStyleEnum], Awaitable]
    voice = CONFIG.voice.instance()

    def __init__(
        self,
        background_tasks: BackgroundTasks,
        call: CallStateModel,
        cancellation_callback: Callable[[], Awaitable],
        post_call_intelligence: Callable[[CallStateModel], None],
        user_callback: Callable[[str, MessageStyleEnum], Awaitable],
    ):
        self.background_tasks = background_tasks
        self.call = call
        self.cancellation_callback = cancellation_callback
        self.post_call_intelligence = post_call_intelligence
        self.user_callback = user_callback

    async def hangup(self) -> str:
        """
        Use this if the customer said they want to end the call.

        # Behavior
        1. Hangup the call for everyone
        2. The call with Assistant is ended

        # Rules
        - Requires an explicit verbal validation from the customer
        - Never use this action directly after a recall

        # Examples
        - 'Goodbye, see you tomorrow'
        - 'I want to hangup'
        """
        await self.cancellation_callback()
        await self.voice.aplay_text(
            background_tasks=self.background_tasks,
            call=self.call,
            context=VoiceContextEnum.GOODBYE,
            text=await CONFIG.prompts.tts.goodbye(self.call),
        )
        return "Call ended"

    async def new_conversation(
        self,
        customer_response: Annotated[
            str,
            "Phrase used to confirm the creation of a new conversation. This phrase will be spoken to the user. Describe what you're doing in one sentence. Example: 'I am creating a new conversation for a car accident.', 'A new conversation for a stolen watch is being created.'.",
        ],
    ) -> str:
        """
        Use this if the customer wants to create a new conversation.

        # Behavior
        1. Old conversation is stored but not accessible anymore
        2. Reset the Assistant conversation

        # Rules
        - Approval from the customer must be explicitely given (e.g. 'I want to create a new conversation')
        - This should be used only when the subject is totally different
        """
        await self.user_callback(customer_response, self.style)
        # Launch post-call intelligence for the current call
        self.post_call_intelligence(self.call)
        # Store the last message and use it at first message of the new conversation
        last_message = self.call.messages[-1]
        call = CallStateModel(initiate=self.call.initiate)
        call.messages.append(last_message)
        return "Customer file, reminders and messages reset"

    async def create_update_reminder(
        self,
        customer_response: Annotated[
            str,
            "Phrase used to confirm the update. This phrase will be spoken to the user. Describe what you're doing in one sentence. Example: 'I am creating a reminder for next week to call you back.', 'A todo for next week is planned.'.",
        ],
        description: Annotated[
            str,
            "Description of the reminder. Should be detailed enough to be understood by anyone. Example: 'Call back customer to get more details about the accident', 'Send analysis report to the customer'.",
        ],
        due_date_time: Annotated[
            str,
            "Datetime when the reminder should be triggered. Should be in the future, in the ISO format.",
        ],
        owner: Annotated[
            str,
            "The owner of the reminder. Can be 'customer', 'assistant', or a third party from the conversation. Try to be as specific as possible, with a name. Example: 'customer', 'assistant', 'contact', 'witness', 'police'.",
        ],
        title: Annotated[
            str,
            "Short title of the reminder. Should be short and concise, in the format 'Verb + Subject'. Title is unique and allows the reminder to be updated. Example: 'Call back customer', 'Send analysis report', 'Study replacement estimates for the stolen watch'.",
        ],
    ) -> str:
        """
        Use this if you think there is something important to do.

        # Behavior
        1. Create a reminder with the given values
        2. Return a confirmation message

        # Rules
        - A reminder should be as specific as possible
        - If a reminder already exists, it will be updated with the new values
        - The due date should be in the future
        """
        await self.user_callback(customer_response, self.style)

        # Check if reminder already exists, if so update it
        for reminder in self.call.reminders:
            if reminder.title == title:
                try:
                    reminder.description = description
                    reminder.due_date_time = due_date_time  # type: ignore
                    reminder.owner = owner
                    return f'Reminder "{title}" updated.'
                except ValidationError as e:
                    return f'Failed to edit reminder "{title}": {e.json()}'

        # Create new reminder
        try:
            reminder = ReminderModel(
                description=description,
                due_date_time=due_date_time,  # type: ignore
                owner=owner,
                title=title,
            )
            self.call.reminders.append(reminder)
            return f'Reminder "{title}" created.'
        except ValidationError as e:
            return f'Failed to create reminder "{title}": {e.json()}'

    async def update_customer_file(
        self,
        customer_response: Annotated[
            str,
            "Phrase used to confirm the update. This phrase will be spoken to the user. Describe what you're doing in one sentence. Example: 'I am updating the your name to Marie-Jeanne Duchemin and your email to mariejeanne@gmail.com.'.",
        ],
        updates: Annotated[
            list[UpdateCustomerFileDict],
            """
            The field to update.

            # Available fields
            {% for name, info in call.initiate.customer_file_model().model_fields.items() %}
            {% if not info.description %}
            - {{ name }}
            {% else %}
            - '{{ name }}', {{ info.description }}
            {% endif %}
            {% endfor %}

            # Data format
            - For dates, use YYYY-MM-DD HH:MM format (e.g. 2024-02-01 18:58)
            - For phone numbers, use E164 format (e.g. +33612345678)

            # Response format
            [{'field': '[field]', 'value': '[value]'}]

            # Examples
            - [{'field': 'caller_email', 'value': 'mariejeanne@gmail.com'}]
            - [{'field': 'caller_name', 'value': 'Marie-Jeanne Duchemin'}, {'field': 'caller_phone', 'value': '+33612345678'}]
            """,
        ],
    ) -> str:
        """
        Use this if the customer wants to update one or more fields in customer file.

        # Behavior
        1. Update the customer file with the new values
        2. Return a confirmation message

        # Rules
        - For values, it is OK to approximate dates if the customer is not precise (e.g., "last night" -> today 04h, "I'm stuck on the highway" -> now)
        - It is best to update multiple fields at once
        """
        test = self.call.initiate.customer_file_model().model_fields
        await self.user_callback(customer_response, self.style)
        # Update all field in customer file
        res = "# Updated fields"
        for field in updates:
            res += f"\n- {self._update_customer_file_worker(field)}"
        return res

    def _update_customer_file_worker(self, update: UpdateCustomerFileDict) -> str:
        field = update["field"]
        value = update["value"]
        if not field in self.call.initiate.customer_file_model().model_fields:
            return f'Field "{field}" does not exist, please use a valid field.'
        try:
            self.call.customer_file[field] = value
            return f'Updated field "{field}" with value "{value}".'
        except ValidationError as e:  # Catch error to create a feedback to the LLM
            return f'Failed to edit field "{field}": {e.json()}'

    async def transfer_to_human_agent(self) -> str:
        """
        Use this if the customer wants to talk to a human and Assistant is unable to help.

        # Behavior
        1. Transfer the customer to an human agent
        2. The call with Assistant is ended

        # Rules
        - Requires an explicit verbal validation from the customer
        - Never use this action directly after a recall

        # Examples
        - 'I want to talk to a human'
        - 'I want to talk to a real person'
        """
        await self.cancellation_callback()
        await self.voice.aplay_text(
            background_tasks=self.background_tasks,
            call=self.call,
            context=VoiceContextEnum.CONNECT_AGENT,
            text=await CONFIG.prompts.tts.end_call_to_connect_agent(self.call),
        )
        return "Transferring to human agent"

    async def search_documentation(
        self,
        customer_response: Annotated[
            str,
            "Phrase used to confirm the search. This phrase will be spoken to the user. Describe what you're doing in one sentence. Example: 'I am searching for the document about the car accident.', 'I am looking for the contract details.'.",
        ],
        queries: Annotated[
            list[str],
            "The text queries to perform the search. Example: ['How much does it cost to repair a broken window?', 'What are the requirements to ask for a cyber attack insurance?']",
        ],
    ) -> str:
        """
        Use this if the customer wants to search for a public specific information you don't have.

        # Rules
        - Multiple queries should be used with different viewpoints and vocabulary
        - The search should be as specific as possible

        # Searchable topics
        contract, law, regulation, article, procedure, guide
        """
        await self.user_callback(customer_response, self.style)
        # Execute in parallel
        tasks = await asyncio.gather(
            *[_search.training_asearch_all(query, self.call) for query in queries]
        )
        # Flatten, remove duplicates, and sort by score
        trainings = sorted(set(training for task in tasks for training in task or []))
        # Format results
        res = "# Search results"
        for training in trainings:
            res += f"\n- {training.title}: {training.content}"
        return res

    async def notify_emergencies(
        self,
        customer_response: Annotated[
            str,
            "Phrase used to confirm the action. This phrase will be spoken to the user. Describe what you're doing in one sentence. Example: 'I am notifying the emergency services.', 'Police number is confirmed.'.",
        ],
        reason: Annotated[
            str,
            "The reason to notify the emergency services. Should be detailed enough to be understood by anyone. Example: 'A person is having a heart attack', 'A child is being attacked by a dog'.",
        ],
        location: Annotated[
            str,
            "The location of the emergency. Should be detailed enough to be understood by anyone. Should contains details like the floor, the building, the code to enter, etc. Example: '123 rue de la paix 75000 Paris, Building A, 3rd floor, code 1234', '12 avenue de la manivelle 13000 Marseille, behind the red door'.",
        ],
        contact: Annotated[
            str,
            "The local contact of a person on site. Should be detailed enough to be understood by anyone. Should contains details like the name, the phone number, etc. Example: 'Marie-Jeanne, +33735119775', 'Jean-Pierre, wear a red hat'.",
        ],
        service: Annotated[
            Literal["police", "firefighters", "pharmacy", "veterinarian", "hospital"],
            "The emergency service to notify.",
        ],
    ) -> str:
        """
        Use this if the customer wants to notify the emergency services for a specific reason.

        # Behavior
        1. A record is stored in the system
        2. A notification is sent to the emergency services

        # Rules
        - Use it only if the situation is critical and requires immediate intervention

        # Examples
        - 'A child is lying on the ground and is not moving'
        - 'I am stuck in a car in fire'
        - 'My neighbor is having a heart attack'
        """
        await self.user_callback(customer_response, self.style)
        # TODO: Implement notification to emergency services for production usage
        _logger.info(
            f"Notifying {service}, location {location}, contact {contact}, reason {reason}."
        )
        return f"Notifying {service} for {reason}."

    @staticmethod
    async def to_openai(call: CallStateModel) -> list[ChatCompletionToolParam]:
        return await asyncio.gather(
            *[
                function_schema(type, call=call)
                for name, type in getmembers(LlmPlugins, isfunction)
                if not name.startswith("_") and name != "to_openai"
            ]
        )
