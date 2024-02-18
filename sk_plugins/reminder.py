from azure.communication.callautomation import CallConnectionClient
from models.call import CallModel
from semantic_kernel.orchestration.kernel_context import KernelContext
from semantic_kernel.plugin_definition import (
    kernel_function,
    kernel_function_context_parameter,
)
from pydantic import ValidationError
from models.reminder import ReminderModel
from datetime import datetime


class ReminderPlugin:
    _call: CallModel
    _client: CallConnectionClient

    def __init__(self, call: CallModel, client: CallConnectionClient):
        self._call = call
        self._client = client

    @kernel_function(
        description="Use this if you think there is something important to do in the future, and you want to be reminded about it. If it already exists, it will be updated with the new values. Example: 'Remind Assitant thuesday at 10am to call back the customer', 'Remind Assitant next week to send the report', 'Remind the customer next week to send the documents by the end of the month'.",
        name="NewOrUpdatedReminder",
    )
    @kernel_function_context_parameter(
        description="The text to be read to the customer to confirm the reminder. Only speak about this action. Use an imperative sentence. Example: 'I am creating a reminder for next week to call back the customer', 'I am creating a reminder for next week to send the report'.",
        name="customer_response",
    )
    @kernel_function_context_parameter(
        description="Contextual description of the reminder. Should be detailed enough to be understood by anyone. Example: 'Watch model is Rolex Submariner 116610LN', 'User said the witnesses car was red but the police report says it was blue. Double check with the involved parties'.",
        name="description",
    )
    @kernel_function_context_parameter(
        description="Datetime when the reminder should be triggered. Should be in the future, in the ISO format.",
        name="due_date_time",
    )
    @kernel_function_context_parameter(
        description="The owner of the reminder. Can be 'customer', 'assistant', or a third party from the claim. Try to be as specific as possible, with a name. Example: 'customer', 'assistant', 'contact', 'witness', 'police'.",
        name="owner",
    )
    @kernel_function_context_parameter(
        description="Short title of the reminder. Should be short and concise, in the format 'Verb + Subject'. Title is unique and allows the reminder to be updated. Example: 'Call back customer', 'Send analysis report', 'Study replacement estimates for the stolen watch'.",
        name="title",
    )
    async def new_or_updated_reminder(
        self,
        context: KernelContext,
        customer_response: str,
        description: str,
        due_date_time: str,
        owner: str,
        title: str,
    ) -> str:
        # TODO: user_callback
        # await user_callback(local_content, default_style)

        for reminder in self._call.reminders:
            if reminder.title == title:
                try:
                    reminder.description = description
                    reminder.due_date_time = datetime.fromisoformat(due_date_time)
                    reminder.owner = owner
                    return f'Reminder "{title}" updated.'
                except ValidationError as e:  # Catch error to inform LLM
                    return f'Failed to edit reminder "{title}": {e.json()}'

        try:
            reminder = ReminderModel(
                description=description,
                due_date_time=datetime.fromisoformat(due_date_time),
                owner=owner,
                title=title,
            )
            self._call.reminders.append(reminder)
            return f'Reminder "{title}" created.'
        except ValidationError as e:  # Catch error to inform LLM
            return f'Failed to create reminder "{title}": {e.json()}'
