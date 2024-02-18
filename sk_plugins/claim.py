from typing import Awaitable, Callable
from models.call import CallModel
from models.claim import ClaimModel
from semantic_kernel.orchestration.kernel_context import KernelContext
from semantic_kernel.plugin_definition import (
    kernel_function_context_parameter,
    kernel_function,
)
from fastapi import BackgroundTasks
from pydantic import ValidationError


class ClaimPlugin:
    _background_tasks: BackgroundTasks
    _call: CallModel
    _post_call_next: Callable[[CallModel], Awaitable[None]]
    _post_call_synthesis: Callable[[CallModel], Awaitable[None]]

    def __init__(
        self,
        background_tasks: BackgroundTasks,
        call: CallModel,
        post_call_next: Callable[[CallModel], Awaitable[None]],
        post_call_synthesis: Callable[[CallModel], Awaitable[None]],
    ):
        self._background_tasks = background_tasks
        self._call = call
        self._post_call_next = post_call_next
        self._post_call_synthesis = post_call_synthesis

    @kernel_function(
        description="Use this if the user wants to end the call, or if the user said goodbye in the current call. Be warnging that the call will be ended immediately. Never use this action directly after a recall. Example: 'I want to hang up', 'Good bye, see you soon', 'We are done here', 'We will talk again later'.",
        name="NewClaim",
    )
    @kernel_function_context_parameter(
        description="The text to be read to the customer to confirm the update. Only speak about this action. Use an imperative sentence. Example: 'I am updating the involved parties to Marie-Jeanne and Jean-Pierre', 'I am updating the contact contact info to 123 rue de la paix 75000 Paris, +33735119775, only call after 6pm'.",
        name="customer_response",
        required=True,
    )
    async def new_claim(self, context: KernelContext, customer_response: str) -> str:
        # TODO: user_callback
        # await user_callback(local_content, default_style)

        # Generate next action
        self._background_tasks.add_task(self._post_call_next, self._call)
        # Generate synthesis
        self._background_tasks.add_task(self._post_call_synthesis, self._call)

        return "Claim, reminders and messages reset."

    @kernel_function(
        description="Use this if the user wants to update a claim field with a new value. Example: 'Update claim explanation to: I was driving on the highway when a car hit me from behind', 'Update contact contact info to: 123 rue de la paix 75000 Paris, +33735119775, only call after 6pm'.",
        name="UpdatedClaim",
    )
    @kernel_function_context_parameter(
        description="The text to be read to the customer to confirm the update. Only speak about this action. Use an imperative sentence. Example: 'I am updating the involved parties to Marie-Jeanne and Jean-Pierre', 'I am updating the contact contact info to 123 rue de la paix 75000 Paris, +33735119775, only call after 6pm'.",
        name="customer_response",
        required=True,
    )
    @kernel_function_context_parameter(
        description="The claim field to update.",
        name="field",
        required=True,
    )
    @kernel_function_context_parameter(
        description="The claim field value to update. For dates, use YYYY-MM-DD HH:MM format (e.g. 2024-02-01 18:58). For phone numbers, use E164 format (e.g. +33612345678).",
        name="value",
        required=True,
    )
    async def updated_claim(
        self,
        context: KernelContext,
        customer_response: str,
        field: str,
        value: str,
    ) -> str:
        # TODO: user_callback
        # await user_callback(local_content, default_style)

        if not field in ClaimModel.editable_fields():
            return f'Failed to update a non-editable field "{field}".'
        else:
            try:
                # Define the field and force to trigger validation
                copy = self._call.claim.model_dump()
                copy[field] = value
                self._call.claim = ClaimModel.model_validate(copy)
                return f'Updated claim field "{field}" with value "{value}".'
            except ValidationError as e:  # Catch error to inform LLM
                return f'Failed to edit field "{field}": {e.json()}'
