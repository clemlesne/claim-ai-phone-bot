from azure.communication.callautomation import CallConnectionClient
from helpers.call import ContextEnum, handle_play
from helpers.config import CONFIG
from models.call import CallModel
from semantic_kernel.plugin_definition import kernel_function


class CallPlugin:
    _call: CallModel
    _client: CallConnectionClient

    def __init__(self, call: CallModel, client: CallConnectionClient):
        self._call = call
        self._client = client

    @kernel_function(
        description="Use this if the user wants to end the call, or if the user said goodbye in the current call. Be warnging that the call will be ended immediately. Never use this action directly after a recall. Example: 'I want to hang up', 'Good bye, see you soon', 'We are done here', 'We will talk again later'.",
        name="Hangup",
    )
    async def hangup(self) -> None:
        await handle_play(
            call=self._call,
            client=self._client,
            context=ContextEnum.GOODBYE,
            text=await CONFIG.prompts.tts.goodbye(self._call),
        )

    @kernel_function(
        description="Use this if the user wants to talk to a human and Assistant is unable to help. This will transfer the customer to an human agent. Approval from the customer must be explicitely given. Never use this action directly after a recall. Example: 'I want to talk to a human', 'I want to talk to a real person'.",
        name="TalkToHuman",
    )
    async def talk_to_human(self) -> None:
        await handle_play(
            call=self._call,
            client=self._client,
            context=ContextEnum.CONNECT_AGENT,
            text=await CONFIG.prompts.tts.end_call_to_connect_agent(self._call),
        )
