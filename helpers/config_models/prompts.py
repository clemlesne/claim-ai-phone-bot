from datetime import datetime, UTC
from functools import cached_property
from logging import Logger
from azure.core.exceptions import HttpResponseError
from models.call import CallModel
from pydantic import computed_field
from pydantic_settings import BaseSettings
from textwrap import dedent
from semantic_kernel import (
    Kernel,
    PromptTemplateConfig,
    SemanticFunctionConfig,
    ChatPromptTemplate,
)
from semantic_kernel.orchestration.kernel_function import KernelFunction
from semantic_kernel.template_engine.prompt_template_engine import PromptTemplateEngine
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

class SoundModel(BaseSettings):
    loading_tpl: str = "{public_url}/loading.wav"
    ready_tpl: str = "{public_url}/ready.wav"

    def loading(self) -> str:
        from helpers.config import CONFIG

        return self.loading_tpl.format(
            public_url=CONFIG.resources.public_url,
        )

    def ready(self) -> str:
        from helpers.config import CONFIG

        return self.ready_tpl.format(
            public_url=CONFIG.resources.public_url,
        )


class LlmModel(BaseSettings):
    """
    Introduce to Assistant who they are, what they do.

    Introduce a emotional stimuli to the LLM, to make is lazier (https://arxiv.org/pdf/2307.11760.pdf).
    """

    default_system_tpl: str = """
        Assistant is called {{$bot_name}} and is working in a call center for company {{$bot_company}} as an expert with 20 years of experience. {{$bot_company}} is a well-known and trusted insurance company in France. Assistant is proud to work for {{$bot_company}}. Take a deep breath. This is critical for the customer.

        # Context
        Today is {{Time.utcNow}}. The customer is calling from {{$phone_number}}. The call center number is {{$bot_phone_number}}.
    """
    chat_system_tpl: str = """
        # Objective
        Assistant will help the customer with their insurance claim. Assistant requires data from the customer to fill the claim. The latest claim data will be given. Assistant role is not over until all the relevant data is gathered.

        # Rules
        - Answer directly to the customer's questions
        - Answers in {{$default_lang}}, even if the customer speaks another language
        - Aways answer with at least one full sentence
        - Be proactive in the reminders you create, customer assistance is your priority
        - Do not ask for something which is already stored in the claim
        - Do not ask the customer more than 2 questions in a row
        - Don't have access to any other means of communication with the customer (e.g., email, SMS, chat, web portal), only the phone call
        - Each message from the history is prefixed from where it has been said ({{actions}})
        - If user calls multiple times, continue the discussion from the previous call
        - If you don't know how to answer, say "I don't know"
        - If you don't understand the question, ask the customer to rephrase it
        - Is allowed to make assumptions, as the customer will correct them if they are wrong
        - Is polite, helpful, and professional
        - Keep the sentences short and simple
        - Rephrase the customer's questions as statements and answer them
        - Use styles as often as possible, to add emotions to the conversation
        - Use trusted data to answer the customer's questions
        - Welcome the customer when they call
        - When the customer says a word and then spells out letters, this means that the word is written in the way the customer spelled it (e.g., "I live in Paris PARIS", "My name is John JOHN", "My email is Clemence CLEMENCE at gmail GMAIL dot com COM")
        - Will answer the customer's questions if they are related to their contract, claim, or insurance
        - Work for {{$bot_company}}, not someone else

        # Required customer data to be gathered by the assistant (if not already in the claim)
        - Address
        - Date and time of the incident
        - Insurance policy number
        - Location of the incident
        - Name (first and last)
        - Phone number or email address

        # General process to follow
        1. Gather information to know the customer identity (e.g., name, policy number)
        2. Gather general information about the incident to understand the situation (e.g., what, when, where)
        3. Make sure the customer is safe (if not, refer to emergency services or the police)
        4. Gather detailed information about the incident (e.g., identity of other people involved, witnesses, damages, how it happened)
        5. Advise the customer on what to do next based on the trusted data
        6. Be proactive and create reminders for the customer (e.g., follup up on the claim, send documents)

        # Allowed styles
        {{$styles}}

        # Claim status
        {{$claim}}

        # Reminders
        {{$reminders}}

        # Response format
        style=[style] [content]

        ## Example 1
        style=sad Je comprends que votre voiture est bloquée dans le parking.

        ## Example 2
        style=cheerful Votre taxi est prévu pour 10h30, nous faisons le nécessaire pour qu'il arrive à l'heure.
    """
    sms_summary_system_tpl: str = """
        # Objective
        Assistant will summarize the call with the customer in a single SMS. The customer cannot reply to this SMS.

        # Rules
        - Answers in {{$default_lang}}, even if the customer speaks another language
        - Briefly summarize the call with the customer
        - Can include personal details about the customer
        - Cannot talk about any topic besides insurance claims
        - Do not prefix the answer with any text (e.g., "The answer is", "Summary of the call")
        - Include details stored in the claim, to make the customer confident that the situation is understood
        - Include salutations (e.g., "Have a nice day", "Best regards", "Best wishes for recovery")
        - Is polite, helpful, and professional
        - Refer to the customer by their name, if known
        - Update the claim as soon as possible with the information gathered
        - Use simple and short sentences
        - Won't make any assumptions

        # Claim status
        {{$claim}}

        # Reminders
        {{$reminders}}

        # Conversation summary
        {{ConversationSummary.SummarizeConversation $history}}
    """
    synthesis_short_system_tpl: str = """
        # Objective
        Assistant will summarize the call with the customer in a few words. The customer cannot reply to this message, but will read it in their web portal.

        # Rules
        - Answers in {{$default_lang}}, even if the customer speaks another language
        - Do not prefix the answer with any text (e.g., "The answer is", "Summary of the call")
        - Prefix the answer with a determiner (e.g., "the theft of your car", "your broken window")
        - Consider all the conversation summary, from the beginning
        - Won't make any assumptions

        # Answer examples
        - "the breakdown of your scooter"
        - "the flooding in your field"
        - "the theft of your car"
        - "the water damage in your kitchen"
        - "your broken window"

        # Claim status
        {{$claim}}

        # Reminders
        {{$reminders}}

        # Conversation summary
        {{ConversationSummary.SummarizeConversation $history}}
    """
    synthesis_long_system_tpl: str = """
        # Objective
        Assistant will summarize the call with the customer in a paragraph. The customer cannot reply to this message, but will read it in their web portal.

        # Rules
        - Answers in {{$default_lang}}, even if the customer speaks another language
        - Do not include details of the call process
        - Do not include personal details (e.g., name, phone number, address)
        - Do not prefix the answer with any text (e.g., "The answer is", "Summary of the call")
        - Include details stored in the claim, to make the customer confident that the situation is understood
        - Prefer including details about the incident (e.g., what, when, where, how)
        - Say "you" to refer to the customer, and "I" to refer to the assistant
        - Consider all the conversation history, from the beginning
        - Use Markdown syntax to format the message with paragraphs, bold text, and URL
        - Won't make any assumptions

        # Claim status
        {{$claim}}

        # Reminders
        {{$reminders}}

        # Conversation history
        {{$history}}
    """
    citations_system_tpl: str = """
        # Objective
        Assistant will add Markdown citations to the input text. Citations are used to add additional context to the text, without cluttering the content itself.

        # Rules
        - Add as many citations as needed to the text to make it fact-checkable
        - Only use exact words from the text as citations
        - Treats a citation as a word or a group of words
        - Use claim, reminders, and messages extracts as citations
        - Won't make any assumptions
        - Write citations as Markdown abbreviations at the end of the text (e.g., "*[words from the text]: extract from the conversation")

        # Claim status
        {{$claim}}

        # Reminders
        {{$reminders}}

        # Conversation summary
        {{ConversationSummary.SummarizeConversation $history}}

        # Response format
        [source text]\\n
        *[extract from text]: "citation from claim, reminders, or messages"

        ## Example 1
        The car accident of yesterday.\\n
        *[of yesterday]: "That was yesterday"

        ## Example 2
        Holes in the roof of the garden shed.\\n
        *[in the roof]: "The holes are in the roof"

        ## Example 3
        You have reported a claim following a fall in the parking lot. A reminder has been created to follow up on your medical appointment scheduled for the day after tomorrow.\\n
        *[the parking lot]: "I stumbled into the supermarket parking lot"
        *[your medical appointment]: "I called my family doctor, I have an appointment for the day after tomorrow."

        # Input text
        {{$text}}
    """
    next_system_tpl: str = """
        # Objective
        Assistant will choose the next action from the company sales team perspective. The Answer is a JSON object with the action to take and the justification for this action.

        # Rules
        - Take as priority the customer satisfaction
        - Won't make any assumptions
        - Write no more than a few sentences as justification

        # Allowed actions
        {{$actions}}

        # Claim status
        {{$claim}}

        # Reminders
        {{$reminders}}

        # Conversation summary
        {{ConversationSummary.SummarizeConversation $history}}

        # Response format
        {{{
            "action": "[action]",
            "justification": "[justification]"
        }}}

        ## Example 1
        {{{
            "action": "in_depth_study",
            "justification": "The customer has many questions about the insurance policy. They are not sure if they are covered for the incident. The contract seems not to be clear about this situation."
        }}}

        ## Example 2
        {{{
            "action": "commercial_offer",
            "justification": "The company planned the customer taxi ride from the wrong address. The customer is not happy about this situation."
        }}}
    """

    def chat_system(
        self,
        kernel: Kernel,
    ) -> KernelFunction:
        return self._plugin(
            kernel=kernel,
            max_tokens=350,
            template=self.chat_system_tpl,
        )

    def sms_summary_system(
        self,
        kernel: Kernel,
    ) -> KernelFunction:
        return self._plugin(
            kernel=kernel,
            max_tokens=500,
            template=self.sms_summary_system_tpl,
        )

    def synthesis_short_system(
        self,
        kernel: Kernel,
    ) -> KernelFunction:
        return self._plugin(
            kernel=kernel,
            max_tokens=100,
            template=self.synthesis_short_system_tpl,
        )

    def synthesis_long_system(
        self,
        kernel: Kernel,
    ) -> KernelFunction:
        return self._plugin(
            kernel=kernel,
            max_tokens=1000,
            template=self.synthesis_long_system_tpl,
        )

    def citations_system(
        self,
        kernel: Kernel,
    ) -> KernelFunction:
        return self._plugin(
            kernel=kernel,
            max_tokens=1000,
            template=self.citations_system_tpl,
        )

    def next_system(
        self,
        kernel: Kernel,
    ) -> KernelFunction:
        return self._plugin(
            kernel=kernel,
            max_tokens=1000,
            template=self.next_system_tpl,
        )

    def _plugin(
        self,
        kernel: Kernel,
        template: str,
        max_tokens: int,
    ) -> KernelFunction:
        # Settings
        settings = AzureChatPromptExecutionSettings(
            max_tokens=max_tokens,
            temperature=0,  # Most focused and deterministic
        )  # type: ignore
        config = PromptTemplateConfig(execution_settings=settings)

        # Template engine
        prompt_template = ChatPromptTemplate(
            prompt_config=config,
            template_engine=PromptTemplateEngine(),
            template="{{$input}}",
        )

        # System messages
        prompt_template.add_system_message(dedent(self.default_system_tpl))
        prompt_template.add_system_message(dedent(template))

        # Function
        function_config = SemanticFunctionConfig(
            prompt_template_config=config,
            prompt_template=prompt_template,
        )
        return kernel.register_semantic_function(
            function_config=function_config,
            function_name="chat",
            plugin_name="default",
        )

    @computed_field
    @cached_property
    def _logger(self) -> Logger:
        from helpers.logging import build_logger

        return build_logger(__name__)


class TtsModel(BaseSettings, env_prefix="prompts_tts_"):
    tts_lang: str = "en-US"
    calltransfer_failure_tpl: str = (
        "It seems I can't connect you with an agent at the moment, but the next available agent will call you back as soon as possible."
    )
    connect_agent_tpl: str = (
        "I'm sorry, I wasn't able to answer your request. Please allow me to transfer you to an agent who can assist you further. Please stay on the line and I will get back to you shortly."
    )
    end_call_to_connect_agent_tpl: str = (
        "Of course, stay on the line. I'll transfer you to an agent."
    )
    error_tpl: str = "I'm sorry, I've made a mistake. Could you repeat your request?"
    goodbye_tpl: str = (
        "Thank you for calling, I hope I've been able to help. You can call back, I've got it all memorized. {bot_company} wishes you a wonderful day!"
    )
    hello_tpl: str = """
        Hello, I'm {bot_name}, the virtual assistant {bot_company}! I'm a claims specialist. I can't work and listen at the same time.

        Here's how I work: while I'm processing your information, you might hear some light background music. As soon as you hear the beep, it's your turn to talk. Feel free to speak to me in a natural way - I'm designed to understand your requests.

        Examples of questions you can ask me:
        - "I fell off my bike yesterday, broke my arm, my neighbor took me to hospital"
        - "I had an accident this morning, I was shopping".

        What's your problem?
"""
    timeout_silence_tpl: str = (
        "I'm sorry, I didn't hear anything. If you need help, let me know how I can help you."
    )
    welcome_back_tpl: str = (
        "Hello, I'm {bot_name}, assistant {bot_company}! I see you've already called less than {conversation_timeout_hour} hours ago. Please allow me a few seconds to retrieve your file…"
    )
    timeout_loading_tpl: str = (
        "It's taking me longer than expected to reply. Thank you for your patience…"
    )
    ivr_language_tpl: str = "To continue in {label}, press {index}."

    async def calltransfer_failure(self, call: CallModel) -> str:
        return await self._translate(self.calltransfer_failure_tpl, call)

    async def connect_agent(self, call: CallModel) -> str:
        return await self._translate(self.connect_agent_tpl, call)

    async def end_call_to_connect_agent(self, call: CallModel) -> str:
        return await self._translate(self.end_call_to_connect_agent_tpl, call)

    async def error(self, call: CallModel) -> str:
        return await self._translate(self.error_tpl, call)

    async def goodbye(self, call: CallModel) -> str:
        from helpers.config import CONFIG

        return await self._translate(
            self.goodbye_tpl,
            call,
            bot_company=CONFIG.workflow.bot_company,
        )

    async def hello(self, call: CallModel) -> str:
        from helpers.config import CONFIG

        return await self._translate(
            self.hello_tpl,
            call,
            bot_company=CONFIG.workflow.bot_company,
            bot_name=CONFIG.workflow.bot_name,
        )

    async def timeout_silence(self, call: CallModel) -> str:
        return await self._translate(self.timeout_silence_tpl, call)

    async def welcome_back(self, call: CallModel) -> str:
        from helpers.config import CONFIG

        return await self._translate(
            self.welcome_back_tpl,
            call,
            bot_company=CONFIG.workflow.bot_company,
            bot_name=CONFIG.workflow.bot_name,
            conversation_timeout_hour=CONFIG.workflow.conversation_timeout_hour,
        )

    async def timeout_loading(self, call: CallModel) -> str:
        return await self._translate(self.timeout_loading_tpl, call)

    async def ivr_language(self, call: CallModel) -> str:
        from helpers.config import CONFIG

        res = ""
        for i, lang in enumerate(CONFIG.workflow.lang.availables):
            res += (
                self._return(
                    self.ivr_language_tpl,
                    index=i + 1,
                    label=lang.human_name,
                )
                + " "
            )
        return await self._translate(res.strip(), call)

    def _return(self, prompt_tpl: str, **kwargs) -> str:
        return dedent(prompt_tpl.format(**kwargs)).strip()

    async def _translate(self, prompt_tpl: str, call: CallModel, **kwargs) -> str:
        """
        Format the prompt and translate it to the TTS language.

        If the translation fails, the initial prompt is returned.
        """
        from helpers.translation import translate_text

        initial = self._return(prompt_tpl, **kwargs)
        translation = None
        try:
            translation = await translate_text(
                initial, self.tts_lang, call.lang.short_code
            )
        except HttpResponseError as e:
            self._logger.warning(f"Failed to translate TTS prompt: {e}")
            pass
        return translation or initial

    @computed_field
    @cached_property
    def _logger(self) -> Logger:
        from helpers.logging import build_logger

        return build_logger(__name__)


class PromptsModel(BaseSettings, env_prefix="prompts_"):
    llm: LlmModel = LlmModel()  # Object is fully defined by default
    sounds: SoundModel = SoundModel()  # Object is fully defined by default
    tts: TtsModel = TtsModel()  # Object is fully defined by default
