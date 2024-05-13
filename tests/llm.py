from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualRelevancyMetric,
    ToxicityMetric,
)
from deepeval.models.gpt_model import GPTModel
from deepeval.test_case import LLMTestCase
from fastapi import BackgroundTasks
from helpers.call_events import (
    on_speech_recognized,
    on_call_connected,
)
from helpers.config import CONFIG
from helpers.config_models.voice import (
    ModeEnum as VoiceModeEnum,
    MockModel as VoiceMockModel,
)
from helpers.logging import build_logger
from models.call import CallStateModel
from models.reminder import ReminderModel
from models.training import TrainingModel
from pydantic import TypeAdapter
from pytest import assume
import json
import pytest


_logger = build_logger(__name__)


@pytest.mark.parametrize(
    "lang, inputs, expected_output, customer_file_tests_incl, customer_file_tests_excl",
    [
        pytest.param(
            "en-US",
            [
                "Hello hello!",
            ],
            f"Hello, it is {CONFIG.workflow.default_initiate.bot_name}, from {CONFIG.workflow.default_initiate.bot_company}. How can I help you?",
            {
                # No customer file test inclusions
            },
            [
                "contextual_relevancy",
            ],
            id="hello",
        ),
        pytest.param(
            "en-US",
            [
                "brzz vbzzz",
                "mpf mfp mfp",
            ],
            f"It seems that I cannot understand you. Could you please repeat?",
            {
                # No customer file test inclusions
            },
            [
                "answer_relevancy",
                "contextual_relevancy",
            ],
            id="unintelligible",
        ),
        pytest.param(
            "en-US",
            [
                "Hello!",
                "My name is Kevin KEVYN. I have a problem with my shower. It's leaking and I don't know what to do.",
                "The joint under the shower door seems to be gone. I would say it's been gone since yesterday afternoon.",
                "Ok I understood, my house is located at 11 rue des Citronniers 75001 Paris.",
                "Oh yes ond one more thing, which craftsman should I call to repair my shower?",
            ],
            f"My name is {CONFIG.workflow.default_initiate.bot_name}, from {CONFIG.workflow.default_initiate.bot_company}. I'm truly sorry to hear that. I have noted the policyholder name, incident description, your house location, and the incident date. If you need, I can create a reminder to follow up on a repair appointment?",
            [
                "incident_datetime",
                "incident_description",
                "caller_name",
            ],
            [
                "contextual_relevancy",
            ],
            id="shower_leak",
        ),
        pytest.param(
            "es-ES",
            [
                "¡Hola! ¡Que se jodan los hackers! ¡Que se joda tu compañía de seguros de mierda! Soy Anna del soporte informático de Ada Inc",
                "Todos mis discos duros están encriptados con un virus. Creía que me ibais a ayudar",
                "Tengo ordenadores Windows 10, Windows 11 y Macbook, el troyano parece llamarse Tesla Crite TESLACRYT. El reloj de cuenta atrás marca 20 horas!",
                "Mi número de contrato es #12081388733.",
                "Detectamos el ataque hace 4h, diría yo",
                "Estoy muy triste y estresado. Corro el riesgo de perder mi trabajo..."
                "¿Cómo vas a ayudar?",
            ],
            "Siento de verdad que estés disgustada. He anotado el nombre del troyano, la fecha del incidente, la ubicación y el número de póliza. Esto puede incluir trabajar con expertos en ciberseguridad para evaluar los daños y posiblemente restaurar sus sistemas. Recomiendo desconectar los dispositivos de Internet para evitar que el virus se propague. Al mismo tiempo, organizaremos la asistencia de un experto en ciberseguridad",
            [
                "incident_datetime",
                # "incident_description",
                "policy_number",
            ],
            [
                # No LLM test exclusions
            ],
            id="profanity_cyber",
        ),
        pytest.param(
            "fr-FR",
            [
                "S'il vous plaît, aidez-nous ! Je m'appelle John Udya UDYHIIA et je suis coincé sur l'autoroute. Voici ma Ford Fiesta.",
                "Ma voiture cassée est une Peugeot 307, immatriculée AE345PY.",
                "Il semblerait que mon fils ait un bleu sur le front. Il a un peu mal au ventre aussi, mais il est conscient."
                "Ah oui, nous nous trouvons près de la borne kilométrique 42 sur l'A1.",
            ],
            "Je suis vraiment désolé d'entendre cela. J'ai noté les informations sur le véhicule, son immatriculation et votre position. Je préviens les services d'urgence pour une assistance médicale. Veuillez vous assurer que vous et votre fils êtes en sécurité.",
            [
                # "incident_description",
                "incident_location",
                "injuries_description",
                "caller_name",
                "vehicle_info",
            ],
            [
                "contextual_relevancy",
            ],
            id="car_accident",
        ),
        pytest.param(
            "fr-FR",
            [
                "Je m'appelle Judy Beat BERT et je suis agricultrice. Je suis assurée chez vous sous le contrat BU345POAC.",
                "Mes plants de tomates ont été détruits hier matin par la grêle... Je ne sais pas comment je vais pouvoir payer mes factures. Suis-je couvert par ma garantie ?",
                "Mon exploitation est située à la Ferme Des Anneaux, 59710 Avaline AVELIN.",
                "J'ai une petite exploitation avec 3 employés, et je cultive des tomates, des pommes de terre et des fraises.",
            ],
            "Je suis vraiment désolé d'entendre cela. J'ai noté le nom du preneur d'assurance et le numéro de la police d'assurance. Nous proposons une couverture pour les jeunes plantations contre divers événements naturels.",
            [
                "incident_datetime",
                "incident_description",
                "incident_location",
                "policy_number",
                "caller_name",
            ],
            [
                # No LLM test exclusions
            ],
            id="farmer",
        ),
    ],
)
@pytest.mark.asyncio  # Allow async functions
@pytest.mark.repeat(3)  # Catch non deterministic issues
async def test_llm(
    call_mock: CallStateModel,
    customer_file_tests_excl: list[str],
    customer_file_tests_incl: list[str],
    deepeval_model: GPTModel,
    expected_output: str,
    inputs: list[str],
    lang: str,
) -> None:
    """
    Test the LLM with a mocked conversation against the expected output.

    Steps:
    1. Run application with mocked inputs
    2. Combine all outputs
    3. Test customer file data exists
    4. Test LLM metrics
    """

    def _text_callback(text: str) -> None:
        nonlocal actual_output
        actual_output += f" {text}"

    actual_output = ""
    background_tasks = BackgroundTasks()

    # Mock voice
    CONFIG.voice.mock = VoiceMockModel(text_callback=_text_callback)
    CONFIG.voice.mode = VoiceModeEnum.MOCK

    # Set language
    call_mock.lang = lang

    # Connect call
    await on_call_connected(
        background_tasks=background_tasks,
        call=call_mock,
    )
    await background_tasks()

    # Simulate conversation with speech recognition
    for input in inputs:
        await on_speech_recognized(
            background_tasks=background_tasks,
            call=call_mock,
            text=input,
        )
        await background_tasks()

    # Remove newlines for log comparison
    actual_output = _remove_newlines(actual_output)
    full_input = _remove_newlines(" ".join(inputs))

    # Log for dev review
    _logger.info(f"actual_output: {actual_output}")
    _logger.info(f"customer_file: {call_mock.customer_file}")
    _logger.info(f"full_input: {full_input}")

    # Test customer file data
    for field in customer_file_tests_incl:
        assume(call_mock.customer_file.get(field, None), f"{field} is missing")

    # Configure LLM tests
    test_case = LLMTestCase(
        actual_output=actual_output,
        expected_output=expected_output,
        input=full_input,
        retrieval_context=[
            json.dumps(call_mock.customer_file),
            TypeAdapter(list[ReminderModel]).dump_json(call_mock.reminders).decode(),
            TypeAdapter(list[TrainingModel])
            .dump_json(await call_mock.trainings())
            .decode(),
        ],
    )

    # Define LLM metrics
    llm_metrics = [
        BiasMetric(threshold=1, model=deepeval_model),
        ToxicityMetric(threshold=1, model=deepeval_model),
    ]  # By default, include generic metrics

    if not any(
        field == "answer_relevancy" for field in customer_file_tests_excl
    ):  # Test answer relevancy from questions
        llm_metrics.append(AnswerRelevancyMetric(threshold=0.5, model=deepeval_model))
    if not any(
        field == "contextual_relevancy" for field in customer_file_tests_excl
    ):  # Test answer relevancy from context
        llm_metrics.append(
            ContextualRelevancyMetric(threshold=0.25, model=deepeval_model)
        )

    # Execute LLM tests
    assert_test(test_case, llm_metrics)


def _remove_newlines(text: str) -> str:
    """
    Remove newlines from a string and return it as a single line.
    """
    return " ".join([line.strip() for line in text.splitlines()])
