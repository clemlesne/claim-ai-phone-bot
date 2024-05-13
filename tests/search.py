from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.models.gpt_model import GPTModel
from deepeval.test_case import LLMTestCase
from helpers.config import CONFIG
from helpers.logging import build_logger
from models.call import CallStateModel
from typing import Optional
import asyncio
import pytest


_logger = build_logger(__name__)


class RagRelevancyMetric(BaseMetric):
    model: GPTModel
    score: Optional[float] = 0
    success: bool = False
    threshold: float

    def __init__(
        self,
        model: GPTModel,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.model = model

    async def a_measure(self, test_case: LLMTestCase) -> float:
        assert test_case.input
        assert test_case.retrieval_context
        # Measure each document in parallel
        with metric_progress_indicator(self, async_mode=True):
            scores = await asyncio.gather(
                *[
                    self._measure_single(test_case.input, document)
                    for document in test_case.retrieval_context
                ]
            )
            # Score is the average
            self.score = sum(scores) / len(scores)
        # Test against the threshold
        self.success = self.score >= self.threshold
        return self.score

    async def _measure_single(self, message: str, document: str) -> float:
        score = 0
        llm_res, _ = await self.model.a_generate(
            f"""
            Assistant is a data analyst expert with 20 years of experience.

            # Objective
            Assistant will analyze a document and decide whether it would be useful to respond to the user message.

            # Context
            The document comes from a database. It has been stemmed. It may contains technical data and jargon a specialist would use.

            # Rules
            - Answer only with the float value, never add other text
            - Response 0.0 means not useful at all, 1.0 means totally useful

            # Message
            {message}

            # Document
            {document}

            # Response format
            A float from 0.0 to 1.0

            ## Example 1
            Message: I love bananas
            Document: bananas are yellow, apples are red
            Assistant: 1

            ## Example 2
            Message: The sky is blue
            Document: mouse is a rodent, mouse is a computer peripheral
            Assistant: 0

            ## Example 3
            Message: my car is stuck in the mud
            Document: car accidents must be reported within 24 hours, car accidents are dangerous
            Assistant: 0.7
        """
        )
        try:
            score = float(llm_res)
        except ValueError:
            raise ValueError(f"LLM response is not a number: {llm_res}")
        return score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "RAG Relevancy"


@pytest.mark.parametrize(
    "user_message, user_lang",
    [
        pytest.param(
            "accident de voiture sur autoroute",
            "fr-FR",
            id="car_fr",
        ),
        pytest.param(
            "cyber-attaque par DDOS sur mon entreprise",
            "fr-FR",
            id="cyberattack_fr",
        ),
        pytest.param(
            "refus passage expert souhaite contre-expertise",
            "fr-FR",
            id="expertise_fr",
        ),
        pytest.param(
            "grêlons champ réculte tomates perdue indemnisation",
            "fr-FR",
            id="hail_fr",
        ),
    ],
)
@pytest.mark.asyncio  # Allow async functions
@pytest.mark.repeat(10)  # Catch multi-threading and concurrency issues
async def test_relevancy(
    call_mock: CallStateModel,
    deepeval_model: GPTModel,
    user_lang: str,
    user_message: str,
) -> None:
    """
    Test the relevancy of the Message to the training data.

    Steps:
    1. Search for training data
    2. Ask the LLM for relevancy score
    3. Assert the relevancy score is above 0.5

    Test is repeated 10 times to catch multi-threading and concurrency issues.
    """
    search = CONFIG.ai_search.instance()
    call_mock.lang = user_lang

    # Init data
    res_models = await search.training_asearch_all(user_message, call_mock)
    res_list = [d.content for d in res_models or []]

    _logger.info(f"Message: {user_message}")
    _logger.info(f"Data: {res_list}")

    # Configure LLM tests
    test_case = LLMTestCase(
        actual_output="",  # Not used
        input=user_message,
        retrieval_context=res_list,
    )

    # Define LLM metrics
    llm_metrics = [
        RagRelevancyMetric(
            threshold=0.5, model=deepeval_model
        ),  # Compare input to the retrieval context
    ]

    # Execute LLM tests
    assert_test(test_case, llm_metrics)
