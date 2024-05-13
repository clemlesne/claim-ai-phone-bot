from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from deepeval.models.gpt_model import GPTModel
from helpers.config import CONFIG
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from models.call import CallStateModel, CallInitiateModel
from textwrap import dedent
from typing import Any, Tuple
import hashlib
import pytest
import random
import string


@pytest.fixture
def random_text() -> str:
    text = "".join(random.choice(string.printable) for _ in range(100))
    return text


@pytest.fixture
def call_mock() -> CallStateModel:
    call = CallStateModel(
        initiate=CallInitiateModel(
            **CONFIG.workflow.default_initiate.model_dump(),
            phone_number="+33612345678",  # type: ignore
        )
    )
    return call


@pytest.fixture
def deepeval_model(cache: pytest.Cache) -> GPTModel:
    return DeepEvalAzureOpenAI(cache)


class DeepEvalAzureOpenAI(GPTModel):
    _cache: pytest.Cache
    _langchain_kwargs: dict[str, Any]
    _model: BaseChatModel

    def __init__(
        self,
        cache: pytest.Cache,
        **kwargs,
    ):
        platform = CONFIG.llm.backup.azure_openai
        assert platform

        _langchain_kwargs = {
            # Repeatability
            "model_kwargs": {
                "seed": 42,
            },
            "temperature": 0,
            # Azure deployment
            "api_version": "2023-12-01-preview",
            "azure_deployment": platform.deployment,
            "azure_endpoint": platform.endpoint,
            "model": platform.model,
            # Authentication, either RBAC or API
            "api_key": platform.api_key.get_secret_value() if platform.api_key else None,  # type: ignore
            "azure_ad_token_provider": (
                get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                if not platform.api_key
                else None
            ),
            # DeepEval
            **kwargs,
        }
        self._cache = cache
        self._model = AzureChatOpenAI(**_langchain_kwargs)

    def generate(self, prompt: str) -> Tuple[str, float]:
        raise NotImplementedError

    async def a_generate(self, prompt: str) -> Tuple[str, float]:
        prompt = dedent(prompt).strip()
        cache_key = self._cache_key(prompt)
        # Try cache
        content: Tuple[str, float] = self._cache.get(cache_key, None)
        if content:
            return content
        # Try live
        res = await super().a_generate(prompt)
        # Update cache
        self._cache.set(cache_key, res)
        return res

    def get_model_name(self) -> str:
        return "Azure OpenAI"

    def load_model(self) -> BaseChatModel:
        return self._model

    def should_use_azure_openai(self) -> bool:
        return True

    def _cache_key(self, prompt: str) -> str:
        llm_string = self._model._get_llm_string(input=prompt)
        llm_hash = hashlib.sha256(llm_string.encode(), usedforsecurity=False).digest()
        return f"call-center-ai/{llm_hash}"
