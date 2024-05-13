from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from contextlib import asynccontextmanager
from helpers.config_models.ai_search import AiSearchModel
from helpers.logging import build_logger
from models.call import CallStateModel
from models.readiness import ReadinessStatus
from models.training import TrainingModel
from persistence.icache import ICache
from persistence.isearch import ISearch
from pydantic import TypeAdapter
from pydantic import ValidationError
from typing import AsyncGenerator, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)


_logger = build_logger(__name__)


class AiSearchSearch(ISearch):
    _config: AiSearchModel

    def __init__(self, cache: ICache, config: AiSearchModel):
        _logger.info(f"Using AI Search {config.endpoint} with index {config.index}")
        _logger.info(
            f"Note: At ~300 chars /doc, each LLM call will use approx {300 * config.top_k * config.expansion_k / 4} tokens (without tools)"
        )
        self._config = config
        super().__init__(cache)

    async def areadiness(self) -> ReadinessStatus:
        """
        Check the readiness of the AI Search service.
        """
        try:
            async with self._use_db() as db:
                await db.get_document_count()
            return ReadinessStatus.OK
        except HttpResponseError as e:
            _logger.error(f"Error requesting AI Search, {e}")
        except ServiceRequestError as e:
            _logger.error(f"Error connecting to AI Search, {e}")
        return ReadinessStatus.FAIL

    @retry(
        reraise=True,
        retry=retry_if_exception_type(ServiceResponseError),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.5, max=30),
    )
    async def training_asearch_all(
        self, text: str, call: CallStateModel
    ) -> Optional[list[TrainingModel]]:
        _logger.debug(f'Searching training data for "{text}"')
        if not text:
            return None

        # Try cache
        cache_key = f"{self.__class__.__name__}-training_asearch_all-{text}"
        cached = await self._cache.aget(cache_key)
        if cached:
            try:
                return TypeAdapter(list[TrainingModel]).validate_json(cached)
            except ValidationError as e:
                _logger.debug(f"Parsing error: {e.errors()}")

        # Try live
        trainings: list[TrainingModel] = []
        try:
            async with self._use_db() as db:
                results = await db.search(
                    # Full text search
                    query_type="semantic",
                    semantic_configuration_name=self._config.semantic_configuration,
                    search_fields=[
                        "content",
                        "title",
                    ],
                    search_text=text,
                    # Spell correction
                    query_language=call.lang.short_code,
                    query_speller="lexicon",
                    # Vector search
                    vector_queries=[
                        VectorizableTextQuery(
                            fields="vectors",
                            text=text,
                        )
                    ],
                    # Return fields
                    select=[
                        "id",
                        "content",
                        "source_uri",
                        "title",
                    ],
                    top=self._config.top_k,
                )
                async for result in results:
                    try:
                        trainings.append(
                            TrainingModel.model_validate(
                                {
                                    **result,
                                    "score": result[
                                        "@search.score"
                                    ],  # TODO: Use score from semantic ranking with "@search.rerankerScore"
                                }
                            )
                        )
                    except ValidationError as e:
                        _logger.debug(f"Parsing error: {e.errors()}")
        except HttpResponseError as e:
            _logger.error(f"Error requesting AI Search, {e}")
        except ServiceRequestError as e:
            _logger.error(f"Error connecting to AI Search, {e}")

        # Update cache
        await self._cache.aset(
            cache_key,
            (
                TypeAdapter(list[TrainingModel]).dump_json(trainings)
                if trainings
                else None
            ),
        )

        return trainings or None

    @asynccontextmanager
    async def _use_db(self) -> AsyncGenerator[SearchClient, None]:
        """
        Generate the AI Search client and close it after use.
        """
        db = SearchClient(
            credential=AzureKeyCredential(self._config.access_key.get_secret_value()),
            endpoint=self._config.endpoint,
            index_name=self._config.index,
        )
        try:
            yield db
        finally:
            await db.close()
