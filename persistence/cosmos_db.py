from azure.cosmos.aio import CosmosClient, ContainerProxy
from azure.cosmos.exceptions import CosmosHttpResponseError
from contextlib import asynccontextmanager
from helpers.config import CONFIG
from helpers.config_models.database import CosmosDbModel
from helpers.logging import build_logger
from models.call import CallStateModel
from models.readiness import ReadinessStatus
from persistence.istore import IStore
from pydantic import ValidationError
from typing import AsyncGenerator, Optional
from uuid import UUID, uuid4


_logger = build_logger(__name__)


class CosmosDbStore(IStore):
    _config: CosmosDbModel

    def __init__(self, config: CosmosDbModel):
        _logger.info(f"Using Cosmos DB {config.database}/{config.container}")
        self._config = config

    async def areadiness(self) -> ReadinessStatus:
        """
        Check the readiness of the Cosmos DB service.

        This will validate the ACID properties of the database: Create, Read, Update, Delete.
        """
        test_id = str(uuid4())
        test_partition = "+33612345678"
        test_dict = {
            "id": test_id,  # unique id
            "phone_number": test_partition,  # partition key
            "test": "test",
        }
        try:
            # Test the item does not exist
            if await self._item_exists(test_id, test_partition):
                return ReadinessStatus.FAIL
            async with self._use_db() as db:
                # Create a new item
                await db.upsert_item(body=test_dict)
                # Test the item is the same
                read_item = await db.read_item(
                    item=test_id, partition_key=test_partition
                )
                assert {
                    k: v for k, v in read_item.items() if k in test_dict
                } == test_dict  # Check only the relevant fields, Cosmos DB adds metadata
                # Delete the item
                await db.delete_item(item=test_id, partition_key=test_partition)
            # Test the item does not exist
            if await self._item_exists(test_id, test_partition):
                return ReadinessStatus.FAIL
            return ReadinessStatus.OK
        except AssertionError:
            _logger.error("Readiness test failed", exc_info=True)
        except CosmosHttpResponseError as e:
            _logger.error(f"Error requesting CosmosDB, {e}")
        return ReadinessStatus.FAIL

    async def _item_exists(self, test_id: str, partition_key: str) -> bool:
        exist = False
        async with self._use_db() as db:
            try:
                await db.read_item(item=test_id, partition_key=partition_key)
                exist = True
            except CosmosHttpResponseError as e:
                if e.status_code != 404:
                    _logger.error(f"Error requesting CosmosDB, {e}")
                    exist = True
        return exist

    async def call_aget(self, call_id: UUID) -> Optional[CallStateModel]:
        _logger.debug(f"Loading call {call_id}")
        call = None
        try:
            async with self._use_db() as db:
                items = db.query_items(
                    query="SELECT * FROM c WHERE STRINGEQUALS(c.id, @id)",
                    parameters=[{"name": "@id", "value": str(call_id)}],
                )
                raw = await anext(items)
                try:
                    call = CallStateModel.model_validate(raw)
                except ValidationError as e:
                    _logger.warning(f"Error parsing call {str(call_id)}")
                    _logger.debug(f"Parsing error: {e.errors()}")
        except StopAsyncIteration:
            pass
        except CosmosHttpResponseError as e:
            _logger.error(f"Error accessing CosmosDB, {e}")
        return call

    async def call_aset(self, call: CallStateModel) -> bool:
        data = call.model_dump(mode="json", exclude_none=True)
        data["id"] = str(call.call_id)  # CosmosDB requires an id field
        _logger.debug(f"Saving call {call.call_id}: {data}")
        try:
            async with self._use_db() as db:
                await db.upsert_item(body=data)
            return True
        except CosmosHttpResponseError as e:
            _logger.error(f"Error accessing CosmosDB: {e}")
            return False

    async def call_asearch_one(self, phone_number: str) -> Optional[CallStateModel]:
        _logger.debug(f"Loading last call for {phone_number}")
        call = None
        try:
            async with self._use_db() as db:
                items = db.query_items(
                    max_item_count=1,
                    query=f"SELECT * FROM c WHERE (STRINGEQUALS(c.initiate.phone_number, @phone_number, true) OR STRINGEQUALS(c.customer_file.caller_phone, @phone_number, true)) AND c.created_at >= DATETIMEADD('hh', -{CONFIG.workflow.conversation_timeout_hour}, GETCURRENTDATETIME()) ORDER BY c.created_at DESC",
                    parameters=[
                        {
                            "name": "@phone_number",
                            "value": phone_number,
                        }
                    ],
                )
                raw = await anext(items)
                try:
                    call = CallStateModel.model_validate(raw)
                except ValidationError as e:
                    _logger.debug(f"Parsing error: {e.errors()}")
        except StopAsyncIteration:
            pass
        except CosmosHttpResponseError as e:
            _logger.error(f"Error accessing CosmosDB: {e}")
        return call

    async def call_asearch_all(
        self, phone_number: str
    ) -> Optional[list[CallStateModel]]:
        _logger.debug(f"Loading all calls for {phone_number}")
        calls = []
        try:
            async with self._use_db() as db:
                items = db.query_items(
                    query="SELECT * FROM c WHERE STRINGEQUALS(c.initiate.phone_number, @phone_number, true) OR STRINGEQUALS(c.customer_file.caller_phone, @phone_number, true) ORDER BY c.created_at DESC",
                    parameters=[
                        {
                            "name": "@phone_number",
                            "value": phone_number,
                        },
                    ],
                )
                async for raw in items:
                    try:
                        calls.append(CallStateModel.model_validate(raw))
                    except ValidationError as e:
                        _logger.debug(f"Parsing error: {e.errors()}")
        except CosmosHttpResponseError as e:
            _logger.error(f"Error accessing CosmosDB, {e}")
        return calls or None

    @asynccontextmanager
    async def _use_db(self) -> AsyncGenerator[ContainerProxy, None]:
        """
        Generate the Cosmos DB client and close it after use.
        """
        client = CosmosClient(
            # Reliability
            connection_timeout=10,
            retry_backoff_factor=0.5,
            retry_backoff_max=30,
            retry_total=3,
            # Azure deployment
            url=self._config.endpoint,
            # Authentication with API key
            credential=self._config.access_key.get_secret_value(),
        )
        try:
            database = client.get_database_client(self._config.database)
            yield database.get_container_client(self._config.container)
        finally:
            await client.close()
