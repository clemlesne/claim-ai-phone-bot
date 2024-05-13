from helpers.config import CONFIG
from helpers.config_models.database import ModeEnum as DatabaseModeEnum
from models.call import CallStateModel
from pytest import assume
import pytest


@pytest.mark.parametrize(
    "database_mode",
    [
        pytest.param(
            DatabaseModeEnum.SQLITE,
            id="sqlite",
        ),
        pytest.param(
            DatabaseModeEnum.COSMOS_DB,
            id="cosmos_db",
        ),
    ],
)
@pytest.mark.asyncio  # Allow async functions
@pytest.mark.repeat(10)  # Catch multi-threading and concurrency issues
async def test_acid(call_mock: CallStateModel, database_mode: DatabaseModeEnum) -> None:
    """
    Test ACID properties of the database backend.

    Steps:
    1. Create a mock data
    2. Test not exists
    3. Insert test data
    4. Check it exists

    Test is repeated 10 times to catch multi-threading and concurrency issues.
    """
    # Set database mode
    CONFIG.database.mode = database_mode
    db = CONFIG.database.instance()

    # Check not exists
    assume(not await db.call_aget(call_mock.call_id))
    assume(await db.call_asearch_one(call_mock.initiate.phone_number) != call_mock)
    assume(
        call_mock
        not in (await db.call_asearch_all(call_mock.initiate.phone_number) or [])
    )

    # Insert test call
    await db.call_aset(call_mock)

    # Check point read
    assume(await db.call_aget(call_mock.call_id) == call_mock)
    # Check search one
    assume(await db.call_asearch_one(call_mock.initiate.phone_number) == call_mock)
    # Check search all
    assume(
        call_mock in (await db.call_asearch_all(call_mock.initiate.phone_number) or [])
    )
