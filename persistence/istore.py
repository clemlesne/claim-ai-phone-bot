from abc import ABC, abstractmethod
from models.call import CallStateModel
from models.readiness import ReadinessStatus
from typing import Optional
from uuid import UUID


class IStore(ABC):

    @abstractmethod
    async def areadiness(self) -> ReadinessStatus:
        pass

    @abstractmethod
    async def call_aget(self, call_id: UUID) -> Optional[CallStateModel]:
        pass

    @abstractmethod
    async def call_aset(self, call: CallStateModel) -> bool:
        pass

    @abstractmethod
    async def call_asearch_one(self, phone_number: str) -> Optional[CallStateModel]:
        pass

    @abstractmethod
    async def call_asearch_all(
        self, phone_number: str
    ) -> Optional[list[CallStateModel]]:
        pass
