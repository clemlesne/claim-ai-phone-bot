from models.call import CallModel
from semantic_kernel.orchestration.kernel_context import KernelContext
from semantic_kernel.plugin_definition import (
    kernel_function,
    kernel_function_context_parameter,
)
from persistence.isearch import ISearch


class TrainingPlugin:
    _call: CallModel
    _search: ISearch

    def __init__(self, call: CallModel, search: ISearch):
        self._call = call
        self._search = search

    @kernel_function(
        description="Use this if you want to search for a document in the training database. Example: 'A document about the new car insurance policy', 'A document about the new car insurance policy'.",
        name="Search",
    )
    @kernel_function_context_parameter(
        description="Query to search for.",
        name="query",
    )
    async def search(self, context: KernelContext, query: str) -> str:
        trainings = await self._search.training_asearch_all(query, self._call)

        res = ""
        for i, training in enumerate(trainings or []):
            res += f"\n\n# Result {i + 1}"
            res += f"\nTitle: {training.title}"
            res += f"\nContent: {training.content}"

        return res.strip()
