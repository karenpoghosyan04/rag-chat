import asyncio
from abc import ABC, abstractmethod

from app.models.research import ResearchState


class ResearchRepository(ABC):
    @abstractmethod
    async def create(self, state: ResearchState) -> ResearchState: ...

    @abstractmethod
    async def get(self, task_id: str) -> ResearchState | None: ...

    @abstractmethod
    async def save(self, state: ResearchState) -> None: ...


class InMemoryResearchRepository(ResearchRepository):
    def __init__(self) -> None:
        self._tasks: dict[str, ResearchState] = {}
        self._lock = asyncio.Lock()

    async def create(self, state: ResearchState) -> ResearchState:
        async with self._lock:
            self._tasks[state.task_id] = state
        return state

    async def get(self, task_id: str) -> ResearchState | None:
        return self._tasks.get(task_id)

    async def save(self, state: ResearchState) -> None:
        async with self._lock:
            self._tasks[state.task_id] = state

