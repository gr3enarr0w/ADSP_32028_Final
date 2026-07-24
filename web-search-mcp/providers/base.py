"""Base provider interface for search providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    provider: str = ""
    raw: dict | None = None


class BaseProvider(ABC):
    name: str = ""
    supported_modes: set[str] = set()

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

    async def deep_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        return await self.search(query, max_results=max_results)

    def supports_mode(self, mode: str) -> bool:
        return mode in self.supported_modes
