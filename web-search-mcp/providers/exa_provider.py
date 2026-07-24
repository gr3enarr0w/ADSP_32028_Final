"""Exa search provider — neural search with structured output support."""

import os

from exa_py import AsyncExa

from .base import BaseProvider, SearchResult


def _to_results(resp, provider_name: str, snippet_len: int = 500) -> list[SearchResult]:
    return [
        SearchResult(
            title=r.title or "",
            url=r.url or "",
            snippet=(r.text or "")[:snippet_len],
            score=getattr(r, "score", 0.0) or 0.8,
            provider=provider_name,
        )
        for r in resp.results
    ]


class ExaProvider(BaseProvider):
    name = "exa"
    supported_modes = {"quick", "comprehensive", "academic", "code", "company", "people", "deep", "news"}

    def __init__(self):
        self._client = AsyncExa(api_key=os.environ["EXA_API_KEY"])

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.search_and_contents(
            query,
            type="auto",
            num_results=max_results,
            text={"max_characters": 8000},
        )
        return _to_results(resp, self.name, 500)

    async def deep_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        resp = await self._client.search_and_contents(
            query,
            type="deep",
            num_results=max_results,
            text={"max_characters": 12000},
        )
        return _to_results(resp, self.name, 800)

    async def code_search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.search_and_contents(
            query,
            type="auto",
            num_results=max_results,
            text={"max_characters": 10000, "include_html_tags": True},
            include_domains=["github.com", "stackoverflow.com", "dev.to", "medium.com"],
        )
        return _to_results(resp, self.name, 800)

    async def company_search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.search_and_contents(
            query,
            type="auto",
            num_results=max_results,
            category="company",
            text={"max_characters": 8000},
        )
        return _to_results(resp, self.name, 500)

    async def people_search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.search_and_contents(
            query,
            type="auto",
            num_results=max_results,
            category="people",
            text={"max_characters": 6000},
        )
        return _to_results(resp, self.name, 500)
