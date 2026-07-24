"""Linkup search provider — AI-powered web search.

API docs: https://docs.linkup.so/pages/documentation/api-reference/endpoint/post-search
Auth: Bearer token in Authorization header
Depths: fast (sub-second), standard (agentic), deep (multi-iteration)
Response: { results: [{ name, url, content, type, favicon }] }
"""

import os

import httpx

from .base import BaseProvider, SearchResult

LINKUP_SEARCH_URL = "https://api.linkup.so/v1/search"


class LinkupProvider(BaseProvider):
    name = "linkup"
    supported_modes = {"quick", "comprehensive", "deep"}

    def __init__(self):
        self._api_key = os.environ["LINKUP_API_KEY"]
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                LINKUP_SEARCH_URL,
                json={
                    "q": query,
                    "depth": "standard",
                    "outputType": "searchResults",
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
                score=0.7,
                provider=self.name,
            ))
        return results

    async def deep_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                LINKUP_SEARCH_URL,
                json={
                    "q": query,
                    "depth": "deep",
                    "outputType": "searchResults",
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:800],
                score=0.7,
                provider=self.name,
            ))
        return results
