"""Brave Search provider — web search and AI answers."""

import os

import httpx

from .base import BaseProvider, SearchResult

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveProvider(BaseProvider):
    name = "brave"
    supported_modes = {"quick", "comprehensive", "news"}

    def __init__(self):
        self._search_key = os.environ["BRAVE_SEARCH_API_KEY"]
        self._answers_key = os.environ.get("BRAVE_ANSWERS_API_KEY", "")

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                BRAVE_SEARCH_URL,
                params={"q": query, "count": max_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._search_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                score=0.7,
                provider=self.name,
            ))
        return results

    async def news_search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                BRAVE_SEARCH_URL,
                params={"q": query, "count": max_results, "freshness": "pw"},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._search_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("news", {}).get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                score=0.75,
                provider=self.name,
            ))
        if not results:
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    score=0.7,
                    provider=self.name,
                ))
        return results
