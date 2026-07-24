"""Newsdata.io provider — dedicated news search API using official Python SDK."""

import asyncio
import os

from newsdataapi import NewsDataApiClient

from .base import BaseProvider, SearchResult


class NewsdataProvider(BaseProvider):
    name = "newsdata"
    supported_modes = {"news"}

    def __init__(self):
        self._client = NewsDataApiClient(apikey=os.environ["NEWSDATA_API_KEY"])

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        return await self.news_search(query, max_results)

    async def news_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        data = await asyncio.to_thread(
            self._client.latest_api,
            q=query,
            language="en",
            size=min(max_results, 10),
            prioritydomain="top",
            removeduplicate=True,
        )
        if data.get("status") == "error":
            raise RuntimeError(data.get("message", "newsdata API error"))
        results = []
        for item in (data.get("results") or [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("description") or (item.get("content") or "")[:500],
                score=0.75,
                provider=self.name,
            ))
        return results
