"""Tavily search provider — optimized for AI agent workflows."""

import os

import httpx

from .base import BaseProvider, SearchResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
TAVILY_MAP_URL = "https://api.tavily.com/map"
TAVILY_CRAWL_URL = "https://api.tavily.com/crawl"


class TavilyProvider(BaseProvider):
    name = "tavily"
    supported_modes = {"quick", "comprehensive", "academic", "news", "deep"}

    def __init__(self):
        self._api_key = os.environ["TAVILY_API_KEY"]

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
                score=item.get("score", 0.7),
                provider=self.name,
                raw={"answer": data.get("answer", "")},
            ))
        return results

    async def deep_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "advanced",
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:800],
                score=item.get("score", 0.7),
                provider=self.name,
                raw={"answer": data.get("answer", "")},
            ))
        return results

    async def extract(self, urls: list[str], query: str = "") -> dict:
        payload = {"api_key": self._api_key, "urls": urls}
        if query:
            payload["query"] = query
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(TAVILY_EXTRACT_URL, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def crawl(self, url: str, max_pages: int = 5) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                TAVILY_CRAWL_URL,
                json={"api_key": self._api_key, "url": url, "max_depth": 1, "limit": max_pages},
            )
            resp.raise_for_status()
            return resp.json()

    async def map_site(self, url: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                TAVILY_MAP_URL,
                json={"api_key": self._api_key, "url": url},
            )
            resp.raise_for_status()
            return resp.json()
