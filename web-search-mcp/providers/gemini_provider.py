"""Gemini + Google Search grounding provider."""

import asyncio
import os

from google import genai
from google.genai import types
from google.oauth2 import service_account

from .base import BaseProvider, SearchResult


class GeminiProvider(BaseProvider):
    name = "gemini"
    supported_modes = {"quick", "comprehensive", "academic", "deep"}

    @staticmethod
    def is_available() -> bool:
        sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
        return os.path.exists(sa_path)

    def __init__(self):
        sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
        if not os.path.exists(sa_path):
            raise FileNotFoundError(f"Service account not found: {sa_path}")
        project = os.environ.get("GEMINI_PROJECT")
        if not project:
            raise ValueError("GEMINI_PROJECT environment variable is required")
        location = os.environ.get("GEMINI_LOCATION", "global")
        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._client = genai.Client(vertexai=True, project=project, location=location, credentials=creds)
        self._model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

    def _extract_sources(self, response) -> list[SearchResult]:
        results = []
        seen = set()
        try:
            metadata = response.candidates[0].grounding_metadata
            if metadata and metadata.grounding_chunks:
                for chunk in metadata.grounding_chunks:
                    if chunk.web and chunk.web.uri and chunk.web.uri not in seen:
                        seen.add(chunk.web.uri)
                        results.append(SearchResult(
                            title=chunk.web.title or "",
                            url=chunk.web.uri,
                            snippet="",
                            score=0.75,
                            provider=self.name,
                        ))
        except (AttributeError, IndexError):
            pass

        try:
            metadata = response.candidates[0].grounding_metadata
            if metadata and metadata.grounding_supports:
                for support in metadata.grounding_supports:
                    if support.segment and support.segment.text:
                        for ref in (support.grounding_chunk_indices or []):
                            if ref < len(results) and not results[ref].snippet:
                                results[ref].snippet = support.segment.text[:200]
        except (AttributeError, IndexError):
            pass

        return results

    def _generate(self, contents: str) -> object:
        return self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = await asyncio.to_thread(self._generate, query)
        results = self._extract_sources(response)[:max_results]
        answer = response.text or ""
        if results:
            results[0].raw = {"answer": answer}
        elif answer:
            results.append(SearchResult(
                title="Gemini Answer",
                url="",
                snippet=answer[:500],
                score=0.7,
                provider=self.name,
                raw={"answer": answer},
            ))
        return results

    async def academic_search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = await asyncio.to_thread(
            self._generate, f"Find academic research papers, peer-reviewed studies, and scholarly sources on: {query}"
        )
        results = self._extract_sources(response)[:max_results]
        answer = response.text or ""
        if results:
            results[0].raw = {"answer": answer}
        elif answer:
            results.append(SearchResult(
                title="Gemini Answer",
                url="",
                snippet=answer[:500],
                score=0.7,
                provider=self.name,
                raw={"answer": answer},
            ))
        return results

    async def deep_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        all_results = []
        seen = set()

        response = await asyncio.to_thread(
            self._generate, f"Research this topic thoroughly using web search: {query}"
        )
        for r in self._extract_sources(response):
            if r.url not in seen:
                seen.add(r.url)
                all_results.append(r)

        if len(all_results) < max_results:
            context = response.text or ""
            follow_up = (
                f"Based on this research:\n\n{context}\n\n"
                f"Search for additional information about: {query}\n"
                f"Focus on specific data, statistics, and expert analysis."
            )
            response2 = await asyncio.to_thread(self._generate, follow_up)
            for r in self._extract_sources(response2):
                if r.url not in seen:
                    seen.add(r.url)
                    all_results.append(r)

        return all_results[:max_results]
