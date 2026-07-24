"""Apify provider — web scraping and content extraction via actor runs."""

import os

import httpx

APIFY_BASE_URL = "https://api.apify.com/v2"
DEFAULT_CRAWLER = "apify~website-content-crawler"


class ApifyProvider:
    name = "apify"

    def __init__(self):
        self._api_key = os.environ["APIFY_API_KEY"]
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def extract(self, urls: list[str], max_pages: int = 1) -> dict:
        """Extract content from URLs using the website-content-crawler actor."""
        run_input = {
            "startUrls": [{"url": u} for u in urls],
            "maxCrawlPages": max_pages,
            "crawlerType": "cheerio",
        }
        return await self.run_actor(DEFAULT_CRAWLER, run_input, timeout_secs=60)

    async def crawl(self, url: str, max_pages: int = 5) -> dict:
        """Crawl a website section using the website-content-crawler actor."""
        run_input = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": max_pages,
            "crawlerType": "cheerio",
        }
        return await self.run_actor(DEFAULT_CRAWLER, run_input, timeout_secs=90)

    async def run_actor(self, actor_id: str, run_input: dict, timeout_secs: int = 120) -> dict:
        actor_id = actor_id.replace("/", "~")
        async with httpx.AsyncClient(timeout=timeout_secs + 10) as client:
            resp = await client.post(
                f"{APIFY_BASE_URL}/acts/{actor_id}/runs",
                json=run_input,
                headers=self._headers,
                params={"timeout": timeout_secs, "waitForFinish": timeout_secs},
            )
            resp.raise_for_status()
            run_data = resp.json().get("data", {})

            status = run_data.get("status")
            if status != "SUCCEEDED":
                return {"status": status, "error": f"Actor run {status}", "run_id": run_data.get("id")}

            dataset_id = run_data.get("defaultDatasetId")
            if not dataset_id:
                return {"status": status, "items": [], "run_id": run_data.get("id")}

            items_resp = await client.get(
                f"{APIFY_BASE_URL}/datasets/{dataset_id}/items",
                headers=self._headers,
                params={"limit": 100},
            )
            items_resp.raise_for_status()
            items = items_resp.json()

            return {
                "status": "SUCCEEDED",
                "items": items if isinstance(items, list) else [],
                "run_id": run_data.get("id"),
                "item_count": len(items) if isinstance(items, list) else 0,
            }
