"""Research Tool MCP — cost-optimized single-provider search orchestration."""

import logging
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from fastmcp import FastMCP

import orchestrator
from providers.apify_provider import ApifyProvider

mcp = FastMCP("research-tool")


@mcp.tool()
async def research(query: str, mode: str = "auto") -> dict:
    """Search the web using the best provider for this query type.

    Auto-classifies the query and routes to ONE optimal provider.
    Returns a synthesized answer (when available) plus source list.
    Includes usage tracking and escalation hints.

    Args:
        query: Search query or research question.
        mode: Override auto-routing. Options: auto, quick, code, academic,
              company, people, news, deep, general.

    Returns:
        Dict with answer, sources, provider_used, mode, usage stats,
        and escalation_hint if results are thin.
    """
    return await orchestrator.search(query, mode=mode)


@mcp.tool()
async def research_escalate(query: str, mode: str = "auto", exclude_providers: list[str] | None = None) -> dict:
    """Try the next-best provider after initial results were thin.

    Only call this when the user approves escalation. Picks the fallback
    provider, excluding any already tried.

    Args:
        query: Same query from the initial search.
        mode: Same mode (or auto to re-classify).
        exclude_providers: Providers already tried (from previous results).

    Returns:
        Dict with additional sources from a different provider.
    """
    return await orchestrator.escalate(query, mode=mode, exclude_providers=exclude_providers)


@mcp.tool()
async def extract(urls: list[str], max_pages: int = 1) -> dict:
    """Extract content from URLs or crawl a website section.

    Uses Apify's website-content-crawler. Set max_pages=1 to extract
    content from specific URLs, or higher to crawl linked pages.

    Args:
        urls: URLs to extract content from (or starting URLs for crawling).
        max_pages: Pages to collect per URL. 1 = extract only, >1 = crawl.

    Returns:
        Dict with status, items (page content), and item_count.
    """
    provider = ApifyProvider()
    if max_pages <= 1:
        return await provider.extract(urls)
    return await provider.crawl(urls[0], max_pages=max_pages)


@mcp.tool()
async def scrape(actor_id: str, run_input: dict, timeout_secs: int = 120) -> dict:
    """Run an Apify actor for custom web scraping.

    For advanced scraping needs — structured data, JS-heavy sites,
    pagination, or specialized extractors.

    Args:
        actor_id: Apify actor ID (e.g., 'apify/web-scraper').
        run_input: Input configuration for the actor.
        timeout_secs: Max seconds to wait for completion (default 120).

    Returns:
        Dict with status, items (scraped data), and run_id.
    """
    provider = ApifyProvider()
    return await provider.run_actor(actor_id, run_input, timeout_secs=timeout_secs)


if __name__ == "__main__":
    mcp.run()
