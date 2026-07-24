# Web Search MCP

Cost-optimized, single-provider web search MCP server. Auto-routes each query to the best provider based on query type, tracks usage to conserve API credits, and offers escalation options when results are thin.

## Architecture

- `server.py` — FastMCP entry point with 4 tools
- `orchestrator.py` — Query classification, single-provider routing, escalation chain
- `usage_tracker.py` — Persists API call counts to `~/.claude/web-search-mcp-usage.json`
- `providers/` — Individual provider implementations (Exa, Brave, Tavily, Gemini, Linkup, Newsdata, Apify)

## Tools

| Tool | Provider | Purpose |
|------|----------|---------|
| `research` | Auto-routed | Main search — one query, one provider |
| `research_escalate` | Auto-fallback | Try next provider (user-approved only) |
| `extract` | Apify | Extract content from URLs or crawl a site section |
| `scrape` | Apify | Run a custom Apify actor for advanced scraping |

## Supported Providers

- **Exa** — High-quality search results with semantic ranking
- **Brave** — Privacy-focused search with fact-checking
- **Tavily** — AI-optimized search for research
- **Linkup** — Company and organizational research
- **Newsdata** — News aggregation and current events
- **Gemini** — Google's generative AI with web grounding (requires GCP project)
- **Apify** — Web scraping and content extraction

## Configuration

This MCP requires API keys for the providers you want to use. All API keys are configured via environment variables — see `.env.example` for the complete list.

### Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file with your API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys from each provider
   ```

3. **Run the server:**
   ```bash
   python server.py
   ```

### API Key Sources

- **EXA_API_KEY**: https://exa.ai
- **BRAVE_SEARCH_API_KEY**: https://api.search.brave.com
- **TAVILY_API_KEY**: https://tavily.com
- **LINKUP_API_KEY**: https://linkup.so
- **NEWSDATA_API_KEY**: https://newsdata.io
- **APIFY_API_KEY**: https://apify.com
- **GEMINI_PROJECT**: Your GCP project ID (set `GOOGLE_SERVICE_ACCOUNT_JSON` to path of service account JSON file)

### Optional Configuration

- `GEMINI_LOCATION`: GCP location for Gemini (default: `global`)
- `GEMINI_MODEL`: Model to use for Gemini searches (default: `gemini-3.1-pro-preview`)
- `BRAVE_ANSWERS_API_KEY`: Optional separate API key for Brave Answers API

## Usage

Register as an MCP server in your Claude client's MCP configuration. The server auto-selects the best provider for your query type.
