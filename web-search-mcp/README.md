# Web Search MCP

A cost-optimized, single-provider web search MCP (Model Context Protocol) server that intelligently routes search queries to the most appropriate provider based on query type. Supports 7 different search/research providers with usage tracking and escalation options.

## Features

- **Smart Provider Routing**: Automatically selects the best search provider (Brave, Tavily, Exa, Gemini, Linkup, Newsdata) based on query type
- **Cost Optimization**: Tracks API usage to help conserve credits and respects free tier limits
- **Escalation Support**: When initial results are thin, optionally escalate to a fallback provider
- **Multi-Mode Search**: Supports quick facts, code search, academic research, news, people search, company research, and deep investigation
- **Content Extraction**: Built-in web scraping and URL content extraction via Apify
- **Generic and Extensible**: Works with any combination of providers — just add your API keys

## Supported Providers

| Provider | Specialty | API Source |
|----------|-----------|-----------|
| **Brave** | Fast general/quick searches, privacy-focused | https://api.search.brave.com |
| **Tavily** | AI-optimized search, code/technical content | https://tavily.com |
| **Exa** | Semantic search, high-quality results | https://exa.ai |
| **Linkup** | Company and organizational research | https://linkup.so |
| **Newsdata** | News aggregation and current events | https://newsdata.io |
| **Gemini** | Academic research with grounding, deep investigation (requires GCP) | https://cloud.google.com |
| **Apify** | Web scraping, content extraction, custom crawling | https://apify.com |

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env and add your API keys from each provider you want to use
   # You only need to configure the providers you plan to use
   nano .env
   ```

4. **Verify setup:**
   ```bash
   python server.py
   ```
   You should see the FastMCP server start without errors.

## Configuration

### Required Environment Variables

At minimum, configure at least one provider's API key. The server will work with any subset of providers.

```bash
# At least one of these search providers should be configured:
EXA_API_KEY=your_key
BRAVE_SEARCH_API_KEY=your_key
TAVILY_API_KEY=your_key
LINKUP_API_KEY=your_key
NEWSDATA_API_KEY=your_key
APIFY_API_KEY=your_key

# For Gemini (optional, enables academic/deep search mode)
GEMINI_PROJECT=your-gcp-project-id
GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service_account.json
```

### Optional Environment Variables

```bash
# Brave (optional second API key)
BRAVE_ANSWERS_API_KEY=your_key

# Gemini customization
GEMINI_LOCATION=us-central1  # default: global
GEMINI_MODEL=gemini-3.1-pro-preview  # default: gemini-3.1-pro-preview
```

## Getting API Keys

- **Exa**: https://exa.ai (free tier available)
- **Brave Search**: https://api.search.brave.com (free tier: $5 credit)
- **Tavily**: https://tavily.com (free tier: 1,000 API calls/month)
- **Linkup**: https://linkup.so (free tier: 1,000 calls/month)
- **Newsdata**: https://newsdata.io (free tier: 200 queries/day)
- **Apify**: https://apify.com (free tier available)
- **Google Gemini/Vertex**: Requires active GCP account with Vertex AI enabled

## Usage

### Running the Server

```bash
python server.py
```

### Tools

#### 1. `research(query, mode="auto")`

Performs a search using the optimal provider for the query type.

**Parameters:**
- `query` (string, required): Your search query or research question
- `mode` (string, optional): Force a specific search mode. Options:
  - `auto` (default): Auto-detect based on query content
  - `quick`: Fast factual searches
  - `code`: Technical and programming content
  - `academic`: Scholarly research and peer-reviewed papers
  - `company`: Company and organizational information
  - `people`: Person/biography searches
  - `news`: News and current events
  - `deep`: Comprehensive, multi-faceted research
  - `general`: General purpose search

**Returns:**
- `answer`: AI-synthesized answer (when available)
- `sources`: List of source links with snippets
- `provider_used`: Which provider was selected
- `mode`: The search mode used
- `usage_stats`: Current API usage counts
- `escalation_hint`: Suggestion to escalate if results are thin

#### 2. `research_escalate(query, mode="auto", exclude_providers=None)`

Escalates to the fallback provider when initial results are insufficient.

**Parameters:**
- `query` (string, required): Same query from initial search
- `mode` (string, optional): Search mode
- `exclude_providers` (list, optional): Providers already tried

**Returns:** Same format as `research()`

#### 3. `extract(urls, max_pages=1)`

Extracts text content from web pages or crawls a website section.

**Parameters:**
- `urls` (list of strings, required): URLs to extract content from
- `max_pages` (integer, optional): Number of pages to crawl. Set to 1 for extraction only.

**Returns:**
- `status`: Success/failure status
- `items`: Extracted content from each page
- `item_count`: Total items extracted

#### 4. `scrape(actor_id, run_input, timeout_secs=120)`

Runs a custom Apify actor for advanced scraping scenarios.

**Parameters:**
- `actor_id` (string, required): Apify actor ID (e.g., 'apify/web-scraper')
- `run_input` (object, required): Configuration for the actor
- `timeout_secs` (integer, optional): Timeout in seconds (default: 120)

**Returns:**
- `status`: Success/failure status
- `items`: Scraped data
- `run_id`: Apify run ID for reference

## Architecture

- **server.py** — FastMCP entry point with tool definitions
- **orchestrator.py** — Query classification, provider routing logic, escalation chain
- **usage_tracker.py** — API usage counting and persistence (`~/.claude/web-search-mcp-usage.json`)
- **providers/** — Individual provider client implementations
  - `base.py` — Base provider interface
  - `brave_provider.py` — Brave Search implementation
  - `tavily_provider.py` — Tavily implementation
  - `exa_provider.py` — Exa implementation
  - `linkup_provider.py` — Linkup implementation
  - `newsdata_provider.py` — Newsdata implementation
  - `gemini_provider.py` — Google Gemini/Vertex AI implementation
  - `apify_provider.py` — Apify web scraping implementation

## Provider Selection Logic

The router automatically selects providers based on query characteristics:

| Query Type | Primary Provider | Fallback |
|-----------|-----------------|----------|
| Quick facts/definitions | Brave | Tavily |
| Code/technical | Tavily | Exa |
| Academic/research papers | Gemini | Exa |
| Company/org info | Linkup | Exa |
| People/biography | Tavily | Exa |
| News/current events | Newsdata | Brave |
| Deep/comprehensive | Gemini | Exa |
| General purpose | Brave | Linkup |

## Usage Tracking

The server tracks API calls to help you manage free tier limits:

```bash
# Check usage stats
cat ~/.claude/web-search-mcp-usage.json
```

Output example:
```json
{
  "brave": 5,
  "tavily": 3,
  "exa": 1,
  "gemini": 2,
  "linkup": 0,
  "newsdata": 1,
  "apify": 0
}
```

## Troubleshooting

### Server won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that `.env` file exists and is readable
- Try running with explicit Python path: `python3 server.py`

### "Environment variable X is required"
- Add the missing API key to `.env`
- You only need to configure providers you plan to use

### Provider returns empty results
- Check that the API key is valid and has remaining quota
- Try escalating to a fallback provider
- Verify your search query is specific enough

### Gemini provider errors
- Ensure `GEMINI_PROJECT` is set to a valid GCP project ID
- Verify `GOOGLE_SERVICE_ACCOUNT_JSON` points to a valid service account file
- Check that the service account has Vertex AI permissions in GCP

### Rate limiting
- Respect provider rate limits (vary per provider)
- Free tier limits typically reset daily/monthly
- Monitor usage with the tracking JSON file

## Development

### Running Tests

See `TEST_PLAN.md` for comprehensive testing procedures.

### Adding a New Provider

1. Create a new file in `providers/` implementing the `BaseProvider` interface
2. Add provider import and initialization in `server.py`
3. Update routing logic in `orchestrator.py` if needed
4. Add usage tracking in `usage_tracker.py`
5. Update documentation

## License

This MCP is provided as-is for general-purpose web search integration.

## Support

For issues specific to individual providers, refer to their documentation:
- Brave: https://api.search.brave.com
- Tavily: https://docs.tavily.com
- Exa: https://docs.exa.ai
- Linkup: https://docs.linkup.so
- Newsdata: https://newsdata.io/docs
- Apify: https://apify.com/docs
- Google Gemini: https://cloud.google.com/vertex-ai/docs

---

**Note**: This is a generic, reusable implementation. All company-specific configurations, hardcoded paths, and credentials have been removed. Configure with your own API keys as needed.
