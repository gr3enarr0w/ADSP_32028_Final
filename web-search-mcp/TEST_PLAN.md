# Web Search MCP — Test Plan

Run these tests to verify the MCP server connects and all tools work correctly.

## 1. Setup Verification

Ensure all dependencies are installed and `.env` file is configured:
```bash
pip install -r requirements.txt
python server.py
```

**Expected:** Server starts without errors and listens for MCP requests.

## 2. Routing Tests

### 2a. Quick fact → Brave
```
research "What is the capital of France?"
```
**Verify:** `provider_used: brave`, `mode: quick`

### 2b. Code → Tavily
```
research "How to implement async generators in Python"
```
**Verify:** `provider_used: tavily`, `mode: code`

### 2c. News → Newsdata
```
research "latest news about AI regulations"
```
**Verify:** `provider_used: newsdata`, `mode: news`, `sources` list has >0 results

### 2d. People → Tavily
```
research "who is [notable person]"
```
**Verify:** `provider_used: tavily`, `mode: people`

### 2e. Company → Linkup
```
research "[Company name] company valuation funding"
```
**Verify:** `provider_used: linkup`, `mode: company`

### 2f. Academic → Gemini (requires GCP setup)
```
research "peer-reviewed research papers on [topic]"
```
**Verify:** `provider_used: gemini`, `mode: academic`, `sources` list has >0 results

### 2g. Deep → Gemini (requires GCP setup)
```
research "Research [topic] thoroughly"
```
**Verify:** `provider_used: gemini`, `mode: deep`

### 2h. General → Brave
```
research "best practices for [general topic]"
```
**Verify:** `provider_used: brave`, `mode: general`

## 3. Usage Tracking

After running tests, check the usage file:
```bash
cat ~/.claude/web-search-mcp-usage.json
```

**Expected:** Usage counts for each provider called.

## 4. Escalation Flow

Ask a narrow query that might return thin results, then escalate:
```
research "very specific niche query"
```

If results are thin (<3), the response should include `escalation_hint`. Then:
```
research_escalate with the same query, excluding the provider that was already tried.
```

**Verify:** Different provider is used, results are returned.

## 5. Extract Tool

Test content extraction:
```
extract ["https://example.com"]
```

**Verify:** Returns extracted page content from Apify.

## 6. Scrape Tool

Test custom Apify actor:
```
scrape "apify/web-scraper" {"startUrls": [{"url": "https://example.com"}]}
```

**Verify:** Returns scraped data from custom actor.

## 7. Error Handling

Test missing API keys:
```
# Run with incomplete .env
python server.py
```

**Expected:** Graceful error messages for missing providers.
