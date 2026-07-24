# Setup Guide

Complete step-by-step instructions for configuring and deploying each system.

## Prerequisites

All systems require:
- Python 3.9+
- pip package manager
- git (for version control)
- Internet connection (for API access)

Optional but recommended:
- Docker/Docker Compose (for containerized deployment)
- PostgreSQL (for production databases)
- Qdrant Cloud account (for managed vector database)

## Quick Start

### 1. Clone and Initialize

```bash
cd /path/to/final
# Each system is ready to use immediately
```

### 2. Install Dependencies

For each system, install Python dependencies:

```bash
# Helpdesk Agent
cd helpdesk-agent
pip install -r requirements.txt
cd ..

# Web Search MCP
cd web-search-mcp
pip install -r requirements.txt
cd ..

# RAG System
cd rag-system
pip install -r requirements.txt
cd ..
```

## System-Specific Setup

### AI Helpdesk Agent

#### Prerequisites
- Atlassian Cloud instance (Jira Service Management)
- Google Cloud project with Vertex AI enabled
- OAuth application credentials
- (Optional) Google Sheets API for FAQ source

#### Configuration Steps

1. **Create Atlassian OAuth App**
   - Go to https://developer.atlassian.com/console
   - Create new OAuth 2.0 app
   - Configure scopes (see docs/atlassian-scopes.md)
   - Get Client ID and Client Secret

2. **Set Up Google Cloud**
   - Create GCP project
   - Enable Vertex AI API
   - Create service account
   - Download service account key as JSON

3. **Configure Environment**
   ```bash
   cd helpdesk-agent
   cp .env.example .env
   ```

4. **Update `.env` with Your Values**
   ```env
   JSM_CLOUD_URL=https://your-domain.atlassian.net
   ATLASSIAN_OAUTH_CLIENT_ID=your-client-id
   ATLASSIAN_OAUTH_CLIENT_SECRET=your-secret
   ATLASSIAN_API_TOKEN=your-api-token
   ATLASSIAN_EMAIL=your-email@your-company.com
   
   GEMINI_PROJECT=your-gcp-project
   PROJECT_KEYS=YOUR_PROJECT_KEY
   
   # Optional
   GOOGLE_SHEET_ID=your-google-sheet-id
   OPSGENIE_TOKEN=your-opsgenie-token
   ```

5. **Place Service Account Key**
   ```bash
   # Copy the downloaded JSON file
   cp /path/to/service_account.json ./service_account.json
   ```

6. **Test Configuration**
   ```bash
   python -c "from config import Config; print('✓ Config loaded')"
   ```

7. **Run Application**
   ```bash
   python main.py
   ```

#### Database Setup

**Local Development (SQLite):**
No additional setup needed. Uses `jsm_data.db` by default.

**Production (PostgreSQL):**
```bash
# Install PostgreSQL and create database
createdb helpdesk

# Update .env
DATABASE_URL=postgresql://user:password@localhost:5432/helpdesk

# Run migrations
python -m alembic upgrade head
```

#### Docker Deployment
```bash
# Build image
docker build -t helpdesk-agent .

# Run with env file
docker run --env-file .env helpdesk-agent
```

### Web Search MCP

#### Prerequisites
- Google Cloud project
- Google Search API enabled
- (Optional) Apify account

#### Configuration Steps

1. **Get Google Search API Key**
   - Go to https://console.cloud.google.com/
   - Create new project or use existing
   - Enable Custom Search API
   - Create API key in Credentials

2. **Create Custom Search Engine (Optional)**
   - Go to https://programmablesearchengine.google.com/
   - Create new search engine
   - Get search engine ID (CX parameter)

3. **Configure Environment**
   ```bash
   cd web-search-mcp
   cp .env.example .env
   ```

4. **Update `.env`**
   ```env
   GOOGLE_SEARCH_API_KEY=your-api-key
   GOOGLE_SEARCH_ENGINE_ID=your-cx-id
   # Optional
   APIFY_API_KEY=your-apify-key
   ```

5. **Test Configuration**
   ```bash
   python server.py
   # Should start MCP server on default port
   ```

#### Integration with Claude

The MCP server can be integrated with Claude via `.mcp.json`:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "GOOGLE_SEARCH_API_KEY": "your-api-key"
      }
    }
  }
}
```

### RAG System with Qdrant

#### Prerequisites
- Qdrant instance (local or cloud)
- Document sources (local files or URLs)
- Optional: LLM API keys (OpenAI, Anthropic, etc.)

#### Configuration Steps

1. **Set Up Qdrant**

   **Option A: Local (Development)**
   ```bash
   # Using Docker
   docker run -p 6333:6333 \
     -v qdrant_storage:/qdrant/storage \
     qdrant/qdrant:latest
   ```

   **Option B: Cloud (Production)**
   - Create account at https://qdrant.tech/
   - Create cluster
   - Get cluster URL and API key

2. **Configure Environment**
   ```bash
   cd rag-system
   cp .env.example .env
   ```

3. **Update `.env`**
   ```env
   # Local Qdrant
   QDRANT_URL=http://localhost:6333
   
   # Or cloud Qdrant
   # QDRANT_URL=https://your-cluster.qdrant.io
   # QDRANT_API_KEY=your-api-key
   
   # Embedding
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   EMBEDDING_PROVIDER=local
   
   # LLM (choose one)
   LLM_PROVIDER=local  # or: openai, anthropic, gemini
   # LLM_API_KEY=your-key
   # LLM_MODEL=gpt-4  # or your model
   
   # Data
   DATA_SOURCE_PATH=./data
   DATABASE_URL=sqlite:///./rag.db
   ```

4. **Prepare Documents**
   ```bash
   # Create data directory
   mkdir -p data
   
   # Add your documents (PDF, TXT, MD, etc.)
   cp /path/to/your/docs/* data/
   ```

5. **Index Documents**
   ```bash
   python scripts/ingest.py --source-path ./data
   ```

6. **Test RAG**
   ```bash
   python scripts/test_rag.py --query "your test query"
   ```

#### Production Deployment

**Using PostgreSQL:**
```env
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
```

**Using Cloud Qdrant:**
```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

**Using API-based Embeddings:**
```env
EMBEDDING_PROVIDER=openai  # or: anthropic, cohere
EMBEDDING_API_KEY=your-key
```

## Testing

Each system includes test suites:

```bash
# Helpdesk Agent
cd helpdesk-agent
pytest tests/

# Web Search MCP
cd web-search-mcp
pytest providers/

# RAG System
cd rag-system
pytest tests/
```

## Troubleshooting

### Common Issues

**Missing API Key**
- Ensure `.env` file exists and has all required keys
- Check `.env.example` for required variables
- Verify files are not committed to git

**Connection Refused**
- For local services, ensure Docker containers are running
- Check port numbers (Qdrant default: 6333)
- Verify firewall rules for remote services

**Authentication Failed**
- Check API keys haven't expired
- Verify credentials are correct in `.env`
- Check scopes/permissions for OAuth apps

**Import Errors**
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.9+): `python --version`
- Use virtual environment: `python -m venv venv && source venv/bin/activate`

## Monitoring

### Helpdesk Agent
- Check logs: `tail -f logs/helpdesk.log`
- Monitor Jira metrics in dashboards
- Check API usage in Atlassian Cloud

### Web Search MCP
- Monitor MCP server logs
- Track API calls to Google Search
- Monitor Apify quota usage

### RAG System
- Monitor Qdrant collection stats
- Check embedding quality metrics
- Monitor retrieval latency
- Track LLM API costs

## Security Checklist

- [ ] `.env` file created from `.env.example`
- [ ] No credentials committed to git
- [ ] `.gitignore` includes all secret files
- [ ] Service accounts have minimal required permissions
- [ ] API keys rotated regularly (quarterly recommended)
- [ ] Database passwords changed from defaults
- [ ] HTTPS enabled for all network traffic
- [ ] Firewall rules restrict API access to authorized IPs
- [ ] Monitoring and alerting configured
- [ ] Regular backups of Qdrant collections

## Next Steps

1. **Customize for Your Needs**
   - Update configuration files in each system
   - Modify prompts and response generation
   - Integrate with your data sources

2. **Deploy to Production**
   - Use Docker/Kubernetes for containerization
   - Set up CI/CD pipelines
   - Configure monitoring and alerting

3. **Integrate Systems**
   - Connect helpdesk agent to web search
   - Integrate RAG system for semantic search
   - Create unified dashboard

## Additional Resources

- `docs/ARCHITECTURE_OVERVIEW.md` - System architecture
- `helpdesk-agent/docs/` - Helpdesk-specific documentation
- `web-search-mcp/README.md` - Web search details
- `rag-system/README.md` - RAG system details
