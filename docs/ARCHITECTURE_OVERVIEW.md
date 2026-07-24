# Architecture Overview

This repository contains three interconnected AI/ML systems designed to work together or independently. All systems are completely generic and can be deployed to any organization.

## System Components

### 1. AI Helpdesk Agent (`helpdesk-agent/`)

A production-grade intelligent helpdesk assistant built on Atlassian Jira Service Management (JSM). Provides automated ticket analysis, response drafting, and FAQ integration.

**Key Features:**
- Automated ticket routing and prioritization
- AI-powered response generation using Gemini
- FAQ integration from Confluence knowledge base
- Google Sheets FAQ source management
- Service desk metrics and analytics
- Multi-project support with centralized management
- OpsGenie incident alerting integration

**Technology Stack:**
- Python 3.11+
- Atlassian Cloud APIs (JSM, Confluence, Jira)
- Google Gemini 2.5 Flash (via Vertex AI)
- SQLite/PostgreSQL for persistence
- FastAPI for service APIs

**Configuration:**
- Copy `.env.example` to `.env`
- Replace all `<YOUR_...>` placeholders with your actual values
- Requires: Atlassian Cloud instance, GCP project with Vertex AI enabled, OAuth credentials

### 2. Web Search MCP (`web-search-mcp/`)

A Model Context Protocol (MCP) server providing web search capabilities for Claude and other AI systems. Enables agents to perform real-time internet searches and page scraping.

**Key Features:**
- Real-time web search via Google Search API
- Web page content extraction (Apify integration)
- Multi-provider search support
- Integration with Rovo search framework
- Custom search engine support

**Technology Stack:**
- Python 3.9+
- Google Search API
- Apify web scraping
- FastMCP (Model Context Protocol)

**Configuration:**
- Copy `.env.example` to `.env`
- Requires: Google Search API key, Custom Search Engine ID (optional)
- Optional: Apify account for enhanced web scraping

### 3. RAG System with Qdrant (`rag-system/`)

A retrieval-augmented generation (RAG) system using Qdrant vector database. Enables semantic search and context-aware AI responses over document collections.

**Key Features:**
- Semantic document retrieval with Qdrant
- Pluggable embedding models (local or API-based)
- Support for multiple LLM providers
- Document chunking and indexing strategies
- Metadata filtering and hybrid search
- Production-ready scaling patterns

**Technology Stack:**
- Python 3.9+
- Qdrant vector database (local or cloud)
- Sentence Transformers for embeddings
- Support for OpenAI, Anthropic, Ollama, Gemini LLMs
- PostgreSQL/SQLite for metadata

**Configuration:**
- Copy `.env.example` to `.env`
- Set up Qdrant (local or cloud instance)
- Configure embedding and LLM providers
- Prepare document source (local files or remote URLs)

## Integration Patterns

### Pattern 1: Helpdesk + Web Search
Use the Web Search MCP within the helpdesk agent to search for resolution steps or documentation before generating responses.

### Pattern 2: Helpdesk + RAG
Integrate the RAG system to provide semantic search over company documentation, knowledge bases, or FAQ collections.

### Pattern 3: Search + RAG
Combine web search results with internal document RAG to provide comprehensive answers.

### Pattern 4: Standalone Systems
Each system is designed to work independently. Deploy only what you need.

## Deployment Architectures

### Local Development
All systems can run locally with minimal dependencies:
- Use SQLite for databases
- Run Qdrant locally via Docker
- Use local embedding models (Sentence Transformers)

### Cloud Deployment
Production deployments typically use:
- Cloud Qdrant for vector storage
- Managed PostgreSQL/Cloud SQL for metadata
- Cloud LLM APIs (Gemini, OpenAI, Anthropic)
- Containerized services (Docker/Kubernetes)

### Hybrid Configuration
Mix local and cloud components based on your requirements.

## Security & Secrets Management

All systems follow these practices:

1. **No Hardcoded Credentials**
   - All sensitive values are environment variables
   - `.env.example` provided as template
   - `.gitignore` prevents `.env` from being committed

2. **Service Accounts**
   - Use service accounts instead of personal credentials
   - Rotate credentials regularly
   - Use OAuth 2.0 where available

3. **API Key Management**
   - Store in secure vaults (Bitwarden, AWS Secrets, etc.)
   - Rotate periodically
   - Use minimal required scopes/permissions

## Configuration Hierarchy

Each system supports configuration via:
1. `.env` files (highest priority)
2. Environment variables
3. Config files (YAML/JSON)
4. Defaults in code (lowest priority)

## Monitoring & Logging

All systems implement:
- Structured logging (JSON format recommended)
- Request/response tracing
- Error tracking and alerts
- Performance metrics
- Audit logs for compliance

## Development Workflow

1. Clone the repository
2. Copy `.env.example` to `.env` for each system you're using
3. Install dependencies: `pip install -r requirements.txt`
4. Configure services in `.env`
5. Run tests: `pytest`
6. Start services locally
7. Deploy to your target environment

## License & Attribution

These systems were originally developed for internal Red Hat tooling but have been fully genericized for public use. All system-specific references have been replaced with placeholders.

## Support & Contributing

For issues, questions, or contributions:
1. Check existing documentation in each system's directory
2. Review the SETUP_GUIDE.md for configuration issues
3. Consult each system's README.md for detailed information
