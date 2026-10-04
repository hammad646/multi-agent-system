# Multi-Agent Assistant

An enterprise-grade, local-first multi-agent assistant built with **LangGraph**, **FastMCP**, and **Google Gemini** (`gemini-2.5-flash`), integrating:
1. **RAG Agent**: PDF ingestion, dense vector search via Pinecone, local BM25 keyword search, and Reciprocal Rank Fusion (RRF) reranking with exact page citations.
2. **GitHub Agent**: FastMCP stdio server providing repository navigation and issue inspection.
3. **Google Calendar Agent**: Availability queries, conflict detection, duplicate prevention, and timezone-aware event scheduling.
4. **Gmail Agent & Background Scheduler**: MIME email drafting, recipient capping, and persistent background job execution with atomic claims.
5. **Supervisor Router & Multi-Step Workflows**: Intent routing, cross-domain workflow orchestration (`schedule_and_email`), and strict Human-in-the-Loop (HITL) approval gates.
6. **FastAPI & Minimal Web UI**: Single-file static web application with real-time approval cards, scheduled email management, and document ingestion.

---

## Architecture Diagram

```mermaid
flowchart TD
    User([User / Browser / CLI]) --> Router[Supervisor Router / Service Layer]
    
    subgraph "Core StateGraph"
        Router -->|Intent: rag| RAG[RAG Agent]
        Router -->|Intent: github| GH[GitHub Agent]
        Router -->|Intent: calendar| Cal[Calendar Agent]
        Router -->|Intent: email| Mail[Gmail Agent]
        Router -->|Intent: workflow| WF[Workflow Manager: schedule_and_email]
        Router -->|Intent: chitchat| Responder[Responder Node]
    end

    subgraph "Human-in-the-Loop & Persistence"
        Cal -.->|interrupt()| Gate[Approval Guard]
        Mail -.->|interrupt()| Gate
        WF -.->|interrupt()| Gate
        Gate --> Checkpointer[(SQLite / Postgres Saver)]
    end

    subgraph "MCP Tool Servers & External APIs"
        RAG --> Pinecone[(Pinecone Vector DB)]
        RAG --> SQLiteFTS[(SQLite BM25 Chunks)]
        GH --> MCP_GH[GitHub MCP Server]
        Cal --> MCP_Cal[Google Calendar API]
        Mail --> MCP_Gmail[Gmail API]
        Mail --> SchedulerDB[(Scheduled Emails DB)]
    end

    SchedulerWorker[Background Worker Process] -->|Atomic Claim & Send| SchedulerDB
    SchedulerWorker --> MCP_Gmail
```

---

## Directory Structure

```text
├── app/
│   ├── agents/               # ReAct Worker Agents
│   │   ├── calendar_agent.py # Google Calendar agent
│   │   ├── email_agent.py    # Gmail agent (draft & schedule)
│   │   ├── github_agent.py   # GitHub repository agent
│   │   └── rag_agent.py      # PDF RAG retrieval agent
│   ├── api/                  # FastAPI Web API
│   │   ├── auth.py           # X-API-Key verification dependency
│   │   ├── routes.py         # HTTP endpoints (/chat, /approve, /reject, etc.)
│   │   └── server.py         # FastAPI application factory & CORS setup
│   ├── db/                   # Database models & repositories (SQLAlchemy 2.0)
│   │   ├── base.py           # SQLite / Postgres engine & session factory
│   │   ├── models.py         # Tables: documents, chunks, scheduled_emails, workflows
│   │   └── repositories.py   # Data access repositories (All SQL lives here)
│   ├── mcp_servers/          # FastMCP Tool Servers (stdio communication)
│   │   ├── calendar_server.py# Google Calendar tools (list, check, create)
│   │   ├── common.py         # Structured error handlers & wrappers
│   │   ├── github_server.py  # GitHub tools (issues, readme, repos)
│   │   ├── gmail_server.py   # Gmail tools (draft, send, validate)
│   │   └── google_auth.py    # Desktop OAuth token credentials helper
│   ├── rag/                  # RAG Ingestion & Hybrid Retrieval
│   │   ├── ingest.py         # PDF parsing, chunking, Pinecone embeddings
│   │   ├── models.py         # Ingestion & retrieval Pydantic models
│   │   ├── rerank.py         # Reciprocal Rank Fusion (RRF) & reranking
│   │   └── retrieve.py       # Dense + BM25 hybrid search pipeline
│   ├── scheduler/            # Persistent Email Scheduler
│   │   ├── service.py        # Scheduling service interface
│   │   └── worker.py         # Independent worker with atomic polling & claim
│   ├── web/                  # Minimal Web UI
│   │   └── index.html        # Single-file HTML/CSS/JS frontend (no Node/npm)
│   ├── workflows/            # Multi-Step Workflows
│   │   ├── manager.py        # Workflow registry
│   │   └── schedule_and_email.py # Deterministic multi-approval workflow
│   ├── cli.py                # Interactive Terminal CLI
│   ├── config.py             # Pydantic Settings (.env configuration)
│   ├── errors.py             # Standardized error taxonomy
│   ├── graph.py              # LangGraph StateGraph builder
│   ├── guard.py              # Tool call approval guards via interrupt()
│   ├── logging.py            # Structured JSON logging (structlog)
│   ├── policy.py             # Human approval policy definitions
│   ├── responder.py          # Final user-facing response synthesis
│   ├── router.py             # Single-call LLM router with fallback heuristics
│   ├── service.py            # ChatService orchestrator (CLI & API entrypoint)
│   ├── state.py              # LangGraph AgentState definitions
│   └── telemetry.py          # LangSmith tracing integration helper
├── data/                     # Local SQLite databases & vector checkpoints
├── tests/                    # Comprehensive Pytest Suite (100% mocked offline)
├── demo.py                   # Automated end-to-end runnable demo script
├── requirements.txt          # Production & testing dependencies
├── SPEC.md                   # System Architecture Specification (v2)
├── AGENTS.md                 # Agent development guidelines & Hard Rules
└── README.md                 # Project documentation
```

---

## Prerequisites & Installation

- **Python**: Version **3.11** or **3.12** required.
- **Git** installed.
- No Node.js or npm required (Web UI runs as a single static file).

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone <repo-url>
cd finalproject

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (`.env`)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```ini
# LLM Configuration (Google Gemini Free Tier)
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
MODEL=gemini-2.5-flash

# Pinecone Vector DB (RAG)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=pdf-rag

# GitHub MCP Server
GITHUB_PAT=your_github_personal_access_token_here
GITHUB_ALLOW_WRITE=false

# Google OAuth (Calendar & Gmail)
GOOGLE_OAUTH_CREDENTIALS=./gcp-oauth.keys.json
GOOGLE_TOKEN_PATH=./token.json

# Timezone & System Settings
TIMEZONE=Asia/Karachi
DATABASE_URL=sqlite:///./data/app.db
CHECKPOINT_DB=./data/checkpoints.db
APPROVE_EVENT_CREATE=true
MAX_EMAIL_RECIPIENTS=10
MAX_LATE_MINUTES=60
LOG_EMAIL_BODIES=false

# Offline Mock Mode (Runs full flow without external API keys!)
MOCK_MODE=false

# API Security (Optional, protects endpoints with X-API-Key)
API_KEY=

# LangSmith Tracing (Optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=assistant
```

---

## Service Integrations Setup

### 1. Google Gemini API Setup
1. Visit [Google AI Studio](https://aistudio.google.com/).
2. Create an API key.
3. Set `GEMINI_API_KEY` and `GOOGLE_API_KEY` in `.env`.

### 2. Pinecone Vector DB Setup
1. Sign up at [Pinecone](https://www.pinecone.io/).
2. Create an index named `pdf-rag` with dimension `1024` and metric `cosine`.
3. Set `PINECONE_API_KEY` and `PINECONE_INDEX=pdf-rag` in `.env`.

### 3. GitHub Personal Access Token (PAT)
1. Go to GitHub Settings -> Developer Settings -> Personal access tokens (classic or fine-grained).
2. Create a token with `repo` or `public_repo` read scopes.
3. Set `GITHUB_PAT` in `.env`.

### 4. Google Cloud OAuth (Calendar & Gmail)
1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable **Google Calendar API** and **Gmail API**.
3. Configure the **OAuth Consent Screen**:
   - User Type: **External**
   - Scopes:
     - `https://www.googleapis.com/auth/calendar`
     - `https://www.googleapis.com/auth/calendar.events`
     - `https://www.googleapis.com/auth/gmail.modify`
     - `https://www.googleapis.com/auth/gmail.send`
     - `https://www.googleapis.com/auth/gmail.compose`
   - Add your Google account as a **Test User**.
4. Create Credentials -> **OAuth client ID** -> Application type: **Desktop app**.
5. Download JSON and save as `gcp-oauth.keys.json` in the project root.
6. Generate token file:
   ```bash
   python -m app.mcp_servers.google_auth
   ```
   Follow the browser login prompt to authorize access. `token.json` will be saved.

> [!WARNING]
> ### Before a Long-Running or Online Demo
> While the Google OAuth app is in **Testing** mode, Google automatically expires refresh tokens after **7 days** (`invalid_grant`).
> For a multi-week deployment or public demo:
> 1. Switch the Google OAuth consent screen from **Testing** to **In Production** (using a personal account).
> 2. Accept the one-time "Unverified App" warning during authentication.
> 3. Re-run `python -m app.mcp_servers.google_auth` to generate a persistent refresh token.
>
> *Note: This behavior was confirmed from Google Cloud OAuth documentation.*

---

## Running the Application

### Option A: Automated Demo
Run the complete end-to-end automated demo:
```bash
# Mock mode (no credentials required)
python demo.py

# Live mode (using real configured credentials)
python demo.py --live
```

### Option B: Interactive Terminal CLI
```bash
python -m app.cli
```
- Type natural language queries.
- When an approval is required (e.g. creating a calendar event or sending an email), the CLI pauses and prompts: `[A]pprove / [R]eject / [E]dit / [Q]uit`.
- Process can be interrupted (`Ctrl+C`), and restarted later; the pending approval will be resumed seamlessly via the SQLite checkpointer.

### Option C: FastAPI Web Server & Minimal UI
1. Start the server:
   ```bash
   uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
   ```
2. Open your browser at:
   ```
   http://localhost:8000
   ```
3. The Web UI provides:
   - Real-time chat with human approval cards for pending actions.
   - Distinct visual separation of **PROPOSED** vs **COMPLETED** actions.
   - **Approve**, **Reject**, and **Edit** buttons.
   - Document upload for RAG PDF ingestion.
   - Scheduled email queue management with **Cancel** capability.
   - API Key modal prompt when `API_KEY` is configured in `.env`.

### Option D: Background Scheduler Worker
Run the independent background email scheduler in a separate terminal:
```bash
python -m app.scheduler.worker
```
- Polls SQLite/Postgres for scheduled emails due for delivery.
- Uses atomic database claims (`UPDATE ... SET status='sending', attempts=attempts+1 WHERE status='pending' AND scheduled_at <= ...`) to guarantee zero duplicate sends across multiple workers.
- Recovers stuck sending jobs on worker startup.
- Enforces `MAX_LATE_MINUTES` catch-up window.

---

## Example Prompts per Agent

| Capability | Example Prompt | Description |
|---|---|---|
| **RAG** | `"What are the key findings on page 4 of the architectural report?"` | Retrieves dense & BM25 passages, reranks with RRF, cites `[filename, p.N]`. |
| **GitHub** | `"Show open issues in octocat/Hello-World"` | Queries GitHub MCP server, formats open issues. |
| **Calendar** | `"Am I free tomorrow at 3 PM?"` | Queries calendar for conflicts within timezone `TIMEZONE`. |
| **Gmail Draft** | `"Draft an email to sam@example.com about the project roadmap"` | Validates recipient, creates MIME draft in Gmail. Never sends. |
| **Scheduler** | `"Schedule an email to team@example.com tomorrow at 10 AM saying standup is cancelled"` | Persists scheduled email in SQLite queue for background delivery. |
| **Workflow** | `"Schedule a meeting with client@example.com for Sprint Planning tomorrow at 2pm and email him"` | Multi-step workflow with two distinct human approvals. |

---

## Human-in-the-Loop (HITL) Guardrails

Hard Rules enforced across the system:
1. **Never Send Without Approval**: Sending email, scheduling email, deleting email, updating events, deleting events, and GitHub write actions **always** require explicit human approval via LangGraph `interrupt()`.
2. **Drafting is Safe**: "Write email" or "draft email" creates a draft in Gmail. It never sends.
3. **Rejection Semantics**:
   - Rejecting calendar event creation terminates the workflow and creates nothing.
   - Rejecting email sending keeps the calendar event created and sends no email.
4. **Timezone Awareness**: All datetimes are timezone-aware ISO 8601 strings. Naive datetimes are strictly rejected.
5. **Untrusted Data**: Content retrieved from emails, GitHub issues, and PDFs is treated strictly as data, never as execution instructions.

---

## Testing

Run the full automated test suite (all external services mocked):
```bash
pytest -q
```

Coverage:
- `test_foundation.py`: Configuration, logging, models, repositories.
- `test_rag.py`: Ingestion, vector embedding, hybrid BM25 + dense RRF reranking, citation extraction.
- `test_github.py`: MCP tool definitions, read-only safety, error taxonomy.
- `test_calendar.py`: Availability, duplicate checks, naive datetime rejection, approval routing.
- `test_gmail.py`: MIME construction, recipient limits, draft creation.
- `test_scheduler.py`: Atomic claims, missed window recovery, crash handling.
- `test_supervisor.py`: Single LLM call router, context continuity, intent classification.
- `test_approvals.py`: Approval interruptions, rejection semantics, process restart persistence.
- `test_workflows.py`: Multi-step `schedule_and_email` workflow, idempotency, multi-interrupt replay.
- `test_api.py`: FastAPI endpoints, HTTP approval resumes, `X-API-Key` protection, document uploads.
- `test_hardening.py`: PostgreSQL DDL dialect compilation, LangSmith configuration, error formatting.

---

## Local-First to Online-Later (Roadmap)

Per SPEC.md Section 17, this project is designed local-first, deployment-ready for cloud:

| Component | Local Implementation (Current) | Online Production Transition |
|---|---|---|
| **Interface** | Terminal CLI + Local FastAPI Web UI | Hosted FastAPI behind Reverse Proxy (Caddy / Nginx) |
| **Database** | SQLite (`app.db`) | PostgreSQL (`DATABASE_URL=postgresql://...`) |
| **Checkpointer**| `SqliteSaver` (`checkpoints.db`) | `PostgresSaver` |
| **OAuth** | Desktop OAuth Flow (`token.json`) | Web OAuth 2.0 flow with encrypted DB token vault |
| **Worker** | Local subprocess (`python -m app.scheduler.worker`) | Systemd service / background container |
| **MCP Servers** | Local stdio child processes | Stdio or remote SSE HTTP transport |
| **Security** | `.env` file + optional `API_KEY` header | Vault / AWS Secrets Manager / Cloud KMS |

---

## Known Limits

- **Google OAuth in Testing Mode**: Requires re-authorization every 7 days unless switched to "In Production".
- **Scheduler Delivery**: Scheduled emails are dispatched by the worker process; if the worker is offline, jobs wait until restart and are processed if within `MAX_LATE_MINUTES`.
- **Hybrid Search**: Implemented via dense Pinecone vector embeddings + SQLite BM25 reciprocal rank fusion (RRF) rather than a single native sparse-dense cloud index.
