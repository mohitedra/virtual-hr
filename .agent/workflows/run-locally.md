---
description: How to run the Virtual HR system locally
---

# Running Virtual HR Locally

This guide covers all the steps needed to run the Virtual HR multi-agent system on your local machine.

## Prerequisites

1. **Python 3.10+** installed
2. **Docker Desktop** installed and running (for Milvus vector database)
3. **API Keys**:
   - OpenAI API key
   - Anthropic API key (optional, for Claude responses)
4. **Google Cloud Service Account**:
   - A Google Cloud project with Sheets API and Drive API enabled
   - A service account with a downloaded JSON credentials file

---

## Step 1: Navigate to the Project Directory

```bash
cd /Users/mohitruwatia/Virtual\ HR/virtual-hr
```

---

## Step 2: Set Up Python Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` and set the following required values:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | Your Anthropic/Claude API key |
| `LEAVE_TRACKER_SHEET_ID` | Google Sheet ID for leave tracking |
| `FEEDBACK_TRACKER_SHEET_ID` | Google Sheet ID for feedback tracking |

**Finding Google Sheet IDs:**
From the Sheet URL `https://docs.google.com/spreadsheets/d/SHEET_ID/edit`, the `SHEET_ID` is the long string between `/d/` and `/edit`.

---

## Step 5: Set Up Google Sheets Credentials

1. Create a Google Cloud Project at https://console.cloud.google.com
2. Enable **Google Sheets API** and **Google Drive API**
3. Create a **Service Account** under APIs & Services → Credentials
4. Download the JSON key file and save it as `credentials.json` in the project root
5. **Share your Google Sheets** with the service account email (ends with `@*.iam.gserviceaccount.com`)

---

## Step 6: Start Milvus Vector Database (Docker)

The RAG agent requires Milvus for document embeddings. Start it with Docker Compose:

// turbo
```bash
docker compose up -d
```

Wait for all containers to be healthy (this may take 1-2 minutes on first run):

// turbo
```bash
docker compose ps
```

You should see three containers running:
- `milvus-etcd`
- `milvus-minio`
- `milvus-standalone`

---

## Step 7: Run the Application

// turbo
```bash
python main.py
```

The API server will start at `http://localhost:8000`

---

## Step 8: Verify the System

Open your browser or use curl to check the health endpoint:

// turbo
```bash
curl http://localhost:8000/health
```

Or visit the interactive API docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Using the API

### Send a Chat Message

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the leave policy for marriage?",
    "employee_id": "EMP001",
    "employee_name": "John Doe"
  }'
```

### Example Queries

| Query Type | Example |
|------------|---------|
| **Policy Questions** | "What is the remote work policy?" |
| **Leave Application** | "I want to apply for 2 days annual leave starting 2026-01-15" |
| **Leave Balance** | "Check my leave balance. My employee ID is EMP001" |
| **Feedback** | "I want to give feedback: The new office setup is great!" |

---

## Stopping the System

1. Stop the API server with `Ctrl+C`
2. Stop Milvus containers:

```bash
docker compose down
```

To also remove stored data:

```bash
docker compose down -v
rm -rf volumes/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Milvus connection error | Ensure Docker is running and containers are up |
| Google Sheets permission denied | Share the sheets with your service account email |
| OpenAI rate limit | Wait a few seconds and retry |

---

## Project Structure

```
virtual-hr/
├── main.py                 # FastAPI application entry point
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Milvus database setup
├── credentials.json        # Google Sheets service account key
├── .env                    # Environment variables
├── agents/
│   ├── orchestrator/       # Request routing agent
│   ├── rag_agent/          # Policy Q&A (with data/ folder for PDFs)
│   ├── leave_agent/        # Leave management
│   └── feedback_agent/     # Feedback collection
└── utils/                  # Utility functions
```
