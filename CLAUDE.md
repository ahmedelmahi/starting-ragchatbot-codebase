# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

**Always use `uv` to manage all dependencies and run the server. Do not use `pip` directly.**

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Run a Python file
uv run python <file.py>

# Run the application (from project root)
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app runs at http://localhost:8000 with API docs at http://localhost:8000/docs.

## Environment Setup

Create `.env` in project root:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot for querying course materials.

### Request Flow

```
Frontend (script.js)
    → POST /api/query
    → app.py (FastAPI)
    → rag_system.py (orchestrator)
    → ai_generator.py (Claude API with tool calling)
    → search_tools.py (CourseSearchTool)
    → vector_store.py (ChromaDB queries)
```

### Key Components

**Backend (`backend/`):**
- `app.py` - FastAPI endpoints: `/api/query` (process questions), `/api/courses` (stats)
- `rag_system.py` - Main orchestrator connecting all components
- `ai_generator.py` - Claude API integration with tool execution loop
- `vector_store.py` - ChromaDB with two collections:
  - `course_catalog` - course metadata for fuzzy name resolution
  - `course_content` - chunked content for semantic search
- `search_tools.py` - Tool definitions for Claude's tool_use capability
- `document_processor.py` - Parses course files, chunks text with overlap
- `session_manager.py` - In-memory conversation history per session

**Frontend (`frontend/`):** Vanilla HTML/CSS/JS with markdown rendering (marked.js)

### Document Format

Course files in `docs/` follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[content...]

Lesson 1: [lesson title]
[content...]
```

### Configuration

Settings in `backend/config.py`:
- `CHUNK_SIZE`: 800 chars per chunk
- `CHUNK_OVERLAP`: 100 chars overlap
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
