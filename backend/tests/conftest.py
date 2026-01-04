"""Shared fixtures for RAG chatbot tests"""
import pytest
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API testing"""
    mock = MagicMock()

    # Mock session manager
    mock.session_manager.create_session.return_value = "test-session-123"
    mock.session_manager.clear_session.return_value = None

    # Mock query method
    mock.query.return_value = (
        "MCP stands for Model Context Protocol.",
        [{"text": "Introduction to MCP, Lesson 1", "url": "https://example.com/lesson1"}]
    )

    # Mock course analytics
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Introduction to MCP", "Advanced Claude", "AI Fundamentals"]
    }

    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create fresh app for testing
    app = FastAPI(title="Course Materials RAG System - Test")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models (inline to avoid import issues)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class SessionClearRequest(BaseModel):
        session_id: str

    class SessionClearResponse(BaseModel):
        success: bool
        message: str

    # Store mock in app state for access in endpoints
    app.state.rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            rag = app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            rag = app.state.rag_system
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear", response_model=SessionClearResponse)
    async def clear_session(request: SessionClearRequest):
        try:
            rag = app.state.rag_system
            rag.session_manager.clear_session(request.session_id)
            return SessionClearResponse(success=True, message="Session cleared successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"status": "ok", "message": "RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """Create TestClient for API testing"""
    from starlette.testclient import TestClient
    return TestClient(test_app)


@pytest.fixture
def mock_search_results_with_data():
    """SearchResults with actual course content"""
    return SearchResults(
        documents=[
            "MCP stands for Model Context Protocol. It enables AI models to access external tools.",
            "To install MCP servers, you need to configure them in your settings file."
        ],
        metadata=[
            {"course_title": "Introduction to MCP", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to MCP", "lesson_number": 2, "chunk_index": 1}
        ],
        distances=[0.2, 0.4]
    )


@pytest.fixture
def mock_search_results_empty():
    """Empty SearchResults"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def mock_search_results_with_error():
    """SearchResults with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'nonexistent course'"
    )


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing CourseSearchTool"""
    mock = MagicMock()
    mock.max_results = 5
    return mock


@pytest.fixture
def mock_anthropic_response_text():
    """Mock Anthropic response with just text (no tool use)"""
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "This is a direct response without tool use."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_anthropic_response_tool_use():
    """Mock Anthropic response requesting tool use"""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"

    # Text block
    mock_text = MagicMock()
    mock_text.type = "text"
    mock_text.text = ""

    # Tool use block
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.id = "tool_123"
    mock_tool.name = "search_course_content"
    mock_tool.input = {"query": "What is MCP?", "course_name": "Introduction to MCP"}

    mock_response.content = [mock_text, mock_tool]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Mock final Anthropic response after tool execution"""
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "MCP stands for Model Context Protocol. It enables AI models to access external tools and data sources."
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def sample_tool_definitions():
    """Sample tool definitions for testing"""
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]
