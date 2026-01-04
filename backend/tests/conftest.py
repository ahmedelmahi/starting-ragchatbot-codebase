"""Shared fixtures for RAG chatbot tests"""
import pytest
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults


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
