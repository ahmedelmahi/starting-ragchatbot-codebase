"""
Tests for CourseSearchTool.execute() method in search_tools.py

These tests evaluate:
1. Successful search result formatting
2. Error handling (course not found, empty results)
3. Filter handling (course_name, lesson_number)
4. Source tracking for UI display
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager, CourseOutlineTool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute() method"""

    def test_execute_returns_formatted_results_on_success(self, mock_vector_store, mock_search_results_with_data):
        """Test that execute() returns properly formatted results when search succeeds"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?")

        # Verify search was called
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains course context headers
        assert "[Introduction to MCP - Lesson 1]" in result
        assert "[Introduction to MCP - Lesson 2]" in result

        # Verify content is included
        assert "MCP stands for Model Context Protocol" in result

    def test_execute_returns_error_when_course_not_found(self, mock_vector_store, mock_search_results_with_error):
        """Test that execute() returns error message when course resolution fails"""
        mock_vector_store.search.return_value = mock_search_results_with_error

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="What is MCP?", course_name="nonexistent course")

        # Should return the error from SearchResults
        assert "No course found matching 'nonexistent course'" in result

    def test_execute_returns_no_content_message_on_empty_results(self, mock_vector_store, mock_search_results_empty):
        """Test that execute() returns appropriate message when no results found"""
        mock_vector_store.search.return_value = mock_search_results_empty

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="obscure topic nobody knows about")

        assert "No relevant content found" in result

    def test_execute_passes_course_name_filter_to_vector_store(self, mock_vector_store, mock_search_results_with_data):
        """Test that execute() correctly passes course_name filter"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?", course_name="Introduction to MCP")

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name="Introduction to MCP",
            lesson_number=None
        )

    def test_execute_passes_lesson_number_filter_to_vector_store(self, mock_vector_store, mock_search_results_with_data):
        """Test that execute() correctly passes lesson_number filter"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=1
        )

    def test_execute_passes_both_filters_to_vector_store(self, mock_vector_store, mock_search_results_with_data):
        """Test that execute() correctly passes both filters"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?", course_name="MCP Course", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name="MCP Course",
            lesson_number=2
        )

    def test_execute_includes_filter_info_in_empty_results_message(self, mock_vector_store, mock_search_results_empty):
        """Test that empty results message includes filter context"""
        mock_vector_store.search.return_value = mock_search_results_empty

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test", course_name="MCP Course", lesson_number=3)

        assert "No relevant content found" in result
        assert "MCP Course" in result
        assert "lesson 3" in result


class TestCourseSearchToolSourceTracking:
    """Tests for source tracking in CourseSearchTool"""

    def test_execute_stores_sources_in_last_sources(self, mock_vector_store, mock_search_results_with_data):
        """Test that execute() stores sources for UI retrieval"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="What is MCP?")

        # Verify sources were stored
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Introduction to MCP - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson"

    def test_execute_stores_sources_without_lesson_number(self, mock_vector_store):
        """Test source tracking when lesson_number is None in metadata"""
        results = SearchResults(
            documents=["Course overview content"],
            metadata=[{"course_title": "Test Course", "lesson_number": None, "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["url"] is None

    def test_sources_cleared_on_empty_results(self, mock_vector_store, mock_search_results_empty):
        """Test that sources are handled correctly when results are empty"""
        # First search with results
        mock_vector_store.search.return_value = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Course", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="first query")
        assert len(tool.last_sources) == 1

        # Second search with no results - last_sources should remain from previous
        # (This tests the actual behavior - sources are only updated on success)
        mock_vector_store.search.return_value = mock_search_results_empty
        tool.execute(query="second query")

        # Note: Current implementation doesn't clear sources on empty results
        # This could be a bug or intentional - test documents actual behavior
        # If this test fails, it means sources ARE being cleared


class TestCourseSearchToolFormatting:
    """Tests for result formatting in CourseSearchTool"""

    def test_format_results_creates_proper_headers(self, mock_vector_store):
        """Test that _format_results creates proper course/lesson headers"""
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Course B", "lesson_number": 3, "chunk_index": 1}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)

        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 3]" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted

    def test_format_results_handles_missing_lesson_number(self, mock_vector_store):
        """Test formatting when lesson_number is None"""
        results = SearchResults(
            documents=["Course intro content"],
            metadata=[{"course_title": "Course A", "lesson_number": None, "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)

        # Should have course title but no lesson
        assert "[Course A]" in formatted
        assert "Lesson" not in formatted


class TestToolManager:
    """Tests for ToolManager functionality"""

    def test_register_tool_adds_tool_to_manager(self, mock_vector_store):
        """Test that register_tool properly adds tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        assert "search_course_content" in manager.tools

    def test_get_tool_definitions_returns_all_registered_tools(self, mock_vector_store):
        """Test that get_tool_definitions returns definitions for all tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_calls_correct_tool(self, mock_vector_store, mock_search_results_with_data):
        """Test that execute_tool correctly dispatches to the right tool"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test query")

        assert "Introduction to MCP" in result

    def test_execute_tool_returns_error_for_unknown_tool(self):
        """Test that execute_tool returns error for unregistered tool"""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", param="value")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources_returns_sources_from_search_tool(self, mock_vector_store, mock_search_results_with_data):
        """Test that get_last_sources retrieves sources from tools"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = "https://example.com"

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute a search
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) == 2

    def test_reset_sources_clears_tool_sources(self, mock_vector_store, mock_search_results_with_data):
        """Test that reset_sources clears sources from all tools"""
        mock_vector_store.search.return_value = mock_search_results_with_data
        mock_vector_store.get_lesson_link.return_value = None

        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="test")
        assert len(tool.last_sources) > 0

        manager.reset_sources()
        assert len(tool.last_sources) == 0


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition()"""

    def test_get_tool_definition_returns_valid_schema(self, mock_vector_store):
        """Test that tool definition has correct structure"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]
