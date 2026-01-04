"""
Integration tests for RAG system components

These tests use real components (not mocks) to identify integration issues:
1. VectorStore with actual ChromaDB
2. CourseSearchTool with real VectorStore
3. Full query flow testing
"""

import pytest
import tempfile
import shutil
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from models import Course, Lesson, CourseChunk


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with real ChromaDB"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def vector_store(self, temp_chroma_path):
        """Create a real VectorStore with temporary storage"""
        return VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

    @pytest.fixture
    def sample_course(self):
        """Create a sample course for testing"""
        return Course(
            title="Introduction to MCP",
            course_link="https://example.com/mcp-course",
            instructor="John Doe",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="What is MCP?",
                    lesson_link="https://example.com/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Installing MCP",
                    lesson_link="https://example.com/lesson2",
                ),
                Lesson(
                    lesson_number=3,
                    title="Advanced MCP",
                    lesson_link="https://example.com/lesson3",
                ),
            ],
        )

    @pytest.fixture
    def sample_chunks(self, sample_course):
        """Create sample course chunks for testing"""
        return [
            CourseChunk(
                content="MCP stands for Model Context Protocol. It allows AI models to interact with external tools and data sources.",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="To install MCP, you need to configure your settings.json file with the server definitions.",
                course_title=sample_course.title,
                lesson_number=2,
                chunk_index=1,
            ),
            CourseChunk(
                content="Advanced MCP usage includes creating custom tools, handling authentication, and error management.",
                course_title=sample_course.title,
                lesson_number=3,
                chunk_index=2,
            ),
        ]

    def test_add_and_search_course_content(
        self, vector_store, sample_course, sample_chunks
    ):
        """Test adding course content and searching it"""
        # Add course metadata and content
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search for content
        results = vector_store.search(query="What is MCP?")

        assert not results.is_empty()
        assert len(results.documents) > 0
        assert "MCP" in results.documents[0]

    def test_search_with_course_filter(
        self, vector_store, sample_course, sample_chunks
    ):
        """Test that course name filter works correctly"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with course filter
        results = vector_store.search(
            query="installation", course_name="MCP"  # Partial match should work
        )

        assert not results.is_empty()
        # All results should be from the filtered course
        for meta in results.metadata:
            assert meta["course_title"] == "Introduction to MCP"

    def test_search_with_lesson_filter(
        self, vector_store, sample_course, sample_chunks
    ):
        """Test that lesson number filter works correctly"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with lesson filter
        results = vector_store.search(query="MCP", lesson_number=1)

        # All results should be from lesson 1
        for meta in results.metadata:
            assert meta["lesson_number"] == 1

    def test_course_name_resolution(self, vector_store, sample_course, sample_chunks):
        """Test that fuzzy course name matching works"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Test various partial matches
        resolved = vector_store._resolve_course_name("MCP")
        assert resolved == "Introduction to MCP"

        resolved = vector_store._resolve_course_name("Introduction")
        assert resolved == "Introduction to MCP"

    def test_search_nonexistent_course_returns_error(
        self, vector_store, sample_course, sample_chunks
    ):
        """Test that searching for nonexistent course returns error"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        results = vector_store.search(
            query="test", course_name="Completely Nonexistent Course That Doesn't Match"
        )

        assert results.error is not None
        assert "No course found" in results.error

    def test_get_lesson_link(self, vector_store, sample_course):
        """Test getting lesson link from course metadata"""
        vector_store.add_course_metadata(sample_course)

        link = vector_store.get_lesson_link("Introduction to MCP", 1)
        assert link == "https://example.com/lesson1"

        link = vector_store.get_lesson_link("Introduction to MCP", 2)
        assert link == "https://example.com/lesson2"

    def test_get_lesson_link_nonexistent(self, vector_store, sample_course):
        """Test getting link for nonexistent lesson"""
        vector_store.add_course_metadata(sample_course)

        link = vector_store.get_lesson_link("Introduction to MCP", 999)
        assert link is None


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with real VectorStore"""

    @pytest.fixture
    def temp_chroma_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def vector_store(self, temp_chroma_path):
        return VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

    @pytest.fixture
    def populated_vector_store(self, vector_store):
        """Vector store with sample data"""
        course = Course(
            title="Python Basics",
            course_link="https://example.com/python",
            instructor="Jane Smith",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Variables",
                    lesson_link="https://example.com/python/1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Functions",
                    lesson_link="https://example.com/python/2",
                ),
            ],
        )
        chunks = [
            CourseChunk(
                content="Python variables store data. Use = for assignment.",
                course_title="Python Basics",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Functions are defined with def keyword. They organize code.",
                course_title="Python Basics",
                lesson_number=2,
                chunk_index=1,
            ),
        ]

        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)
        return vector_store

    def test_execute_returns_formatted_search_results(self, populated_vector_store):
        """Test that execute returns properly formatted results"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="What are Python variables?")

        assert "[Python Basics" in result
        assert "variables" in result.lower()

    def test_execute_populates_last_sources(self, populated_vector_store):
        """Test that sources are tracked after search"""
        tool = CourseSearchTool(populated_vector_store)
        tool.execute(query="What are functions?")

        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["text"] is not None

    def test_execute_with_invalid_course_returns_error(self, populated_vector_store):
        """Test error handling for invalid course filter"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute(query="test", course_name="Nonexistent Course XYZ")

        assert "No course found" in result


class TestToolManagerIntegration:
    """Integration tests for ToolManager with real tools"""

    @pytest.fixture
    def temp_chroma_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def tool_manager_with_data(self, temp_chroma_path):
        """Create ToolManager with populated VectorStore"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

        # Add sample data
        course = Course(
            title="Test Course",
            course_link="https://example.com/test",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Lesson 1",
                    lesson_link="https://example.com/test/1",
                ),
            ],
        )
        chunks = [
            CourseChunk(
                content="Test content about programming concepts.",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            ),
        ]
        vector_store.add_course_metadata(course)
        vector_store.add_course_content(chunks)

        # Create manager with tools
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(vector_store))
        manager.register_tool(CourseOutlineTool(vector_store))

        return manager

    def test_execute_search_tool_through_manager(self, tool_manager_with_data):
        """Test executing search tool through manager"""
        result = tool_manager_with_data.execute_tool(
            "search_course_content", query="programming"
        )

        assert "Test Course" in result

    def test_sources_available_after_search(self, tool_manager_with_data):
        """Test that sources can be retrieved after search"""
        tool_manager_with_data.execute_tool(
            "search_course_content", query="programming"
        )
        sources = tool_manager_with_data.get_last_sources()

        assert len(sources) > 0

    def test_reset_sources_works(self, tool_manager_with_data):
        """Test that sources can be reset"""
        tool_manager_with_data.execute_tool(
            "search_course_content", query="programming"
        )
        tool_manager_with_data.reset_sources()
        sources = tool_manager_with_data.get_last_sources()

        assert len(sources) == 0


class TestCourseOutlineToolIntegration:
    """Integration tests for CourseOutlineTool"""

    @pytest.fixture
    def temp_chroma_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def vector_store_with_course(self, temp_chroma_path):
        """Vector store with a course for outline testing"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

        course = Course(
            title="Advanced Python",
            course_link="https://example.com/adv-python",
            instructor="Dr. Expert",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Decorators",
                    lesson_link="https://example.com/adv/1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Generators",
                    lesson_link="https://example.com/adv/2",
                ),
                Lesson(
                    lesson_number=3,
                    title="Context Managers",
                    lesson_link="https://example.com/adv/3",
                ),
            ],
        )
        vector_store.add_course_metadata(course)
        return vector_store

    def test_get_course_outline_returns_full_structure(self, vector_store_with_course):
        """Test that outline includes all course info"""
        tool = CourseOutlineTool(vector_store_with_course)
        result = tool.execute(course_name="Advanced Python")

        assert "Advanced Python" in result
        assert "https://example.com/adv-python" in result
        assert "Lesson 1: Decorators" in result
        assert "Lesson 2: Generators" in result
        assert "Lesson 3: Context Managers" in result

    def test_get_course_outline_with_partial_name(self, vector_store_with_course):
        """Test that outline works with partial course name"""
        tool = CourseOutlineTool(vector_store_with_course)
        result = tool.execute(course_name="Advanced")

        assert "Advanced Python" in result

    def test_get_course_outline_nonexistent_course(self, vector_store_with_course):
        """Test error handling for nonexistent course"""
        tool = CourseOutlineTool(vector_store_with_course)
        result = tool.execute(course_name="Nonexistent Course")

        assert "No course found" in result


class TestSearchResultsEdgeCases:
    """Test edge cases in search results handling"""

    def test_search_results_from_chroma_with_empty_lists(self):
        """Test handling of empty ChromaDB results"""
        chroma_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        results = SearchResults.from_chroma(chroma_results)

        assert results.is_empty()
        assert len(results.documents) == 0

    def test_search_results_from_chroma_with_none_values(self):
        """Test handling when ChromaDB returns None-like values"""
        chroma_results = {"documents": None, "metadatas": None, "distances": None}

        # This should handle None gracefully
        try:
            results = SearchResults.from_chroma(chroma_results)
            assert results.is_empty()
        except TypeError:
            # If it raises TypeError, that's a bug to fix
            pytest.fail("SearchResults.from_chroma doesn't handle None values properly")

    def test_search_results_empty_constructor(self):
        """Test creating empty results with error message"""
        results = SearchResults.empty("Custom error message")

        assert results.is_empty()
        assert results.error == "Custom error message"
