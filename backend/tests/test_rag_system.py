"""
Tests for RAG system content-query handling in rag_system.py

These tests evaluate:
1. Query flow through the RAG system
2. Tool integration with AIGenerator
3. Source retrieval and reset
4. Session management during queries
"""
import pytest
from unittest.mock import MagicMock, patch, Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    CHROMA_PATH = "./test_chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RESULTS = 5
    ANTHROPIC_API_KEY = "test_key"
    ANTHROPIC_MODEL = "claude-3-sonnet"
    MAX_HISTORY = 2


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_passes_tools_to_ai_generator(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that query() passes tool definitions to AIGenerator"""
        from rag_system import RAGSystem

        # Setup mocks
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "AI response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        # Create RAG system
        config = MockConfig()
        rag = RAGSystem(config)

        # Execute query
        rag.query("What is MCP?", session_id="test_session")

        # Verify tools were passed
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args.kwargs.get('tools') is not None
        assert call_args.kwargs.get('tool_manager') is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_retrieves_and_returns_sources(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that query() retrieves sources from tool_manager"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "AI response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_store_instance.search.return_value = MagicMock(
            documents=["content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_store_instance.get_lesson_link.return_value = "https://example.com"
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        # Manually set sources on the search tool to simulate what happens after tool execution
        rag.search_tool.last_sources = [{"text": "Test - Lesson 1", "url": "https://example.com"}]

        response, sources = rag.query("What is MCP?")

        assert sources is not None
        assert len(sources) == 1
        assert sources[0]["text"] == "Test - Lesson 1"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_resets_sources_after_retrieval(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that sources are reset after being retrieved"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "AI response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        # Set sources
        rag.search_tool.last_sources = [{"text": "Source 1", "url": None}]

        rag.query("First query")

        # Sources should be reset after query
        assert len(rag.search_tool.last_sources) == 0

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_updates_session_history(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that query() updates conversation history"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "The answer is 42"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        rag.query("What is the meaning of life?", session_id="session_123")

        # Verify add_exchange was called
        mock_session_instance.add_exchange.assert_called_once()
        call_args = mock_session_instance.add_exchange.call_args[0]
        assert "What is the meaning of life?" in call_args[1]
        assert "The answer is 42" in call_args[2]

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_includes_conversation_history_in_prompt(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that previous conversation is passed to AIGenerator"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = "User: Previous question\nAssistant: Previous answer"
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        rag.query("Follow up question", session_id="session_123")

        call_args = mock_ai_instance.generate_response.call_args
        history = call_args.kwargs.get('conversation_history')
        assert history is not None
        assert "Previous question" in history

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_works_without_session_id(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that query() works when no session_id is provided"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        response, sources = rag.query("Question without session")

        # Should not try to get history or add exchange
        mock_session_instance.get_conversation_history.assert_not_called()
        mock_session_instance.add_exchange.assert_not_called()

        assert response == "Response"


class TestRAGSystemToolIntegration:
    """Tests for tool registration and execution flow"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_search_tool_registered_in_tool_manager(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that CourseSearchTool is registered in ToolManager"""
        from rag_system import RAGSystem

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        config = MockConfig()
        rag = RAGSystem(config)

        # Verify tool is registered
        assert "search_course_content" in rag.tool_manager.tools

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_outline_tool_registered_in_tool_manager(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that CourseOutlineTool is registered in ToolManager"""
        from rag_system import RAGSystem

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        config = MockConfig()
        rag = RAGSystem(config)

        # Verify tool is registered
        assert "get_course_outline" in rag.tool_manager.tools

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_tool_definitions_passed_to_api(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that tool definitions include required schema"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        rag.query("Test query", session_id="test")

        call_args = mock_ai_instance.generate_response.call_args
        tools = call_args.kwargs.get('tools')

        # Should have both tools
        assert len(tools) == 2

        # Find search tool definition
        search_tool_def = next(t for t in tools if t['name'] == 'search_course_content')
        assert 'input_schema' in search_tool_def
        assert 'query' in search_tool_def['input_schema']['properties']


class TestRAGSystemPromptFormatting:
    """Tests for query prompt formatting"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_wrapped_in_proper_prompt(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that user query is wrapped in instruction prompt"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        rag.query("What is MCP?", session_id="test")

        call_args = mock_ai_instance.generate_response.call_args
        query_param = call_args.kwargs.get('query')

        # Query should include course materials context
        assert "course materials" in query_param
        assert "What is MCP?" in query_param


class TestRAGSystemSourceHandling:
    """Tests for source handling edge cases"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_empty_sources_handled_gracefully(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that empty sources list is returned when no search performed"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Direct answer without search"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        response, sources = rag.query("What is Python?", session_id="test")

        # Should return empty list, not None or error
        assert sources == []
        assert response == "Direct answer without search"

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_sources_include_url_when_available(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that sources include lesson URLs when available"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "Response"
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        # Simulate sources with URLs
        rag.search_tool.last_sources = [
            {"text": "Course A - Lesson 1", "url": "https://example.com/lesson1"},
            {"text": "Course A - Lesson 2", "url": None}  # Some lessons might not have URLs
        ]

        response, sources = rag.query("Test", session_id="test")

        assert len(sources) == 2
        assert sources[0]["url"] == "https://example.com/lesson1"
        assert sources[1]["url"] is None


class TestRAGSystemErrorHandling:
    """Tests for error handling in RAG system"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_handles_ai_generator_error(
        self, mock_session_mgr, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that query handles AI generator errors gracefully"""
        from rag_system import RAGSystem

        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.side_effect = Exception("API Error")
        mock_ai_gen.return_value = mock_ai_instance

        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        mock_session_instance = MagicMock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session_mgr.return_value = mock_session_instance

        config = MockConfig()
        rag = RAGSystem(config)

        # This should raise the exception (current behavior)
        # In a real system, you might want to catch and handle this
        with pytest.raises(Exception) as exc_info:
            rag.query("Test query", session_id="test")

        assert "API Error" in str(exc_info.value)
