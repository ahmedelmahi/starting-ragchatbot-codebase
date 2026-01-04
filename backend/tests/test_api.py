"""Tests for FastAPI endpoints"""
import pytest


class TestRootEndpoint:
    """Tests for the root endpoint"""

    def test_root_returns_ok(self, test_client):
        """Root endpoint returns status ok"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_with_new_session(self, test_client, mock_rag_system):
        """Query without session_id creates new session"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session(self, test_client, mock_rag_system):
        """Query with session_id uses existing session"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_returns_answer_and_sources(self, test_client, mock_rag_system):
        """Query returns proper answer and sources structure"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "MCP stands for Model Context Protocol."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Introduction to MCP, Lesson 1"
        assert data["sources"][0]["url"] == "https://example.com/lesson1"

    def test_query_calls_rag_system(self, test_client, mock_rag_system):
        """Query endpoint calls RAG system with correct parameters"""
        test_client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "my-session"}
        )
        mock_rag_system.query.assert_called_once_with("What is MCP?", "my-session")

    def test_query_missing_query_field(self, test_client):
        """Query without query field returns 422"""
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422

    def test_query_invalid_json(self, test_client):
        """Query with invalid JSON returns 422"""
        response = test_client.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_query_error_returns_500(self, test_client, mock_rag_system):
        """Query returns 500 when RAG system raises exception"""
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_stats(self, test_client):
        """Courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to MCP" in data["course_titles"]

    def test_courses_calls_analytics(self, test_client, mock_rag_system):
        """Courses endpoint calls get_course_analytics"""
        test_client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_error_returns_500(self, test_client, mock_rag_system):
        """Courses returns 500 when analytics fails"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Collection not found")
        response = test_client.get("/api/courses")
        assert response.status_code == 500
        assert "Collection not found" in response.json()["detail"]


class TestSessionClearEndpoint:
    """Tests for POST /api/session/clear endpoint"""

    def test_clear_session_success(self, test_client):
        """Clear session returns success response"""
        response = test_client.post(
            "/api/session/clear",
            json={"session_id": "session-to-clear"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Session cleared successfully"

    def test_clear_session_calls_manager(self, test_client, mock_rag_system):
        """Clear session calls session manager"""
        test_client.post(
            "/api/session/clear",
            json={"session_id": "session-to-clear"}
        )
        mock_rag_system.session_manager.clear_session.assert_called_once_with("session-to-clear")

    def test_clear_session_missing_id(self, test_client):
        """Clear session without session_id returns 422"""
        response = test_client.post("/api/session/clear", json={})
        assert response.status_code == 422

    def test_clear_session_error_returns_500(self, test_client, mock_rag_system):
        """Clear session returns 500 when manager fails"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session not found")
        response = test_client.post(
            "/api/session/clear",
            json={"session_id": "invalid-session"}
        )
        assert response.status_code == 500
        assert "Session not found" in response.json()["detail"]
