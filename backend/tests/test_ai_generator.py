"""
Tests for AIGenerator in ai_generator.py

These tests evaluate:
1. Direct response generation (no tool use)
2. Tool use detection and execution
3. Tool result handling and follow-up response
4. Correct API parameter passing
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorDirectResponse:
    """Tests for direct response generation (no tool use)"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_returns_text_for_simple_query(
        self, mock_anthropic_class
    ):
        """Test that generate_response returns text when no tool use is needed"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Python is a programming language."
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        # Test
        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        result = generator.generate_response(query="What is Python?")

        assert result == "Python is a programming language."

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_includes_conversation_history(
        self, mock_anthropic_class
    ):
        """Test that conversation history is included in system prompt"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        generator.generate_response(
            query="Follow up question",
            conversation_history="User: What is MCP?\nAssistant: MCP is Model Context Protocol.",
        )

        # Verify history was included
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs.get("system", "")
        assert "Previous conversation" in system_content
        assert "What is MCP?" in system_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_passes_tools_when_provided(self, mock_anthropic_class):
        """Test that tools are passed to API when provided"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        tools = [{"name": "test_tool", "description": "A test tool"}]

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        generator.generate_response(query="Query", tools=tools)

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs.get("tools") == tools
        assert call_args.kwargs.get("tool_choice") == {"type": "auto"}


class TestAIGeneratorToolExecution:
    """Tests for tool use detection and execution"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_detects_tool_use_request(self, mock_anthropic_class):
        """Test that tool use is detected when stop_reason is 'tool_use'"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First response requests tool use
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = ""
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_abc123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "What is MCP?"}
        mock_tool_response.content = [mock_text, mock_tool_block]

        # Final response after tool execution
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = MagicMock()
        mock_final_content.text = "MCP is Model Context Protocol."
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Setup tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result: MCP information"

        tools = [{"name": "search_course_content"}]

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        result = generator.generate_response(
            query="What is MCP?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="What is MCP?"
        )

        # Verify final response is returned
        assert result == "MCP is Model Context Protocol."

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_passes_correct_parameters(self, mock_anthropic_class):
        """Test that tool parameters are passed correctly to tool_manager"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Tool use response with multiple parameters
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {
            "query": "installation steps",
            "course_name": "MCP Introduction",
            "lesson_number": 2,
        }
        mock_tool_response.content = [mock_tool_block]

        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = MagicMock()
        mock_final_content.text = "Final answer"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool results"

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify all parameters were passed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="installation steps",
            course_name="MCP Introduction",
            lesson_number=2,
        )

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_result_sent_back_to_api(self, mock_anthropic_class):
        """Test that tool results are sent back to API correctly"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_xyz"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_block]

        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = MagicMock()
        mock_final_content.text = "Final"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search found: MCP content"

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Check second API call includes tool result
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call.kwargs.get("messages", [])

        # Find tool_result message
        tool_result_message = None
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if content.get("type") == "tool_result":
                        tool_result_message = content
                        break

        assert tool_result_message is not None
        assert tool_result_message["tool_use_id"] == "tool_xyz"
        assert tool_result_message["content"] == "Search found: MCP content"


class TestAIGeneratorMultipleToolCalls:
    """Tests for handling multiple tool calls in one response"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_handles_multiple_tool_calls(self, mock_anthropic_class):
        """Test that multiple tool calls in one response are all executed"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Response with two tool calls
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"

        mock_tool_1 = MagicMock()
        mock_tool_1.type = "tool_use"
        mock_tool_1.id = "tool_1"
        mock_tool_1.name = "search_course_content"
        mock_tool_1.input = {"query": "query 1"}

        mock_tool_2 = MagicMock()
        mock_tool_2.type = "tool_use"
        mock_tool_2.id = "tool_2"
        mock_tool_2.name = "get_course_outline"
        mock_tool_2.input = {"course_name": "MCP"}

        mock_tool_response.content = [mock_tool_1, mock_tool_2]

        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_content = MagicMock()
        mock_final_content.text = "Combined answer"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")
        result = generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager,
        )

        # Both tools should be called
        assert mock_tool_manager.execute_tool.call_count == 2
        assert result == "Combined answer"


class TestAIGeneratorNoToolManager:
    """Tests for behavior when no tool_manager is provided"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_returns_empty_response_when_tool_use_without_manager(
        self, mock_anthropic_class
    ):
        """Test behavior when tool_use happens but no tool_manager provided"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Response requests tool use
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "I need to search for this."
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_response.content = [mock_text, mock_tool_block]

        mock_client.messages.create.return_value = mock_tool_response

        generator = AIGenerator(api_key="test_key", model="claude-3-sonnet")

        # Without tool_manager, should return the text content
        result = generator.generate_response(
            query="Query", tools=[{"name": "search_course_content"}], tool_manager=None
        )

        # Current implementation returns first content's text
        # This tests actual behavior
        assert result == "I need to search for this."


class TestAIGeneratorAPIParameters:
    """Tests for correct API parameter configuration"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_uses_correct_model(self, mock_anthropic_class):
        """Test that the configured model is used"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        generator.generate_response(query="Test")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs.get("model") == "claude-sonnet-4-20250514"

    @patch("ai_generator.anthropic.Anthropic")
    def test_uses_correct_max_tokens(self, mock_anthropic_class):
        """Test that max_tokens is set correctly"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="test-model")
        generator.generate_response(query="Test")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs.get("max_tokens") == 800

    @patch("ai_generator.anthropic.Anthropic")
    def test_uses_zero_temperature(self, mock_anthropic_class):
        """Test that temperature is set to 0 for deterministic responses"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="test-model")
        generator.generate_response(query="Test")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs.get("temperature") == 0

    @patch("ai_generator.anthropic.Anthropic")
    def test_system_prompt_includes_key_instructions(self, mock_anthropic_class):
        """Test that system prompt contains necessary instructions"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test_key", model="test-model")
        generator.generate_response(query="Test")

        call_args = mock_client.messages.create.call_args
        system = call_args.kwargs.get("system", "")

        # Check for key instructions
        assert "search_course_content" in system or "course" in system.lower()
        assert "tool" in system.lower()


class TestAIGeneratorSequentialToolCalls:
    """Tests for sequential tool calling (up to 2 rounds)"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_two_sequential_tool_calls_succeed(self, mock_anthropic_class):
        """Test that two sequential tool calls work correctly"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: First tool use
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_1 = MagicMock()
        mock_tool_1.type = "tool_use"
        mock_tool_1.id = "tool_round1"
        mock_tool_1.name = "get_course_outline"
        mock_tool_1.input = {"course_name": "MCP"}
        mock_response_1.content = [mock_tool_1]

        # Round 2: Second tool use
        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "tool_use"
        mock_tool_2 = MagicMock()
        mock_tool_2.type = "tool_use"
        mock_tool_2.id = "tool_round2"
        mock_tool_2.name = "search_course_content"
        mock_tool_2.input = {"query": "lesson 2 details", "course_name": "MCP"}
        mock_response_2.content = [mock_tool_2]

        # Final response after max rounds
        mock_final = MagicMock()
        mock_final.stop_reason = "end_turn"
        mock_final_text = MagicMock()
        mock_final_text.text = "Here is the comprehensive answer."
        mock_final.content = [mock_final_text]

        mock_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1, Lesson 2",
            "Lesson 2 content details",
        ]

        generator = AIGenerator(api_key="test", model="test-model")
        result = generator.generate_response(
            query="Tell me about MCP lesson 2",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify 2 tool executions
        assert mock_tool_manager.execute_tool.call_count == 2
        # Verify 3 API calls (2 tool rounds + 1 final)
        assert mock_client.messages.create.call_count == 3
        assert result == "Here is the comprehensive answer."

    @patch("ai_generator.anthropic.Anthropic")
    def test_stops_after_first_round_if_no_tool_use(self, mock_anthropic_class):
        """Test that loop exits early if Claude doesn't request tool use"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # First call uses tool
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tool_1"
        mock_tool.name = "search_course_content"
        mock_tool.input = {"query": "test"}
        mock_response_1.content = [mock_tool]

        # Second call returns text (no more tools needed)
        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.text = "Final answer after one tool."
        mock_response_2.content = [mock_text]

        mock_client.messages.create.side_effect = [mock_response_1, mock_response_2]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator(api_key="test", model="test-model")
        result = generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Only 1 tool execution, 2 API calls
        assert mock_tool_manager.execute_tool.call_count == 1
        assert mock_client.messages.create.call_count == 2
        assert result == "Final answer after one tool."

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_failure_stops_loop(self, mock_anthropic_class):
        """Test that tool failure terminates the loop"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tool_fail"
        mock_tool.name = "search_course_content"
        mock_tool.input = {"query": "test"}
        mock_response.content = [mock_tool]

        mock_client.messages.create.return_value = mock_response

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = Exception("Connection error")

        generator = AIGenerator(api_key="test", model="test-model")
        result = generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        assert "Tool execution failed" in result
        # Only 1 API call before failure
        assert mock_client.messages.create.call_count == 1

    @patch("ai_generator.anthropic.Anthropic")
    def test_max_rounds_forces_final_response(self, mock_anthropic_class):
        """Test that reaching max rounds forces a text response"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Both rounds request tools
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tool_x"
        mock_tool.name = "search_course_content"
        mock_tool.input = {"query": "test"}
        mock_tool_response.content = [mock_tool]

        mock_final = MagicMock()
        mock_final.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.text = "Forced final response"
        mock_final.content = [mock_text]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_tool_response,
            mock_final,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator = AIGenerator(api_key="test", model="test-model")
        result = generator.generate_response(
            query="Query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
        )

        # Verify tool_choice was set to "none" on final call
        final_call = mock_client.messages.create.call_args_list[-1]
        assert final_call.kwargs.get("tool_choice") == {"type": "none"}
        assert result == "Forced final response"

    @patch("ai_generator.anthropic.Anthropic")
    def test_tools_remain_available_in_second_round(self, mock_anthropic_class):
        """Test that tools are included in the second API call"""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "tool_1"
        mock_tool.name = "search_course_content"
        mock_tool.input = {"query": "test"}
        mock_tool_response.content = [mock_tool]

        mock_final = MagicMock()
        mock_final.stop_reason = "end_turn"
        mock_text = MagicMock()
        mock_text.text = "Done"
        mock_final.content = [mock_text]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final]

        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Result"

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]

        generator = AIGenerator(api_key="test", model="test-model")
        generator.generate_response(
            query="Query", tools=tools, tool_manager=mock_tool_manager
        )

        # Check second call still has tools
        second_call = mock_client.messages.create.call_args_list[1]
        assert second_call.kwargs.get("tools") == tools
