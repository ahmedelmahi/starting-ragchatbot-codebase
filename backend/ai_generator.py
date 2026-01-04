import anthropic
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ToolExecutionResult:
    """Result of executing tools in a single round"""

    updated_messages: List[Dict[str, Any]]
    error: bool = False
    error_message: str = ""


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **Content search tool (search_course_content)**: Use for questions about specific course content or detailed educational materials
- **Course outline tool (get_course_outline)**: Use for questions about course structure, syllabus, lesson lists, or what topics a course covers. Always include the course title, course link, and all lesson numbers with their titles in your response.
- **Up to 2 sequential tool calls allowed** - Use a second tool only when the first result requires additional lookup
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Maximum number of sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to MAX_TOOL_ROUNDS sequential tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages
        messages = [{"role": "user", "content": query}]

        # Prepare API call parameters
        api_params = {**self.base_params, "system": system_content}

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Tool execution loop
        for _ in range(self.MAX_TOOL_ROUNDS):
            api_params["messages"] = messages
            response = self.client.messages.create(**api_params)

            # No tool use - return response
            if response.stop_reason != "tool_use" or not tool_manager:
                return self._extract_text_response(response)

            # Execute tools
            result = self._execute_tool_round(response, messages, tool_manager)

            if result.error:
                return result.error_message

            messages = result.updated_messages

        # Max rounds reached - force text response
        api_params["messages"] = messages
        api_params["tool_choice"] = {"type": "none"}
        final_response = self.client.messages.create(**api_params)
        return self._extract_text_response(final_response)

    def _extract_text_response(self, response) -> str:
        """Extract text content from API response."""
        for content_block in response.content:
            if hasattr(content_block, "text") and content_block.text:
                return content_block.text
        return ""

    def _execute_tool_round(
        self, response, current_messages: List[Dict[str, Any]], tool_manager
    ) -> ToolExecutionResult:
        """
        Execute all tool calls in a single response round.

        Args:
            response: API response containing tool use requests
            current_messages: Current message list
            tool_manager: Manager to execute tools

        Returns:
            ToolExecutionResult with updated messages or error info
        """
        # Copy messages to avoid mutating original
        messages = current_messages.copy()

        # Add assistant's response (contains tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Tool execution failed
                    return ToolExecutionResult(
                        updated_messages=messages,
                        error=True,
                        error_message=f"Tool execution failed: {str(e)}",
                    )

        # Add tool results as user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return ToolExecutionResult(updated_messages=messages)
