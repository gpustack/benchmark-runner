"""
Custom response handler that fixes TTFT and ITL calculation for models with reasoning tokens.

This handler extends guidellm's ChatCompletionsResponseHandler to properly handle
both regular content tokens and reasoning_content tokens, ensuring accurate timing metrics.

Usage:
    To use this handler, pass it via backend_kwargs when running benchmarks:

    benchmark-runner benchmark run \\
        --target http://localhost:8000/v1 \\
        --backend openai_http \\
        --backend-kwargs '{"response_handlers": {"chat_completions": "chat_completions_with_reasoning"}}' \\
        --model your-model-name \\
        --data your-dataset

    Or in a scenario config file:
    {
        "backend_kwargs": {
            "response_handlers": {
                "chat_completions": "chat_completions_with_reasoning"
            }
        }
    }
"""

from guidellm.backends.response_handlers import (
    ChatCompletionsResponseHandler,
    GenerationResponseHandlerFactory,
)


@GenerationResponseHandlerFactory.register("chat_completions_with_reasoning")
class ChatCompletionsWithReasoningResponseHandler(ChatCompletionsResponseHandler):
    """
    Response handler for chat completions that supports reasoning tokens.

    This handler extends the standard ChatCompletionsResponseHandler to properly
    track both regular content tokens and reasoning_content tokens. This ensures
    that TTFT (Time To First Token) and ITL (Inter-Token Latency) are calculated
    correctly for models that output reasoning tokens before regular content.

    Key differences from the base handler:
    - Tracks both delta.content and delta.reasoning_content in streaming responses
    - Ensures first_token_iteration is set when ANY token arrives (not just content)
    - Fixes ITL calculation by properly tracking all token arrivals

    Example:
    ::
        handler = ChatCompletionsWithReasoningResponseHandler()
        response = handler.compile_streaming(request)
    """

    def __json__(self):
        """
        Return JSON-serializable representation of this handler class.

        This method is called by custom JSON encoders to serialize the handler
        class to its registered name.

        :return: The registered name of this handler
        """
        return "chat_completions_with_reasoning"

    @classmethod
    def __class_json__(cls):
        """
        Return JSON-serializable representation of this handler class.

        This class method is called when the class itself (not an instance)
        needs to be serialized.

        :return: The registered name of this handler
        """
        return "chat_completions_with_reasoning"

    def add_streaming_line(self, line: str) -> int | None:
        """
        Process a single line from a chat completion streaming response.

        Handles both regular content and reasoning_content tokens to ensure
        accurate timing metrics (TTFT and ITL).

        :param line: Raw SSE line from the streaming response
        :return: 1 if any token was extracted, 0 if line ignored, None if done
        """
        if not (data := self.extract_line_data(line)):
            return None if data is None else 0

        if "id" in data and self.streaming_response_id is None:
            self.streaming_response_id = data["id"]

        updated = False
        choices, usage = self.extract_choices_and_usage(data)
        choice: dict[str, dict] = choices[0] if choices else {}

        # Support both regular content and reasoning_content tokens
        # This ensures TTFT and ITL are calculated correctly for models with reasoning
        if choices:
            delta = choice.get("delta", {})
            content = delta.get("content")
            reasoning_content = delta.get("reasoning_content")

            # Track any token arrival (content or reasoning)
            # IMPORTANT: Check if field exists (not if it's truthy) to handle empty strings
            # The first chunk often has content="" which should still count for TTFT
            if content is not None or reasoning_content is not None:
                # Append whichever content is present (prioritize regular content)
                self.streaming_texts.append(content or reasoning_content or "")
                updated = True

        if usage:
            self.streaming_usage = usage

        return 1 if updated else 0
