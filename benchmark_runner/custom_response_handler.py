"""
Custom response handler that fixes TTFT and ITL calculation for models with reasoning tokens.

This handler extends guidellm's ChatCompletionsResponseHandler to properly handle
both regular content tokens and reasoning_content tokens, ensuring accurate timing metrics.

guidellm 0.7.1 note:
- Handlers still live in ``guidellm.backends.openai.request_handlers`` and are
  registered on ``OpenAIRequestHandlerFactory``. The stock factory registers the
  built-in handlers by API PATH (e.g. ``/v1/chat/completions``); we register this
  one by the distinct NAME ``chat_completions_with_reasoning`` and select it via
  the backend's ``request_handlers`` override (path -> handler name).
- v0.7.1's ``ChatCompletionsRequestHandler`` already fires TTFT on reasoning
  deltas; this subclass preserves the 0.6.0 behavior where reasoning content is
  also accumulated as generated text so ITL is measured across every token.

Usage:
    Select this handler through the ``openai_http_error_detail`` backend, keying
    ``request_handlers`` by API path -> registered handler name:

    benchmark-runner benchmark run \\
        --target http://localhost:8000 \\
        --backend openai_http_error_detail \\
        --backend-kwargs \\
          '{"request_handlers": {"/v1/chat/completions": "chat_completions_with_reasoning"}}' \\
        --model your-model-name \\
        --data your-dataset

    Or in a scenario config file (spec.backend):
    {
        "kind": "openai_http_error_detail",
        "request_handlers": {
            "/v1/chat/completions": "chat_completions_with_reasoning"
        }
    }
"""

from guidellm.backends.openai.request_handlers import (
    ChatCompletionsRequestHandler,
    OpenAIRequestHandlerFactory,
)


@OpenAIRequestHandlerFactory.register("chat_completions_with_reasoning")
class ChatCompletionsWithReasoningResponseHandler(ChatCompletionsRequestHandler):
    """
    Response handler for chat completions that supports reasoning tokens.

    This handler extends the standard ChatCompletionsResponseHandler to properly
    track both regular content tokens and reasoning_content tokens. This ensures
    that TTFT (Time To First Token) and ITL (Inter-Token Latency) are calculated
    correctly for models that output reasoning tokens before regular content.

    Key difference from the 0.7.1 base handler:
    - A chunk's reasoning delta is accumulated as generated text as well, so the
      inter-token latency series spans reasoning tokens instead of starting at the
      first post-reasoning content token.

    Everything else (TTFT on reasoning deltas, the TTFOT content flag, tool-call
    deltas, separate reasoning_text) is the base handler's, by delegation.

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

        Delegates to the 0.7.1 base handler and then ALSO counts this chunk's
        reasoning delta as generated text, which is the one behavior this subclass
        exists for: ITL is then measured across every token the model emitted, not
        only the post-reasoning content.

        Why delegate instead of reimplementing the loop (which is what this method
        used to do): the base handler maintains three pieces of state that a full
        override silently dropped —

        * ``_last_iteration_had_content`` — the flag the HTTP layer reads to stamp
          ``first_output_token_iteration`` (TTFOT). Never set by the old override,
          so TTFOT was ALWAYS None for exactly the backend+handler pairing this
          project ships and gpustack sends unconditionally.
        * ``streaming_tool_calls`` — tool-call deltas, reassembled by index.
        * ``streaming_reasoning_texts`` — reasoning kept separately, so the response
          carries ``reasoning_text`` instead of an empty field.

        Reasoning therefore now appears in BOTH ``text`` (this subclass's purpose)
        and ``reasoning_text`` (the base's own accounting). Output token counts are
        unaffected: they come from ``usage.completion_tokens``, and the text-derived
        word/char counts are a fallback that behaved this way before too.

        One deliberate behavior change: the old override treated ``content == ""``
        as a token arrival ("the first chunk often has content='' which should still
        count for TTFT"). In 0.7.1 that reason is gone — the base already fires TTFT
        on a reasoning delta — and counting an empty role-only chunk as the first
        token UNDER-reports TTFT for non-reasoning models. Empty deltas are now
        ignored, matching the base.

        :param line: Raw SSE line from the streaming response
        :return: 1 if any token was extracted, 0 if line ignored, None if done
        """
        reasoning_seen = len(self.streaming_reasoning_texts)
        iterations = super().add_streaming_line(line)

        # Count this chunk's reasoning delta as generated text as well, so the
        # inter-token latency series spans reasoning tokens too.
        new_reasoning = self.streaming_reasoning_texts[reasoning_seen:]
        if new_reasoning:
            self.streaming_texts.extend(new_reasoning)

        return iterations
