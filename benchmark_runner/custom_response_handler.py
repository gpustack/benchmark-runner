"""
Chat-completions handler that counts a reasoning model's thinking as output text.

What this handler is for CHANGED at guidellm 0.7.1, so the old description ("fixes
TTFT and ITL for reasoning models") no longer applies — see below.

guidellm 0.7.x note:
- Handlers still live in ``guidellm.backends.openai.request_handlers`` and are
  registered on ``OpenAIRequestHandlerFactory``. The stock factory registers the
  built-in handlers by API PATH (e.g. ``/v1/chat/completions``); we register this
  one by the distinct NAME ``chat_completions_with_reasoning`` and select it via
  the backend's ``request_handlers`` override (path -> handler name).

- TIMING IS NO LONGER THIS HANDLER'S JOB. 0.5.x's base handler only looked at
  ``delta.content``, so a reasoning-only chunk returned 0, the backend never
  stamped ``first_token_iteration`` on it, and ``token_iterations`` skipped it:
  TTFT came out as "request start -> first token AFTER all the thinking" and the
  ITL series began there too. That is what the original override fixed, and the
  mechanism was its RETURN VALUE (``updated = True``), not the text it appended.
  0.7.x does this upstream — the base sets ``updated`` on
  ``delta.reasoning``/``delta.reasoning_content`` — and adds what 0.5.x had no
  concept of: reasoning kept separately (``reasoning_text``) and TTFOT, the first
  CONTENT token, as its own metric. The ITL denominator is likewise untouched by
  this class: it is ``output_tokens``, i.e. ``usage.completion_tokens`` (which
  includes reasoning tokens), falling back to ``token_iterations``.

- WHAT IS LEFT is the text accounting, and it is not cosmetic. The base files
  reasoning ONLY under ``reasoning_text``, so ``text`` — and the word/character
  metrics derived from it (``output_metrics.text_words`` /
  ``text_characters``, aggregated into ``metrics.text.words`` /
  ``metrics.text.characters``) — cover the post-thinking content alone. For a
  reasoning model benchmarked with a small ``output_tokens``, there IS no
  post-thinking content: guidellm forces ``max_completion_tokens=output_tokens``
  plus ``ignore_eos=True``, so the whole budget goes to thinking and the model is
  cut off mid-thought. Measured on four real gpustack runs (1024/128 synthetic,
  61136 successful requests): only 12 requests — 0.02% — ever emitted a content
  token, while all 61136 report ~104 words / ~600 characters of output. Without
  this class those figures are 0 and the sampled ``output`` is an empty string,
  i.e. the report claims the model produced nothing at all.

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
    Chat-completions handler that also counts reasoning as generated output text.

    Key difference from the 0.7.x base handler — exactly one:
    - A chunk's reasoning delta is appended to ``streaming_texts`` as well, so the
      response's ``text`` and the word/character metrics derived from it include
      the thinking. The base counts only post-thinking content, which is empty for
      a reasoning model whose ``output_tokens`` budget is spent thinking.

    NOT this class's doing (all inherited, all already correct in 0.7.x): TTFT
    firing on a reasoning delta, the ITL series and its ``output_tokens``
    denominator, the TTFOT content flag, tool-call deltas, and the separate
    ``reasoning_text``. See the module docstring for why the timing story moved
    upstream and what that leaves here.

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

        Delegates to the 0.7.x base handler and then ALSO counts this chunk's
        reasoning delta as generated text, which is the one behavior this subclass
        exists for. Note what that does NOT buy: the base already returns 1 on a
        reasoning-only chunk, so TTFT and the ITL series already span the thinking
        phase without this. What it buys is that ``text`` (and the word/character
        metrics read off it) is non-empty for a response that never got past
        thinking — the common case at a small ``output_tokens``.

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

        Reasoning therefore appears in BOTH ``text`` (this subclass's purpose) and
        ``reasoning_text`` (the base's own accounting) — deliberate duplication, so
        that a consumer reading either field sees what the model produced. Token
        counts are unaffected either way: they come from
        ``usage.completion_tokens``, never from the text. The word/character counts
        ARE affected, and that is the point — they are reported metrics
        (``metrics.text.words`` / ``metrics.text.characters``), not a fallback.

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
