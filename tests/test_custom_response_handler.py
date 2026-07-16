"""The reasoning-aware chat handler must not lose the 0.7.1 base handler's state.

This handler is not optional in practice: gpustack sends
``request_handlers={"/v1/chat/completions": "chat_completions_with_reasoning"}``
on every benchmark, so whatever it drops is dropped for every run. It used to
reimplement ``add_streaming_line`` wholesale, which silently discarded three
things the base maintains — the TTFOT content flag, tool-call deltas, and the
separate reasoning text — while the backend's TTFOT code read the very flag that
was never set.

What it is still supposed to do differently: count reasoning tokens as generated
text, so the inter-token latency series spans them.
"""

import json

from benchmark_runner.custom_response_handler import (
    ChatCompletionsWithReasoningResponseHandler,
)


def sse(**delta) -> str:
    """One chat-completions SSE chunk carrying ``delta``."""
    return "data: " + json.dumps(
        {"id": "resp-1", "choices": [{"index": 0, "delta": delta}]}
    )


class TestTtfotFlag:
    """``last_iteration_had_content`` is what the backend stamps TTFOT from.

    Regression: the old override never assigned ``_last_iteration_had_content``, so
    it stayed False for the whole response and
    ``request_info.timings.first_output_token_iteration`` was never set — TTFOT was
    always null for the one handler this project ships.
    """

    def test_reasoning_only_chunk_is_not_content(self):
        h = ChatCompletionsWithReasoningResponseHandler()
        assert h.add_streaming_line(sse(reasoning_content="thinking")) == 1
        # Counted as a token (drives TTFT/ITL) but NOT as output content.
        assert h.last_iteration_had_content is False

    def test_content_chunk_sets_the_flag(self):
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(sse(reasoning_content="thinking"))
        h.add_streaming_line(sse(content="hello"))
        assert h.last_iteration_had_content is True

    def test_the_flag_survives_a_non_token_line(self):
        # A line that yields no tokens must not reset the flag, or TTFOT could be
        # re-stamped later than it happened.
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(sse(content="hello"))
        h.add_streaming_line("data: " + json.dumps({"choices": []}))
        assert h.last_iteration_had_content is True


class TestReasoningIsCountedAsGeneratedText:
    """The one behavior this subclass exists for."""

    def test_reasoning_lands_in_the_text_stream(self):
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(sse(reasoning_content="think "))
        h.add_streaming_line(sse(content="answer"))
        assert "".join(h.streaming_texts) == "think answer"

    def test_reasoning_is_also_kept_separately(self):
        # The base's own accounting is preserved, so the response carries
        # reasoning_text instead of an empty field.
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(sse(reasoning_content="think"))
        assert h.streaming_reasoning_texts == ["think"]

    def test_the_openai_reasoning_key_works_too(self):
        # The base accepts delta.reasoning as well as delta.reasoning_content.
        h = ChatCompletionsWithReasoningResponseHandler()
        assert h.add_streaming_line(sse(reasoning="think")) == 1
        assert "".join(h.streaming_texts) == "think"


class TestBaseStateIsNotDropped:
    def test_tool_call_deltas_are_accumulated(self):
        # The old override ignored tool_calls entirely: the response then carried
        # none, and the backend's tool-call expectation check would fail the turn.
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(
            sse(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ]
            )
        )
        assert 0 in h.streaming_tool_calls
        assert h.last_iteration_had_content is True

    def test_response_id_is_captured(self):
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(sse(content="x"))
        assert h.streaming_response_id == "resp-1"

    def test_usage_is_captured(self):
        h = ChatCompletionsWithReasoningResponseHandler()
        h.add_streaming_line(
            "data: "
            + json.dumps(
                {
                    "id": "resp-1",
                    "choices": [{"index": 0, "delta": {"content": "x"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                }
            )
        )
        assert h.streaming_usage["completion_tokens"] == 3

    def test_done_still_terminates(self):
        h = ChatCompletionsWithReasoningResponseHandler()
        assert h.add_streaming_line("data: [DONE]") is None


def test_empty_content_is_no_longer_a_token():
    # Deliberate change: the old override counted content="" as a token arrival so
    # a role-only first chunk would set TTFT. In 0.7.1 reasoning deltas already do
    # that, and counting an empty chunk UNDER-reports TTFT for plain models.
    h = ChatCompletionsWithReasoningResponseHandler()
    assert h.add_streaming_line(sse(content="")) == 0
    assert h.streaming_texts == []


def test_registered_name_is_stable():
    # gpustack selects this handler by name; renaming the registration breaks every
    # benchmark with an "Unknown request handler" ValueError.
    from guidellm.backends.openai.request_handlers import OpenAIRequestHandlerFactory

    assert (
        OpenAIRequestHandlerFactory.get_registered_object(
            "chat_completions_with_reasoning"
        )
        is ChatCompletionsWithReasoningResponseHandler
    )
