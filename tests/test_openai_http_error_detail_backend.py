import asyncio

import httpx
import pytest
from guidellm.schemas import (
    GenerationRequest,
    GenerationRequestArguments,
    GenerationResponse,
    RequestInfo,
)

from benchmark_runner.openai_http_error_detail_backend import (
    OpenAIHTTPErrorDetailBackend,
    OpenAIHTTPErrorDetailBackendArgs,
    format_http_status_error,
    format_http_status_error_async,
)


def _build_http_status_error(
    status_code: int,
    *,
    json_body=None,
    text_body: str | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response_kwargs = {"status_code": status_code, "request": request}
    if json_body is not None:
        response_kwargs["json"] = json_body
    elif text_body is not None:
        response_kwargs["content"] = text_body.encode("utf-8")
    response = httpx.Response(**response_kwargs)
    return httpx.HTTPStatusError("request failed", request=request, response=response)


def test_format_http_status_error_with_openai_error_body():
    exc = _build_http_status_error(
        400,
        json_body={
            "error": {
                "message": "The input is longer than the model's context length.",
                "type": "BadRequestError",
                "code": 400,
            }
        },
    )

    message = format_http_status_error(exc)

    assert message.startswith("HTTP 400:")
    assert "The input is longer than the model's context length." in message
    assert "type=BadRequestError" in message
    assert "code=400" in message


def test_format_http_status_error_with_non_json_fallback():
    exc = _build_http_status_error(
        502,
        text_body="upstream gateway timeout",
    )

    message = format_http_status_error(exc)

    assert message.startswith("HTTP 502:")
    assert "upstream gateway timeout" in message


def test_format_http_status_error_without_response():
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    exc = httpx.HTTPStatusError("request failed", request=request, response=None)

    message = format_http_status_error(exc)

    assert message.startswith("HTTP request failed:")


def test_format_http_status_error_async_with_unread_stream_response():
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(
            b'{"error":{"message":"too long","type":"BadRequestError","code":400}}'
        ),
    )
    exc = httpx.HTTPStatusError("request failed", request=request, response=response)

    message = asyncio.run(format_http_status_error_async(exc))

    assert message.startswith("HTTP 400:")
    assert "too long" in message
    assert "type=BadRequestError" in message
    assert "code=400" in message


class _RecordingHandler:
    """Minimal request handler that records whether ``post_validation`` ran.

    ``post_validation`` is the last step of the handler lifecycle (added in
    guidellm 0.7.3): it rejects a compiled response with no text, no tool calls
    and no output tokens. Our backend mirrors upstream's ``_resolve_*`` bodies, so
    it has to call it at the same point — otherwise an unusable response is
    silently counted as a successful request with zero output.
    """

    def __init__(self, reject: bool = False):
        self.reject = reject
        self.validated: list[GenerationResponse] = []
        self.last_iteration_had_content = True

    def _response(self) -> GenerationResponse:
        return GenerationResponse(request_id="req-1", request_args="{}")

    def compile_non_streaming(self, request, arguments, data) -> GenerationResponse:
        return self._response()

    def compile_streaming(self, request, arguments) -> GenerationResponse:
        return self._response()

    def add_streaming_line(self, line: str) -> int | None:
        return None if line.strip() == "data: [DONE]" else 1

    def post_validation(self, response: GenerationResponse) -> None:
        self.validated.append(response)
        if self.reject:
            raise ValueError(
                "[UNUSABLE_BACKEND_RESPONSE] backend resolved with empty "
                "response payload"
            )


def _started_backend(handler: httpx.MockTransport) -> OpenAIHTTPErrorDetailBackend:
    backend = OpenAIHTTPErrorDetailBackend(
        OpenAIHTTPErrorDetailBackendArgs(target="http://localhost:8000", model="m")
    )
    backend._async_client = httpx.AsyncClient(transport=handler)
    return backend


async def _drain(agen) -> list:
    return [item async for item in agen]


@pytest.mark.parametrize("reject", [False, True])
def test_resolve_non_streaming_runs_post_validation(reject):
    handler = _RecordingHandler(reject=reject)
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json={"id": "1"}))
    backend = _started_backend(transport)
    request_info = RequestInfo(request_id="req-1")

    async def run():
        return await _drain(
            backend._resolve_non_streaming(
                GenerationRequest(),
                request_info,
                handler,
                GenerationRequestArguments(),
                {"method": "POST", "url": "http://localhost:8000/v1/chat/completions"},
            )
        )

    if reject:
        # The ValueError must escape so the scheduler records the request as
        # errored instead of banking a zero-output success.
        with pytest.raises(ValueError, match="UNUSABLE_BACKEND_RESPONSE"):
            asyncio.run(run())
    else:
        yielded = asyncio.run(run())
        assert [response for response, _ in yielded] == handler.validated

    assert len(handler.validated) == 1


@pytest.mark.parametrize("reject", [False, True])
def test_resolve_streaming_runs_post_validation(reject):
    handler = _RecordingHandler(reject=reject)
    body = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, content=body))
    backend = _started_backend(transport)
    request_info = RequestInfo(request_id="req-1")

    async def run():
        return await _drain(
            backend._resolve_streaming(
                GenerationRequest(),
                request_info,
                handler,
                GenerationRequestArguments(),
                {"method": "POST", "url": "http://localhost:8000/v1/chat/completions"},
            )
        )

    if reject:
        with pytest.raises(ValueError, match="UNUSABLE_BACKEND_RESPONSE"):
            asyncio.run(run())
    else:
        yielded = asyncio.run(run())
        # First yield is the TTFT marker (None), the last is the compiled response.
        assert yielded[0][0] is None
        assert [yielded[-1][0]] == handler.validated

    assert len(handler.validated) == 1
