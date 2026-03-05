import httpx

from benchmark_runner.openai_http_error_detail_backend import (
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

    import asyncio

    message = asyncio.run(format_http_status_error_async(exc))

    assert message.startswith("HTTP 400:")
    assert "too long" in message
    assert "type=BadRequestError" in message
    assert "code=400" in message
