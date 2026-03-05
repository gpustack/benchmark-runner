"""
The core flow in this module is adapted from GuideLLM's OpenAI backend:
- Source: https://github.com/vllm-project/guidellm/blob/v0.5.3/src/guidellm/backends/openai.py
- Reference implementation: `guidellm/backends/openai.py::OpenAIHTTPBackend.resolve`

Adjustments in this module:
1) Register a new backend type `openai_http_error_detail`.
2) Keep the request/stream processing flow aligned with GuideLLM.
3) For HTTP error responses, read body early (`aread()`) before stream closes.
4) Extract OpenAI-style error fields (`error.message/type/code`) with safe fallback.
5) Re-raise as concise `RuntimeError` so scheduler `RequestInfo.error` keeps detail.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any
import httpx

from guidellm.backends import Backend
from guidellm.backends.response_handlers import GenerationResponseHandlerFactory
from guidellm.schemas import GenerationRequest, GenerationResponse, RequestInfo
from guidellm.backends.openai import OpenAIHTTPBackend

ERROR_DETAIL_BACKEND_TYPE = "openai_http_error_detail"
MAX_ERROR_DETAIL_LENGTH = 2048
__all__ = [
    "ERROR_DETAIL_BACKEND_TYPE",
    "MAX_ERROR_DETAIL_LENGTH",
    "OpenAIHTTPErrorDetailBackend",
    "format_http_error_response",
    "format_http_status_error_async",
    "format_http_status_error",
]


def _truncate(text: str, max_length: int = MAX_ERROR_DETAIL_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def format_http_status_error(exc: httpx.HTTPStatusError) -> str:
    response = exc.response
    if response is None:
        return _truncate(f"HTTP request failed: {exc}")

    status_code = response.status_code
    message: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    body_fallback: str | None = None

    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = _stringify(error_obj.get("message"))
                error_type = _stringify(error_obj.get("type"))
                error_code = _stringify(error_obj.get("code"))
            else:
                message = _stringify(payload.get("message"))
                error_type = _stringify(payload.get("type"))
                error_code = _stringify(payload.get("code"))

        if message is None:
            body_fallback = _truncate(json.dumps(payload, ensure_ascii=False))
    except Exception:
        try:
            body_fallback = _truncate(response.text)
        except Exception:
            body_fallback = _stringify(exc)

    detail = message or body_fallback or _stringify(exc) or "unknown error"
    suffix_parts = []
    if error_type:
        suffix_parts.append(f"type={error_type}")
    if error_code:
        suffix_parts.append(f"code={error_code}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return _truncate(f"HTTP {status_code}: {detail}{suffix}")


async def format_http_error_response(
    response: httpx.Response, fallback: str | None = None
) -> str:
    """
    Format HTTP error detail directly from a response object.

    This function is safe for streaming responses and attempts to read the body
    before parsing OpenAI-style `error.message/type/code`.
    """
    try:
        await response.aread()
    except Exception:
        pass

    payload: Any = None
    message: str | None = None
    error_type: str | None = None
    error_code: str | None = None
    body_fallback: str | None = None

    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                message = _stringify(error_obj.get("message"))
                error_type = _stringify(error_obj.get("type"))
                error_code = _stringify(error_obj.get("code"))
            else:
                message = _stringify(payload.get("message"))
                error_type = _stringify(payload.get("type"))
                error_code = _stringify(payload.get("code"))

        if message is None and payload is not None:
            body_fallback = _truncate(json.dumps(payload, ensure_ascii=False))
    except Exception:
        try:
            body_fallback = _truncate(response.text)
        except Exception:
            body_fallback = None

    detail = message or body_fallback or _stringify(fallback) or "unknown error"
    suffix_parts = []
    if error_type:
        suffix_parts.append(f"type={error_type}")
    if error_code:
        suffix_parts.append(f"code={error_code}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    return _truncate(f"HTTP {response.status_code}: {detail}{suffix}")


async def format_http_status_error_async(exc: httpx.HTTPStatusError) -> str:
    """
    Async-safe formatter for streaming responses.

    When httpx raises from a stream context, response content may not be read yet.
    We call `aread()` first so JSON/text extraction is available.
    """
    response = exc.response
    if response is None:
        return format_http_status_error(exc)
    return await format_http_error_response(response, fallback=str(exc))


@Backend.register(ERROR_DETAIL_BACKEND_TYPE)
class OpenAIHTTPErrorDetailBackend(OpenAIHTTPBackend):
    """
    OpenAI HTTP backend that enriches HTTP errors with response-body details.
    """

    async def resolve(
        self,
        request: GenerationRequest,
        request_info: RequestInfo,
        history: list[tuple[GenerationRequest, GenerationResponse]] | None = None,
    ) -> AsyncIterator[tuple[GenerationResponse | None, RequestInfo]]:
        if self._async_client is None:
            raise RuntimeError("Backend not started up for process.")

        if history is not None:
            raise NotImplementedError("Multi-turn requests not yet supported")

        if (request_path := self.api_routes.get(request.request_type)) is None:
            raise ValueError(f"Unsupported request type '{request.request_type}'")

        request_url = f"{self.target}/{request_path}"
        request_files = (
            {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in request.arguments.files.items()
            }
            if request.arguments.files
            else None
        )
        request_json = request.arguments.body if not request_files else None
        request_data = request.arguments.body if request_files else None
        response_handler = GenerationResponseHandlerFactory.create(
            request.request_type, handler_overrides=self.response_handlers
        )

        if not request.arguments.stream:
            request_info.timings.request_start = time.time()
            response = await self._async_client.request(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=self._build_headers(request.arguments.headers),
                json=request_json,
                data=request_data,
                files=request_files,
            )
            request_info.timings.request_end = time.time()
            if response.is_error:
                raise RuntimeError(
                    await format_http_error_response(
                        response, fallback="request failed"
                    )
                )
            data = response.json()
            yield response_handler.compile_non_streaming(request, data), request_info
            return

        try:
            request_info.timings.request_start = time.time()

            async with self._async_client.stream(
                request.arguments.method or "POST",
                request_url,
                params=request.arguments.params,
                headers=self._build_headers(request.arguments.headers),
                json=request_json,
                data=request_data,
                files=request_files,
            ) as stream:
                if stream.is_error:
                    request_info.timings.request_end = time.time()
                    raise RuntimeError(
                        await format_http_error_response(
                            stream, fallback="request failed"
                        )
                    )
                end_reached = False

                async for chunk in stream.aiter_lines():
                    iter_time = time.time()

                    if request_info.timings.first_request_iteration is None:
                        request_info.timings.first_request_iteration = iter_time
                    request_info.timings.last_request_iteration = iter_time
                    request_info.timings.request_iterations += 1

                    iterations = response_handler.add_streaming_line(chunk)
                    if iterations is None or iterations <= 0 or end_reached:
                        end_reached = end_reached or iterations is None
                        continue

                    if request_info.timings.first_token_iteration is None:
                        request_info.timings.first_token_iteration = iter_time
                        request_info.timings.token_iterations = 0

                    request_info.timings.last_token_iteration = iter_time
                    request_info.timings.token_iterations += iterations

            request_info.timings.request_end = time.time()
            yield response_handler.compile_streaming(request), request_info
        except asyncio.CancelledError as err:
            yield response_handler.compile_streaming(request), request_info
            raise err
