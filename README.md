Benchmark Runner
================

Benchmark Runner is a thin wrapper around GuideLLM that provides a simplified CLI,
custom progress reporting, and ShareGPT dataset preparation for benchmarking
generative models.

What it adds
------------
- A streamlined `benchmark-runner` CLI focused on benchmark and config commands.
- Optional server-side progress updates during benchmarks.
- ShareGPT dataset conversion to GuideLLM-compatible JSONL.
- A JSON summary output format for benchmark reports.
- Custom response handler for accurate TTFT/ITL metrics with reasoning tokens (e.g., DeepSeek-R1).
- Optional backend mode to preserve HTTP error details (`message/type/code`) in failed request records.

Install
-------
Python 3.10+ is required.

```bash
pip install -e .
```

Usage
-----
Show available commands:

```bash
benchmark-runner --help
```

Run a benchmark:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH
```

Progress reporting
------------------
You can send progress updates to a server endpoint during a benchmark:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH \
  --progress-url https://example.com/api/progress/123 \
  --progress-auth YOUR_TOKEN
```

HTTP Error Details for Failed Requests
--------------------------------------
GuideLLM's default `openai_http` backend does not always preserve response-body
error payloads in request-level benchmark errors. Benchmark Runner provides an
opt-in backend type that enriches failed request errors using OpenAI-style error
fields (`error.message`, `error.type`, `error.code`):

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000/v1 \
  --backend openai_http_error_detail \
  --profile constant \
  --rate 10 \
  --max-requests 100 \
  --sample-requests 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH
```

When a request fails, `requests.errored[*].info.error` in benchmark outputs will
contain text similar to:
`HTTP 400: ... (type=BadRequestError, code=400)`.

Note: if `--sample-requests 0` is used, request-level samples are omitted by design,
including failed request details.

ShareGPT dataset support
------------------------
If a dataset filename contains "sharegpt" and ends with `.json` or `.jsonl`,
Benchmark Runner will convert it to a GuideLLM-compatible JSONL file before running
the benchmark.

Example:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --processor PROCESSOR_PATH \
  --data ./ShareGPT_V3_unfiltered_cleaned_split.json
```

Outputs
-------
Benchmark Runner supports GuideLLM outputs plus a JSON summary output.
To save summary JSON:

```bash
benchmark-runner benchmark \
  --target http://localhost:8000 \
  --profile constant \
  --rate 10 \
  --max-seconds 20 \
  --data "prompt_tokens=128,output_tokens=256" \
  --processor PROCESSOR_PATH \
  --outputs summary_json \
  --output-dir ./benchmarks
```

Reasoning Tokens Support
-------------------------
For models that output reasoning tokens (e.g., DeepSeek-R1, o1-preview), use the custom
response handler to get accurate TTFT and ITL metrics:

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000/v1 \
  --backend openai_http \
  --backend-kwargs '{"response_handlers": {"chat_completions": "chat_completions_with_reasoning"}}' \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data your-dataset \
  --max-requests 100
```

Docker
------
This repository includes a Dockerfile used to build a runtime image.

```bash
docker build -t benchmark-runner .
```

Development
-----------
Install development dependencies:

```bash
pip install -e ".[dev]"
```

macOS Notes
-----------
Benchmark Runner applies two macOS-only runtime defaults to avoid known
multiprocessing hangs:
- switch GuideLLM multiprocessing context from `fork` to `spawn` (unless
  `GUIDELLM__MP_CONTEXT_TYPE` is explicitly set)
- default `--data-num-workers` to `0` unless provided on the CLI

References:
- https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- https://bugs.python.org/issue33725

To disable these defaults for debugging/experiments:

```bash
BENCHMARK_RUNNER_DISABLE_MACOS_WORKAROUNDS=1 benchmark-runner benchmark run ...
```

License
-------
See repository license information.
