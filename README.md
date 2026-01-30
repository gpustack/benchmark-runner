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

License
-------
See repository license information.
