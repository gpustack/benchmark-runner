Benchmark Runner
================

Benchmark Runner is a thin wrapper around GuideLLM that provides a simplified CLI,
custom progress reporting, and ShareGPT dataset preparation for benchmarking
generative models.

What it adds
------------
- A streamlined `benchmark-runner` CLI focused on benchmark and config commands.
- An **adaptive auto-tune ramp** (`--auto-tune`): probes one deterministic answer —
  peak throughput, or the maximum load meeting a latency SLA — instead of asking
  you to guess a load value. See "Auto-tune" below.
- **Manual stages** (`--stages`): one single-strategy run per stage, each with its
  own request/duration limits.
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

Auto-tune (adaptive ramp)
-------------------------
`--auto-tune` replaces "guess a load value and read a curve" with probing a single
deterministic answer. It runs one single-strategy GuideLLM benchmark per measured
point and reads the metrics back to choose the next one:

- **Phase 1** — geometric bracket: double the knob until throughput stops climbing
  (or the SLA breaks, or the server overloads).
- **Phase 2** — for an SLA target, bisect the pass/fail bracket for the maximum knob
  still within SLA; for a throughput target, run a unimodal (ternary/golden hybrid)
  search inside the bracket for the throughput argmax.

The **target is derived**, not configured: set any `--sla-*` threshold and the
answer becomes "the maximum load meeting that SLA"; set none and it becomes "peak
throughput".

`--axis` picks the load axis, which decides what the knob means:

| `--axis` | GuideLLM profile | Knob | Loop |
|---|---|---|---|
| `rate` (default) | `constant` | requests/second offered | open |
| `concurrency` | `concurrent` | in-flight streams | closed |

```bash
# Peak throughput on the request-rate axis.
benchmark-runner benchmark run \
  --target http://localhost:8000 \
  --auto-tune --axis rate \
  --lower-bound 4 --upper-bound 1024 \
  --max-points 12 --max-total-seconds 3600 \
  --data "prompt_tokens=1024,output_tokens=128" \
  --processor PROCESSOR_PATH

# Maximum concurrency that keeps avg TTFT <= 500ms and p95 TPOT <= 50ms.
benchmark-runner benchmark run \
  --target http://localhost:8000 \
  --auto-tune --axis concurrency \
  --sla-avg-ttft-ms 500 --sla-p95-tpot-ms 50 \
  --data "prompt_tokens=128,output_tokens=128" \
  --processor PROCESSOR_PATH
```

**SLA thresholds** — 3 metrics x 3 aggregations = 9 optional `<=` targets, all in
milliseconds: `--sla-{avg,p95,p99}-{ttft,tpot,latency}-ms`. Any subset may be set; a
point passes when **every** set threshold holds (AND) and its success rate is
>= 95%. (GuideLLM reports end-to-end latency in seconds; it is converted
internally so every threshold is in ms.)

**Search range is hard** — `[--lower-bound, --upper-bound]` is the range you asked
for, and the ramp measures **nothing outside either end**. Hitting the upper bound
while throughput is still climbing stops the sweep (rather than quietly probing
higher); symmetrically, a server that is *already* saturated at the lower bound is
**reported** rather than searched below — the ramp stops, and gpustack raises a
`saturated_at_lower_bound` warning carrying the measured sustainable rate so you can
lower the range and re-run. Defaults are 4 / 1024, identical to the gpustack presets.

**Budget** — `--max-points` and `--max-total-seconds` (default 3600) cap the whole
run; per-point request counts are derived as `max(--min-requests, knob * multiplier)`
with the multiplier defaulting by axis (10 concurrency / 30 rate), so every point
below saturation gets a comparable ~30s measurement window.

The duration cap binds from *inside* a point as well as between points: each run
(the saturation probe included) is given whatever is left of the budget as its own
duration limit, so a stalled point — an unresponsive server, or an offered rate the
server answers at a trickle — cannot run past the cap while working through its
request count. A point cut short this way still reports the metrics it gathered, and
the ramp ends with `stop_reason: budget_seconds`. The auto-tune mode takes neither
`--max-requests` nor `--max-seconds`: it derives both per point.

The search range and budget are validated up front — `--lower-bound` must be
positive and not exceed `--upper-bound`, and `--min-requests` / `--max-points` /
`--max-total-seconds` / `--multiplier` must be positive. A zero lower bound would
otherwise leave the knob pinned at 0 for every point.

**Saturation probe** — on the rate axis with a throughput target, one up-front
`throughput` run measures the server's ceiling rps and caps the ramp at
~`ceil(ceiling * 1.2)`: past the ceiling every point reports the same throughput
with worse latency, while costing geometrically more wall clock. That cap is *soft* —
reaching it while the server is still keeping up means the estimate read low, so it
doubles rather than truncating a real curve — and it never overrides `--upper-bound`.
The probe writes a `__satprobe` output file and is not counted as a measured point.
Disable with `--no-probe-saturation`.

Each measured point writes its own output pair, `{base}__p{index}.{ext}`, and uses a
distinct random seed (`--random-seed` as the base, `+1` per point) so cached prompts
and KV/prefix reuse cannot inflate later points. This needs no flag and happens
whether or not you pass `--random-seed` (the base defaults to 42): the ramp's points
are a doubling apart and measured back to back, so cache reuse would inflate the
throughput curve the peak is read off. `--no-seed-increment` pins one seed for every
point instead. (Manual `--stages` deliberately does the opposite by default — see
"Manual stages" below.)

The point files are dual-JSON pairs derived from the base id of the **first**
`--outputs` entry, so auto-tune cannot honor additional output kinds; passing more
than one logs which entries it ignored rather than quietly emitting only the first.

**Why the search stopped** — the ramp also writes `{base}__ramp.json`:

```json
{
  "version": 1,
  "bracket_reason": "capacity_plateau",
  "stop_reason": "converged",
  "stopped_at": 256.0,
  "points_measured": 7, "max_points": 12,
  "elapsed_seconds": 54.32, "max_total_seconds": 3600.0,
  "sla_bracket": [256.0, null],
  "probe_ceiling": null
}
```

`bracket_reason` is why the geometric bracket ended — which of the SLA, the server's
capacity, your search range, or the budget bounded the answer. `stop_reason` is why
the ramp as a whole ended, Phase 2 included; the two differ whenever a bracket found
its answer and the following search then converged normally. One of
`sla_failed` · `capacity_plateau` · `overloaded` · `upper_bound` · `budget_points` ·
`budget_seconds` · `converged` · `point_failed`.

This is reported rather than left to be inferred from the measured curve because
several terminations leave an **identical** curve behind: a run ended by
`--max-total-seconds` looks exactly like one that stopped of its own accord to
anybody counting points, and a `capacity_plateau` stop under a loose SLA looks
exactly like a threshold breaking at the top — with opposite advice attached in both
cases. The file appears when the ramp returns, so its absence means "still running,
or not a ramp at all"; a write failure is logged and does not fail the run.

Manual stages
-------------
`--stages` runs one single-strategy benchmark per stage, each carrying its own
constraints, writing `{base}__stage{i}.{ext}`. `--axis` selects what each stage's
`rate` means, exactly as it does for the ramp:

```bash
benchmark-runner benchmark run \
  --target http://localhost:8000 \
  --axis rate \
  --stages '[{"rate": 2, "max_requests": 60},
             {"rate": 4, "max_requests": 120},
             {"rate": 8, "max_seconds": 30}]' \
  --data "prompt_tokens=128,output_tokens=128" \
  --processor PROCESSOR_PATH
```

**Stage data is held fixed by default** — every stage uses the same seed, so the
same synthetic prompts are replayed at each load and the only thing that changes
between stages is the offered load. That is usually what a stage list is for.

To give **each stage its own prompts**, pass `--random-seed` explicitly — any value,
`42` included — and each stage gets `random_seed + stage_index`:

```bash
# stages get seeds 7, 8, 9
benchmark-runner benchmark run \
  --stages '[{"rate": 2}, {"rate": 4}, {"rate": 8}]' \
  --random-seed 7 \
  --data "prompt_tokens=128,output_tokens=128" ...
```

Passing the option is what opts in: left out, the seed stays at its default and no
per-stage variation happens. `--no-seed-increment` pins one seed for every stage even
when `--random-seed` is given. This only affects the Random synthetic dataset —
file datasets (including ShareGPT) are read in file order regardless of seed.

Vary the seed when you want each stage measured on cold prompts, since replaying
identical prompts lets the server answer later stages out of its prefix/KV cache and
flatters them. Keep it fixed when you want the stages to be directly comparable.
Note this differs from `--auto-tune`, which **always** varies its seed and needs no
flag to do so: its points are a doubling apart and measured back to back, so cache
reuse would inflate the throughput curve the peak is read off — there it is a
correctness requirement, not a preference.

`--auto-tune`, `--stages` and `--profile` are mutually exclusive.

Concurrency cap
---------------
`--max-concurrency` bounds in-flight requests. It is **required** by GuideLLM's
`throughput` profile (and `sweep`'s throughput pass, including the auto-tune
saturation probe), where it defaults to 512 to match GuideLLM's own sweep default.
It is optional for the open-loop rate profiles (`constant`/`poisson`/`async`), where
it bounds pile-up when the server cannot keep up with the offered rate, and ignored
by `concurrent`/`synchronous`.

Constraints
-----------
Per-benchmark stop conditions, all optional and passed straight through to
GuideLLM's constraint list: `--max-seconds`, `--max-requests`, `--max-errors`,
`--max-error-rate`, `--max-global-error-rate`.

`--max-error-rate` is a **fraction strictly between 0 and 1** (GuideLLM rejects both
endpoints). Omit it for no error-rate constraint — `1` would mean "tolerate every
failure" and `0` is not expressible as a rate (use `--max-errors` for a count).

`--detect-saturation` (alias `--default-over-saturation`) enables GuideLLM's native
`OverSaturationConstraint` in `enforce` mode: the run stops once throughput stops
keeping up with the offered load. `--over-saturation '{"enabled": true, ...}'` takes
the same detector with explicit settings — any `OverSaturationConstraint` field is
forwarded (`min_seconds`, `max_window_seconds`, `minimum_window_size`,
`moe_threshold`, `minimum_ttft`, `maximum_window_ratio`, `confidence`, and `mode`,
which can be set to `monitor` to report without stopping). Passing the option at all
enables the detector; `'{"enabled": false}'` is the way to ask for it off. It is a **generic runtime constraint that
applies to every mode**, not just auto-tune; on the ramp's short per-point runs it
simply won't fire (GuideLLM guards with `min_seconds=30` / `minimum_window_size=5`).

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

The conversion is **single-turn**: `sharegpt_to_guidellm.extract_first_turn` keeps
only the first human->gpt pair of each conversation. Multi-turn is therefore
synthetic-only — `--data "...,turns=N"` on GuideLLM's `synthetic_text` source. Passing
`turns` alongside a ShareGPT file has no effect.

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
  --backend openai_http_error_detail \
  --backend-kwargs '{"request_handlers": {"/v1/chat/completions": "chat_completions_with_reasoning"}}' \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data your-dataset \
  --max-requests 100
```

GuideLLM 0.7.1 registers request handlers by **API path**, so `request_handlers` is
keyed by path and its value is a registered handler *name* (a string the backend
resolves to a class). A legacy `response_handlers` dict keyed by request type
(`{"chat_completions": ...}`) is still accepted and translated, but the path-keyed
form above is the current one.

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
