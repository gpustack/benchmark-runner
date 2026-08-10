"""
Adaptive ramp auto-tune engine.

A benchmark task = automatically probe ONE deterministic answer (peak throughput
OR the SLA-capacity boundary) instead of asking the user to guess a load value.
The engine orchestrates repeated single-strategy guidellm runs (each one knob
point) and reads the metrics back to decide the next point:

    Phase 1  geometric bracket   (knob *= 2 until a stop criterion trips, clamped
                                  to the bound in force — the search range
                                  [lower_bound, upper_bound] is HARD at both ends)
    Phase 2  depends on the target, because the two answers have different shapes:
             * SLA        — the pass/fail predicate is MONOTONE in the knob, so
                            bisect the (last_pass, first_fail) bracket for the
                            largest knob still inside SLA.
             * saturation — throughput is UNIMODAL, not monotone, so bisection has
                            nothing to bisect on. Run a three-point (ternary /
                            golden-section hybrid) search over Phase 1's
                            (prev_prev_knob, knob) bracket for the argmax, keeping
                            a < b < c with f(b) the running max and shrinking by
                            LOCAL comparison. The earlier "bisect the last doubling
                            gap and walk right when a point beats the global best"
                            scheme is gone: it only finds the peak when the peak
                            sits at the bracket's left end (see Phase 2 below).

Each ramp point is ONE ``benchmark_generative_text`` invocation with a single
``concurrent``/``constant`` strategy, a ``max_requests`` constraint, and a
DISTINCT random seed (defeats prefix/KV cache reuse across points). Each
point writes its own dual_json pair ``{base}__p{index}.json`` /
``{base}__p{index}.full.json`` which the gpustack manager globs.

This module is invoked from ``benchmark_runner.main`` when ``--auto-tune`` is set;
non-auto-tune modes (``--profile`` / ``--stages`` / ``--rate``) are untouched.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from guidellm.benchmark import BenchmarkScenario
from guidellm.benchmark.entrypoints import benchmark_generative_text

from benchmark_runner.scenario_builder import (
    DEFAULT_MAX_CONCURRENCY,
    build_scenario_args,
)

# 95% success floor shared by both stop targets. Peak probing uses
# a loose floor (not 100%) so a stray timeout doesn't prematurely cut the peak.
SUCCESS_FLOOR = 0.95
# Throughput plateau threshold: <5% gain over previous point = saturated.
PLATEAU_GAIN = 0.05
# Max concurrency cap for the saturation probe so an unbounded offered load can't
# pile up in-flight requests and drag the server down — which would bias the very
# ceiling the probe is measuring. Re-exported from the scenario builder, which
# applies the same cap to any `throughput` profile.
DEFAULT_RATE_MAX_CONCURRENCY = DEFAULT_MAX_CONCURRENCY

# ── Why the search stopped ────────────────────────────────────────────────────
# A closed set, one entry per `break` (plus the two loop predicates). Reported to
# the consumer instead of being left to be re-derived from the shape of the
# measured curve: several of these terminations produce IDENTICAL grids, so the
# reader cannot tell them apart afterwards.
#
# Two concrete cases that motivated reporting it:
#   * SLA target, thresholds never breached. `capacity_plateau` (we stopped
#     because throughput flattened) and `sla_failed`-at-the-top look the same in
#     rate space — both end with "the highest knob measured met the SLA".
#   * `budget_seconds` vs `budget_points`. A consumer counting points can only see
#     the second one; a run cut short by the time cap looks exactly like a run
#     that stopped of its own accord, and the advice for the two is different
#     (raise the duration cap vs. nothing to fix).
STOP_SLA_FAILED = "sla_failed"  # a point breached the SLA -> bracket found
STOP_CAPACITY_PLATEAU = "capacity_plateau"  # throughput stopped climbing
STOP_OVERLOADED = "overloaded"  # success rate fell under the floor
STOP_UPPER_BOUND = "upper_bound"  # reached the top of the search range
# Reached the probe's soft cap with no measured point to overrule it — only
# possible when lower_bound already sits at/above the cap, i.e. the FIRST point
# tripped it. Distinct from `upper_bound` because the advice is opposite: raising
# the range does nothing here, the probe re-derives its cap on every run.
STOP_PROBE_BOUND = "probe_bound"
STOP_BUDGET_POINTS = "budget_points"  # max_points spent
STOP_BUDGET_SECONDS = "budget_seconds"  # max_total_seconds spent
STOP_CONVERGED = "converged"  # the search interval closed
STOP_POINT_FAILED = "point_failed"  # a point produced no benchmark

# Schema version of the ramp outcome, so a consumer reading an older sidecar can
# tell "field absent" from "field not yet invented".
RAMP_OUTCOME_VERSION = 1

# ── Saturation probe sizing ───────────────────────────────────────────────────
# The probe answers one question: roughly how many requests per second can this
# server retire? Two knobs decide the answer's quality, doing different jobs.
#
# PROBE_SECONDS is the measurement WINDOW. Sizing by request count alone
# (max_requests=50) does not bound time at all — the run ends when the batch
# drains, so its duration is one request latency — and, worse, it makes the window
# exactly one batch: measured 1.71s against a 1.65s mean latency, i.e. 1.04
# turnovers. Every rate in that window then describes a burst rather than a rate
# (the prompt-token rate came back 3.4x the true value, TTFT 6x the steady-state
# value at the same concurrency).
#
# PROBE_MAX_CONCURRENCY decides WHERE ON THE CURVE we measure, which is what
# actually determines whether the estimate is usable. It used to be implicit and
# accidental: max_requests=50 caps in-flight at 50 whatever max_concurrency says,
# so the probe measured ~48 in-flight while claiming an unbounded ceiling. Lifting
# the request cap without pinning this would let in-flight climb to 512 — deep in
# the overloaded branch, where the achieved rate DECLINES (measured on one server:
# 30.1 rps at 53 in-flight, 25.5 at 221, 24.9 at 276). A stable measurement of the
# overloaded rate is a WORSE input than a noisy one near the knee, because the cap
# derived from it can sit below the real peak. So the accidental operating point is
# kept — deliberately this time.
PROBE_SECONDS = 10.0
PROBE_MAX_CONCURRENCY = 64
# Backstop, sized so that hitting it is harmless. What makes the measurement a rate
# rather than a batch is TURNOVERS = requests / concurrency, not requests / time —
# so 16x the concurrency is 16 turnovers however fast the server is. Derived rather
# than a round literal, because the invariant is against PROBE_MAX_CONCURRENCY: a
# fixed 5000 would silently stop meaning "16 turnovers" the moment that changes.
#
# It can only bind on a server retiring >100 rps (sub-0.64s latency at 64 in
# flight), and there 16 turnovers arrive well inside the window anyway. Keeping it
# this small also stays under the ShareGPT conversion cache, which is sized from
# `upper_bound * multiplier` and can be under 1000 for a narrow range — a bigger
# backstop would end the probe on `requests_exhausted` instead of on its window.
PROBE_MAX_REQUESTS = PROBE_MAX_CONCURRENCY * 16

# Floor on the per-point duration cap derived from the remaining budget. A point
# handed ~0 seconds would be stopped before a single response landed and report no
# metrics at all, which reads downstream as a failed point rather than a truncated
# one. See _remaining_seconds.
_MIN_POINT_SECONDS = 5.0


@dataclass
class PointMetrics:
    """Normalized metrics for one measured knob point.

    All latency-family fields are stored in MILLISECONDS so they compare directly
    to the ms-denominated SLA thresholds. TTFT/TPOT are already ms in guidellm;
    request latency is SECONDS in guidellm and is converted to ms here (x1000).
    """

    knob: float
    index: int
    output_tps: float  # metrics.output_tokens_per_second.successful.mean
    ttft_ms: float  # metrics.time_to_first_token_ms.successful.mean
    ttft_p95_ms: float  # ...time_to_first_token_ms.successful.percentiles.p95
    ttft_p99_ms: float  # ...time_to_first_token_ms.successful.percentiles.p99
    # TPOT = DECODE-ONLY time per output token, which guidellm files under
    # `inter_token_latency_ms`: (last_token - first_token) / (output_tokens - 1).
    # That is the industry's TPOT (vLLM, genai-perf) and what the server's
    # SLA_THRESHOLDS evaluates, so the bracketing here and the stored verdict
    # agree. guidellm's `time_per_output_token_ms` divides by output_tokens from
    # request_start instead, folding TTFT into the decode average — it used to
    # feed these three fields, which made a TPOT threshold tighten with queueing
    # (the error is TTFT / (n * TPOT): ~5% at 128 output tokens, ~40% at 16).
    #
    # It remains the FALLBACK, because the decode-only reading needs two token
    # timestamps: a response delivered as one chunk (whole output at once, common
    # at low load) has first_iteration == last_iteration and guidellm reports 0.
    # Falling back keeps the threshold evaluable there; 0 would clear every budget
    # and failing outright would bracket the ramp on its first point.
    tpot_ms: float  # metrics.inter_token_latency_ms.successful.mean
    tpot_p95_ms: float  # ...inter_token_latency_ms.successful.percentiles.p95
    tpot_p99_ms: float  # ...inter_token_latency_ms.successful.percentiles.p99
    latency_ms: float  # metrics.request_latency.successful.mean * 1000 (s -> ms)
    latency_p95_ms: float  # ...request_latency.successful.percentiles.p95 * 1000
    latency_p99_ms: float  # ...request_latency.successful.percentiles.p99 * 1000
    achieved_rate: float  # metrics.requests_per_second.successful.mean
    success: float  # successful / total


@dataclass
class AutoTuneConfig:
    """Ramp engine inputs (all supplied via CLI).

    SLA is a set of up to 9 OPTIONAL "<=" latency thresholds (all in ms): avg + p95
    + p99 of TTFT, TPOT, and end-to-end latency. Any subset may be set; the target
    becomes "sla" iff at least one is set, and a point passes iff every SET
    threshold holds.
    """

    axis: str  # "rate" | "concurrency"
    # [lower_bound, upper_bound] is a HARD range: the ramp starts at lower_bound and
    # measures nothing outside either end (a peak below the floor is REPORTED, not
    # searched for — see the Phase-1 plateau branch). Defaults are kept identical to
    # the CLI options and the gpustack presets/UI; 4 skips the low-info 1/2 points.
    lower_bound: float = 4.0
    upper_bound: float = 1024.0
    multiplier: Optional[float] = None  # default resolved by axis (10 conc / 30 rate)
    # Per-point request floor. 100 rather than 30 because a percentile is only as
    # good as the samples above it: with n samples, p99 has n/100 above it, so at
    # n=30 (or 40) p99 IS the maximum — one outlier defines the tail. On the
    # concurrency axis `knob * 10` puts every knob under 10 in that regime, which is
    # exactly where a p95/p99 SLA threshold can end a run on its first point.
    #
    # 100 does not make p99 trustworthy (it makes it the second-largest sample —
    # ~1000 is needed for a real tail estimate). It removes the degenerate
    # "p99 == max" case and keeps the cheap stages honest; the report still flags
    # stages whose sample count cannot support the tail it is showing.
    min_requests: int = 100
    # Cap on MEASURED points. The saturation probe is not one of them (it writes a
    # "__satprobe" file the consumer's point glob never matches and never enters an
    # aggregate), so it does not spend this budget — only wall clock.
    max_points: int = 12
    max_total_seconds: float = 3600.0  # 1h; kept in sync with the CLI default
    # SLA thresholds (all optional, all in ms, all "<=" comparisons).
    sla_avg_ttft_ms: Optional[float] = None
    sla_p95_ttft_ms: Optional[float] = None
    sla_p99_ttft_ms: Optional[float] = None
    sla_avg_tpot_ms: Optional[float] = None
    sla_p95_tpot_ms: Optional[float] = None
    sla_p99_tpot_ms: Optional[float] = None
    sla_avg_latency_ms: Optional[float] = None
    sla_p95_latency_ms: Optional[float] = None
    sla_p99_latency_ms: Optional[float] = None
    random_seed_base: int = 42
    # True: each point's seed = base + index (points differ, spreading prefix/KV
    # cache reuse). False: every point uses the base seed.
    seed_increment: bool = True
    # Saturation bounding (rate axis + saturation target only): probe the server's
    # ceiling rps once up front and use it as a SOFT cap on the ramp, so the sweep
    # doesn't spend its budget doubling far past saturation where every point
    # reports the same throughput. The ramp still starts at lower_bound, and the
    # soft cap yields when the ceiling estimate turns out to be too low. See
    # run_ramp for the probe + bounding logic.
    probe_saturation: bool = True

    @property
    def resolved_multiplier(self) -> float:
        if self.multiplier is not None:
            return self.multiplier
        # concurrency: requests-per-slot; rate: seconds of test time.
        return 10.0 if self.axis == "concurrency" else 30.0

    def sla_pairs(self, m: PointMetrics) -> list[tuple[Optional[float], float]]:
        """(threshold, metric_value) pairs for the 9 SLA dimensions.

        Threshold is None when unset (ignored by the pass predicate). All values
        are in ms, aligned with the thresholds.
        """
        return [
            (self.sla_avg_ttft_ms, m.ttft_ms),
            (self.sla_p95_ttft_ms, m.ttft_p95_ms),
            (self.sla_p99_ttft_ms, m.ttft_p99_ms),
            (self.sla_avg_tpot_ms, m.tpot_ms),
            (self.sla_p95_tpot_ms, m.tpot_p95_ms),
            (self.sla_p99_tpot_ms, m.tpot_p99_ms),
            (self.sla_avg_latency_ms, m.latency_ms),
            (self.sla_p95_latency_ms, m.latency_p95_ms),
            (self.sla_p99_latency_ms, m.latency_p99_ms),
        ]

    @property
    def target(self) -> str:
        """ "sla" if ANY of the 9 SLA thresholds is set, else "saturation"."""
        any_set = any(
            t is not None
            for t in (
                self.sla_avg_ttft_ms,
                self.sla_p95_ttft_ms,
                self.sla_p99_ttft_ms,
                self.sla_avg_tpot_ms,
                self.sla_p95_tpot_ms,
                self.sla_p99_tpot_ms,
                self.sla_avg_latency_ms,
                self.sla_p95_latency_ms,
                self.sla_p99_latency_ms,
            )
        )
        return "sla" if any_set else "saturation"


@dataclass
class RampOutcome:
    """The measured points plus WHY the search ended where it did.

    Two reasons, not one, because they answer different questions:

    * ``bracket_reason`` — why the geometric bracket (Phase 1) ended. This is the
      one that says whether the SLA, the server's capacity, or the user's own
      search range bounded the answer.
    * ``stop_reason`` — why the ramp as a whole ended, Phase 2 included. Equal to
      ``bracket_reason`` when Phase 2 never ran.

    A run that brackets on ``capacity_plateau`` and then converges reports
    ``bracket_reason=capacity_plateau, stop_reason=converged``: the search
    completed normally AND the thing that limited it was capacity. Collapsing the
    two into one field loses exactly the distinction the consumer needs.
    """

    points: list[PointMetrics]
    bracket_reason: str
    stop_reason: str
    target: str
    axis: str
    # Highest knob actually measured, so the consumer can see the search ended
    # below the range it was given without recomputing it from the points.
    stopped_at: Optional[float]
    lower_bound: float
    upper_bound: float
    max_points: int
    max_total_seconds: float
    elapsed_seconds: float
    # (last_pass, first_fail) — first_fail is None when no point ever breached the
    # SLA, i.e. no boundary was located and the SLA number is a floor, not an edge.
    sla_bracket: Optional[tuple[Optional[float], Optional[float]]] = None
    # Ceiling rps measured by the saturation probe (rate axis + saturation target),
    # None when no probe ran.
    probe_ceiling: Optional[float] = None
    # What the probe's soft cap actually DID. `probe_ceiling` alone cannot say:
    # three outcomes are indistinguishable downstream without these two.
    #
    #   relaxed > 0                      the probe read low; the cap gave way
    #   relaxed == 0 && stopped_at == bound   the cap clamped the overshoot point
    #                                         (its whole reason for existing)
    #   relaxed == 0 && stopped_at < bound    the cap never bound anything
    #
    # Only the middle one earns the probe's cost, and "should --probe-saturation
    # stay on for this deployment?" is answerable only by counting which of the
    # three keeps happening. Re-deriving it downstream would mean reimplementing
    # the cap formula, the Phase-1/2 split and the clamp rule — three things that
    # would then have to stay in sync with this file forever.
    probe_relaxed: int = 0
    probe_bound: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable facts (the points travel in their own report files)."""
        return {
            "version": RAMP_OUTCOME_VERSION,
            "bracket_reason": self.bracket_reason,
            "stop_reason": self.stop_reason,
            "target": self.target,
            "axis": self.axis,
            "stopped_at": self.stopped_at,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "points_measured": len(self.points),
            "max_points": self.max_points,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "max_total_seconds": self.max_total_seconds,
            "sla_bracket": (
                list(self.sla_bracket) if self.sla_bracket is not None else None
            ),
            "probe_ceiling": self.probe_ceiling,
            "probe_relaxed": self.probe_relaxed,
            "probe_bound": self.probe_bound,
        }


def _mean(node: Any, *path: str) -> float:
    """Safely walk ``node.a.b.c`` returning 0.0 on any missing attribute/None."""
    cur = node
    for attr in path:
        cur = getattr(cur, attr, None)
        if cur is None:
            return 0.0
    return float(cur or 0.0)


def _normalize(benchmark: Any, knob: float, index: int) -> PointMetrics:
    """Map a guidellm ``benchmarks[0]`` result to our flat PointMetrics.

    Note: ``request_latency`` is in SECONDS in guidellm; the SLA thresholds are in
    ms, so its mean/p99 are multiplied by 1000 here. TTFT/TPOT are already ms.
    """
    m = benchmark.metrics
    totals = m.request_totals
    total = getattr(totals, "total", 0) or 0
    successful = getattr(totals, "successful", 0) or 0
    success = (successful / total) if total else 0.0
    return PointMetrics(
        knob=knob,
        index=index,
        output_tps=_mean(m, "output_tokens_per_second", "successful", "mean"),
        ttft_ms=_mean(m, "time_to_first_token_ms", "successful", "mean"),
        ttft_p95_ms=_mean(
            m, "time_to_first_token_ms", "successful", "percentiles", "p95"
        ),
        ttft_p99_ms=_mean(
            m, "time_to_first_token_ms", "successful", "percentiles", "p99"
        ),
        # Decode-only, i.e. the industry TPOT — see PointMetrics.tpot_ms. `or`
        # takes the includes-TTFT reading when the decode-only one is 0, which is
        # what a non-incrementally streamed response leaves behind.
        tpot_ms=(
            _mean(m, "inter_token_latency_ms", "successful", "mean")
            or _mean(m, "time_per_output_token_ms", "successful", "mean")
        ),
        tpot_p95_ms=(
            _mean(m, "inter_token_latency_ms", "successful", "percentiles", "p95")
            or _mean(m, "time_per_output_token_ms", "successful", "percentiles", "p95")
        ),
        tpot_p99_ms=(
            _mean(m, "inter_token_latency_ms", "successful", "percentiles", "p99")
            or _mean(m, "time_per_output_token_ms", "successful", "percentiles", "p99")
        ),
        # request_latency is seconds -> convert to ms to match the SLA thresholds.
        latency_ms=_mean(m, "request_latency", "successful", "mean") * 1000.0,
        latency_p95_ms=_mean(m, "request_latency", "successful", "percentiles", "p95")
        * 1000.0,
        latency_p99_ms=_mean(m, "request_latency", "successful", "percentiles", "p99")
        * 1000.0,
        achieved_rate=_mean(m, "requests_per_second", "successful", "mean"),
        success=success,
    )


def _passes_sla(m: PointMetrics, cfg: AutoTuneConfig) -> bool:
    """SLA-pass = success>=95% AND every SET threshold holds (<=).

    AND is taken over SET thresholds only; unset (None) thresholds are ignored.
    Up to 9 dimensions: avg+p95+p99 of TTFT, TPOT, and end-to-end latency (all ms).
    """
    if m.success < SUCCESS_FLOOR:
        return False
    for threshold, value in cfg.sla_pairs(m):
        if threshold is None:
            continue
        # A non-positive value means the metric was not measured, not that it took
        # 0 ms: `_normalize` returns 0.0 for anything missing from the report, and
        # the decode-only TPOT is genuinely undefined when requests emit a single
        # token (no second token to time), which guidellm reports as 0.0. Waiving
        # the threshold there would let a max_tokens=1 run pass every point and
        # ramp to the upper bound. Failing instead surfaces it as sla_never_met.
        if value <= 0 or value > threshold:
            return False
    return True


def _capacity_saturated(
    m: PointMetrics,
    prev_tps: Optional[float],
    prev_achieved: Optional[float],
    axis: str,
) -> bool:
    """Has the server stopped turning more offered load into more delivered work?

    Two signals, either one is enough:

    * ``plateau`` — output token throughput stopped climbing (<5% over the previous
      point). Applies to both axes: it is the quantity the whole sweep is about.
    * ``cant_keepup`` — the ACHIEVED request rate stopped climbing. Rate axis only,
      where the knob IS an offered rate. Note this is a growth comparison between
      consecutive points, NOT a single-point achieved-vs-offered ratio: with a
      finite max_requests the drain tail leaves achieved a ~constant fraction below
      offered (achieved ~= offered / (1 + latency/window)) even on an idle server,
      so an absolute floor false-trips on the very first, unsaturated point.

    Both targets use this, for different reasons. For the saturation target it IS
    the stop criterion (the peak is here). For the SLA target it is a SECONDARY
    stop: the SLA may still hold far past this point, but past saturation more load
    buys no more work — only queueing — so a "capacity" answer taken from up there
    is a load nobody would ever choose to run at. See the Phase-1 SLA branch.
    """
    cant_keepup = (
        axis == "rate"
        and prev_achieved is not None
        and prev_achieved > 0
        and (m.achieved_rate / prev_achieved - 1.0) < PLATEAU_GAIN
    )
    plateau = (
        prev_tps is not None
        and prev_tps > 0
        and (m.output_tps / prev_tps - 1.0) < PLATEAU_GAIN
    )
    return cant_keepup or plateau


# A run_point callable executes ONE measured point for (knob, index) and returns
# its normalized PointMetrics (or None when the point produced no benchmark). The
# per-point seed is derived internally from the index. It is injectable so tests
# can drive the ramp with a synthetic throughput curve instead of a live server.
RunPointFn = Callable[[float, int], Awaitable[Optional["PointMetrics"]]]


async def run_ramp(  # noqa: C901
    cfg: AutoTuneConfig,
    base_kwargs: dict[str, Any],
    output_base: str,
    server_progress: Any = None,
    progress: Any = None,
    console: Any = None,
    run_point: Optional[RunPointFn] = None,
    saturation_probe: Optional[Callable[[], Awaitable[float]]] = None,
) -> RampOutcome:
    """
    Execute the adaptive ramp and return the measured points.

    :param cfg: Ramp configuration (axis, bounds, budget, SLA).
    :param base_kwargs: Common guidellm kwargs shared by every point (target, data,
        backend, processor, ...). Per-point profile/rate/seed/outputs/max_requests
        are layered on top for each run.
    :param output_base: Dual_json base id (from --outputs, e.g. "123"); each point
        writes ``{output_base}__p{index}.dual_json`` -> ``__p{index}.json`` + full.
    """
    start = time.monotonic()

    def _elapsed() -> float:
        return time.monotonic() - start

    def _budget_reason(points_done: int) -> Optional[str]:
        """Which budget is spent, or None while there is room for another point.

        The two caps are reported separately: a run cut short by the clock looks
        identical to one that stopped on its own to anybody counting points, and
        the fix differs (raise the duration cap vs. nothing to fix).

        Only MEASURED points count. The saturation probe is a run of its own but
        is not one of them — it is a sizing aid, it writes to a "__satprobe" file
        the consumer's point glob never matches, and its result never enters an
        aggregate. Charging it to max_points made `--max-points 1` legal at the
        CLI yet produce ZERO measured points (the probe ate the only slot), i.e.
        exactly the "no points, no bracket" outcome the up-front validation
        exists to prevent. Its wall clock is still bounded: it runs after `start`,
        so `_elapsed` already contains it, and its own run is capped by
        `_remaining_seconds`.
        """
        if points_done >= cfg.max_points:
            return STOP_BUDGET_POINTS
        if _elapsed() >= cfg.max_total_seconds:
            return STOP_BUDGET_SECONDS
        return None

    def _budget_ok(points_done: int) -> bool:
        return _budget_reason(points_done) is None

    def _remaining_seconds() -> float:
        """Wall clock left in the total budget, floored at a usable minimum.

        Handed to each run as a ``max_duration`` constraint so the budget bounds
        the run from INSIDE a point, not merely between points. Without it
        max_total_seconds was advisory: it is only consulted by the loop
        predicates, so a point that stalls (an unresponsive server, an offered
        rate the server answers at a trickle) runs for as long as its
        max_requests takes — arbitrarily past the cap the caller set, with no
        second backstop, since auto-tune deliberately drops the global
        --max-seconds.

        The floor keeps a nearly-spent budget from requesting a 0-second run
        (guidellm would stop it before a single response landed, producing a
        point with no metrics); the loop predicate has already decided there IS
        room for this point.
        """
        return max(_MIN_POINT_SECONDS, cfg.max_total_seconds - _elapsed())

    # Slices of the overall bar consumed before the first measured point. The
    # saturation probe (below) is a full guidellm run of its own, so it claims
    # slice 0 and every measured point shifts one slice right.
    probe_slices = 0

    def _prep_progress(index: int, remaining_est: int) -> None:
        # Smooth (within-point) server progress: guidellm's per-point
        # on_benchmark_update callbacks fire continuously during a run, computing
        # overall = (run_index + point_fraction) / run_total. Setting run_index to
        # the count of completed points maps this point's live fraction onto its
        # slice of the overall bar, so the bar creeps instead of jumping once per
        # point. run_total is kept >= index + 2 so on_benchmark_complete can never
        # reach 100 (the ramp pushes the final 100 itself); remaining_est (points
        # left, incl. this one) shrinks as the sweep nears the end, keeping the bar
        # roughly proportional. The count is adaptive, so this is an estimate.
        if server_progress is None:
            return
        base = probe_slices + index
        server_progress.run_index = base
        server_progress.run_total = base + max(2, remaining_est)

    def _phase1_remaining(knob: float) -> int:
        # This point + remaining doublings to the bound in force + a few Phase-2
        # steps. Uses the effective bound (which the probe may have tightened) so
        # the estimate doesn't imply a long climb the ramp will never make.
        doublings = max(0, math.floor(math.log2(max(_bound() / max(knob, 1e-9), 1.0))))
        return 1 + doublings + 3

    def _phase2_remaining(lo: float, hi: float) -> int:
        # Steps left to close the (lo, hi) bracket. Both Phase-2 searches shrink it
        # by roughly half per point (SLA bisection exactly; the unimodal search by
        # whichever side survives the local comparison), so log2 estimates both.
        return max(1, math.ceil(math.log2(max(2.0, hi - lo))))

    async def _default_run_point(knob: float, index: int) -> Optional[PointMetrics]:
        # number = max(min_requests, round(knob * multiplier)) -> max_requests
        number = max(cfg.min_requests, round(knob * cfg.resolved_multiplier))
        # Per-point seed: increment by index so points differ, unless the
        # user pinned a fixed seed (seed_increment=False → same seed each point).
        seed = (
            cfg.random_seed_base + index if cfg.seed_increment else cfg.random_seed_base
        )

        local = dict(base_kwargs)
        local["random_seed"] = seed
        local["max_requests"] = number
        # Bound this point by whatever is left of the total budget, so a stalled
        # point cannot overrun it (see _remaining_seconds).
        local["max_seconds"] = _remaining_seconds()
        local["outputs"] = [f"{output_base}__p{index}.dual_json"]
        if cfg.axis == "concurrency":
            local["profile"] = "concurrent"
            local["rate"] = [float(int(knob))]  # streams = int(knob)
        else:
            local["profile"] = "constant"
            local["rate"] = [float(knob)]
            # In-flight requests on this (open-loop) axis are bounded only if the
            # caller passed --max-concurrency; base_kwargs carries it through to
            # the constant profile untouched (see scenario_builder).

        args = _build_args(local)
        # max_requests is carried in the scenario spec.constraints (built from
        # local["max_requests"]); no need to also pass it as a **constraints kwarg.
        report, _ = await benchmark_generative_text(
            args=args, progress=progress, console=console
        )
        if not report.benchmarks:
            return None
        return _normalize(report.benchmarks[0], knob, index)

    # Live runs use _default_run_point; tests inject a synthetic curve.
    _run_point = run_point if run_point is not None else _default_run_point

    points: list[PointMetrics] = []
    knob = float(cfg.lower_bound)
    prev_tps: Optional[float] = None
    prev_achieved: Optional[float] = None
    prev_knob: Optional[float] = None  # last knob whose tps still improved
    prev_prev_knob: Optional[float] = None  # the geometric point before prev_knob
    last_pass: Optional[float] = None
    first_fail: Optional[float] = None
    # Saturation bracket for the Phase-2 peak search: (lo, hi) where lo is the
    # last still-improving knob (holds the running-best tps) and hi is the knob
    # that stopped improving / overloaded. Left None when the sweep hit the upper
    # bound while still climbing (peak is at the bound; nothing above to search).
    sat_bracket: Optional[tuple[float, float]] = None
    # Probe-derived soft cap on the knob (see the saturation probe below). None =
    # no cap beyond the user's cfg.upper_bound. probe_ceiling keeps the raw
    # measurement the cap came from, so the ramp can tell whether it was wrong.
    saturation_bound: Optional[float] = None
    probe_ceiling: Optional[float] = None
    # How many times the soft cap doubled because the server outran the probe.
    probe_relaxed = 0

    def _bound() -> float:
        """Knob ceiling in force right now: the tighter of the user's hard
        upper_bound and the probe's soft saturation cap."""
        if saturation_bound is None:
            return float(cfg.upper_bound)
        return min(float(cfg.upper_bound), saturation_bound)

    # ── Saturation-bounded ceiling (rate axis + saturation target only) ──────
    # One throughput probe measures the server's ceiling request rate, which
    # becomes a SOFT cap on the ramp: past ~20% above the ceiling there is nothing
    # left to learn (achieved rate is pinned, so every further point reports the
    # same throughput with worse latency) while each doubling costs geometrically
    # MORE wall clock — per-point requests scale with the knob but the drain rate
    # is stuck at the ceiling, so a point at 32x the ceiling takes ~32x as long as
    # one at it. The plateau / can't-keep-up detectors normally stop the ramp one
    # point past saturation on their own; this cap bounds how expensive that one
    # overshoot point can get.
    #
    # Three deliberate properties:
    #
    #  * It is NOT written back onto cfg.upper_bound. That is the user's stated
    #    search range, and silently narrowing it makes the "not saturated — raise
    #    the bound and re-run" verdict nonsense: the bound WAS 1024, we stopped at
    #    38 of our own accord, and raising it changes nothing because the probe
    #    re-derives the cap every run.
    #  * It YIELDS to evidence — to ANY measured point, not just to one that beats
    #    the probe's own number. Reaching the cap means the growth criteria just
    #    passed on that point, so the cap doubles rather than truncating a curve
    #    the measurements say is still climbing. It only gets to stop the search
    #    when there is no measured point to consult (lower_bound already above the
    #    cap), where it reports `probe_bound` rather than `upper_bound`. Only
    #    cfg.upper_bound is a hard stop.
    #  * It does NOT lift the ramp's start. Starting at ceiling/4 saved about one
    #    point (~30s) at lower_bound=4 — every sub-saturation point costs the same
    #    ~30s window, since requests and drain rate scale together — and cost three
    #    things: the low-load end of the curve (the unloaded latency baseline the
    #    decision chart is read against), immunity to a bad ceiling estimate (the
    #    probe measures for ~2s; if it OVERestimates, a lifted start lands past the
    #    peak, the very first point plateaus, and Phase 1 exits with no bracket at
    #    all), and any chance of finding a peak that sits below the lifted start.
    #
    # The probe writes a "__satprobe" file (NOT matched by the manager's
    # "{id}__p{index}" glob) and is NOT counted as a measured point. Injectable so
    # tests can supply a ceiling.
    async def _default_probe() -> float:
        local = dict(base_kwargs)
        local["random_seed"] = cfg.random_seed_base
        local["max_requests"] = PROBE_MAX_REQUESTS
        # Two different jobs, so the smaller of the two wins: PROBE_SECONDS is the
        # measurement window (long enough to be a rate rather than one batch), and
        # the remaining budget is the safety cap — the probe runs BEFORE the first
        # point and is not counted as one, but it spends the same clock, so it must
        # not eat a budget the ramp then has to honor.
        local["max_seconds"] = min(PROBE_SECONDS, _remaining_seconds())
        local["outputs"] = [f"{output_base}__satprobe.dual_json"]
        # Offer load as fast as the server accepts it, capped at
        # PROBE_MAX_CONCURRENCY, and read back the achieved rps as the ceiling.
        # A concurrency cap is REQUIRED by guidellm's ThroughputProfileArgs, not
        # merely advisory — see the scenario builder.
        local["profile"] = "throughput"
        local["max_concurrency"] = PROBE_MAX_CONCURRENCY
        report, _ = await benchmark_generative_text(
            args=_build_args(local), progress=progress, console=console
        )
        if not report.benchmarks:
            return 0.0
        return _normalize(report.benchmarks[0], 0.0, 0).achieved_rate

    _probe = saturation_probe if saturation_probe is not None else _default_probe

    if (
        cfg.probe_saturation
        and cfg.target == "saturation"
        and cfg.axis == "rate"
        and _budget_ok(len(points))
    ):
        # The probe is a full guidellm run, so its on_benchmark_update callbacks
        # drive the same server bar as a measured point does — but it runs BEFORE
        # any _prep_progress call, i.e. with ServerBenchmarkerProgress' defaults
        # (run_index=0, run_total=1). That maps the probe's own 0..1 fraction onto
        # the WHOLE bar: progress hit 100% within seconds of start, and since the
        # server clamps progress monotonically it stayed pinned at 100% for the
        # rest of the ramp. Give the probe its own slice instead.
        if server_progress is not None:
            server_progress.run_index = 0
            server_progress.run_total = 1 + max(2, _phase1_remaining(knob))
        ceiling = await _probe()
        probe_slices = 1
        if ceiling and ceiling > 0:
            # Soft cap ~20% above the measured ceiling (the throughput peak sits
            # at/just below it). The ramp still starts at lower_bound.
            probe_ceiling = float(ceiling)
            saturation_bound = min(cfg.upper_bound, max(2.0, math.ceil(ceiling * 1.2)))

    # ── Phase 1: geometric bracket ──────────────────────────────────────────
    # `bracket_reason` is set at every exit from this loop; a fall-through means a
    # budget cap ended it (see below). Left None only if the loop never runs, which
    # a positive max_points/max_total_seconds makes impossible.
    bracket_reason: Optional[str] = None
    while _budget_ok(len(points)):
        _prep_progress(len(points), _phase1_remaining(knob))
        m = await _run_point(knob, len(points))
        if m is None:
            bracket_reason = STOP_POINT_FAILED
            break
        points.append(m)

        if cfg.target == "sla":
            if _passes_sla(m, cfg):
                last_pass = knob
                # SECONDARY termination: the SLA still holds, but the server has
                # stopped converting more load into more delivered work. Without
                # this the SLA branch has NO saturation criterion at all (only the
                # 95% success floor inside _passes_sla), so a threshold set loosely
                # enough never to break lets the ramp double all the way to
                # upper_bound through the saturated region — measuring points that
                # deliver LESS throughput at an order of magnitude worse latency,
                # and then reporting the top of the range as the "SLA capacity".
                # Observed on a real run: throughput peaked at 256 streams
                # (TTFT 142ms) yet the answer came back 1024 (TTFT 5809ms, 6% LESS
                # throughput) because a 10s TTFT threshold was never violated.
                #
                # Deliberately NOT treated as an SLA failure: writing first_fail
                # here would make Phase 2 bisect (last_pass, saturation_knob) and
                # return a SATURATION point dressed up as a latency boundary. The
                # honest statement is "the SLA never bound this sweep; capacity
                # did", which the aggregation side reports as `sla_not_binding`
                # (and takes the throughput peak as the operating point).
                if _capacity_saturated(m, prev_tps, prev_achieved, cfg.axis):
                    bracket_reason = STOP_CAPACITY_PLATEAU
                    break
                if knob >= cfg.upper_bound:
                    bracket_reason = STOP_UPPER_BOUND
                    break
                prev_tps = m.output_tps
                prev_achieved = m.achieved_rate
                # Clamped for the same reason as the saturation branch: an
                # unclamped doubling would measure a load OUTSIDE the range the
                # user asked for.
                knob = min(knob * 2, float(cfg.upper_bound))
            else:
                first_fail = knob  # bracket = (last_pass, first_fail)
                bracket_reason = STOP_SLA_FAILED
                break
        else:  # saturation (peak throughput)
            # Here saturation IS the stop criterion (the peak is at it), so the
            # shared detector decides the bracket. See _capacity_saturated for why
            # both signals compare CONSECUTIVE points instead of achieved-vs-offered.
            overloaded = m.success < SUCCESS_FLOOR
            if overloaded or _capacity_saturated(m, prev_tps, prev_achieved, cfg.axis):
                bracket_reason = (
                    STOP_OVERLOADED if overloaded else STOP_CAPACITY_PLATEAU
                )
                # Throughput stopped climbing (or the server overloaded). Under a
                # unimodal assumption the argmax lies between the highest sampled
                # point's LEFT and RIGHT geometric neighbours. The highest so far
                # is prev_knob; its left neighbour is prev_prev_knob (the ramp
                # doubles, so prev_knob/2 when there is none). Bracketing only
                # (prev_knob, knob) would drop the (prev_prev, prev) half and miss
                # a peak that sits there — e.g. a 16->32 jump that overshoots a
                # true peak at ~24. So the bracket spans (prev_prev, knob).
                #
                # The floor is CLAMPED to lower_bound: [lower_bound, upper_bound]
                # is the range the user asked for, and the ramp never measures
                # outside it — symmetrically with the upper end, where hitting
                # upper_bound while still climbing stops the sweep and reports
                # `not_saturated` instead of quietly probing higher. When the
                # server turns out to be saturated at lower_bound already, the
                # answer is to SAY SO (validity `saturated_at_lower_bound`, which
                # carries the measured ceiling) and let the user lower the range —
                # not to search a region they did not ask about.
                if prev_knob is not None:
                    lo = (
                        prev_prev_knob
                        if prev_prev_knob is not None
                        else max(float(cfg.lower_bound), prev_knob / 2.0)
                    )
                    sat_bracket = (lo, knob)
                break
            prev_tps = m.output_tps
            prev_achieved = m.achieved_rate
            prev_prev_knob = prev_knob
            prev_knob = knob
            bound = _bound()
            if knob >= bound:
                # Reached a bound. Which one, and does it get to end the search?
                #
                # THE SOFT CAP DOES NOT END A MULTI-POINT SEARCH. Reaching this
                # line means every measured criterion above just passed on THIS
                # point — throughput grew, the server kept up, nothing overloaded —
                # so the measurements say the run is still climbing. A 10s burst
                # estimate does not get to overrule a 30s steady-state measurement
                # of the very quantity it was estimating.
                #
                # It also cannot be argued with on screen. The probe is kept out of
                # the point grid on purpose (its numbers are burst artefacts), so a
                # run that stops here leaves a chart still rising at its last point
                # with no visible reason — it reads as a bug, and the coverage
                # verdict then tells the user to raise a bound that was never the
                # constraint. Measured: a run stopped at 31 req/s because achieved
                # 25.87 missed ceiling*1.05 = 26.88 by one percent. A manual ladder
                # over the same deployment confirmed 31 WAS the peak — and proved
                # it with a single point at 32 (-2.6% throughput, 2x the TTFT).
                # One measured point past the top is what makes a peak a peak.
                #
                # Cost of letting it run: bounded by construction. The previous
                # point cleared the growth criteria, so it sits at or below
                # saturation, and geometric doubling puts the next one at most 2x
                # there — the "32x the ceiling" point the cap was written to
                # prevent cannot happen while these criteria work, because they
                # fire on the FIRST point that stops growing.
                #
                # The first point is the exception, and the reason the cap still
                # exists: with nothing measured before it, none of the criteria can
                # fire (they all compare against a previous point), so doubling
                # blind is unbounded — lower_bound 512 against a 25 rps server puts
                # the next point at 1024, i.e. 1024*30 requests draining at 25 rps,
                # about twenty minutes. There the probe is the only evidence there
                # is, so there it still decides.
                soft_cap = saturation_bound is not None and bound < cfg.upper_bound
                still_keeping_up = len(points) > 1 or (
                    probe_ceiling is not None
                    and m.achieved_rate > probe_ceiling * (1.0 + PLATEAU_GAIN)
                )
                if soft_cap and still_keeping_up:
                    saturation_bound = min(
                        float(cfg.upper_bound), saturation_bound * 2.0
                    )
                    probe_relaxed += 1
                    bound = _bound()
                else:
                    # Stop on the bound. No bracket when lower_bound itself is
                    # already at/above the bound (a single measured point): the
                    # optimum lies BELOW the range the user gave, which is reported
                    # via validity rather than searched for — see the comment in the
                    # plateau branch above.
                    #
                    # Named apart, because the advice differs: `upper_bound` means
                    # the user's range ended the search and raising it would help,
                    # while `probe_bound` means OUR estimate did and raising the
                    # range would change nothing (the probe re-derives the cap every
                    # run). Reporting both as `upper_bound` is what produced the
                    # "raise your 1024" verdict for a run that stopped at 31.
                    bracket_reason = STOP_PROBE_BOUND if soft_cap else STOP_UPPER_BOUND
                    break
            # Clamp the doubling to the bound instead of stepping over it. The
            # bound check above runs AFTER a point is measured, so an unclamped
            # `knob *= 2` lets the LAST Phase-1 point land anywhere up to (just
            # under) 2x the bound — e.g. bound 38 from a measured ceiling of 31
            # still probed 56. That breaks the contract upper_bound states, and the
            # overshoot point is the least informative one in the sweep: it sits
            # far past saturation, so it only measures how badly the server
            # degrades, at the highest per-point cost of any point.
            knob = min(knob * 2, bound)

    if bracket_reason is None:
        # The loop predicate went false: one of the two budgets is spent.
        bracket_reason = _budget_reason(len(points)) or STOP_BUDGET_POINTS
    # Phase 2 may end for its own reason; until it runs, the bracket's reason is
    # also the ramp's.
    stop_reason = bracket_reason

    # ── Phase 2: bisection on the SLA bracket (SLA target only) ─────────────
    # Bisection is valid HERE and only here: "passes the SLA" is monotone in the
    # knob, so a single pass/fail bracket collapses to the boundary. The saturation
    # target gets a different Phase 2 (below) because throughput is not monotone.
    if cfg.target == "sla" and last_pass is not None and first_fail is not None:
        lo, hi = last_pass, first_fail
        stop_reason = STOP_CONVERGED
        # The budget is consulted only AFTER a next point has been established, so
        # it is blamed only when it actually prevented a measurement. Testing it in
        # the loop predicate instead relabelled a search that had FINISHED as
        # `budget_points` whenever the point count happened to land on the cap:
        # `--max-points 6` and `--max-points 12` produced the same six points and
        # the same boundary, yet reported "truncated, raise the cap" vs "converged".
        while hi - lo > 1:
            mid = math.floor((lo + hi) / 2)
            if mid <= lo:
                break  # nothing left on the integer grid -> converged
            if (spent := _budget_reason(len(points))) is not None:
                stop_reason = spent
                break
            _prep_progress(len(points), _phase2_remaining(lo, hi))
            m = await _run_point(float(mid), len(points))
            if m is None:
                stop_reason = STOP_POINT_FAILED
                break
            points.append(m)
            if _passes_sla(m, cfg):
                lo = float(mid)
                last_pass = lo
            else:
                hi = float(mid)
                first_fail = hi

    # ── Phase 2: peak-seeking unimodal search (saturation target only) ───────
    # The goal is the single knob that MAXIMISES throughput, not a dense curve.
    # Phase 1's geometric bracket (prev_prev, knob) is dropped onto a three-point
    # search that keeps a < b < c with f(b) the running max, then probes the
    # MIDPOINT of the wider side and shrinks toward the peak (a golden-section /
    # ternary hybrid on the integer knob, reusing Phase-1 points).
    #
    # This replaces evalscope's (and our former) "bisect and only walk right when
    # a point beats the running best" scheme: that judged direction by comparing
    # to the GLOBAL best, which is only correct when the peak sits at the bracket's
    # LEFT end. Here direction comes from the LOCAL comparison f(x) vs f(b), so a
    # peak anywhere inside (a, c) — including the (prev_prev, prev) half the old
    # bracket omitted — is found. Ties keep the incumbent b (the cheaper/lower
    # knob at equal throughput). The final peak is argmax over ALL measured points.
    if cfg.target == "saturation" and sat_bracket is not None:
        a, c = sat_bracket
        b = prev_knob if prev_knob is not None else a
        tps_by_knob = {p.knob: p.output_tps for p in points}
        fb = tps_by_knob.get(b, prev_tps if prev_tps is not None else -1.0)
        measured = set(tps_by_knob)

        async def _probe_point(x: float) -> Optional[float]:
            _prep_progress(len(points), _phase2_remaining(a, c))
            m = await _run_point(x, len(points))
            if m is None:
                return None
            points.append(m)
            tps_by_knob[x] = m.output_tps
            measured.add(x)
            return m.output_tps

        stop_reason = STOP_CONVERGED
        # Budget checked after a next point is established — see the SLA bisection
        # above for why the loop predicate is the wrong place for it. Here it matters
        # twice over, because this search can also run out of INTEGER GRID while the
        # bracket is still 2 wide, which is convergence and not a budget problem.
        while (c - a) > 1:
            # Midpoint of each side; must be a fresh interior integer knob.
            xl = math.floor((a + b) / 2)
            xr = math.ceil((b + c) / 2)
            left_ok = a < xl < b and float(xl) not in measured
            right_ok = b < xr < c and float(xr) not in measured
            if left_ok and right_ok:
                use_left = (b - a) >= (c - b)  # probe the wider side first
            elif left_ok:
                use_left = True
            elif right_ok:
                use_left = False
            else:
                break  # integer grid exhausted on both sides -> converged
            if (spent := _budget_reason(len(points))) is not None:
                stop_reason = spent
                break
            x = float(xl if use_left else xr)
            fx = await _probe_point(x)
            if fx is None:
                stop_reason = STOP_POINT_FAILED
                break
            if use_left:
                if fx > fb:
                    c, b, fb = b, x, fx  # higher on the left -> peak in (a, b)
                else:
                    a = x  # not higher -> peak in (x, c)
            else:
                if fx > fb:
                    a, b, fb = b, x, fx  # higher on the right -> peak in (b, c)
                else:
                    c = x  # not higher -> peak in (a, x)

    # Converged: push progress to 100 once (server clamps monotonically), then
    # close the session. server_progress IS inside the per-point runs' progress
    # chain (that is what makes the bar creep within a point), so every point's
    # on_finalize already closed its session and _ensure_session reopened it on the
    # next write; the ramp still owns the FINAL 100 + close, because no single
    # point knows it is the last one.
    if server_progress is not None:
        try:
            await server_progress._update_progress(100.0)
        except Exception:
            pass
        try:
            await server_progress.on_finalize()
        except Exception:
            pass

    return RampOutcome(
        points=points,
        bracket_reason=bracket_reason,
        stop_reason=stop_reason,
        target=cfg.target,
        axis=cfg.axis,
        stopped_at=max((p.knob for p in points), default=None),
        lower_bound=float(cfg.lower_bound),
        upper_bound=float(cfg.upper_bound),
        max_points=int(cfg.max_points),
        max_total_seconds=float(cfg.max_total_seconds),
        elapsed_seconds=_elapsed(),
        # Reported even when no boundary was found: first_fail=None is the fact
        # that the SLA number is a floor ("still met at the top we reached"), not
        # an edge ("breaks just above").
        sla_bracket=((last_pass, first_fail) if cfg.target == "sla" else None),
        probe_ceiling=probe_ceiling,
        probe_relaxed=probe_relaxed,
        probe_bound=saturation_bound,
    )


def _build_args(local_kwargs: dict[str, Any]) -> BenchmarkScenario:
    """Build a guidellm ``BenchmarkScenario`` for one ramp point.

    Mirrors main.py's ``_run_once``: delegates to the shared spec builder, which
    maps the flat CLI kwargs onto ``spec`` (backend/profile/data/seed/outputs/
    constraints) and normalizes data sources (e.g. ShareGPT -> guidellm jsonl).
    The per-point profile ({kind: concurrent, streams:[N]} for the concurrency
    axis, {kind: constant, rate:[R]} for the rate axis), the per-point seed, and
    the ``max_requests`` constraint are already set on ``local_kwargs`` by the
    caller.
    """
    return build_scenario_args(local_kwargs)
