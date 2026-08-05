"""CLI dispatch guards for the `benchmark run` command.

These cover the validation/wiring that happens in ``main.run`` *before* any
benchmark is executed, so they need no live server: the mutual-exclusion guard,
the per-stage ``rate`` requirement, that an explicit ``--random-seed 0`` survives
into the ramp config (a ``... or 42`` would silently drop it), and that the ramp
outcome lands on disk next to the point files.
"""

import json

from click.testing import CliRunner

import benchmark_runner.main as main
from benchmark_runner.auto_tune import RampOutcome
from benchmark_runner.main import cli


def test_auto_tune_rejects_explicit_profile():
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--profile", "constant"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_auto_tune_rejects_stages():
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--stages", '[{"rate": 2}]']
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_stages_rejects_explicit_profile():
    # The third pair, which used to slip through: the stages branch overwrites
    # profile with constant/concurrent per --axis, so `--stages ... --profile
    # throughput` exited 0 while running a constant-rate sweep and saying nothing
    # about the profile it discarded. README documents all three as exclusive.
    result = CliRunner().invoke(
        cli,
        ["benchmark", "run", "--stages", '[{"rate": 2}]', "--profile", "throughput"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert "--profile" in result.output


def test_all_three_modes_at_once_is_rejected():
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--stages",
            '[{"rate": 2}]',
            "--profile",
            "throughput",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


class TestAutoTuneRangeIsValidated:
    """A bad range/budget must fail at the CLI, not silently waste a benchmark.

    None of these are rejected anywhere else: click's ``float``/``int`` types accept
    them and the gpustack API stores both bounds as unconstrained
    ``Optional[float]``, so a bad value travels all the way into the ramp — where it
    produces a full run's worth of useless points and a reason visible only by
    reading the ramp's arithmetic.
    """

    def _invoke(self, *args):
        return CliRunner().invoke(cli, ["benchmark", "run", "--auto-tune", *args])

    def test_zero_lower_bound_is_rejected(self):
        # knob = min(0 * 2, bound) == 0 forever: every one of max_points points
        # would measure the same invalid zero load.
        result = self._invoke("--lower-bound", "0")
        assert result.exit_code != 0
        assert "--lower-bound" in result.output
        assert "greater than 0" in result.output
        assert "Traceback" not in result.output

    def test_negative_lower_bound_is_rejected(self):
        result = self._invoke("--lower-bound", "-4")
        assert result.exit_code != 0
        assert "greater than 0" in result.output

    def test_inverted_range_is_rejected(self):
        # The ramp documents [lower, upper] as hard at BOTH ends, then measures its
        # first point at lower_bound -- outside the range when the range is
        # inverted.
        result = self._invoke("--lower-bound", "100", "--upper-bound", "50")
        assert result.exit_code != 0
        assert "must not exceed" in result.output
        assert "Traceback" not in result.output

    def test_zero_max_points_is_rejected(self):
        # Phase 1's predicate is false on entry -> zero points, no bracket, no
        # answer.
        result = self._invoke("--max-points", "0")
        assert result.exit_code != 0
        assert "greater than 0" in result.output

    def test_zero_budget_is_rejected(self):
        result = self._invoke("--max-total-seconds", "0")
        assert result.exit_code != 0
        assert "greater than 0" in result.output

    def test_zero_multiplier_is_rejected(self):
        result = self._invoke("--multiplier", "0")
        assert result.exit_code != 0
        assert "--multiplier" in result.output

    def test_equal_bounds_are_allowed(self, monkeypatch, tmp_path):
        # A single-point probe at one exact load is a legitimate request.
        captured = {}

        async def fake_run_ramp(cfg, **kwargs):
            captured["cfg"] = cfg
            return _fake_outcome(cfg)

        monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--auto-tune",
                "--lower-bound",
                "8",
                "--upper-bound",
                "8",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].lower_bound == 8.0

    def test_the_range_is_not_checked_without_auto_tune(self):
        # These options are auto-tune's; a non-ramp run that happens to carry a
        # leftover value must not be blocked by a check that cannot apply to it.
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--stages",
                '[{"max_requests": 10}]',
                "--lower-bound",
                "0",
            ],
        )
        # Still fails, but on the stage's missing 'rate' -- not on the bound.
        assert "greater than 0" not in result.output


def test_stage_missing_rate_is_a_friendly_error():
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--stages", '[{"max_requests": 10}]']
    )
    assert result.exit_code != 0
    # BadParameter names the option and the missing key, not a raw KeyError.
    assert "rate" in result.output
    assert "Traceback" not in result.output


class TestStageSeedPolicy:
    """Stages hold the data FIXED by default; --random-seed opts into variation.

    This is a deliberate difference from the ramp, not an oversight, so it is
    pinned here — the asymmetry looks like a bug on a quick read and would
    otherwise get "fixed" into ramp behavior, silently changing what every
    existing stage list measures.

    Why they differ: the ramp's points sit a doubling apart and run back to back,
    so replaying prompts lets the server answer later points from its prefix/KV
    cache and inflates the curve the peak is read off — varying the seed there is
    a correctness requirement. A stage list is user-picked load points, usually
    written to vary ONE thing (offered load), which argues for identical prompts.

    Mechanism worth knowing: ``set_if_not_default`` drops --random-seed while it
    sits at its 42 default, so "absent" means "not requested". Passing the option
    is the opt-in, even at the same value 42.
    """

    def _stage_seeds(self, monkeypatch, *extra):
        """Seeds the three stages actually hand to guidellm."""
        seeds = []

        def spy(local_kwargs):
            seeds.append(local_kwargs.get("random_seed"))
            return object()  # stands in for the BenchmarkScenario

        async def noop(*a, **kw):
            return None, None

        monkeypatch.setattr(main, "build_scenario_args", spy)
        monkeypatch.setattr(main, "benchmark_generative_text", noop)
        result = CliRunner().invoke(
            cli,
            [
                "benchmark",
                "run",
                "--stages",
                '[{"rate": 2}, {"rate": 4}, {"rate": 8}]',
                "--data",
                "prompt_tokens=8,output_tokens=8",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(seeds) == 3, f"expected 3 stages, saw {seeds}"
        return seeds

    def test_no_seed_by_default_so_stages_are_comparable(self, monkeypatch):
        # Every stage replays the same synthetic prompts; only the load differs.
        assert self._stage_seeds(monkeypatch) == [None, None, None]

    def test_passing_the_option_opts_into_per_stage_variation(self, monkeypatch):
        assert self._stage_seeds(monkeypatch, "--random-seed", "7") == [7, 8, 9]

    def test_the_default_value_passed_explicitly_still_opts_in(self, monkeypatch):
        # 42 IS the default, but typing it is a request -- this is the one that
        # reads as surprising, hence the test.
        assert self._stage_seeds(monkeypatch, "--random-seed", "42") == [42, 43, 44]

    def test_no_seed_increment_pins_one_seed(self, monkeypatch):
        seeds = self._stage_seeds(
            monkeypatch, "--random-seed", "7", "--no-seed-increment"
        )
        assert seeds == [7, 7, 7]


def test_auto_tune_preserves_explicit_seed_zero(monkeypatch, tmp_path):
    # --output-dir is passed only to keep the ramp sidecar out of the repo root:
    # with no output dir the writer falls back to cwd (gpustack always passes one).
    captured = {}

    async def fake_run_ramp(cfg, **kwargs):
        captured["cfg"] = cfg
        return RampOutcome(
            points=[],
            bracket_reason="point_failed",
            stop_reason="point_failed",
            target=cfg.target,
            axis=cfg.axis,
            stopped_at=None,
            lower_bound=cfg.lower_bound,
            upper_bound=cfg.upper_bound,
            max_points=cfg.max_points,
            max_total_seconds=cfg.max_total_seconds,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--random-seed",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["cfg"].random_seed_base == 0


def _fake_outcome(cfg, **facts):
    """A RampOutcome with no points — enough to exercise the sidecar writer."""
    base = dict(
        points=[],
        bracket_reason="capacity_plateau",
        stop_reason="converged",
        target=cfg.target,
        axis=cfg.axis,
        stopped_at=256.0,
        lower_bound=cfg.lower_bound,
        upper_bound=cfg.upper_bound,
        max_points=cfg.max_points,
        max_total_seconds=cfg.max_total_seconds,
        elapsed_seconds=54.321,
        sla_bracket=(256.0, None),
    )
    base.update(facts)
    return RampOutcome(**base)


def test_ramp_outcome_is_written_beside_the_point_files(monkeypatch, tmp_path):
    # The consumer globs "{base}__p{index}.json" for points; the outcome goes to
    # "{base}__ramp.json", which that glob must not pick up.
    async def fake_run_ramp(cfg, **kwargs):
        return _fake_outcome(cfg)

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "run",
            "--auto-tune",
            "--output-dir",
            str(tmp_path),
            "--outputs",
            "42.dual_json",
        ],
    )
    assert result.exit_code == 0, result.output

    path = tmp_path / "42__ramp.json"
    assert path.exists(), sorted(p.name for p in tmp_path.iterdir())
    facts = json.loads(path.read_text())
    assert facts["bracket_reason"] == "capacity_plateau"
    assert facts["stop_reason"] == "converged"
    assert facts["stopped_at"] == 256.0
    assert facts["sla_bracket"] == [256.0, None]
    assert facts["version"] == 1
    # Not matched by the point glob.
    assert "__p" not in path.name.replace("__ramp", "")


def test_a_sidecar_that_cannot_be_written_does_not_fail_the_run(monkeypatch, tmp_path):
    # The points ARE the measurement; a diagnostic file must never take a run down.
    async def fake_run_ramp(cfg, **kwargs):
        return _fake_outcome(cfg)

    def boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(main, "run_ramp", fake_run_ramp)
    monkeypatch.setattr(main.Path, "write_text", boom)
    result = CliRunner().invoke(
        cli, ["benchmark", "run", "--auto-tune", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Could not write ramp outcome" in result.output
