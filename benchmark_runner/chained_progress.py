import asyncio
from typing import Sequence
from guidellm.benchmark.progress import (
    BenchmarkerProgress,
    GenerativeBenchmarkAccumulator,
    GenerativeBenchmark,
    Profile,
    SchedulingStrategy,
    SchedulerState,
)


class ChainedBenchmarkerProgress(
    BenchmarkerProgress[GenerativeBenchmarkAccumulator, GenerativeBenchmark]
):
    def __init__(self, progresses: Sequence[BenchmarkerProgress]):
        super().__init__()
        self.progresses = progresses

    async def on_initialize(self, profile: Profile):
        await self._gather("on_initialize", profile)

    async def on_benchmark_start(self, strategy: SchedulingStrategy):
        await self._gather("on_benchmark_start", strategy)

    async def on_benchmark_update(
        self,
        accumulator: GenerativeBenchmarkAccumulator,
        scheduler_state: SchedulerState,
    ):
        await self._gather("on_benchmark_update", accumulator, scheduler_state)

    async def on_benchmark_complete(self, benchmark: GenerativeBenchmark):
        await self._gather("on_benchmark_complete", benchmark)

    async def on_finalize(self):
        await self._gather("on_finalize")

    async def _gather(self, method, *args):
        await asyncio.gather(*(getattr(p, method)(*args) for p in self.progresses))
