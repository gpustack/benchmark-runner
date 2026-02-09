"""
Output handler for serializing generative benchmark reports to JSON.

This module implements a file-based output for saving benchmark results in JSON format.
It extends GenerativeBenchmarkerOutput and supports both directory and explicit file path
output, automatically creating parent directories as needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field


from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport

__all__ = ["GenerativeBenchmarkerSummaryJson"]


@GenerativeBenchmarkerOutput.register("summary_json")
class GenerativeBenchmarkerSummaryJson(GenerativeBenchmarkerOutput):
    """
    Output handler for serializing benchmark reports to JSON files.

    This class saves generative benchmark reports to a specified file or directory in JSON format.
    If a directory is provided, a default filename is used. Certain fields can be excluded from the output.

    Example:
        output = GenerativeBenchmarkerSummaryJson(output_path="/path/to/output.json")
        result_path = await output.finalize(report)
    """

    DEFAULT_FILE: ClassVar[str] = "benchmarks.json"
    EXCLUDE_FIELDS: ClassVar[dict[str, dict[str, Any]]] = {
        "benchmarks": {
            "__all__": {
                "requests": ...,
                "metrics": {"audio", "image", "video"},
            }
        }
    }

    output_path: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Directory or file path for saving the serialized report.",
    )
    error_limit: int | None = Field(
        default=20,
        description="Maximum number of errored requests to include.",
    )
    incomplete_limit: int | None = Field(
        default=20,
        description="Maximum number of incomplete requests to include.",
    )

    @classmethod
    def validated_kwargs(
        cls,
        output_path: str | Path | None,
        error_limit: int | None = None,
        incomplete_limit: int | None = None,
        **_kwargs,
    ) -> dict[str, Any]:
        """
        Validate and normalize keyword arguments for output path.

        Args:
            output_path: Directory or file path for serialization output.
            _kwargs: Additional keyword arguments (ignored).
        Returns:
            Dictionary of validated keyword arguments for class initialization.
        """
        validated: dict[str, Any] = {}
        if output_path is not None:
            output_path = (
                output_path if isinstance(output_path, Path) else Path(output_path)
            )
            if output_path.suffix.lower() == ".summary_json":
                output_path = output_path.with_suffix(".json")
            validated["output_path"] = output_path

        if error_limit is not None:
            validated["error_limit"] = error_limit
        if incomplete_limit is not None:
            validated["incomplete_limit"] = incomplete_limit
        return validated

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Serialize and save the benchmark report to the configured output path in JSON format.

        Args:
            report: The generative benchmarks report to serialize.
        Returns:
            Path to the saved report file.
        """
        output_path = self.output_path
        if output_path.is_dir():
            output_path = output_path / self.DEFAULT_FILE

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Exclude specified fields from the report, but keep a small error sample
        full_dict = report.model_dump()
        summary_dict = report.model_dump(exclude=self.EXCLUDE_FIELDS)
        self._attach_error_samples(summary_dict, full_dict)
        save_str = json.dumps(summary_dict, indent=4)

        with output_path.open("w", encoding="utf-8") as file:
            file.write(save_str)

        return output_path

    def _attach_error_samples(
        self, summary_dict: dict[str, Any], full_dict: dict[str, Any]
    ) -> None:
        summary_benchmarks = summary_dict.get("benchmarks") or []
        full_benchmarks = full_dict.get("benchmarks") or []

        for idx, benchmark in enumerate(summary_benchmarks):
            full_benchmark = full_benchmarks[idx] if idx < len(full_benchmarks) else {}
            requests = (full_benchmark or {}).get("requests") or {}

            errored = self._limit_items(requests.get("errored") or [], self.error_limit)
            incomplete = self._limit_items(
                requests.get("incomplete") or [], self.incomplete_limit
            )

            if errored or incomplete:
                benchmark["requests_truncated"] = {}
                if errored:
                    benchmark["requests_truncated"]["errored"] = errored
                if incomplete:
                    benchmark["requests_truncated"]["incomplete"] = incomplete

    @staticmethod
    def _limit_items(items: list[Any], limit: int | None) -> list[Any]:
        if limit is None:
            return list(items)
        return list(items)[: max(limit, 0)]
