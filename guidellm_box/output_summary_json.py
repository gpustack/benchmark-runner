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

    @classmethod
    def validated_kwargs(
        cls, output_path: str | Path | None, **_kwargs
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
            validated["output_path"] = (
                Path(output_path) if not isinstance(output_path, Path) else output_path
            )
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

        # Exclude specified fields from the report
        model_dict = report.model_dump(exclude=self.EXCLUDE_FIELDS)
        save_str = json.dumps(model_dict, indent=4)

        with output_path.open("w", encoding="utf-8") as file:
            file.write(save_str)

        return output_path
