"""
Output handler for serializing generative benchmark reports to JSON (both summary and full).

This module implements a dual-output JSON handler that saves both:
1. Summary JSON - Excludes large fields like individual requests and detailed metrics
2. Full JSON - Contains complete benchmark data including all requests

Both files are saved to the same directory with clear naming conventions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field


from guidellm.benchmark.outputs.output import GenerativeBenchmarkerOutput
from guidellm.benchmark.schemas import GenerativeBenchmarksReport

__all__ = ["GenerativeBenchmarkerDualJson", "AutoMarshalJSONEncoder"]


class AutoMarshalJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder with auto-marshal support (similar to Golang's MarshalJSON).

    This encoder automatically checks if objects have __class_json__() or __json__()
    methods and calls them for serialization, providing a Golang-like interface for
    custom JSON marshaling in Python.
    """

    def default(self, o):
        """
        Override default serialization for non-serializable objects.

        Args:
            o: Object to serialize.

        Returns:
            Serializable representation of the object.
        """
        # Check if the object has a __class_json__ method (for class objects)
        if isinstance(o, type) and hasattr(o, "__class_json__"):
            return o.__class_json__()

        # Check if the object has a __json__ method (for instances)
        if hasattr(o, "__json__") and callable(getattr(o, "__json__")):
            return o.__json__()

        # Handle class/type objects (like response handler classes)
        if isinstance(o, type):
            # Try to find the registered name for this handler class
            from guidellm.backends.response_handlers import (
                GenerationResponseHandlerFactory,
            )

            registry = GenerationResponseHandlerFactory.registry or {}
            class_to_name = {v: k for k, v in registry.items()}

            handler_name = class_to_name.get(o)
            if handler_name:
                return handler_name
            else:
                # Fallback: use the full class name
                return f"{o.__module__}.{o.__name__}"

        # Let the base class handle other types or raise TypeError
        return super().default(o)


@GenerativeBenchmarkerOutput.register("dual_json")
class GenerativeBenchmarkerDualJson(GenerativeBenchmarkerOutput):
    """
    Output handler for serializing benchmark reports to both summary and full JSON files.

    This class saves two JSON files:
    1. Summary JSON - Excludes large fields (requests, detailed metrics) for quick overview
    2. Full JSON - Contains complete benchmark data including all requests and metrics

    If a directory is provided, default filenames are used. If a file path is provided,
    the summary uses that path and the full version adds a suffix.

    Example:
        # Using directory
        output = GenerativeBenchmarkerDualJson(output_path="/path/to/dir")
        # Creates: /path/to/dir/benchmarks.json (summary)
        #          /path/to/dir/benchmarks.full.json (full)

        # Using file path
        output = GenerativeBenchmarkerDualJson(output_path="/path/to/results.json")
        # Creates: /path/to/results.json (summary)
        #          /path/to/results.full.json (full)
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
            if output_path.suffix.lower() == ".dual_json":
                output_path = output_path.with_suffix(".json")
            validated["output_path"] = output_path

        if error_limit is not None:
            validated["error_limit"] = error_limit
        if incomplete_limit is not None:
            validated["incomplete_limit"] = incomplete_limit
        return validated

    async def finalize(self, report: GenerativeBenchmarksReport) -> Path:
        """
        Serialize and save the benchmark report to both summary and full JSON files.

        Args:
            report: The generative benchmarks report to serialize.
        Returns:
            Path to the saved summary report file.
        """
        # Determine output paths
        summary_path = self.output_path
        if summary_path.is_dir():
            summary_path = summary_path / self.DEFAULT_FILE

        # Create full path by inserting ".full" before the extension
        full_path = (
            summary_path.parent / f"{summary_path.stem}.full{summary_path.suffix}"
        )

        # Ensure parent directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        full_dict = report.model_dump()
        summary_dict = report.model_dump(exclude=self.EXCLUDE_FIELDS)
        self._attach_error_samples(summary_dict, full_dict)

        # Use custom encoder to handle response handler classes
        encoder_cls = AutoMarshalJSONEncoder

        # Save summary JSON
        summary_str = json.dumps(summary_dict, indent=4, cls=encoder_cls)
        with summary_path.open("w", encoding="utf-8") as file:
            file.write(summary_str)

        # Save full JSON
        full_str = json.dumps(full_dict, indent=4, cls=encoder_cls)
        with full_path.open("w", encoding="utf-8") as file:
            file.write(full_str)

        return summary_path

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
