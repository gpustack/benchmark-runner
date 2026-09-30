import hashlib
import json
import os
import tempfile
from pathlib import Path
from benchmark_runner.sharegpt_to_guidellm import convert_sharegpt_to_guidellm


class ShareGPTAdapter:
    def supports(self, source: str) -> bool:
        return (
            source.endswith(".json") or source.endswith(".jsonl")
        ) and "sharegpt" in source.lower()

    def prepare(
        self,
        source: str,
        *,
        tokenizer: str,
        max_items: int | None,
        input_min: int | None = None,
        input_max: int | None = None,
        output_tokens: int | None = None,
    ) -> list[str]:
        source_path = Path(source)
        if max_items is not None:
            max_items = int(max_items * 1.2)  # Convert more

        stat = source_path.stat()
        cache_key = hashlib.sha256(
            json.dumps(
                [
                    str(source_path.resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                    tokenizer,
                    input_min,
                    input_max,
                    output_tokens,
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        output = source_path.parent / f"converted_{source_path.stem}_{cache_key}.jsonl"
        metadata = output.with_suffix(".meta.json")
        if output.exists() and metadata.exists():
            try:
                cache = json.loads(metadata.read_text(encoding="utf-8"))
                if cache["complete"] or (
                    max_items is not None and cache["written"] >= max_items
                ):
                    return [str(output)]
            except (OSError, ValueError, KeyError):
                pass

        fd, temp_name = tempfile.mkstemp(prefix="sharegpt_", dir=source_path.parent)
        os.close(fd)
        temp_output = Path(temp_name)
        try:
            stats = convert_sharegpt_to_guidellm(
                input_file=source_path,
                output_file=temp_output,
                tokenizer_name=tokenizer,
                max_items=max_items,
                input_min=input_min,
                input_max=input_max,
                output_tokens=output_tokens,
            )
            os.replace(temp_output, output)
            metadata.write_text(
                json.dumps(
                    {"written": stats["written"], "complete": stats["complete"]}
                ),
                encoding="utf-8",
            )
        finally:
            temp_output.unlink(missing_ok=True)
        return [str(output)]


dataset_adapters = [
    ShareGPTAdapter(),
]


def prepare_datasets(
    data: list[str],
    *,
    tokenizer: str,
    max_items: int | None,
    input_min: int | None = None,
    input_max: int | None = None,
    output_tokens: int | None = None,
) -> list[str]:
    prepared = []

    for source in data:
        for adapter in dataset_adapters:
            if adapter.supports(source):
                prepared.extend(
                    adapter.prepare(
                        source,
                        tokenizer=tokenizer,
                        max_items=max_items,
                        input_min=input_min,
                        input_max=input_max,
                        output_tokens=output_tokens,
                    )
                )
                break
        else:
            prepared.append(source)

    return prepared
