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
    ) -> list[str]:
        source_path = Path(source)
        output = source_path.parent / f"converted_{source_path.stem}.jsonl"

        if output.exists():
            return [str(output)]

        if max_items is not None:
            max_items = int(max_items * 1.2)  # Convert more

        convert_sharegpt_to_guidellm(
            input_file=Path(source),
            output_file=output,
            tokenizer_name=tokenizer,
            max_items=max_items,
        )
        return [str(output)]


dataset_adapters = [
    ShareGPTAdapter(),
]


def prepare_datasets(
    data: list[str],
    *,
    tokenizer: str,
    max_items: int | None,
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
                    )
                )
                break
        else:
            prepared.append(source)

    return prepared
