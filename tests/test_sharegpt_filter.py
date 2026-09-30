import json
from pathlib import Path

import pytest

from benchmark_runner import sharegpt_to_guidellm
from benchmark_runner.sharegpt_adapter import ShareGPTAdapter


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def _source(tmp_path):
    source = tmp_path / "sharegpt.json"
    source.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": "reference answer"},
                    ]
                }
                for prompt in ("one", "one two", "one two three", "one two")
            ]
        ),
        encoding="utf-8",
    )
    return source


def test_filter_before_sample_cap_and_use_fixed_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sharegpt_to_guidellm, "load_tokenizer", lambda _: Tokenizer())
    output = tmp_path / "converted.jsonl"
    stats = sharegpt_to_guidellm.convert_sharegpt_to_guidellm(
        _source(tmp_path),
        output,
        "fake-tokenizer",
        max_items=2,
        input_min=2,
        input_max=2,
        output_tokens=64,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["text"] for record in records] == ["one two", "one two"]
    assert [record["output_tokens_count"] for record in records] == [64, 64]
    assert stats["written"] == 2


@pytest.mark.parametrize(
    "minimum,maximum,expected_prompts",
    [
        (2, 2, ["one two", "one two"]),
        (2, None, ["one two", "one two three", "one two"]),
        (None, 2, ["one", "one two", "one two"]),
    ],
)
def test_input_limits_without_output_override_use_answer_length(
    tmp_path, monkeypatch, minimum, maximum, expected_prompts
):
    monkeypatch.setattr(sharegpt_to_guidellm, "load_tokenizer", lambda _: Tokenizer())
    output = tmp_path / "converted.jsonl"
    sharegpt_to_guidellm.convert_sharegpt_to_guidellm(
        _source(tmp_path),
        output,
        "fake-tokenizer",
        input_min=minimum,
        input_max=maximum,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["text"] for record in records] == expected_prompts
    assert [record["output_tokens_count"] for record in records] == [2] * len(
        expected_prompts
    )


def test_empty_filter_fails_before_writing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sharegpt_to_guidellm, "load_tokenizer", lambda _: Tokenizer())
    output = tmp_path / "converted.jsonl"
    with pytest.raises(ValueError, match="No ShareGPT samples"):
        sharegpt_to_guidellm.convert_sharegpt_to_guidellm(
            _source(tmp_path), output, "fake-tokenizer", input_min=10
        )
    assert not output.exists()


def test_cache_separates_filters_and_grows_when_more_samples_are_needed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sharegpt_to_guidellm, "load_tokenizer", lambda _: Tokenizer())
    source = _source(tmp_path)
    adapter = ShareGPTAdapter()
    first = adapter.prepare(
        str(source), tokenizer="fake-tokenizer", max_items=1, input_min=2
    )[0]
    larger = adapter.prepare(
        str(source), tokenizer="fake-tokenizer", max_items=3, input_min=2
    )[0]
    assert larger == first
    assert len(Path(larger).read_text(encoding="utf-8").splitlines()) == 3
    other = adapter.prepare(
        str(source), tokenizer="fake-tokenizer", max_items=1, input_max=2
    )[0]
    assert other != first
