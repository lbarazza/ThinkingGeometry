from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"

STEP_PATTERN = re.compile(
    r"(?im)^[ \t]*step[ \t]+(\d+)[ \t]*(?=[:.)\-]|\n|$)"
)

FINAL_ANSWER_PATTERN = re.compile(
    r"(?im)^[ \t]*####[ \t]*final[ \t]+answer[ \t]*$"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def token_before_last_token_before_span(
    offsets: list[tuple[int, int]],
    char_start: int,
) -> int | None:
    token_before_previous_token_index = None
    previous_token_index = None

    for token_index, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= char_start:
            token_before_previous_token_index = previous_token_index
            previous_token_index = token_index
            continue

        break

    return token_before_previous_token_index


def find_step_positions(
    chat_prompt: str,
    model_output: str,
    tokenizer: AutoTokenizer,
) -> list[dict[str, Any]]:
    full_text = chat_prompt + model_output
    output_char_offset = len(chat_prompt)

    encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    step_positions = []

    for match in STEP_PATTERN.finditer(model_output):
        char_start = output_char_offset + match.start()
        char_end = output_char_offset + match.end()
        marker_step_number = int(match.group(1))

        full_token_index = token_before_last_token_before_span(
            offsets=offsets,
            char_start=char_start,
        )

        if full_token_index is None:
            continue

        token_id = token_ids[full_token_index]

        step_positions.append(
            {
                "step_number": marker_step_number - 1,
                "marker_step_number": marker_step_number,
                "marker_matched_text": match.group(0),
                "marker_char_start": char_start,
                "marker_char_end": char_end,
                "full_token_index": full_token_index,
                "token_id": token_id,
                "token_text": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                ),
            }
        )

    for match in FINAL_ANSWER_PATTERN.finditer(model_output):
        char_start = output_char_offset + match.start()
        char_end = output_char_offset + match.end()

        full_token_index = token_before_last_token_before_span(
            offsets=offsets,
            char_start=char_start,
        )

        if full_token_index is None:
            continue

        token_id = token_ids[full_token_index]

        step_positions.append(
            {
                "step_number": -1,
                "marker_step_number": -1,
                "marker_matched_text": match.group(0),
                "marker_char_start": char_start,
                "marker_char_end": char_end,
                "full_token_index": full_token_index,
                "token_id": token_id,
                "token_text": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                ),
            }
        )

    for i, pos in enumerate(step_positions):
        pos["manual_step_number"] = i if pos["step_number"] != -1 else -1

    return step_positions


def annotate_record(
    record: dict[str, Any],
    tokenizer: AutoTokenizer,
    source_response_path: Path,
) -> dict[str, Any]:
    base = {
        "id": record["id"],
        "source_response_path": str(source_response_path),
        "model_id": record.get("model_id", DEFAULT_MODEL_ID),
        "num_prompt_tokens": record.get("num_prompt_tokens"),
        "num_output_tokens": record.get("num_output_tokens"),
    }

    if record.get("error") is not None:
        return {
            **base,
            "step_positions": [],
            "error": record["error"],
        }

    chat_prompt = record.get("chat_prompt")
    model_output = record.get("model_output")

    if chat_prompt is None:
        return {
            **base,
            "step_positions": [],
            "error": "Missing chat_prompt",
        }

    if model_output is None:
        return {
            **base,
            "step_positions": [],
            "error": "Missing model_output",
        }

    return {
        **base,
        "step_positions": find_step_positions(
            chat_prompt=chat_prompt,
            model_output=model_output,
            tokenizer=tokenizer,
        ),
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract token positions of Step i headings from Gemma responses."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Generated response JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output annotation JSONL file.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional tokenizer model id. Defaults to the first model_id in the input.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_jsonl(args.input)

    model_id = args.model_id
    if model_id is None:
        model_id = next(
            (
                record.get("model_id")
                for record in records
                if record.get("model_id") is not None
            ),
            DEFAULT_MODEL_ID,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            annotated = annotate_record(
                record=record,
                tokenizer=tokenizer,
                source_response_path=args.input,
            )
            f.write(json.dumps(annotated, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} annotations to {args.output}")


if __name__ == "__main__":
    main()
