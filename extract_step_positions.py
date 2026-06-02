"""
For each record in a responses JSONL, find the last non-whitespace token
before each "Step N:" marker and before the final answer "####" marker.
Writes one JSON line per record to data/annotations/step_positions.jsonl.
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


SKIP_TOKENS = {"**", "*", "__", "_"}

def last_nonws_token_before(tok_idx, full_ids, tokenizer):
    """Return the index of the last content token before tok_idx, skipping whitespace and markdown punctuation."""
    i = tok_idx - 1
    while i >= 0:
        text = tokenizer.decode([full_ids[i]]).strip()
        if text != "" and text not in SKIP_TOKENS:
            return i
        i -= 1
    return None


def find_char_to_tok(offset_map, char_pos):
    """Return the index of the first token whose span covers char_pos."""
    for i, (s, e) in enumerate(offset_map):
        if s <= char_pos < e:
            return i
    return None


def process_record(record, tokenizer):
    prompt_ids = record.get("prompt_token_ids")
    output_ids = record.get("output_token_ids")
    if not prompt_ids or not output_ids:
        return []

    full_ids = prompt_ids + output_ids
    full_text = tokenizer.decode(full_ids)

    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offset_map = enc["offset_mapping"]

    steps = []

    # Character offset where the output section begins (skip prompt)
    n_prompt = len(prompt_ids)
    output_char_start = offset_map[n_prompt][0] if n_prompt < len(offset_map) else len(full_text)

    # Step N: markers — first occurrence of each N, within output only
    seen_ns = {}
    for m in re.finditer(r'(?<!\w)Step (\d+):', full_text):
        if m.start() < output_char_start:
            continue
        n = int(m.group(1))
        if n not in seen_ns:
            seen_ns[n] = m.start()

    for manual_step_number, (n, char_pos) in enumerate(sorted(seen_ns.items())):
        tok_idx = find_char_to_tok(offset_map, char_pos)
        if tok_idx is None:
            continue
        target = last_nonws_token_before(tok_idx, full_ids, tokenizer)
        if target is not None and target >= n_prompt:
            steps.append({"manual_step_number": manual_step_number, "full_token_index": target})

    # Final answer marker #### — search within output only
    ans_pos = full_text.find("####", output_char_start)
    if ans_pos != -1:
        tok_idx = find_char_to_tok(offset_map, ans_pos)
        if tok_idx is not None:
            target = last_nonws_token_before(tok_idx, full_ids, tokenizer)
            if target is not None:
                steps.append({"manual_step_number": -1, "full_token_index": target})

    return steps, len(full_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to responses_scored.jsonl")
    parser.add_argument("--model-id", default="google/gemma-4-E4B-it")
    parser.add_argument("--output-dir", default="data/annotations/step_positions")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model_slug = args.model_id.split("/")[-1].lower()
    run_slug = Path(args.input).parent.name
    out_path = Path(args.output_dir) / model_slug / f"{run_slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    total = skipped = 0

    with open(input_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            total += 1

            if record.get("error") or not record.get("prompt_token_ids") or not record.get("output_token_ids"):
                skipped += 1
                fout.write(json.dumps({"id": record["id"], "num_tokens": 0, "steps": []}) + "\n")
                continue

            result = process_record(record, tokenizer)
            if result is None:
                skipped += 1
                fout.write(json.dumps({"id": record["id"], "num_tokens": 0, "steps": []}) + "\n")
                continue

            steps, num_tokens = result
            fout.write(json.dumps({"id": record["id"], "num_tokens": num_tokens, "steps": steps}) + "\n")

    print(f"Processed {total} records ({skipped} skipped). Written to {out_path}")


if __name__ == "__main__":
    main()
